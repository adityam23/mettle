use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use turbovec::IdMapIndex;

use crate::error::CoreError;

// ---------------------------------------------------------------------------
// Document / Chunk types
// ---------------------------------------------------------------------------

/// Embedding callback: chunk texts in, flat `[n_chunks * dim]` f32 array out.
pub type EmbedFn = dyn Fn(&[String]) -> Result<Vec<f32>, CoreError>;

/// A document submitted for ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// One stored document as reported by the list endpoint: the id is the
/// consumer's reconciliation key, `chunk_count` its footprint here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub id: String,
    pub chunk_count: u64,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// A chunk produced by splitting a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub text: String,
    pub char_offset: usize,
}

/// A scored search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChunk {
    pub chunk_id: String,
    pub document_id: String,
    pub text: String,
    pub score: f64,
    #[serde(default)]
    pub source_span: Option<(usize, usize)>,
}

// ---------------------------------------------------------------------------
// Search mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Vector-only: turbovec nearest neighbours within the allowlist.
    Dense,
    /// Lexical-only: FTS5 `MATCH` over OR-joined query tokens, ranked by
    /// `bm25()` (higher score = better; returned scores are negated bm25).
    /// No embedding is computed.
    Keyword,
    /// Head-preserving candidate fusion: the dense top-10 ordering is pinned,
    /// then Reciprocal Rank Fusion (k=60, 1:1) over the dense top-40 and the
    /// keyword top-40 fills ranks 11+. Allowlist enforced on both lists before
    /// fusion. Breaking change vs. the pre-2026-09-10 FTS5-candidates ∩
    /// dense-rerank semantics; the pinning is a measured deviation from pure
    /// RRF — see the `search()` notes.
    #[default]
    Hybrid,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SearchMode::Dense => "dense",
            SearchMode::Keyword => "keyword",
            SearchMode::Hybrid => "hybrid",
        })
    }
}

// ---------------------------------------------------------------------------
// Collection search request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub mode: SearchMode,
    #[serde(default)]
    pub allowlist: Option<Vec<String>>,
    #[serde(default)]
    pub filter: Option<HashMap<String, String>>,
}

fn default_k() -> usize {
    10
}

/// L2-normalize one embedding in place. Zero (or near-zero) vectors stay
/// zero so the no-engine fallback keeps its current behaviour.
fn l2_normalize(v: &mut [f32]) {
    let sq: f64 = v.iter().map(|&x| x as f64 * x as f64).sum();
    let norm = sq.sqrt();
    if norm > 1e-10 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// RAG chat request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct RagChatRequest {
    pub model: String,
    pub messages: Vec<crate::ollama_types::ChatMessage>,
    pub collections: Vec<String>,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<HashMap<String, serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// Citation in RAG response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Citation {
    pub chunk_id: String,
    pub document_id: String,
    pub source_span: Option<(usize, usize)>,
    pub retrieval_score: f64,
}

/// Response for /v1/rag/chat — a chat response annotated with retrieval citations.
#[derive(Debug, Clone, Serialize)]
pub struct RagChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: crate::ollama_types::ResponseMessage,
    pub done_reason: String,
    pub done: bool,
    pub citations: Vec<Citation>,
    #[serde(flatten)]
    pub timing: crate::ollama_types::TimingInfo,
}

/// Streaming chunk for /v1/rag/chat. Final chunk carries citations.
#[derive(Debug, Clone, Serialize)]
pub struct RagChatStreamChunk {
    pub model: String,
    pub created_at: String,
    pub message: crate::ollama_types::ChatStreamMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<Citation>>,
}

// ---------------------------------------------------------------------------
// Collection store (Sqlite + turbovec)
// ---------------------------------------------------------------------------

/// Manages all named collections. Each collection has its own turbovec
/// `IdMapIndex` and SQLite database for document/chunk metadata + FTS5.
pub struct CollectionStore {
    collections: HashMap<String, Arc<Mutex<Collection>>>,
    data_dir: PathBuf,
    default_bit_width: usize,
}

/// Maximum collection-name length (bytes; the allowed charset is ASCII).
const MAX_COLLECTION_NAME_LEN: usize = 128;

/// Collection names become directory names under `<data_dir>/collections/`,
/// so they are restricted to `[A-Za-z0-9._-]{1,128}` with no leading dot —
/// no separators, no `..`, no hidden directories. Document ids are NOT
/// constrained by this: they are only ever SQLite keys.
pub fn validate_collection_name(name: &str) -> Result<(), CoreError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_COLLECTION_NAME_LEN
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(CoreError::InvalidRequest(format!(
            "invalid collection name '{name}': must match [A-Za-z0-9._-]{{1,{MAX_COLLECTION_NAME_LEN}}} \
             and must not start with '.'"
        )))
    }
}

impl CollectionStore {
    pub fn new(data_dir: PathBuf, default_bit_width: usize) -> Self {
        Self {
            collections: HashMap::new(),
            data_dir,
            default_bit_width,
        }
    }

    /// Get a handle to an existing collection.
    pub fn get(&self, name: &str) -> Option<&Arc<Mutex<Collection>>> {
        self.collections.get(name)
    }

    pub fn create(
        &mut self,
        name: &str,
        embedding_model: &str,
        dim: usize,
    ) -> Result<&Arc<Mutex<Collection>>, CoreError> {
        validate_collection_name(name)?;
        if self.collections.contains_key(name) {
            return Err(CoreError::Conflict(format!(
                "collection '{}' already exists",
                name
            )));
        }

        let coll_dir = self.data_dir.join("collections").join(name);
        std::fs::create_dir_all(&coll_dir).map_err(|e| {
            CoreError::Retrieval(format!("failed to create collection directory: {}", e))
        })?;

        let collection = Collection::new(
            name,
            dim,
            embedding_model,
            &coll_dir,
            self.default_bit_width,
        )?;

        let handle = Arc::new(Mutex::new(collection));
        self.collections
            .insert(name.to_string(), Arc::clone(&handle));
        Ok(self.collections.get(name).unwrap())
    }

    /// Delete a collection and its data. Only names of loaded collections
    /// are ever turned into a path, so an arbitrary (e.g. percent-decoded
    /// `../..`) name can never reach `remove_dir_all`.
    pub fn delete(&mut self, name: &str) -> Result<(), CoreError> {
        if !self.collections.contains_key(name) {
            return Err(CoreError::CollectionNotFound(name.to_string()));
        }
        let coll_dir = self.data_dir.join("collections").join(name);
        if coll_dir.exists() {
            std::fs::remove_dir_all(&coll_dir).map_err(|e| {
                CoreError::Retrieval(format!("failed to delete collection data: {}", e))
            })?;
        }
        self.collections.remove(name);
        Ok(())
    }

    /// List all collection names.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.collections.keys().cloned().collect();
        names.sort();
        names
    }

    /// Load all existing collections from disk on startup.
    pub fn load_all(&mut self) -> Result<usize, CoreError> {
        let colls_dir = self.data_dir.join("collections");
        if !colls_dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        for entry in std::fs::read_dir(&colls_dir)
            .map_err(|e| CoreError::Retrieval(format!("failed to read collections dir: {}", e)))?
        {
            let entry = entry
                .map_err(|e| CoreError::Retrieval(format!("failed to read dir entry: {}", e)))?;
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Err(e) = validate_collection_name(&name) {
                    tracing::warn!("skipping collection directory: {e}");
                    continue;
                }
                let coll = Collection::load(&name, &entry.path())?;
                self.collections.insert(name, Arc::new(Mutex::new(coll)));
                count += 1;
            }
        }
        Ok(count)
    }

    /// Flush all collections to disk.
    pub fn flush_all(&self) -> Result<(), CoreError> {
        for coll in self.collections.values() {
            if let Ok(c) = coll.lock() {
                c.flush()?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// A single named collection
// ---------------------------------------------------------------------------

/// One named collection: a turbovec `IdMapIndex` + SQLite for metadata.
///
/// The index is stored on disk at `<data_dir>/index.bin` and the SQLite
/// database at `<data_dir>/metadata.db`.
pub struct Collection {
    pub name: String,
    pub dim: usize,
    pub embedding_model: String,
    data_dir: PathBuf,
    index: IdMapIndex,
    /// Counter for allocating sequential chunk IDs for the index.
    next_int_id: u64,
    /// Quantisation width of the turbovec index; needed to rebuild it.
    bit_width: usize,
    /// True when *every* vector in `index` was L2-normalised on write
    /// (`collection_meta.vectors_normalized = "1"`). A collection created
    /// before that change loads as false — its rows score `||v|| * cos(v, q)`
    /// — and the `rebuild_index` repair path is the only fix.
    vectors_normalized: bool,
}

impl Collection {
    /// Create a new, empty collection.
    pub fn new(
        name: &str,
        dim: usize,
        embedding_model: &str,
        data_dir: &Path,
        bit_width: usize,
    ) -> Result<Self, CoreError> {
        std::fs::create_dir_all(data_dir).map_err(|e| {
            CoreError::Retrieval(format!("failed to create collection directory: {}", e))
        })?;

        let index = IdMapIndex::new(dim, bit_width)
            .map_err(|e| CoreError::Retrieval(format!("failed to create turbovec index: {}", e)))?;

        // Initialize the SQLite schema.
        let db_path = data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS collection_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                int_id INTEGER NOT NULL UNIQUE,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                char_offset INTEGER NOT NULL DEFAULT 0
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text
            );
            ",
        )
        .map_err(|e| CoreError::Retrieval(format!("failed to init sqlite schema: {}", e)))?;

        conn.execute(
            "INSERT OR REPLACE INTO collection_meta (key, value) VALUES
                ('embedding_model', ?1), ('dim', ?2), ('bit_width', ?3),
                ('vectors_normalized', '1')",
            rusqlite::params![embedding_model, dim as i64, bit_width as i64],
        )
        .map_err(|e| CoreError::Retrieval(format!("failed to write collection meta: {}", e)))?;

        Ok(Self {
            name: name.to_string(),
            dim,
            embedding_model: embedding_model.to_string(),
            data_dir: data_dir.to_path_buf(),
            index,
            next_int_id: 1,
            bit_width,
            vectors_normalized: true,
        })
    }

    /// Load an existing collection from disk. The embedding dimension is
    /// persisted in `collection_meta` at creation; a collection without it
    /// is corrupt and refused (never silently rebuild an index at a wrong
    /// dim).
    pub fn load(name: &str, data_dir: &Path) -> Result<Self, CoreError> {
        let db_path = data_dir.join("metadata.db");
        if !db_path.exists() {
            return Err(CoreError::CollectionNotFound(format!(
                "collection '{}' has no metadata.db at {}",
                name,
                data_dir.display()
            )));
        }

        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;

        let meta = |key: &str| -> Option<String> {
            conn.query_row(
                "SELECT value FROM collection_meta WHERE key = ?1",
                [key],
                |row| row.get(0),
            )
            .ok()
        };
        let embedding_model = meta("embedding_model").unwrap_or_default();
        let dim = meta("dim").and_then(|v| v.parse().ok()).ok_or_else(|| {
            CoreError::Retrieval(format!("collection '{name}' has no dim in meta"))
        })?;
        let bit_width = meta("bit_width").and_then(|v| v.parse().ok()).unwrap_or(4);

        let index_path = data_dir.join("index.bin");
        let index = if index_path.exists() {
            IdMapIndex::load(&index_path).map_err(|e| {
                CoreError::Retrieval(format!("failed to load turbovec index: {}", e))
            })?
        } else {
            IdMapIndex::new(dim, bit_width).map_err(|e| {
                CoreError::Retrieval(format!("failed to create turbovec index: {}", e))
            })?
        };

        // The next int_id from existing chunks.
        let max_int_id: u64 = conn
            .query_row("SELECT COALESCE(MAX(int_id), 0) FROM chunks", [], |row| {
                row.get::<_, i64>(0).map(|v| v as u64)
            })
            .unwrap_or(0);

        // A missing stamp means the collection predates L2 normalisation
        // (or was written by the zero-vector fallback era): its dense/hybrid
        // scores carry a magnitude bias and only a whole-collection re-embed
        // can repair it. Warn once per collection, and only when there is
        // actually a stored vector to be biased by it.
        let vectors_normalized = meta("vectors_normalized").as_deref() == Some("1");
        if !vectors_normalized && !index.is_empty() {
            tracing::warn!(
                "collection '{}' was written before vectors were L2-normalised: \
                 dense and hybrid scores are scale-biased; repair it with \
                 POST /v1/collections/{}/reindex",
                name,
                name
            );
        }

        Ok(Self {
            name: name.to_string(),
            dim,
            embedding_model,
            data_dir: data_dir.to_path_buf(),
            index,
            next_int_id: max_int_id + 1,
            bit_width,
            vectors_normalized,
        })
    }

    // -----------------------------------------------------------------------
    // Document ingestion
    // -----------------------------------------------------------------------

    /// Add a document: chunk it, generate embeddings via `embed_fn`,
    /// store in turbovec and SQLite. Returns the chunk IDs created.
    ///
    /// `embed_fn` receives a slice of chunk texts and returns a flat
    /// `[f32; n_chunks * dim]` array of embeddings.
    pub fn add_document(
        &mut self,
        doc: &Document,
        embed_fn: &EmbedFn,
        chunk_tokens: usize,
        chunk_overlap: usize,
    ) -> Result<Vec<String>, CoreError> {
        let chunks = chunk_document(doc, chunk_tokens, chunk_overlap);
        let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = embed_fn(&chunk_texts)?;
        self.add_document_chunks(doc, &chunks, &embeddings)
    }

    /// Insert a pre-chunked, pre-embedded document. This is the path the
    /// server uses so embedding can happen asynchronously (no collection
    /// lock held across `.await`).
    pub fn add_document_chunks(
        &mut self,
        doc: &Document,
        chunks: &[Chunk],
        embeddings: &[f32],
    ) -> Result<Vec<String>, CoreError> {
        let per_doc = self.add_documents_chunks(&[(doc.clone(), chunks.to_vec())], embeddings)?;
        Ok(per_doc.into_iter().next().unwrap_or_default())
    }

    /// Insert a batch of pre-chunked, pre-embedded documents **all or
    /// nothing**: any failure (duplicate id — in the batch or already
    /// stored — a vector-count mismatch, a SQLite or turbovec error) leaves
    /// the collection byte-identical. One `add_with_ids`, one SQLite
    /// transaction, one caller-side flush.
    ///
    /// `embeddings` is the flat `[f32; total_chunks * dim]` array for every
    /// document in `docs`, in order. Chunk-less documents write nothing and
    /// come back with an empty id list.
    pub fn add_documents_chunks(
        &mut self,
        docs: &[(Document, Vec<Chunk>)],
        embeddings: &[f32],
    ) -> Result<Vec<Vec<String>>, CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;

        let total_chunks: usize = docs
            .iter()
            .filter(|(_, chunks)| !chunks.is_empty())
            .map(|(_, chunks)| chunks.len())
            .sum();
        if total_chunks == 0 {
            return Ok(docs.iter().map(|_| Vec::new()).collect());
        }
        if embeddings.len() != total_chunks * self.dim {
            return Err(CoreError::Retrieval(format!(
                "embed_fn returned {} floats, expected {} ({} chunks * dim {})",
                embeddings.len(),
                total_chunks * self.dim,
                total_chunks,
                self.dim
            )));
        }

        // Whether the whole index will be normalised once this batch lands:
        // true for a stamped collection, and for an unstamped one that has no
        // vector yet (nothing biased could be left behind). An unstamped
        // collection that *does* hold rows stays mixed — those legacy rows
        // keep their magnitude bias until `rebuild_index` replaces them, so
        // it must not be stamped here.
        let all_normalized_after_insert = self.vectors_normalized || self.index.is_empty();

        // Pre-flight every document: an id that already exists, or that
        // appears twice in the batch, fails the whole batch before a single
        // row is written.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (doc, _) in docs {
            if !seen.insert(doc.id.as_str()) {
                return Err(CoreError::Conflict(format!(
                    "document '{}' appears twice in the batch",
                    doc.id
                )));
            }
            let existing: bool = conn
                .query_row(
                    "SELECT COUNT(*) > 0 FROM documents WHERE id = ?1",
                    [&doc.id],
                    |row| row.get(0),
                )
                .unwrap_or(false);
            if existing {
                return Err(CoreError::Conflict(format!(
                    "document '{}' already exists in collection '{}'",
                    doc.id, self.name
                )));
            }
        }

        // turbovec scores estimate the inner product <v, q>; the served models
        // emit unnormalized embeddings, so normalize here — dense ranking must
        // be cosine or long high-norm chunks dominate regardless of similarity.
        let mut normalized = embeddings.to_vec();
        for row in normalized.chunks_exact_mut(self.dim) {
            l2_normalize(row);
        }

        // Atomic all-or-nothing: every SQLite row in ONE transaction, every
        // vector in ONE turbovec insert, then commit. Any failure before the
        // commit rolls the rows back (the transaction drops) and a failed
        // commit removes the vectors again — no half-written batch survives.
        // One index write per batch is also what makes a ~1000-document
        // ingest feasible: the single-document path rewrites the whole
        // index.bin on every call.
        let base_id = self.next_int_id;
        let int_ids: Vec<u64> = (base_id..base_id + total_chunks as u64).collect();

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| CoreError::Retrieval(format!("sqlite begin: {}", e)))?;

        let mut per_doc = Vec::with_capacity(docs.len());
        let mut next_int = 0usize;
        for (doc, chunks) in docs {
            // A chunk-less document writes no row at all (single-path parity).
            if chunks.is_empty() {
                per_doc.push(Vec::new());
                continue;
            }
            tx.execute(
                "INSERT INTO documents (id, metadata_json) VALUES (?1, ?2)",
                rusqlite::params![
                    doc.id,
                    serde_json::to_string(&doc.metadata).unwrap_or_default()
                ],
            )
            .map_err(|e| CoreError::Retrieval(format!("sqlite insert document: {}", e)))?;

            let mut chunk_ids = Vec::with_capacity(chunks.len());
            for (i, chunk) in chunks.iter().enumerate() {
                let chunk_id = format!("{}/chunk/{}", doc.id, i);
                tx.execute(
                    "INSERT INTO chunks (id, int_id, document_id, text, char_offset) VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![
                        chunk_id,
                        int_ids[next_int] as i64,
                        doc.id,
                        chunk.text,
                        chunk.char_offset as i64,
                    ],
                )
                .map_err(|e| CoreError::Retrieval(format!("sqlite insert chunk: {}", e)))?;

                tx.execute(
                    "INSERT INTO chunks_fts (chunk_id, text) VALUES (?1, ?2)",
                    rusqlite::params![chunk_id, chunk.text],
                )
                .map_err(|e| CoreError::Retrieval(format!("sqlite fts insert: {}", e)))?;

                chunk_ids.push(chunk_id);
                next_int += 1;
            }
            per_doc.push(chunk_ids);
        }

        self.index
            .add_with_ids(&normalized, &int_ids)
            .map_err(|e| CoreError::Retrieval(format!("turbovec add failed: {}", e)))?;

        // The rows above are normalised, so stamp the collection inside the
        // same transaction: a failed commit then leaves neither the rows nor
        // the flag behind.
        if all_normalized_after_insert {
            tx.execute(
                "INSERT OR REPLACE INTO collection_meta (key, value) \
                 VALUES ('vectors_normalized', '1')",
                [],
            )
            .map_err(|e| CoreError::Retrieval(format!("sqlite stamp vectors_normalized: {}", e)))?;
        }

        if let Err(e) = tx.commit() {
            for id in &int_ids {
                self.index.remove(*id);
            }
            return Err(CoreError::Retrieval(format!("sqlite commit: {}", e)));
        }
        self.next_int_id += total_chunks as u64;
        if all_normalized_after_insert {
            self.vectors_normalized = true;
        }

        Ok(per_doc)
    }

    /// Every document in the collection with its chunk count and metadata.
    /// The ids are the reconciliation key for a consumer tracking its own
    /// lineage (list endpoint).
    pub fn list_documents(&self) -> Result<Vec<DocumentSummary>, CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;
        let mut stmt = conn
            .prepare(
                "SELECT d.id, d.metadata_json, \
                 (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) \
                 FROM documents d ORDER BY d.id",
            )
            .map_err(|e| CoreError::Retrieval(format!("sqlite query: {}", e)))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(|e| CoreError::Retrieval(format!("sqlite query: {}", e)))?;
        let mut out = Vec::new();
        for row in rows {
            let (id, metadata_json, chunk_count) =
                row.map_err(|e| CoreError::Retrieval(format!("sqlite row: {}", e)))?;
            out.push(DocumentSummary {
                id,
                chunk_count: chunk_count.max(0) as u64,
                metadata: serde_json::from_str(&metadata_json).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// Delete a document and all its chunks.
    pub fn delete_document(&mut self, doc_id: &str) -> Result<(), CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;

        // Collect integer IDs to remove from turbovec.
        let mut stmt = conn
            .prepare("SELECT int_id FROM chunks WHERE document_id = ?1")
            .map_err(|e| CoreError::Retrieval(format!("sqlite query: {}", e)))?;

        let int_ids: Vec<u64> = stmt
            .query_map([doc_id], |row| row.get::<_, i64>(0).map(|v| v as u64))
            .map_err(|e| CoreError::Retrieval(format!("sqlite query: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        // FTS rows must go first: the subquery reads `chunks`.
        conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?1)",
            [doc_id],
        )
        .map_err(|e| CoreError::Retrieval(format!("fts delete: {}", e)))?;

        conn.execute("DELETE FROM chunks WHERE document_id = ?1", [doc_id])
            .and_then(|_| conn.execute("DELETE FROM documents WHERE id = ?1", [doc_id]))
            .map_err(|e| CoreError::Retrieval(format!("sqlite delete: {}", e)))?;

        // Remove from turbovec (O(1) per id).
        for id in &int_ids {
            self.index.remove(*id);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search the collection. `query_embedding` is the dense embedding of
    /// the query text (dim must match `self.dim`). `allowlist` is an optional
    /// caller-supplied set of document IDs; results outside it are excluded
    /// (in-kernel for the dense leg via turbovec's filtered search, and pushed
    /// into the FTS predicate for the keyword/hybrid leg so the keyword `LIMIT`
    /// counts in-scope rows). `Some(empty)` means nothing is eligible.
    pub fn search(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        mode: SearchMode,
        allowlist: Option<&[String]>,
    ) -> Result<Vec<ScoredChunk>, CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;

        // Resolve the document-id allowlist to index int_ids.
        let allow_ids: Option<Vec<u64>> = match allowlist {
            None => None,
            Some(doc_ids) => {
                let mut stmt = conn
                    .prepare("SELECT int_id FROM chunks WHERE document_id = ?1")
                    .map_err(|e| CoreError::Retrieval(format!("allowlist query: {}", e)))?;
                let mut ids = Vec::new();
                for doc_id in doc_ids {
                    let rows = stmt
                        .query_map([doc_id], |row| row.get::<_, i64>(0).map(|v| v as u64))
                        .map_err(|e| CoreError::Retrieval(format!("allowlist query: {}", e)))?;
                    ids.extend(rows.filter_map(|r| r.ok()));
                }
                Some(ids)
            }
        };

        // `Some(empty)` means nothing is eligible: no document id resolved to a
        // chunk (or the caller passed an empty list). Return early because
        // turbovec's in-kernel filter panics on an empty allowlist; the keyword
        // leg would yield no rows anyway (an empty `json_each` is empty).
        if matches!(&allow_ids, Some(ids) if ids.is_empty()) {
            return Ok(Vec::new());
        }

        // Queries are normalized to the same cosine space as stored vectors.
        let mut q = query_embedding.to_vec();
        l2_normalize(&mut q);

        match mode {
            SearchMode::Dense => {
                let (scores, ids) = self
                    .index
                    .search_with_allowlist(&q, k, allow_ids.as_deref());
                self.lookup_chunks(&conn, &ids, &scores)
            }
            SearchMode::Keyword => {
                // Adoption-contract §1.1 (2026-09-10): OR-mode FTS5 + bm25()
                // ranking. The old AND-expansion measured 0.000 recall on the
                // parity benchmark (FTS5 AND of long natural questions matches
                // nothing); this recipe measures .487@10 vs docket's
                // rank_bm25 .380.
                //
                // Tokenize with the metric's rule: lowercase [a-z0-9]+ runs,
                // OR-joined into the MATCH, no other sanitization. FTS5's
                // AND/OR/NOT operators are uppercase-only, so lowercase
                // [a-z0-9]+ barewords cannot inject operators.
                //
                // Disclosed caveats: FTS5's unicode61 tokenizer vs the
                // metric's [a-z0-9]+ (underscore/unicode edge cases), and
                // indexed text carries the ingest-side chunk cap.
                let tokens = keyword_tokens(query_text);
                if tokens.is_empty() {
                    return Ok(Vec::new());
                }
                let fts_query = tokens.join(" OR ");

                // Scope pushdown: with an allowlist the document filter runs
                // inside SQLite, so `ORDER BY bm25` and `LIMIT` count in-scope
                // rows only. Filtering *after* a global LIMIT silently drops
                // every in-scope keyword hit that ranks below the global pool
                // (the hybrid leg's bug). `None` keeps the original statement.
                let allow_json = allowlist_json(allowlist)?;
                let sql = if allow_json.is_some() {
                    "SELECT c.id, c.document_id, c.text, c.char_offset, bm25(chunks_fts)
                     FROM chunks_fts fts
                     JOIN chunks c ON c.id = fts.chunk_id
                     WHERE chunks_fts MATCH ?1
                       AND c.document_id IN (SELECT value FROM json_each(?2))
                     ORDER BY bm25(chunks_fts)
                     LIMIT ?3"
                } else {
                    "SELECT c.id, c.document_id, c.text, c.char_offset, bm25(chunks_fts)
                     FROM chunks_fts fts
                     JOIN chunks c ON c.id = fts.chunk_id
                     WHERE chunks_fts MATCH ?1
                     ORDER BY bm25(chunks_fts)"
                };
                let mut stmt = conn
                    .prepare(sql)
                    .map_err(|e| CoreError::Retrieval(format!("fts query: {}", e)))?;

                // bm25() is smaller-is-better (SQLite returns negative
                // relevance); negate so score is higher-is-better like dense.
                let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<ScoredChunk> {
                    let bm25: f64 = row.get(4)?;
                    Ok(ScoredChunk {
                        chunk_id: row.get(0)?,
                        document_id: row.get(1)?,
                        text: row.get(2)?,
                        score: -bm25,
                        source_span: Some((row.get::<_, i64>(3)? as usize, 0)),
                    })
                };
                let rows = if let Some(json) = &allow_json {
                    stmt.query_map(rusqlite::params![&fts_query, json, k as i64], map_row)
                } else {
                    stmt.query_map([&fts_query], map_row)
                }
                .map_err(|e| CoreError::Retrieval(format!("fts query: {}", e)))?;

                let results: Vec<ScoredChunk> = rows.filter_map(|r| r.ok()).take(k).collect();

                Ok(results)
            }
            SearchMode::Hybrid => {
                // Adoption-contract §1.2 (2026-09-10), BREAKING semantic
                // change: the old FTS5-candidates ∩ dense-rerank collapsed to
                // dense (hybrid == dense .787@10 both) and starved the
                // reranker of candidate diversity (rerank_hybrid .867 vs
                // docket's diverse-pool .913).
                //
                // Shipped geometry (user decision 2026-09-11, option C of the
                // measured frontier): the dense top-HEAD ordering is pinned
                // first, and Reciprocal Rank Fusion (k=60, 1:1) over the
                // remaining dense top-POOL and the keyword top-POOL
                // (OR+bm25, §1.1) fills the tail. Score-scale-free;
                // fused top-k.
                //
                // DISCLOSED DEVIATION from the grilled pure-RRF spec, with
                // the measured reason (offline replay of all 150 questions;
                // benchmarks/retrieval_parity/fusion_probe.py,
                // diversity_probe.py, fusion_composition.py,
                // fusion_cell_search.py, fusion_head_probe.py):
                //
                // 1. The spec'd docket-identical equal-weight fusion
                //    (top-40/40, 1:1, no head pin) SERVED hybrid .7200@10 —
                //    below its own ≥.787 gate, because OR/bm25 evidence
                //    (keyword .487 vs dense .787) outvotes dense order AND a
                //    dual-list chunk earns BOTH RRF terms, so it jumps into
                //    the top-10 and displaces dense head golds.
                // 2. Suppressing the keyword weight instead (dense 2.0 :
                //    keyword 0.5, top-80 pools) SERVED hybrid .8067@10 —
                //    but puts ZERO keyword-only candidates in the fused
                //    top-40 (the apparent novelty was dense ranks 40-79
                //    leaking through the deeper dense pool, not lexical
                //    diversity), and SERVED rerank_hybrid .8800, missing the
                //    contract's ≥.90 gate: the rerank pool is compositionally
                //    plain dense.
                // 3. Pinning the dense head decouples the two gates: stage-1
                //    top-10 recall is exactly the dense mode's own ordering
                //    (no keyword-induced head churn), while the tail carries
                //    the lexical candidates the reranker needs. Offline the
                //    fused top-40 matches the pool composition that measured
                //    rerank_hybrid .9067 when served (≈10 keyword-only
                //    chunks vs 0.0 in (2)). HEAD=10 is also the recall-optimal
                //    pin: h8/h5/h0 measured .7733/.7467/.7400 offline vs
                //    .7933, because every extra fused head position risks
                //    displacing a dense gold with a keyword one.
                //
                // The allowlist is enforced on BOTH candidate lists before
                // fusion: keyword rows by document id *inside* the SQL
                // statement (so the keyword `LIMIT` counts in-scope rows, not
                // global ones), dense in-kernel via turbovec.
                const RRF_K: f64 = 60.0;
                const POOL: usize = 40;
                const HEAD: usize = 10;
                const DENSE_W: f64 = 1.0;
                const KEYWORD_W: f64 = 1.0;
                // Never shrink a large-k request below the historical pool:
                // the measured geometry is k=40, larger k only widens it.
                let pool = POOL.max(k);

                let tokens = keyword_tokens(query_text);
                let mut rrf: HashMap<u64, f64> = HashMap::new();
                if !tokens.is_empty() {
                    let fts_query = tokens.join(" OR ");
                    // Same scope pushdown as the keyword-only leg: the document
                    // filter is part of the statement, so `LIMIT pool` is the
                    // in-scope top-`pool`. Taking the global top-`pool` and
                    // filtering afterwards empties the keyword leg whenever the
                    // scope's hits all rank below it (hybrid degenerates to
                    // dense-only).
                    let allow_json = allowlist_json(allowlist)?;
                    let sql = if allow_json.is_some() {
                        "SELECT c.int_id, c.document_id
                         FROM chunks_fts fts
                         JOIN chunks c ON c.id = fts.chunk_id
                         WHERE chunks_fts MATCH ?1
                           AND c.document_id IN (SELECT value FROM json_each(?2))
                         ORDER BY bm25(chunks_fts)
                         LIMIT ?3"
                    } else {
                        "SELECT c.int_id, c.document_id
                         FROM chunks_fts fts
                         JOIN chunks c ON c.id = fts.chunk_id
                         WHERE chunks_fts MATCH ?1
                         ORDER BY bm25(chunks_fts)
                         LIMIT ?2"
                    };
                    let mut stmt = conn
                        .prepare(sql)
                        .map_err(|e| CoreError::Retrieval(format!("fts hybrid query: {}", e)))?;

                    let pool = pool as i64;
                    let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<(u64, String)> {
                        Ok((row.get::<_, i64>(0)? as u64, row.get::<_, String>(1)?))
                    };
                    let rows: Vec<(u64, String)> = if let Some(json) = &allow_json {
                        stmt.query_map(rusqlite::params![&fts_query, json, pool], map_row)
                    } else {
                        stmt.query_map(rusqlite::params![&fts_query, pool], map_row)
                    }
                    .map_err(|e| CoreError::Retrieval(format!("fts hybrid query: {}", e)))?
                    .filter_map(|r| r.ok())
                    .collect();
                    for (rank, (int_id, _)) in rows.into_iter().enumerate() {
                        *rrf.entry(int_id).or_default() += KEYWORD_W / (RRF_K + rank as f64 + 1.0);
                    }
                }

                // Dense leg (allowlist enforced in-kernel; empty keyword side
                // just leaves a single-list RRF = dense ranking).
                let (_, d_ids) = self
                    .index
                    .search_with_allowlist(&q, pool, allow_ids.as_deref());
                for (rank, &id) in d_ids.iter().enumerate() {
                    *rrf.entry(id).or_default() += DENSE_W / (RRF_K + rank as f64 + 1.0);
                }

                // Pin the dense head, then fill from the fused ordering. The
                // head's RRF scores are read before `rrf` is consumed.
                let head: Vec<u64> = d_ids.iter().copied().take(HEAD).collect();
                let head_set: std::collections::HashSet<u64> = head.iter().copied().collect();
                let mut ordered: Vec<(u64, f64)> = head
                    .iter()
                    .map(|&id| (id, rrf.get(&id).copied().unwrap_or(0.0)))
                    .collect();

                let mut fused: Vec<(u64, f64)> = rrf.into_iter().collect();
                // Score desc; int_id asc tie-break for determinism.
                fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                ordered.extend(fused.into_iter().filter(|(id, _)| !head_set.contains(id)));
                ordered.truncate(k);
                let ids: Vec<u64> = ordered.iter().map(|(id, _)| *id).collect();
                let scores: Vec<f32> = ordered.iter().map(|(_, s)| *s as f32).collect();
                self.lookup_chunks(&conn, &ids, &scores)
            }
        }
    }

    /// Look up chunk metadata from SQLite by integer IDs.
    fn lookup_chunks(
        &self,
        conn: &rusqlite::Connection,
        int_ids: &[u64],
        scores: &[f32],
    ) -> Result<Vec<ScoredChunk>, CoreError> {
        let mut results = Vec::with_capacity(int_ids.len());
        for (i, &int_id) in int_ids.iter().enumerate() {
            let mut stmt = conn
                .prepare("SELECT id, document_id, text, char_offset FROM chunks WHERE int_id = ?1")
                .map_err(|e| CoreError::Retrieval(format!("chunk lookup: {}", e)))?;

            let chunk: Option<ScoredChunk> = stmt
                .query_row([int_id as i64], |row| {
                    Ok(ScoredChunk {
                        chunk_id: row.get(0)?,
                        document_id: row.get(1)?,
                        text: row.get(2)?,
                        score: scores.get(i).copied().unwrap_or(0.0) as f64,
                        source_span: Some((row.get::<_, i64>(3)? as usize, 0)),
                    })
                })
                .ok();

            if let Some(c) = chunk {
                results.push(c);
            }
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Flush the turbovec index to disk.
    pub fn flush(&self) -> Result<(), CoreError> {
        self.write_index(&self.index)
    }

    /// Write an index (usually the live one) to `<data_dir>/index.bin`.
    fn write_index(&self, index: &IdMapIndex) -> Result<(), CoreError> {
        let path = self.data_dir.join("index.bin");
        index
            .write(&path)
            .map_err(|e| CoreError::Retrieval(format!("failed to write index: {}", e)))
    }

    /// Stamp `collection_meta` as holding only L2-normalised vectors. Written
    /// by every path that normalises (`new`, the insert paths, `rebuild_index`)
    /// so `load` can tell a repaired collection from a legacy one.
    fn stamp_normalized(&self) -> Result<(), CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;
        conn.execute(
            "INSERT OR REPLACE INTO collection_meta (key, value) \
             VALUES ('vectors_normalized', '1')",
            [],
        )
        .map_err(|e| CoreError::Retrieval(format!("failed to stamp vectors_normalized: {}", e)))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Reindex (repair an un-normalised / zero-vector collection)
    // -----------------------------------------------------------------------

    /// True when this collection's stored vectors predate L2 normalisation
    /// (or come from the zero-vector fallback era): dense/hybrid scores carry
    /// a scale bias and [`Self::rebuild_index`] is the repair. A collection
    /// with no stored vector has nothing biased, so it needs no reindex even
    /// when unstamped.
    pub fn needs_reindex(&self) -> bool {
        !self.vectors_normalized && !self.index.is_empty()
    }

    /// Every stored chunk as `(int_id, text)` in ascending `int_id` order —
    /// the input for a re-embed. Locking/SQLite only, no engine access.
    pub fn texts_for_reindex(&self) -> Result<Vec<(u64, String)>, CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;
        let mut stmt = conn
            .prepare("SELECT int_id, text FROM chunks ORDER BY int_id ASC")
            .map_err(|e| CoreError::Retrieval(format!("sqlite query: {}", e)))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, String>(1)?))
            })
            .map_err(|e| CoreError::Retrieval(format!("sqlite query: {}", e)))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| CoreError::Retrieval(format!("sqlite row: {}", e)))?);
        }
        Ok(out)
    }

    /// Replace the whole vector index with `vectors` (row-major, `dim` per
    /// row, one row per entry of `int_ids`, in the same order), L2-normalising
    /// each row exactly as insert does, then flush the index to disk and stamp
    /// the collection as normalised. SQLite rows, FTS5 and `int_id`s are
    /// untouched, so `delete_document` and allowlists keep working.
    ///
    /// Errors — without touching the live index — if `int_ids` is not exactly
    /// the stored chunk set or the vector count is not `int_ids.len() * dim`.
    pub fn rebuild_index(&mut self, vectors: &[f32], int_ids: &[u64]) -> Result<(), CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| CoreError::Retrieval(format!("failed to open sqlite db: {}", e)))?;

        // Validate before a single fallible build step: a mismatch is a caller
        // bug and must leave the previous index in place, never half-replaced.
        let mut stored: Vec<u64> = {
            let mut stmt = conn
                .prepare("SELECT int_id FROM chunks ORDER BY int_id ASC")
                .map_err(|e| CoreError::Retrieval(format!("sqlite query: {}", e)))?;
            let rows = stmt
                .query_map([], |row| row.get::<_, i64>(0).map(|v| v as u64))
                .map_err(|e| CoreError::Retrieval(format!("sqlite query: {}", e)))?;
            let mut ids = Vec::new();
            for row in rows {
                ids.push(row.map_err(|e| CoreError::Retrieval(format!("sqlite row: {}", e)))?);
            }
            ids
        };
        let mut given = int_ids.to_vec();
        given.sort_unstable();
        stored.sort_unstable();
        if given != stored {
            return Err(CoreError::Retrieval(format!(
                "rebuild_index for collection '{}' got {} int_ids, but the stored chunk set \
                 has {} (ids must match exactly)",
                self.name,
                int_ids.len(),
                stored.len()
            )));
        }
        if vectors.len() != int_ids.len() * self.dim {
            return Err(CoreError::Retrieval(format!(
                "rebuild_index for collection '{}' got {} floats, expected {} ({} chunks * dim {})",
                self.name,
                vectors.len(),
                int_ids.len() * self.dim,
                int_ids.len(),
                self.dim
            )));
        }

        // Same normalisation the insert path applies.
        let mut normalized = vectors.to_vec();
        for row in normalized.chunks_exact_mut(self.dim) {
            l2_normalize(row);
        }

        let mut rebuilt = IdMapIndex::new(self.dim, self.bit_width)
            .map_err(|e| CoreError::Retrieval(format!("failed to create turbovec index: {}", e)))?;
        rebuilt
            .add_with_ids(&normalized, int_ids)
            .map_err(|e| CoreError::Retrieval(format!("turbovec add failed: {}", e)))?;

        // Every fallible build step has succeeded; only now overwrite the
        // on-disk index and swap the live one. A failure at the stamp below
        // leaves a repaired but unstamped collection, which simply reports
        // `needs_reindex()` again (idempotent re-run), never a false claim.
        self.write_index(&rebuilt)?;
        self.index = rebuilt;
        self.stamp_normalized()?;
        self.vectors_normalized = true;
        Ok(())
    }

    /// Number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// True when the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

/// Chunk a document with the standard token-window chunker.
pub fn chunk_document(doc: &Document, chunk_tokens: usize, chunk_overlap: usize) -> Vec<Chunk> {
    chunk_text(&doc.content, &doc.id, chunk_tokens, chunk_overlap)
}

/// Simple approximate token-window chunker.
/// Splits on whitespace (each word ≈ one token), aiming for `chunk_tokens`
/// words with `chunk_overlap` words of overlap between consecutive chunks.
fn chunk_text(text: &str, doc_id: &str, chunk_tokens: usize, chunk_overlap: usize) -> Vec<Chunk> {
    // (byte offset, word) pairs so each chunk knows where it starts.
    let words: Vec<(usize, &str)> = text
        .split_whitespace()
        .map(|w| (w.as_ptr() as usize - text.as_ptr() as usize, w))
        .collect();
    if words.is_empty() {
        return Vec::new();
    }

    let step = chunk_tokens.saturating_sub(chunk_overlap).max(1);
    let mut chunks = Vec::new();

    for (chunk_i, start) in (0..words.len()).step_by(step).enumerate() {
        let end = (start + chunk_tokens).min(words.len());
        chunks.push(Chunk {
            id: format!("{}/chunk/{}", doc_id, chunk_i),
            document_id: doc_id.to_string(),
            text: words[start..end]
                .iter()
                .map(|(_, w)| *w)
                .collect::<Vec<_>>()
                .join(" "),
            char_offset: words[start].0,
        });

        if end >= words.len() {
            break;
        }
    }

    chunks
}

/// Tokenize a query with the retrieval metric's rule: lowercase `[a-z0-9]+`
/// runs (adoption-contract §1.1). Unicode/underscore characters split runs;
fn keyword_tokens(query: &str) -> Vec<String> {
    let lowered = query.to_lowercase();
    let bytes = lowered.as_bytes();
    let mut tokens: Vec<String> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        let is_tok = b.is_ascii_lowercase() || b.is_ascii_digit();
        match (is_tok, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                tokens.push(lowered[s..i].to_string());
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        tokens.push(lowered[s..].to_string());
    }
    tokens
}

/// JSON-encode a document-id allowlist for SQLite's `json_each`, the scoping
/// mechanism shared by the two FTS legs (`search`).
///
/// `None` means unrestricted and is returned as-is so callers can keep their
/// original, unfiltered statement; `Some(json)` is passed as a single bound
/// parameter, so the keyword `LIMIT` applies to the filtered set without any
/// dependence on the allowlist's length (`json_each` needs one parameter for
/// an arbitrarily long list, unlike an `IN (?, ?, ...)` list).
fn allowlist_json(allowlist: Option<&[String]>) -> Result<Option<String>, CoreError> {
    match allowlist {
        None => Ok(None),
        Some(ids) => serde_json::to_string(ids)
            .map(Some)
            .map_err(|e| CoreError::Retrieval(format!("allowlist encode: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_small() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, "doc1", 4, 2);
        // step = 2, so: [0..4], [2..6], [4..8], [6..10], [8..10]
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert_eq!(chunk.document_id, "doc1");
        }
    }

    #[test]
    fn test_keyword_tokens() {
        assert_eq!(keyword_tokens("hello world"), vec!["hello", "world"]);
        assert_eq!(
            keyword_tokens("it's \"special\""),
            vec!["it", "s", "special"]
        );
        assert_eq!(
            keyword_tokens("Foo:BAR (x OR y) -baz*"),
            vec!["foo", "bar", "x", "or", "y", "baz"]
        );
        assert_eq!(keyword_tokens("a_b c2"), vec!["a", "b", "c2"]);
        assert!(keyword_tokens("!!!").is_empty());
    }

    // -------------------------------------------------------------------
    // Property tests: ingest → search → delete → restart equivalence
    // -------------------------------------------------------------------

    use tempfile::TempDir;

    fn test_embed_fn(texts: &[String]) -> Result<Vec<f32>, CoreError> {
        Ok(vec![0.1f32; texts.len() * 16])
    }

    fn test_collection(dir: &Path, name: &str) -> Collection {
        Collection::new(name, 16, "test-model", dir, 4).unwrap()
    }

    fn test_document(id: &str, content: &str) -> Document {
        Document {
            id: id.into(),
            content: content.into(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Ingest a document, search for it with dense mode, verify results.
    #[test]
    fn property_ingest_then_search_dense() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "test");

        let doc = test_document("doc1", "The quick brown fox jumps over the lazy dog");
        let chunk_ids = coll.add_document(&doc, &test_embed_fn, 5, 2).unwrap();
        assert!(!chunk_ids.is_empty(), "should produce at least one chunk");

        // Search with a matching embedding (all 0.1 should match any 0.1).
        let q_embedding = vec![0.1f32; 16];
        let results = coll
            .search(&q_embedding, "fox", 10, SearchMode::Dense, None)
            .unwrap();
        assert!(!results.is_empty(), "dense search should return results");
        assert!(
            results.iter().any(|r| r.text.contains("fox")),
            "results should contain 'fox'"
        );
    }

    /// Keyword search via FTS5.
    #[test]
    fn property_keyword_search() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "test");

        let doc = test_document(
            "doc1",
            "machine learning is a subset of artificial intelligence",
        );
        coll.add_document(&doc, &test_embed_fn, 5, 2).unwrap();

        let q_embedding = vec![0.0f32; 16];
        let results = coll
            .search(
                &q_embedding,
                "machine learning",
                10,
                SearchMode::Keyword,
                None,
            )
            .unwrap();
        assert!(!results.is_empty(), "keyword search should find results");
    }

    /// Hybrid mode fuses the keyword top-40 and dense top-40 candidates (RRF
    /// union, adoption-contract §1.2): a document that matches only on
    /// keywords and one that matches only on embeddings both surface. Under
    /// the old FTS→dense intersection, the dense-only document was invisible.
    #[test]
    fn hybrid_rrf_union_diversity() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "test");

        let embed = |texts: &[String]| -> Result<Vec<f32>, CoreError> {
            let mut out = Vec::with_capacity(texts.len() * 16);
            for t in texts {
                // doc2 (quantum) sits in a different region of the space than
                // doc1 (alpha), so the query below is dense-near doc2 only.
                let v = if t.contains("quantum") {
                    0.9f32
                } else {
                    0.1f32
                };
                out.extend(std::iter::repeat(v).take(16));
            }
            Ok(out)
        };
        coll.add_document(
            &test_document("doc1", "alpha beta gamma delta epsilon"),
            &embed,
            5,
            2,
        )
        .unwrap();
        coll.add_document(
            &test_document("doc2", "quantum computing is neither alpha nor beta"),
            &embed,
            5,
            2,
        )
        .unwrap();

        // Query tokens match doc1's text; query embedding is doc2's region.
        let q = vec![0.9f32; 16];
        let results = coll
            .search(&q, "alpha", 10, SearchMode::Hybrid, None)
            .unwrap();
        let docs: std::collections::HashSet<&str> =
            results.iter().map(|r| r.document_id.as_str()).collect();
        assert!(docs.contains("doc1"), "keyword-only doc1 must surface");
        assert!(docs.contains("doc2"), "dense-only doc2 must surface");

        // No keyword side (no tokens): hybrid = single-list RRF over dense.
        let results = coll
            .search(&q, "!!!", 10, SearchMode::Hybrid, None)
            .unwrap();
        assert!(
            results.iter().any(|r| r.document_id == "doc2"),
            "hybrid with empty keyword leg falls back to the dense ranking"
        );
    }
    /// The dense top-10 is pinned ahead of the fused tail (contract §1.2 as
    /// amended): a keyword-only document must never displace a densely
    /// stronger one at the head, because that head churn is what drove the
    /// served `hybrid` arm below its gate (.7200).
    #[test]
    fn hybrid_head_stays_dense_ordered() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "test");

        // Dense ordering a > b > c for the query below.
        let embed = |texts: &[String]| -> Result<Vec<f32>, CoreError> {
            let mut out = Vec::with_capacity(texts.len() * 16);
            for t in texts {
                let v = if t.contains("alpha") {
                    1.0f32
                } else if t.contains("bravo") {
                    0.5
                } else {
                    0.1
                };
                out.extend(std::iter::repeat_n(v, 16));
            }
            Ok(out)
        };
        for (id, text) in [("doc_a", "alpha"), ("doc_b", "bravo"), ("doc_c", "charlie")] {
            coll.add_document(&test_document(id, text), &embed, 5, 0)
                .unwrap();
        }

        let q = vec![0.9f32; 16];
        let dense = coll
            .search(&q, "charlie", 10, SearchMode::Dense, None)
            .unwrap();
        assert_eq!(dense[0].document_id, "doc_a", "dense orders a first");

        // "charlie" matches doc_c in FTS5 only; doc_c is dense-worst. Under
        // additive equal-weight RRF it collects both terms (1/62 + 1/61) and
        // outranks doc_a (1/61) — the head pin must keep doc_a first.
        let fused = coll
            .search(&q, "charlie", 10, SearchMode::Hybrid, None)
            .unwrap();
        let ids: Vec<&str> = fused.iter().map(|r| r.document_id.as_str()).collect();
        assert_eq!(ids[0], "doc_a", "hybrid must not reorder the dense head");
        assert!(ids.contains(&"doc_c"), "keyword-only doc still surfaces");
    }

    /// Regression (allowlist scope pushdown): a scoped keyword/hybrid search
    /// must rank *within* the scope. 50 filler documents each out-rank the 12
    /// scoped documents on `bm25("alpha")`, so every scoped document sits
    /// beyond the global top-`POOL` (40) — exactly the geometry under which the
    /// old hybrid leg took the global top-40 and *then* filtered, leaving the
    /// keyword side empty and degrading hybrid to dense-only. Keyword mode
    /// returns all in-scope matches, and the fused scores carry the keyword
    /// RRF term again: pre-fix every score was a dense-only term (≤ 1/61),
    /// post-fix every scoped document also earns a keyword term (≥ 2/72).
    #[test]
    fn allowlisted_keyword_search_ranks_within_scope() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "test");

        // Embeddings are irrelevant to what this test pins (keyword evidence);
        // one deterministic value per document keeps the dense leg stable.
        let embed = |texts: &[String]| -> Result<Vec<f32>, CoreError> {
            let mut out = Vec::with_capacity(texts.len() * 16);
            for t in texts {
                let v = 0.1f32 + t.matches("zeta").count() as f32 * 0.02;
                out.extend(std::iter::repeat_n(v, 16));
            }
            Ok(out)
        };

        // 50 fillers: 5 `alpha` occurrences in a 5-token document, i.e. the
        // strongest bm25 match available, so they own the global top-40.
        for i in 0..50 {
            coll.add_document(
                &test_document(&format!("filler{:02}", i), "alpha alpha alpha alpha alpha"),
                &embed,
                200,
                0,
            )
            .unwrap();
        }
        // 12 scoped documents: one `alpha` each, 21..76 tokens long. They all
        // rank below every filler globally, so the global top-40 holds none.
        for i in 0..12 {
            let content = format!("{}alpha", "zeta ".repeat(20 + 5 * i));
            coll.add_document(
                &test_document(&format!("scoped{:02}", i), &content),
                &embed,
                200,
                0,
            )
            .unwrap();
        }
        let scoped: Vec<String> = (0..12).map(|i| format!("scoped{:02}", i)).collect();
        let q = vec![0.1f32; 16];

        // Premise: the unrestricted keyword top-40 is fillers only, so every
        // scoped hit lies beyond the global pool.
        let global = coll
            .search(&q, "alpha", 40, SearchMode::Keyword, None)
            .unwrap();
        assert_eq!(global.len(), 40);
        assert!(
            global.iter().all(|r| r.document_id.starts_with("filler")),
            "premise: scoped hits must rank beyond the global top-40"
        );

        // Scoped keyword search: every in-scope match comes back instead of
        // being cut by the global ranking. The order among them is SQLite's
        // bm25 arithmetic, so only the set is asserted.
        let kw = coll
            .search(
                &q,
                "alpha",
                12,
                SearchMode::Keyword,
                Some(scoped.as_slice()),
            )
            .unwrap();
        let kw_ids: Vec<&str> = kw.iter().map(|r| r.document_id.as_str()).collect();
        assert_eq!(
            kw_ids.len(),
            12,
            "scoped keyword search returns every in-scope match"
        );
        let mut got = kw_ids.clone();
        got.sort_unstable();
        let mut want: Vec<&str> = scoped.iter().map(String::as_str).collect();
        want.sort_unstable();
        assert_eq!(got, want, "exactly the in-scope documents are returned");
        assert!(
            !kw_ids.iter().any(|id| id.starts_with("filler")),
            "the global top-ranked fillers must not leak into the scoped result"
        );

        // Scoped hybrid: the fused scores must carry keyword evidence again.
        let hyb = coll
            .search(&q, "alpha", 12, SearchMode::Hybrid, Some(scoped.as_slice()))
            .unwrap();
        assert_eq!(hyb.len(), 12, "the scope has exactly 12 documents");
        let fused_scores: Vec<f64> = hyb.iter().map(|r| r.score).collect();
        assert!(
            hyb.iter().all(|r| r.score > 0.02),
            "scoped keyword evidence missing from the fusion: {fused_scores:?}"
        );

        // A pathologically long allowlist must not fail: the `json_each`
        // binding is one parameter whatever the list length (an `IN (?, ?, …)`
        // list would hit SQLite's parameter cap and abort the query).
        let mut huge = scoped.clone();
        huge.extend((0..2000).map(|i| format!("missing{:04}", i)));
        let kw_huge = coll
            .search(&q, "alpha", 12, SearchMode::Keyword, Some(huge.as_slice()))
            .unwrap();
        assert_eq!(kw_huge.len(), 12, "long allowlists must still work");

        // `Some(empty)` = nothing eligible, in every mode (the dense leg must
        // never reach turbovec's panic-on-empty-allowlist path).
        let empty: Vec<String> = Vec::new();
        for mode in [SearchMode::Keyword, SearchMode::Hybrid, SearchMode::Dense] {
            let results = coll
                .search(&q, "alpha", 12, mode, Some(empty.as_slice()))
                .unwrap();
            assert!(
                results.is_empty(),
                "{mode} must return nothing for an empty allowlist"
            );
        }
    }

    /// Ingest a document, delete it, verify it's gone.
    #[test]
    fn property_ingest_then_delete() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "test");

        let doc = test_document("doc1", "this document will be deleted");
        coll.add_document(&doc, &test_embed_fn, 5, 2).unwrap();
        assert!(
            !coll.is_empty(),
            "collection should not be empty after ingest"
        );

        coll.delete_document("doc1").unwrap();
        assert!(coll.is_empty(), "collection should be empty after delete");
    }

    /// Ingest, flush to disk, load from disk, verify search equivalence.
    #[test]
    fn property_restart_equivalence() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let coll_name = "persist-test";

        let doc = test_document("doc1", "persistence testing with turbovec and sqlite");

        // Phase 1: create, ingest, flush.
        {
            let mut coll = test_collection(&dir, coll_name);
            let chunk_ids = coll.add_document(&doc, &test_embed_fn, 10, 2).unwrap();
            assert!(!chunk_ids.is_empty());
            coll.flush().unwrap();

            let q = vec![0.1f32; 16];
            let results = coll
                .search(&q, "persistence", 10, SearchMode::Dense, None)
                .unwrap();
            assert!(!results.is_empty(), "search should work before restart");
        }

        // Phase 2: load from disk, verify same results.
        {
            let loaded = Collection::load(coll_name, &dir).unwrap();
            assert!(!loaded.is_empty(), "loaded collection should have data");

            let q = vec![0.1f32; 16];
            let results = loaded
                .search(&q, "persistence", 10, SearchMode::Dense, None)
                .unwrap();
            assert!(!results.is_empty(), "search should work after restart");
        }
    }

    /// Duplicate document ingest should fail.
    #[test]
    fn property_duplicate_document_rejected() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "test");

        let doc = test_document("doc1", "unique content");
        coll.add_document(&doc, &test_embed_fn, 5, 2).unwrap();

        let result = coll.add_document(&doc, &test_embed_fn, 5, 2);
        assert!(result.is_err(), "duplicate document should be rejected");
    }

    /// Collection creation with duplicate name should fail.
    #[test]
    fn property_duplicate_collection_rejected() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut store = CollectionStore::new(dir, 4);

        store.create("c1", "test-model", 16).unwrap();
        let result = store.create("c1", "other-model", 16);
        assert!(result.is_err(), "duplicate collection should be rejected");
    }

    /// Deleting a non-existent collection should return an error.
    #[test]
    fn property_delete_nonexistent_collection() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut store = CollectionStore::new(dir, 4);

        // Unknown names must never be turned into a path (traversal guard).
        let result = store.delete("nonexistent");
        assert!(matches!(result, Err(CoreError::CollectionNotFound(_))));
    }

    /// turbovec recall smoke (spec §13): inserting random vectors and querying
    /// with one of them must return that vector among the top hits. Guards
    /// turbovec dependency bumps against silent recall regressions.
    #[test]
    fn turbovec_recall_smoke() {
        use turbovec::IdMapIndex;

        let dim = 32;
        let n = 200;
        // Deterministic pseudo-random vectors (xorshift), no rand dep needed.
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / 16_777_216.0 - 0.5
        };
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();

        let mut index = IdMapIndex::new(dim, 4).unwrap();
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        index.add_with_ids(&flat, &ids).unwrap();

        let mut hits = 0;
        for (i, v) in vectors.iter().enumerate().step_by(10) {
            let (_scores, found) = index.search(v, 10);
            if found.contains(&(i as u64)) {
                hits += 1;
            }
        }
        let queried = vectors.len().div_ceil(10);
        assert!(
            hits * 10 >= queried * 9,
            "recall@10 for exact vectors below 90%: {}/{}",
            hits,
            queried
        );
    }

    // -------------------------------------------------------------------
    // Reindex / repair of un-normalised vectors
    // -------------------------------------------------------------------

    /// Read `collection_meta.vectors_normalized` (`None` = legacy, unstamped).
    fn normalized_stamp(dir: &Path) -> Option<String> {
        let conn = rusqlite::Connection::open(dir.join("metadata.db")).unwrap();
        conn.query_row(
            "SELECT value FROM collection_meta WHERE key = 'vectors_normalized'",
            [],
            |row| row.get(0),
        )
        .ok()
    }

    /// Drop the stamp, i.e. make `dir` look like a collection written before
    /// L2 normalisation landed.
    fn clear_normalized_stamp(dir: &Path) {
        let conn = rusqlite::Connection::open(dir.join("metadata.db")).unwrap();
        conn.execute(
            "DELETE FROM collection_meta WHERE key = 'vectors_normalized'",
            [],
        )
        .unwrap();
    }

    /// A normalising collection is stamped at creation and on insert, and an
    /// unstamped one with chunks is legacy until `rebuild_index` repairs it.
    #[test]
    fn legacy_collection_reports_needs_reindex_until_rebuilt() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "legacy");
        assert_eq!(
            normalized_stamp(&dir).as_deref(),
            Some("1"),
            "collection_meta must be stamped at creation"
        );

        coll.add_document(
            &test_document("doc1", "alpha bravo charlie"),
            &test_embed_fn,
            200,
            0,
        )
        .unwrap();
        coll.flush().unwrap();
        assert_eq!(normalized_stamp(&dir).as_deref(), Some("1"));
        assert!(!coll.needs_reindex());

        // Strip the stamp: this is exactly what a pre-normalisation collection
        // looks like on disk (no flag, no version, no stored raw vectors).
        clear_normalized_stamp(&dir);
        let mut legacy = Collection::load("legacy", &dir).unwrap();
        assert!(
            legacy.needs_reindex(),
            "an unstamped collection with stored vectors is legacy"
        );

        // Repair with deliberately un-normalised magnitudes: rebuild_index
        // normalises them itself.
        let stored = legacy.texts_for_reindex().unwrap();
        let ids: Vec<u64> = stored.iter().map(|(id, _)| *id).collect();
        let vectors = vec![3.0f32; stored.len() * 16];
        legacy.rebuild_index(&vectors, &ids).unwrap();
        assert!(
            !legacy.needs_reindex(),
            "rebuild_index repairs the collection"
        );
        assert_eq!(normalized_stamp(&dir).as_deref(), Some("1"));

        // The stamp is persisted, not just in-memory state.
        legacy.flush().unwrap();
        let reloaded = Collection::load("legacy", &dir).unwrap();
        assert!(!reloaded.needs_reindex(), "the stamp survives a reload");

        // An empty collection has nothing to repair even when unstamped.
        let empty_dir = tmp.path().join("empty-legacy");
        let _empty = test_collection(&empty_dir, "empty-legacy");
        clear_normalized_stamp(&empty_dir);
        let loaded = Collection::load("empty-legacy", &empty_dir).unwrap();
        assert!(loaded.is_empty());
        assert!(
            !loaded.needs_reindex(),
            "zero stored chunks means zero biased vectors"
        );
    }

    /// `texts_for_reindex` returns every stored chunk exactly once, ascending
    /// by int_id, matching the ingested chunk order.
    #[test]
    fn texts_for_reindex_returns_every_chunk_in_int_id_order() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "texts");

        let corpus = [
            ("doc1", "alpha bravo charlie delta echo foxtrot golf hotel"),
            ("doc2", "india juliet kilo lima mike november oscar papa"),
            (
                "doc3",
                "quebec romeo sierra tango uniform victor whiskey xray",
            ),
        ];
        let mut expected: Vec<String> = Vec::new();
        for (id, content) in corpus {
            let chunks = chunk_document(&test_document(id, content), 4, 2);
            assert!(
                chunks.len() > 1,
                "each document must contribute several chunks"
            );
            expected.extend(chunks.iter().map(|c| c.text.clone()));
            coll.add_document(&test_document(id, content), &test_embed_fn, 4, 2)
                .unwrap();
        }

        let stored = coll.texts_for_reindex().unwrap();
        assert_eq!(stored.len(), coll.len(), "one row per stored chunk");
        assert_eq!(stored.len(), expected.len());
        assert!(
            stored.windows(2).all(|w| w[0].0 < w[1].0),
            "int_ids ascend and never repeat: {:?}",
            stored.iter().map(|(id, _)| *id).collect::<Vec<_>>()
        );
        let texts: Vec<String> = stored.iter().map(|(_, t)| t.clone()).collect();
        assert_eq!(texts, expected, "ascending int_id order == ingest order");
    }

    /// `rebuild_index` preserves `len()`, the int_id → vector mapping, and a
    /// later document delete still removes exactly its own chunks.
    #[test]
    fn rebuild_index_preserves_ids_and_delete_mapping() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "rebuild");

        // A 200-token window makes every document exactly one chunk, so the
        // int_ids are the predictable 1, 2, 3 in ingest order.
        for (id, text) in [
            ("doc1", "alpha one"),
            ("doc2", "bravo two"),
            ("doc3", "charlie three"),
        ] {
            coll.add_document(&test_document(id, text), &test_embed_fn, 200, 0)
                .unwrap();
        }

        let stored = coll.texts_for_reindex().unwrap();
        let int_ids: Vec<u64> = stored.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            int_ids,
            vec![1, 2, 3],
            "ingest order allocates ascending int_ids"
        );

        // One-hot rows keyed by int_id: a query on column c can then only hit
        // the row belonging to int_id c, making the mapping observable.
        let mut vectors = vec![0.0f32; stored.len() * 16];
        for (row, (int_id, _)) in stored.iter().enumerate() {
            vectors[row * 16 + *int_id as usize] = 1.0;
        }
        coll.rebuild_index(&vectors, &int_ids).unwrap();

        assert_eq!(coll.len(), int_ids.len(), "len() == int_ids.len()");
        assert!(!coll.needs_reindex());

        let mut q = vec![0.0f32; 16];
        q[2] = 1.0;
        let hits = coll.search(&q, "two", 10, SearchMode::Dense, None).unwrap();
        assert_eq!(
            hits[0].document_id, "doc2",
            "int_id 2's row must still be the nearest to query column 2"
        );

        // int_ids survived the rebuild, so SQLite-side deletes still line up
        // with the index.
        let before = coll.len();
        let doc2_chunks = coll
            .list_documents()
            .unwrap()
            .into_iter()
            .find(|d| d.id == "doc2")
            .unwrap()
            .chunk_count;
        coll.delete_document("doc2").unwrap();
        assert_eq!(coll.len(), before - doc2_chunks as usize);
        let hits = coll.search(&q, "two", 10, SearchMode::Dense, None).unwrap();
        assert!(
            hits.iter().all(|r| r.document_id != "doc2"),
            "a deleted document's chunks must be gone from the rebuilt index"
        );
    }

    /// A count/geometry mismatch is rejected and leaves the previous index
    /// searchable — never a partial swap.
    #[test]
    fn rebuild_index_rejects_mismatch_without_touching_the_index() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "guard");

        let embed = |texts: &[String]| -> Result<Vec<f32>, CoreError> {
            let mut out = Vec::with_capacity(texts.len() * 16);
            for t in texts {
                // Distinct directions survive L2 normalisation (equal
                // coordinates of different magnitudes would not): doc2's
                // chunks are one-hot on column 2, doc1's on column 0.
                let mut v = vec![0.0f32; 16];
                if t.contains("quantum") {
                    v[2] = 1.0;
                } else {
                    v[0] = 1.0;
                }
                out.extend(v);
            }
            Ok(out)
        };
        coll.add_document(&test_document("doc1", "alpha beta"), &embed, 5, 0)
            .unwrap();
        coll.add_document(&test_document("doc2", "quantum computing"), &embed, 5, 0)
            .unwrap();

        let before_len = coll.len();
        let stored = coll.texts_for_reindex().unwrap();
        let ids: Vec<u64> = stored.iter().map(|(id, _)| *id).collect();
        assert!(ids.len() >= 2);
        let mut q = vec![0.0f32; 16];
        q[2] = 1.0;
        let before: Vec<String> = coll
            .search(&q, "quantum", 10, SearchMode::Dense, None)
            .unwrap()
            .into_iter()
            .map(|r| r.document_id)
            .collect();
        assert_eq!(before[0], "doc2", "premise: doc2 is the dense winner");

        // Wrong float count for the given id set.
        let err = coll
            .rebuild_index(&vec![0.9f32; ids.len() * 16 - 1], &ids)
            .unwrap_err();
        assert!(matches!(err, CoreError::Retrieval(_)), "got {err:?}");

        // An int_id the stored chunk set does not contain.
        let mut extra = ids.clone();
        extra.push(9999);
        assert!(
            coll.rebuild_index(&vec![0.9f32; extra.len() * 16], &extra)
                .is_err(),
            "an unknown int_id must be rejected"
        );

        // A stored chunk left out.
        assert!(
            coll.rebuild_index(&vec![0.9f32; (ids.len() - 1) * 16], &ids[..ids.len() - 1])
                .is_err(),
            "a missing int_id must be rejected"
        );

        // Duplicated id (same length, wrong set).
        let mut dup = ids.clone();
        dup[0] = dup[1];
        assert!(
            coll.rebuild_index(&vec![0.9f32; dup.len() * 16], &dup)
                .is_err(),
            "a duplicated int_id must be rejected"
        );

        // The live index is untouched: same size, same ranking, still stamped.
        assert_eq!(coll.len(), before_len, "no partial swap");
        assert!(!coll.needs_reindex());
        let after: Vec<String> = coll
            .search(&q, "quantum", 10, SearchMode::Dense, None)
            .unwrap()
            .into_iter()
            .map(|r| r.document_id)
            .collect();
        assert_eq!(after, before, "the previous index is still searchable");
    }

    // -------------------------------------------------------------------
    // Randomized property tests (spec §13) — proptest
    // -------------------------------------------------------------------

    use proptest::prelude::*;

    fn arb_words() -> impl Strategy<Value = Vec<String>> {
        proptest::collection::vec("[a-z]{1,8}", 1..200)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Chunking covers every word, respects the window size, and reports
        /// char offsets that point at the true chunk start.
        #[test]
        fn chunking_invariants(words in arb_words(), chunk_tokens in 1usize..64, overlap in 0usize..16) {
            prop_assume!(overlap < chunk_tokens);
            let text = words.join(" ");
            let chunks = chunk_text(&text, "d", chunk_tokens, overlap);

            prop_assert!(!chunks.is_empty());
            // Every chunk is at most chunk_tokens words.
            for c in &chunks {
                prop_assert!(c.text.split_whitespace().count() <= chunk_tokens);
                // char_offset points at the chunk's first word.
                prop_assert!(text[c.char_offset..].starts_with(c.text.split_whitespace().next().unwrap()));
            }
            // Concatenated chunks (dropping overlap) cover all words: the
            // total of unique starting positions reaches the last word.
            let last = chunks.last().unwrap();
            let last_words: Vec<&str> = last.text.split_whitespace().collect();
            prop_assert_eq!(*last_words.last().unwrap(), words.last().unwrap().as_str());
        }

        /// Ingest → delete → the collection index is empty again, and
        /// keyword search no longer matches (restart-equivalent state).
        #[test]
        fn ingest_delete_restart_equivalence(
            contents in proptest::collection::vec("[a-z]{2,8}( [a-z]{2,8}){3,20}", 1..5)
        ) {
            let tmp = TempDir::new().unwrap();
            let dir = tmp.path().to_path_buf();

            {
                let mut coll = test_collection(&dir, "prop");
                for (i, content) in contents.iter().enumerate() {
                    let doc = test_document(&format!("doc{}", i), content);
                    coll.add_document(&doc, &test_embed_fn, 8, 2).unwrap();
                }
                coll.flush().unwrap();
            }

            // Restart: same number of vectors, keyword search still works.
            {
                let mut coll = Collection::load("prop", &dir).unwrap();
                let before = coll.len();
                prop_assert!(before > 0);

                // Search for the first word of the first doc.
                let word = contents[0].split_whitespace().next().unwrap();
                let q = vec![0.1f32; 16];
                let results = coll.search(&q, word, 10, SearchMode::Keyword, None).unwrap();
                prop_assert!(results.iter().any(|r| r.document_id == "doc0"));

                // Delete everything; index must be empty and search must miss.
                for i in 0..contents.len() {
                    coll.delete_document(&format!("doc{}", i)).unwrap();
                }
                prop_assert!(coll.is_empty());
                let results = coll.search(&q, word, 10, SearchMode::Keyword, None).unwrap();
                prop_assert!(results.is_empty());
            }
        }
    }
}
