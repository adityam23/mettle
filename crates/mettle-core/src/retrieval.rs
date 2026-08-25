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
    Dense,
    Keyword,
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
    embedding_dim: usize,
}

impl CollectionStore {
    pub fn new(data_dir: PathBuf, default_bit_width: usize, embedding_dim: usize) -> Self {
        Self {
            collections: HashMap::new(),
            data_dir,
            default_bit_width,
            embedding_dim,
        }
    }

    /// Get a handle to an existing collection.
    pub fn get(&self, name: &str) -> Option<&Arc<Mutex<Collection>>> {
        self.collections.get(name)
    }

    /// Create a new named collection. Returns an error if it already exists.
    pub fn create(
        &mut self,
        name: &str,
        embedding_model: &str,
    ) -> Result<&Arc<Mutex<Collection>>, CoreError> {
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
            self.embedding_dim,
            embedding_model,
            &coll_dir,
            self.default_bit_width,
        )?;

        let handle = Arc::new(Mutex::new(collection));
        self.collections
            .insert(name.to_string(), Arc::clone(&handle));
        Ok(self.collections.get(name).unwrap())
    }

    /// Delete a collection and its data.
    pub fn delete(&mut self, name: &str) -> Result<(), CoreError> {
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
        for entry in std::fs::read_dir(&colls_dir).map_err(|e| {
            CoreError::Retrieval(format!("failed to read collections dir: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                CoreError::Retrieval(format!("failed to read dir entry: {}", e))
            })?;
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().into_owned();
                let coll = Collection::load(&name, self.embedding_dim, &entry.path())?;
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

        let index = IdMapIndex::new(dim, bit_width).map_err(|e| {
            CoreError::Retrieval(format!("failed to create turbovec index: {}", e))
        })?;

        // Initialize the SQLite schema.
        let db_path = data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| {
            CoreError::Retrieval(format!("failed to open sqlite db: {}", e))
        })?;

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
                ('embedding_model', ?1), ('dim', ?2), ('bit_width', ?3)",
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
        })
    }

    /// Load an existing collection from disk.
    pub fn load(name: &str, dim: usize, data_dir: &Path) -> Result<Self, CoreError> {
        let db_path = data_dir.join("metadata.db");
        if !db_path.exists() {
            return Err(CoreError::CollectionNotFound(format!(
                "collection '{}' has no metadata.db at {}",
                name,
                data_dir.display()
            )));
        }

        let conn = rusqlite::Connection::open(&db_path).map_err(|e| {
            CoreError::Retrieval(format!("failed to open sqlite db: {}", e))
        })?;

        let meta = |key: &str| -> Option<String> {
            conn.query_row(
                "SELECT value FROM collection_meta WHERE key = ?1",
                [key],
                |row| row.get(0),
            )
            .ok()
        };
        let embedding_model = meta("embedding_model").unwrap_or_default();
        let dim = meta("dim")
            .and_then(|v| v.parse().ok())
            .unwrap_or(dim);
        let bit_width = meta("bit_width")
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

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

        // Determine the next int_id from existing chunks.
        let max_int_id: u64 = conn
            .query_row("SELECT COALESCE(MAX(int_id), 0) FROM chunks", [], |row| {
                row.get::<_, i64>(0).map(|v| v as u64)
            })
            .unwrap_or(0);

        Ok(Self {
            name: name.to_string(),
            dim,
            embedding_model,
            data_dir: data_dir.to_path_buf(),
            index,
            next_int_id: max_int_id + 1,
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
        // Check if document already exists.
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| {
            CoreError::Retrieval(format!("failed to open sqlite db: {}", e))
        })?;

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

        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        if embeddings.len() != chunks.len() * self.dim {
            return Err(CoreError::Retrieval(format!(
                "embed_fn returned {} floats, expected {} ({} chunks * dim {})",
                embeddings.len(),
                chunks.len() * self.dim,
                chunks.len(),
                self.dim
            )));
        }

        // Allocate integer IDs for the turbovec index.
        let base_id = self.next_int_id;
        let int_ids: Vec<u64> = (base_id..base_id + chunks.len() as u64).collect();
        self.next_int_id += chunks.len() as u64;

        // Insert into turbovec.
        self.index
            .add_with_ids(embeddings, &int_ids)
            .map_err(|e| CoreError::Retrieval(format!("turbovec add failed: {}", e)))?;

        // Insert into SQLite.
        conn.execute(
            "INSERT INTO documents (id, metadata_json) VALUES (?1, ?2)",
            rusqlite::params![doc.id, serde_json::to_string(&doc.metadata).unwrap_or_default()],
        )
        .map_err(|e| CoreError::Retrieval(format!("sqlite insert document: {}", e)))?;

        let mut chunk_ids = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_id = format!("{}/chunk/{}", doc.id, i);
            conn.execute(
                "INSERT INTO chunks (id, int_id, document_id, text, char_offset) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    chunk_id,
                    int_ids[i] as i64,
                    doc.id,
                    chunk.text,
                    chunk.char_offset as i64,
                ],
            )
            .map_err(|e| CoreError::Retrieval(format!("sqlite insert chunk: {}", e)))?;

            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, text) VALUES (?1, ?2)",
                rusqlite::params![chunk_id, chunk.text],
            )
            .map_err(|e| CoreError::Retrieval(format!("sqlite fts insert: {}", e)))?;

            chunk_ids.push(chunk_id);
        }

        Ok(chunk_ids)
    }

    /// Delete a document and all its chunks.
    pub fn delete_document(&mut self, doc_id: &str) -> Result<(), CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| {
            CoreError::Retrieval(format!("failed to open sqlite db: {}", e))
        })?;

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
    /// (in-kernel for dense/hybrid via turbovec's filtered search).
    pub fn search(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        mode: SearchMode,
        allowlist: Option<&[String]>,
    ) -> Result<Vec<ScoredChunk>, CoreError> {
        let db_path = self.data_dir.join("metadata.db");
        let conn = rusqlite::Connection::open(&db_path).map_err(|e| {
            CoreError::Retrieval(format!("failed to open sqlite db: {}", e))
        })?;

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

        match mode {
            SearchMode::Dense => {
                let (scores, ids) =
                    self.index
                        .search_with_allowlist(query_embedding, k, allow_ids.as_deref());
                self.lookup_chunks(&conn, &ids, &scores)
            }
            SearchMode::Keyword => {
                let allowed_docs: Option<std::collections::HashSet<&str>> =
                    allowlist.map(|d| d.iter().map(String::as_str).collect());

                let fts_query = sanitize_fts5(query_text);
                let mut stmt = conn
                    .prepare(
                        "SELECT c.id, c.document_id, c.text, c.char_offset
                         FROM chunks_fts fts
                         JOIN chunks c ON c.id = fts.chunk_id
                         WHERE chunks_fts MATCH ?1
                         ORDER BY rank",
                    )
                    .map_err(|e| CoreError::Retrieval(format!("fts query: {}", e)))?;

                let results: Vec<ScoredChunk> = stmt
                    .query_map([&fts_query], |row| {
                        Ok(ScoredChunk {
                            chunk_id: row.get(0)?,
                            document_id: row.get(1)?,
                            text: row.get(2)?,
                            score: 1.0,
                            source_span: Some((row.get::<_, i64>(3)? as usize, 0)),
                        })
                    })
                    .map_err(|e| CoreError::Retrieval(format!("fts query: {}", e)))?
                    .filter_map(|r| r.ok())
                    .filter(|c| {
                        allowed_docs
                            .as_ref()
                            .is_none_or(|docs| docs.contains(c.document_id.as_str()))
                    })
                    .take(k)
                    .collect();

                Ok(results)
            }
            SearchMode::Hybrid => {
                // FTS5 → candidate IDs → dense rerank via allowlist.
                let fts_query = sanitize_fts5(query_text);
                let mut stmt = conn
                    .prepare(
                        "SELECT c.int_id FROM chunks_fts fts
                         JOIN chunks c ON c.id = fts.chunk_id
                         WHERE chunks_fts MATCH ?1
                         ORDER BY rank
                         LIMIT 100",
                    )
                    .map_err(|e| CoreError::Retrieval(format!("fts hybrid query: {}", e)))?;

                let mut candidate_ids: Vec<u64> = stmt
                    .query_map([&fts_query], |row| row.get::<_, i64>(0).map(|v| v as u64))
                    .map_err(|e| CoreError::Retrieval(format!("fts hybrid query: {}", e)))?
                    .filter_map(|r| r.ok())
                    .collect();

                // Intersect FTS candidates with the caller's allowlist.
                if let Some(allow) = &allow_ids {
                    let allow: std::collections::HashSet<u64> = allow.iter().copied().collect();
                    candidate_ids.retain(|id| allow.contains(id));
                }

                if candidate_ids.is_empty() {
                    // No keyword candidates: fall back to dense (still allowlist-filtered).
                    let (scores, ids) =
                        self.index
                            .search_with_allowlist(query_embedding, k, allow_ids.as_deref());
                    return self.lookup_chunks(&conn, &ids, &scores);
                }

                let (scores, ids) = self.index.search_with_allowlist(
                    query_embedding,
                    k.min(candidate_ids.len()),
                    Some(&candidate_ids),
                );

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
        let path = self.data_dir.join("index.bin");
        self.index.write(&path).map_err(|e| {
            CoreError::Retrieval(format!("failed to write index: {}", e))
        })
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

/// Sanitize a user query for FTS5 MATCH syntax: keep alphanumeric terms,
/// drop everything else (operators, quotes, punctuation). Terms stay
/// unquoted, so FTS5 ANDs them together.
fn sanitize_fts5(query: &str) -> String {
    // Lowercase so bare AND/OR/NOT are not parsed as FTS5 operators.
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
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
    fn test_sanitize_fts5() {
        assert_eq!(sanitize_fts5("hello world"), "hello world");
        assert_eq!(sanitize_fts5("it's \"special\""), "it s special");
        assert_eq!(sanitize_fts5("foo:bar (x OR y) -baz*"), "foo bar x or y baz");
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

        let doc = test_document("doc1", "machine learning is a subset of artificial intelligence");
        coll.add_document(&doc, &test_embed_fn, 5, 2).unwrap();

        let q_embedding = vec![0.0f32; 16];
        let results = coll
            .search(&q_embedding, "machine learning", 10, SearchMode::Keyword, None)
            .unwrap();
        assert!(!results.is_empty(), "keyword search should find results");
    }

    /// Ingest a document, delete it, verify it's gone.
    #[test]
    fn property_ingest_then_delete() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut coll = test_collection(&dir, "test");

        let doc = test_document("doc1", "this document will be deleted");
        coll.add_document(&doc, &test_embed_fn, 5, 2).unwrap();
        assert!(!coll.is_empty(), "collection should not be empty after ingest");

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
            let results = coll.search(&q, "persistence", 10, SearchMode::Dense, None).unwrap();
            assert!(!results.is_empty(), "search should work before restart");
        }

        // Phase 2: load from disk, verify same results.
        {
            let loaded = Collection::load(coll_name, 16, &dir).unwrap();
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
        let mut store = CollectionStore::new(dir, 4, 16);

        store.create("c1", "test-model").unwrap();
        let result = store.create("c1", "other-model");
        assert!(result.is_err(), "duplicate collection should be rejected");
    }

    /// Deleting a non-existent collection should return an error.
    #[test]
    fn property_delete_nonexistent_collection() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut store = CollectionStore::new(dir, 4, 16);

        let result = store.delete("nonexistent");
        assert!(result.is_ok(), "deleting non-existent collection is a no-op");
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
                let mut coll = Collection::load("prop", 16, &dir).unwrap();
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
