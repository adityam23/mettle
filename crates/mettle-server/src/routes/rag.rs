use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use mettle_core::engine::{EngineRequest, EngineResponse, GenConstraint, RequestKind};
use mettle_core::hooks::{Hit, RequestCtx};
use mettle_core::retrieval::{
    chunk_document, Citation, Document, DocumentSummary, RagChatRequest, RagChatResponse,
    RagChatStreamChunk, SearchMode, SearchRequest,
};
use mettle_core::stream::{engine_accumulate, engine_stream, Accumulated, FinishEvt, TokenEvt};
use mettle_core::timing::timestamp_iso8601;

use super::error::{api_error, api_error_code};
use crate::llama_bridge::{convert_options, usage_to_timing};
use crate::response_stream::NdJsonStream;
use crate::state::AppState;

/// Map a CoreError to the HTTP status it should produce.
fn core_error_status(e: &mettle_core::error::CoreError) -> StatusCode {
    use mettle_core::error::CoreError;
    match e {
        CoreError::Conflict(_) => StatusCode::CONFLICT,
        CoreError::CollectionNotFound(_)
        | CoreError::DocumentNotFound(_)
        | CoreError::ModelNotFound(_) => StatusCode::NOT_FOUND,
        CoreError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Embed `texts` with the collection's embedding model.
///
/// Fail loud, never degrade (residency contract §1.3): no path substitutes
/// all-zero vectors for a failed embedding, because those are silent
/// corruption — the request succeeds and dense/hybrid search quietly rots.
/// - Embedder not servable (no engine, not resident, dropped, loading):
///   the structured 503 `slot_unavailable` with the slot's real reason.
/// - Embedder resident but the forward pass failed: 503 naming the error.
/// - Dim mismatch between the model and the collection: 409 (the
///   collection needs re-ingest under its own embedder).
///
/// Callers embed BEFORE writing anything, so an error here writes nothing.
async fn embed_for_collection(
    state: &AppState,
    embedding_model: &str,
    collection: &str,
    dim: usize,
    texts: &[String],
) -> Result<Vec<f32>, axum::response::Response> {
    let alias = if embedding_model.is_empty() || embedding_model == "default" {
        state
            .registry
            .read()
            .map_err(|_| {
                api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
            })?
            .default_embedding()
            .alias
            .clone()
    } else {
        embedding_model.to_string()
    };

    // The engine keys loaded models by their canonical registry alias, so the
    // request must carry it: a collection created with a short name
    // (`bge-m3`, resolving to `bge-m3:latest`) would otherwise fail with
    // "model 'bge-m3' is not loaded" on every ingest and dense search.
    let (entry, engine) = state.v1_engine_for(&alias)?;
    let alias = entry.alias.clone();
    let vectors = mettle_core::engine::embed_texts(engine.as_ref(), &alias, texts)
        .await
        .map_err(|e| {
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("embedding via collection embedder '{alias}' failed: {e}"),
            )
            .into_response()
        })?;

    let mut flat = Vec::with_capacity(texts.len() * dim);
    for v in vectors {
        if v.len() != dim {
            return Err(api_error(
                StatusCode::CONFLICT,
                format!(
                    "embedding model '{}' produces {}-d vectors but collection '{}' \
                     expects {}; re-ingest the collection under its own embedding model",
                    alias,
                    v.len(),
                    collection,
                    dim
                ),
            )
            .into_response());
        }
        flat.extend_from_slice(&v);
    }
    Ok(flat)
}

// ---------------------------------------------------------------------------
// Collections CRUD
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    /// Embedding model alias; empty or "default" selects the registry's
    /// default embedding model.
    #[serde(default)]
    pub embedding_model: String,
}

/// POST /v1/collections — create a new collection.
pub async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> axum::response::Response {
    let embedding_model = if req.embedding_model.is_empty() || req.embedding_model == "default" {
        match state.registry.read() {
            Ok(r) => r.default_embedding().alias.clone(),
            Err(_) => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned")
                    .into_response()
            }
        }
    } else {
        req.embedding_model.clone()
    };

    // Canonicalise before storing. `resolve_model` is what applies the
    // `:latest` fallback, and the engine keys loaded models by that canonical
    // alias, so a collection created as "bge-m3" must be stored (and used) as
    // "bge-m3:latest" - the same thing the "default" arm above already does.
    let embedding_model = match state.resolve_model(&embedding_model) {
        Ok(entry) => entry.alias,
        Err(e) => return api_error(StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };

    // The collection's turbovec index is built at the embedding model's real
    // dimension. Resolve it from the GGUF header when the model is on disk,
    // else fall back to the registry's declared dim. Never hardcode 768 —
    // bge-m3/qwen3-embedding are 1024.
    let dim = match resolve_embedding_dim(&state, &embedding_model) {
        Ok(d) => d,
        Err((status, msg)) => return api_error(status, &msg).into_response(),
    };

    match state.collections.lock() {
        Ok(mut store) => match store.create(&req.name, &embedding_model, dim) {
            Ok(_) => StatusCode::CREATED.into_response(),
            Err(e) => api_error(core_error_status(&e), e.to_string()).into_response(),
        },
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response(),
    }
}

/// Resolve an embedding model's vector dimension for collection sizing.
/// Prefers the GGUF header's `{arch}.embedding_length` (authoritative, works
/// for any model), then falls back to the registry's declared dim for the
/// built-in embedders. Errors when the model is unknown / not an embedder.
fn resolve_embedding_dim(
    state: &AppState,
    embedding_model: &str,
) -> Result<usize, (StatusCode, String)> {
    let entry = state
        .resolve_model(embedding_model)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    // Is it an embedding model at all?
    if !matches!(
        entry.loader_kind,
        mettle_core::model_registry::LoaderKind::Embedding { .. }
    ) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("model '{}' is not an embedding model", embedding_model),
        ));
    }

    // GGUF header first (the model's real dim), if the file is resolvable.
    if let Ok(path) = crate::llama_bridge::resolve_gguf_path(&entry) {
        if let Some(d) = mettle_llama::gguf_embedding_dim(&path) {
            return Ok(d as usize);
        }
    }

    // Registry-declared fallback (built-in embedders: 768 / 1024 / 1024).
    if let Some(d) = entry.embedding_dim {
        return Ok(d);
    }

    Err((
        StatusCode::BAD_REQUEST,
        format!(
            "cannot determine embedding dimension for '{}' (GGUF not cached and no declared dim)",
            embedding_model
        ),
    ))
}

/// GET /v1/collections — list all collections.
pub async fn list_collections(State(state): State<Arc<AppState>>) -> axum::response::Response {
    match state.collections.lock() {
        Ok(store) => Json(store.list()).into_response(),
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response(),
    }
}

/// GET /v1/collections/:name — one collection's shape and whether its
/// stored vectors predate the L2-normalisation change (in which case its
/// dense/hybrid scores carry a scale bias and it needs `POST .../reindex`).
pub async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> axum::response::Response {
    let handle = match collection_handle(&state, &name) {
        Ok(h) => h,
        Err((status, msg)) => return api_error(status, &msg).into_response(),
    };
    let collection = match handle.lock() {
        Ok(c) => c,
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };
    let documents = match collection.list_documents() {
        Ok(d) => d.len(),
        Err(e) => return api_error(core_error_status(&e), e.to_string()).into_response(),
    };
    Json(serde_json::json!({
        "name": name,
        "dim": collection.dim,
        "embedding_model": collection.embedding_model.clone(),
        "documents": documents,
        "chunks": collection.len(),
        "needs_reindex": collection.needs_reindex(),
    }))
    .into_response()
}

/// Texts embedded per engine call while reindexing. One call per slice
/// keeps peak memory at `batch * dim` floats for arbitrarily large
/// collections.
const REINDEX_EMBED_BATCH: usize = 64;

/// POST /v1/collections/:name/reindex — re-embed every chunk from its stored
/// text, L2-normalise, and rebuild the turbovec index in place (SQLite rows,
/// FTS5 and document ids are untouched, so `int_id`s are preserved and
/// deletes/allowlists keep working).
///
/// This is the repair path for a collection whose vectors predate
/// normalisation (bug: unnormalised vectors score `||v|| * cos(v, q)` and
/// outrank closer rows) and for chunks written as all-zero vectors by the
/// old silent-fallback ingest. Fail loud: if the collection's embedder is
/// not servable, nothing is written and the 503 is returned unchanged.
pub async fn reindex_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> axum::response::Response {
    let handle = match collection_handle(&state, &name) {
        Ok(h) => h,
        Err((status, msg)) => return api_error(status, &msg).into_response(),
    };
    let (embedding_model, dim) = {
        let collection = match handle.lock() {
            Ok(c) => c,
            Err(_) => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned")
                    .into_response()
            }
        };
        (collection.embedding_model.clone(), collection.dim)
    };

    // Read the stored text in index order; this is the whole corpus, so it
    // must not hold a lock across the embedding awaits below.
    let stored = {
        let collection = match handle.lock() {
            Ok(c) => c,
            Err(_) => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned")
                    .into_response()
            }
        };
        match collection.texts_for_reindex() {
            Ok(t) => t,
            Err(e) => return api_error(core_error_status(&e), e.to_string()).into_response(),
        }
    };
    let int_ids: Vec<u64> = stored.iter().map(|(id, _)| *id).collect();
    let texts: Vec<String> = stored.into_iter().map(|(_, text)| text).collect();

    let mut vectors: Vec<f32> = Vec::with_capacity(texts.len() * dim);
    for slice in texts.chunks(REINDEX_EMBED_BATCH) {
        let flat = match embed_for_collection(&state, &embedding_model, &name, dim, slice).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        vectors.extend(flat);
    }

    let (chunks, documents) = {
        let mut collection = match handle.lock() {
            Ok(c) => c,
            Err(_) => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned")
                    .into_response()
            }
        };
        if let Err(e) = collection.rebuild_index(&vectors, &int_ids) {
            return api_error(core_error_status(&e), e.to_string()).into_response();
        }
        let documents = match collection.list_documents() {
            Ok(d) => d.len(),
            Err(e) => return api_error(core_error_status(&e), e.to_string()).into_response(),
        };
        (collection.len(), documents)
    };

    Json(serde_json::json!({
        "collection": name,
        "documents": documents,
        "chunks": chunks,
        "reindexed": true,
    }))
    .into_response()
}

/// DELETE /v1/collections/:name
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> axum::response::Response {
    match state.collections.lock() {
        Ok(mut store) => match store.delete(&name) {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(e) => api_error(core_error_status(&e), e.to_string()).into_response(),
        },
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response(),
    }
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

type CollectionHandle = Arc<std::sync::Mutex<mettle_core::retrieval::Collection>>;

/// Fetch a collection handle without holding the store lock.
fn collection_handle(
    state: &AppState,
    name: &str,
) -> Result<CollectionHandle, (StatusCode, String)> {
    let store = state.collections.lock().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "lock poisoned".to_string(),
        )
    })?;
    store.get(name).cloned().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("collection '{}' not found", name),
        )
    })
}

/// Parse the ingest request body: JSON `Document`, or multipart with a file
/// field (text/Markdown) and optional `id` field (spec §6).
async fn parse_ingest_request(
    req: axum::extract::Request,
) -> Result<Document, (StatusCode, String)> {
    let content_type = req
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.starts_with("multipart/form-data") {
        use axum::extract::FromRequest;
        let mut multipart = axum::extract::Multipart::from_request(req, &())
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

        let mut id: Option<String> = None;
        let mut content: Option<String> = None;
        let mut metadata = std::collections::HashMap::new();

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
        {
            match field.name().unwrap_or("") {
                "id" => {
                    id = Some(
                        field
                            .text()
                            .await
                            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?,
                    )
                }
                _ => {
                    // Treat any file-bearing field as the document body.
                    // Text and Markdown pass through; PDFs get text-extracted.
                    let filename = field.file_name().map(str::to_string);
                    let bytes = field
                        .bytes()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

                    let is_pdf = filename
                        .as_deref()
                        .is_some_and(|f| f.to_lowercase().ends_with(".pdf"))
                        || bytes.starts_with(b"%PDF-");
                    let text = if is_pdf {
                        pdf_extract::extract_text_from_mem(&bytes).map_err(|e| {
                            (
                                StatusCode::BAD_REQUEST,
                                format!("PDF text extraction failed: {}", e),
                            )
                        })?
                    } else {
                        String::from_utf8_lossy(&bytes).into_owned()
                    };

                    if let Some(f) = filename {
                        metadata.insert("filename".to_string(), f.clone());
                        if id.is_none() {
                            id = Some(f);
                        }
                    }
                    content = Some(text);
                }
            }
        }

        let content = content.ok_or((
            StatusCode::BAD_REQUEST,
            "multipart ingest requires a file field".to_string(),
        ))?;
        let id = id.ok_or((
            StatusCode::BAD_REQUEST,
            "multipart ingest requires an 'id' field or a filename".to_string(),
        ))?;
        Ok(Document {
            id,
            content,
            metadata,
        })
    } else {
        let Json(doc): Json<Document> = {
            use axum::extract::FromRequest;
            Json::from_request(req, &())
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
        };
        Ok(doc)
    }
}

/// POST /v1/collections/:name/documents — ingest a document (JSON or multipart).
pub async fn ingest_document(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    req: axum::extract::Request,
) -> axum::response::Response {
    let doc = match parse_ingest_request(req).await {
        Ok(d) => d,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    let coll_handle = match collection_handle(&state, &collection_name) {
        Ok(h) => h,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    // Read collection parameters without holding the lock across awaits.
    let (dim, embedding_model) = match coll_handle.lock() {
        Ok(c) => (c.dim, c.embedding_model.clone()),
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };

    let chunks = chunk_document(
        &doc,
        state.config.retrieval.chunk_tokens,
        state.config.retrieval.chunk_overlap,
    );
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

    let embeddings = match embed_for_collection(
        &state,
        &embedding_model,
        &collection_name,
        dim,
        &chunk_texts,
    )
    .await
    {
        Ok(e) => e,
        Err(resp) => return resp,
    };

    let mut coll = match coll_handle.lock() {
        Ok(c) => c,
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };

    match coll.add_document_chunks(&doc, &chunks, &embeddings) {
        Ok(chunk_ids) => {
            let _ = coll.flush();
            state
                .metrics
                .ingest_documents_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            state
                .metrics
                .ingest_chunks_total
                .fetch_add(chunk_ids.len() as u64, std::sync::atomic::Ordering::Relaxed);
            Json(serde_json::json!({
                "document_id": doc.id,
                "chunks": chunk_ids.len(),
                "chunk_ids": chunk_ids,
            }))
            .into_response()
        }
        Err(e) => api_error(core_error_status(&e), e.to_string()).into_response(),
    }
}

/// Hard caps on one batch ingest request (docket M3). One filing is the
/// atomic unit: a 10-K at 220-word chunks is ~700 documents / ~1 MB, so
/// 1024 documents and 8 MiB keep a whole filing in one batch.
const MAX_BATCH_DOCUMENTS: usize = 1024;
pub const MAX_BATCH_BYTES: usize = 8 * 1024 * 1024;

#[derive(serde::Deserialize)]
pub struct BatchIngestRequest {
    pub documents: Vec<Document>,
}

/// POST /v1/collections/:name/documents:batch — ingest many documents in one
/// embed pass, one turbovec insert, one SQLite transaction and one flush.
/// **All or nothing**: a duplicate id (in the batch or already stored) or
/// any engine/SQLite error writes nothing.
pub async fn ingest_documents_batch(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    if body.len() > MAX_BATCH_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "batch body is {} bytes; the limit is {} bytes",
                body.len(),
                MAX_BATCH_BYTES
            ),
        )
        .into_response();
    }
    let req: BatchIngestRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("invalid request body: {e}"),
            )
            .into_response()
        }
    };
    if req.documents.len() > MAX_BATCH_DOCUMENTS {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "batch has {} documents; the limit is {}",
                req.documents.len(),
                MAX_BATCH_DOCUMENTS
            ),
        )
        .into_response();
    }

    let coll_handle = match collection_handle(&state, &collection_name) {
        Ok(h) => h,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };
    let (dim, embedding_model) = match coll_handle.lock() {
        Ok(c) => (c.dim, c.embedding_model.clone()),
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };

    // Chunk everything, then embed the whole batch in ONE pass: 1024
    // documents must not become 1024 forward passes.
    let (chunk_tokens, chunk_overlap) = (
        state.config.retrieval.chunk_tokens,
        state.config.retrieval.chunk_overlap,
    );
    let mut chunked: Vec<(Document, Vec<mettle_core::retrieval::Chunk>)> =
        Vec::with_capacity(req.documents.len());
    let mut all_texts: Vec<String> = Vec::new();
    for doc in req.documents {
        let chunks = chunk_document(&doc, chunk_tokens, chunk_overlap);
        all_texts.extend(chunks.iter().map(|c| c.text.clone()));
        chunked.push((doc, chunks));
    }

    let embeddings =
        match embed_for_collection(&state, &embedding_model, &collection_name, dim, &all_texts)
            .await
        {
            Ok(e) => e,
            Err(resp) => return resp,
        };

    let mut coll = match coll_handle.lock() {
        Ok(c) => c,
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };
    match coll.add_documents_chunks(&chunked, &embeddings) {
        Ok(per_doc) => {
            let _ = coll.flush();
            let total_chunks: usize = per_doc.iter().map(|ids| ids.len()).sum();
            state
                .metrics
                .ingest_documents_total
                .fetch_add(chunked.len() as u64, std::sync::atomic::Ordering::Relaxed);
            state
                .metrics
                .ingest_chunks_total
                .fetch_add(total_chunks as u64, std::sync::atomic::Ordering::Relaxed);
            let results: Vec<serde_json::Value> = chunked
                .iter()
                .zip(per_doc)
                .map(|((doc, _), chunk_ids)| {
                    serde_json::json!({
                        "document_id": doc.id,
                        "chunks": chunk_ids.len(),
                        "chunk_ids": chunk_ids,
                    })
                })
                .collect();
            Json(serde_json::json!({
                "documents": results.len(),
                "chunks": total_chunks,
                "results": results,
            }))
            .into_response()
        }
        Err(e) => api_error(core_error_status(&e), e.to_string()).into_response(),
    }
}

/// GET /v1/collections/:name/documents — every stored document with its
/// chunk count and metadata, ordered by id, unpaginated (docket M4).
pub async fn list_documents(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> axum::response::Response {
    let coll_handle = match collection_handle(&state, &collection_name) {
        Ok(h) => h,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };
    let coll = match coll_handle.lock() {
        Ok(c) => c,
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };
    match coll.list_documents() {
        Ok(docs) => Json::<Vec<DocumentSummary>>(docs).into_response(),
        Err(e) => api_error(core_error_status(&e), e.to_string()).into_response(),
    }
}

/// DELETE /v1/collections/:name/documents/:id
pub async fn delete_document(
    State(state): State<Arc<AppState>>,
    Path((collection_name, doc_id)): Path<(String, String)>,
) -> axum::response::Response {
    let coll_handle = match collection_handle(&state, &collection_name) {
        Ok(h) => h,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    let mut coll = match coll_handle.lock() {
        Ok(c) => c,
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };

    match coll.delete_document(&doc_id) {
        Ok(()) => {
            let _ = coll.flush();
            StatusCode::NO_CONTENT.into_response()
        }
        Err(e) => api_error(StatusCode::NOT_FOUND, e.to_string()).into_response(),
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// POST /v1/collections/:name/search
pub async fn search_collection(
    State(state): State<Arc<AppState>>,
    Extension(ctx): Extension<RequestCtx>,
    Path(collection_name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> axum::response::Response {
    let coll_handle = match collection_handle(&state, &collection_name) {
        Ok(h) => h,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    let (dim, embedding_model) = match coll_handle.lock() {
        Ok(c) => (c.dim, c.embedding_model.clone()),
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };

    // Keyword mode never uses the query embedding (pure FTS5/bm25) — skip
    // the forward pass and any embedder failure entirely (Q13a).
    let query_embedding = if req.mode == SearchMode::Keyword {
        Vec::new()
    } else {
        match embed_for_collection(
            &state,
            &embedding_model,
            &collection_name,
            dim,
            std::slice::from_ref(&req.query),
        )
        .await
        {
            Ok(e) => e,
            Err(resp) => return resp,
        }
    };

    let coll = match coll_handle.lock() {
        Ok(c) => c,
        Err(_) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response()
        }
    };

    let started = std::time::Instant::now();
    let searched = coll.search(
        &query_embedding,
        &req.query,
        req.k,
        req.mode,
        req.allowlist.as_deref(),
    );
    state.metrics.observe_search(started.elapsed());

    match searched {
        Ok(mut results) => {
            // Hooks may filter or re-rank the hits (`Hit` == `ScoredChunk`).
            state.hooks.on_retrieval(&ctx, &mut results);
            Json(results).into_response()
        }
        Err(e) => api_error(core_error_status(&e), e.to_string()).into_response(),
    }
}

// ---------------------------------------------------------------------------
// Grounded RAG chat
// ---------------------------------------------------------------------------

/// POST /v1/rag/chat
pub async fn rag_chat_handler(
    State(state): State<Arc<AppState>>,
    Extension(ctx): Extension<RequestCtx>,
    Json(req): Json<RagChatRequest>,
) -> axum::response::Response {
    // 1. Search every requested collection (hybrid mode, server-side).
    let mut hits = match search_collections(&state, &req).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };

    // 2. Run on_retrieval hooks.
    state.hooks.on_retrieval(&ctx, &mut hits);

    // 3. Build citations from hits. A grounded answer without citations is a bug.
    let citations: Vec<Citation> = hits
        .iter()
        .map(|h| Citation {
            chunk_id: h.chunk_id.clone(),
            document_id: h.document_id.clone(),
            source_span: h.source_span,
            retrieval_score: h.score,
        })
        .collect();

    // 4. Augment the system prompt with retrieved context.
    let augmented_messages = augment_prompt(req.messages.clone(), &hits);

    // 5. Send to the model.
    let (entry, sender) = match state.sender_for(&req.model) {
        Ok(v) => v,
        Err((status, msg)) => return api_error(status, msg).into_response(),
    };

    let is_streaming = req.stream.unwrap_or(false);
    let params = convert_options(&req.options);

    let (tx, rx) = tokio::sync::mpsc::channel::<EngineResponse>(256);
    let engine_req = EngineRequest {
        model: entry.alias,
        kind: RequestKind::Chat {
            messages: augmented_messages,
            params,
            constraint: GenConstraint::None,
            streaming: is_streaming,
        },
        response: tx,
    };

    if let Err(e) = sender.send(engine_req).await {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
    }

    let model_name = req.model;
    let hooks = state.hooks.clone();

    if is_streaming {
        let stream = engine_stream(
            rx,
            hooks,
            ctx,
            model_name.clone(),
            {
                let model_name = model_name.clone();
                move |evt: TokenEvt<'_>| Some(rag_token_chunk(model_name.clone(), evt))
            },
            move |evt: FinishEvt| rag_final_chunk(model_name.clone(), citations.clone(), evt),
        );
        NdJsonStream::new(stream).into_response()
    } else {
        match engine_accumulate(rx, hooks, ctx, model_name.clone(), |acc: Accumulated| {
            Ok(rag_response(model_name, citations, acc))
        })
        .await
        {
            Ok(resp) => Json(resp).into_response(),
            Err((status, msg)) => api_error_code(status, msg).into_response(),
        }
    }
}

/// Streaming token chunk; citations ride on the final chunk.
fn rag_token_chunk(model: String, evt: TokenEvt<'_>) -> RagChatStreamChunk {
    RagChatStreamChunk {
        model,
        created_at: timestamp_iso8601(),
        message: mettle_core::ollama_types::ChatStreamMessage {
            role: "assistant".into(),
            content: evt.content.to_string(),
            tool_calls: None,
        },
        done: false,
        citations: None,
    }
}

/// Final streaming chunk carries the retrieval citations (spec §6).
fn rag_final_chunk(model: String, citations: Vec<Citation>, _evt: FinishEvt) -> RagChatStreamChunk {
    RagChatStreamChunk {
        model,
        created_at: timestamp_iso8601(),
        message: mettle_core::ollama_types::ChatStreamMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: None,
        },
        done: true,
        citations: Some(citations),
    }
}

/// Non-streaming RAG chat: accumulate, fire hooks, respond with citations.
fn rag_response(model: String, citations: Vec<Citation>, acc: Accumulated) -> RagChatResponse {
    RagChatResponse {
        model,
        created_at: timestamp_iso8601(),
        message: mettle_core::ollama_types::ResponseMessage {
            role: "assistant".into(),
            content: acc.content,
            tool_calls: None,
        },
        done_reason: acc.finish_reason,
        done: true,
        citations,
        timing: usage_to_timing(&acc.usage),
    }
}

/// Search all requested collections with the last user message as the query.
/// Embeds the query per collection (each may use a different model).
async fn search_collections(
    state: &AppState,
    req: &RagChatRequest,
) -> Result<Vec<Hit>, axum::response::Response> {
    let fail = |status: StatusCode, msg: String| api_error(status, msg).into_response();
    let query = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let mut all_hits = Vec::new();
    for name in &req.collections {
        let handle = collection_handle(state, name).map_err(|(s, m)| fail(s, m))?;

        let (dim, embedding_model) = {
            let coll = handle
                .lock()
                .map_err(|_| fail(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned".into()))?;
            (coll.dim, coll.embedding_model.clone())
        };

        // RAG chat always searches dense/hybrid (server-side default mode),
        // so the query embedding is required.
        let query_embedding = embed_for_collection(
            state,
            &embedding_model,
            name,
            dim,
            std::slice::from_ref(&query),
        )
        .await?;

        let coll = handle
            .lock()
            .map_err(|_| fail(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned".into()))?;

        let started = std::time::Instant::now();
        let results = coll
            .search(&query_embedding, &query, req.k, Default::default(), None)
            .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        state.metrics.observe_search(started.elapsed());
        drop(coll);

        all_hits.extend(results);
    }

    Ok(all_hits)
}

/// Augment system prompt with retrieved context.
fn augment_prompt(
    mut messages: Vec<mettle_core::ollama_types::ChatMessage>,
    hits: &[Hit],
) -> Vec<mettle_core::ollama_types::ChatMessage> {
    if hits.is_empty() {
        return messages;
    }

    let context = hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            format!(
                "[{}] (source: {}, chunk: {}) {}",
                i + 1,
                h.document_id,
                h.chunk_id,
                h.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let system_prompt = format!(
        "You are a helpful assistant. Use the following retrieved context to answer the user's question. \
         Cite sources using [1], [2], etc.\n\nRetrieved context:\n{}",
        context
    );

    if messages.first().map(|m| m.role.as_str()) == Some("system") {
        messages[0].content = system_prompt;
    } else {
        messages.insert(
            0,
            mettle_core::ollama_types::ChatMessage {
                role: "system".into(),
                content: system_prompt,
                images: None,
                tool_calls: None,
            },
        );
    }

    messages
}
