use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Extension, Json,
};
use mettle_core::engine::{EngineRequest, EngineResponse, GenConstraint, RequestKind};
use mettle_core::error::CoreError;
use mettle_core::hooks::{Hit, RequestCtx};
use mettle_core::model_registry::{DefaultRole, RegistryEntry};
use mettle_core::retrieval::{
    chunk_document, Chunk, Citation, Collection, Document, RagChatRequest, RagChatResponse,
    RagChatStreamChunk, ScoredChunk, SearchMode, SearchRequest,
};
use mettle_core::stream::{engine_accumulate, engine_stream, Accumulated, FinishEvt, TokenEvt};
use mettle_core::timing::timestamp_iso8601;

use super::error::{api_error, api_error_code};
use crate::llama_bridge::{convert_options, usage_to_timing};
use crate::response_stream::NdJsonStream;
use crate::state::AppState;

/// A finished error response. Every helper and handler here returns its
/// error this way, so `?` short-circuits straight to the client. Boxed so
/// `Result<_, RouteError>` stays small (a bare `Response` error trips
/// `clippy::result_large_err`).
pub struct RouteError(Box<Response>);

impl From<Response> for RouteError {
    fn from(response: Response) -> Self {
        Self(Box::new(response))
    }
}

impl IntoResponse for RouteError {
    fn into_response(self) -> Response {
        *self.0
    }
}

type RouteResult<T = Response> = Result<T, RouteError>;

/// The standard `{"error": ...}` body as a [`RouteError`].
fn route_error(status: StatusCode, msg: impl Into<String>) -> RouteError {
    api_error(status, msg).into_response().into()
}

/// Map a CoreError to the HTTP status it should produce.
fn core_error_status(e: &CoreError) -> StatusCode {
    match e {
        CoreError::Conflict(_) => StatusCode::CONFLICT,
        CoreError::CollectionNotFound(_)
        | CoreError::DocumentNotFound(_)
        | CoreError::ModelNotFound(_) => StatusCode::NOT_FOUND,
        CoreError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// A CoreError as the error response its status maps to.
fn core_error(e: CoreError) -> RouteError {
    route_error(core_error_status(&e), e.to_string())
}

/// Lock a std mutex, mapping poisoning to the 500 every route returns.
fn lock<T>(mutex: &Mutex<T>) -> RouteResult<MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| route_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned"))
}

/// Run synchronous work (SQLite, index-file I/O, GGUF header reads) on the
/// blocking pool so it never stalls an async executor thread. The closure
/// owns everything it touches, so no `MutexGuard` can cross an `.await`.
async fn run_blocking<R: Send + 'static>(
    f: impl FnOnce() -> RouteResult<R> + Send + 'static,
) -> RouteResult<R> {
    tokio::task::spawn_blocking(f).await.map_err(|e| {
        route_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("blocking task failed: {e}"),
        )
    })?
}

/// Run `f` against the locked collection on the blocking pool.
async fn with_collection<R: Send + 'static>(
    handle: &CollectionHandle,
    f: impl FnOnce(&mut Collection) -> R + Send + 'static,
) -> RouteResult<R> {
    let handle = Arc::clone(handle);
    run_blocking(move || Ok(f(&mut *lock(&handle)?))).await
}

/// The embedder alias a request names: empty or `"default"` selects the
/// registry's default embedding model, anything else passes through.
fn embedding_alias(state: &AppState, name: &str) -> RouteResult<String> {
    let registry = state
        .registry
        .read()
        .map_err(|_| route_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned"))?;
    Ok(registry
        .default_alias_for(name, DefaultRole::Embedding)
        .unwrap_or(name)
        .to_owned())
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
) -> RouteResult<Vec<f32>> {
    let alias = embedding_alias(state, embedding_model)?;

    // The engine keys loaded models by their canonical registry alias, so the
    // request must carry it: a collection created with a short name
    // (`bge-m3`, resolving to `bge-m3:latest`) would otherwise fail with
    // "model 'bge-m3' is not loaded" on every ingest and dense search.
    let (entry, engine) = state.v1_engine_for(&alias)?;
    let alias = entry.alias.clone();
    let vectors = mettle_core::engine::embed_texts(engine.as_ref(), &alias, texts)
        .await
        .map_err(|e| {
            route_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("embedding via collection embedder '{alias}' failed: {e}"),
            )
        })?;

    let mut flat = Vec::with_capacity(texts.len() * dim);
    for v in vectors {
        if v.len() != dim {
            return Err(route_error(
                StatusCode::CONFLICT,
                format!(
                    "embedding model '{}' produces {}-d vectors but collection '{}' \
                     expects {}; re-ingest the collection under its own embedding model",
                    alias,
                    v.len(),
                    collection,
                    dim
                ),
            ));
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
) -> RouteResult {
    // Canonicalise before storing. `resolve_model` is what applies the
    // `:latest` fallback, and the engine keys loaded models by that canonical
    // alias, so a collection created as "bge-m3" must be stored (and used) as
    // "bge-m3:latest" - the same thing the default alias already is.
    let entry = state
        .resolve_model(&embedding_alias(&state, &req.embedding_model)?)
        .map_err(|e| route_error(StatusCode::NOT_FOUND, e.to_string()))?;

    // The GGUF header read and the collection's on-disk creation are both
    // blocking I/O.
    let store = Arc::clone(&state.collections);
    run_blocking(move || {
        let dim = resolve_embedding_dim(&entry)?;
        lock(&store)?
            .create(&req.name, &entry.alias, dim)
            .map(|_| ())
            .map_err(core_error)
    })
    .await?;
    Ok(StatusCode::CREATED.into_response())
}

/// Resolve an embedding model's vector dimension for collection sizing.
/// Prefers the GGUF header's `{arch}.embedding_length` (authoritative, works
/// for any model), then falls back to the registry's declared dim for the
/// built-in embedders. Never hardcode 768 — bge-m3/qwen3-embedding are 1024.
/// Errors when the model is not an embedder or its dim is unknowable.
fn resolve_embedding_dim(entry: &RegistryEntry) -> RouteResult<usize> {
    if !entry.is_embedding() {
        return Err(route_error(
            StatusCode::BAD_REQUEST,
            format!("model '{}' is not an embedding model", entry.alias),
        ));
    }

    // GGUF header first (the model's real dim), if the file is resolvable.
    if let Ok(path) = crate::llama_bridge::resolve_gguf_path(entry) {
        if let Some(d) = mettle_llama::gguf_embedding_dim(&path) {
            return Ok(d as usize);
        }
    }

    entry.declared_embedding_dim().ok_or_else(|| {
        route_error(
            StatusCode::BAD_REQUEST,
            format!(
                "cannot determine embedding dimension for '{}' (GGUF not cached and no declared dim)",
                entry.alias
            ),
        )
    })
}

/// GET /v1/collections — list all collections.
pub async fn list_collections(State(state): State<Arc<AppState>>) -> RouteResult {
    Ok(Json(lock(&state.collections)?.list()).into_response())
}

/// GET /v1/collections/:name — one collection's shape and whether its
/// stored vectors predate the L2-normalisation change (in which case its
/// dense/hybrid scores carry a scale bias and it needs `POST .../reindex`).
pub async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> RouteResult {
    let handle = collection_handle(&state, &name)?;
    let body = with_collection(&handle, move |collection| {
        let documents = collection.list_documents()?.len();
        Ok::<_, CoreError>(serde_json::json!({
            "name": name,
            "dim": collection.dim,
            "embedding_model": collection.embedding_model.clone(),
            "documents": documents,
            "chunks": collection.len(),
            "needs_reindex": collection.needs_reindex(),
        }))
    })
    .await?
    .map_err(core_error)?;
    Ok(Json(body).into_response())
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
) -> RouteResult {
    let handle = collection_handle(&state, &name)?;
    let (dim, embedding_model) = collection_params(&handle)?;

    // Read the stored text in index order; this is the whole corpus, so it
    // must not hold a lock across the embedding awaits below.
    let stored = with_collection(&handle, |c| c.texts_for_reindex())
        .await?
        .map_err(core_error)?;
    let (int_ids, texts): (Vec<u64>, Vec<String>) = stored.into_iter().unzip();

    let mut vectors: Vec<f32> = Vec::with_capacity(texts.len() * dim);
    for slice in texts.chunks(REINDEX_EMBED_BATCH) {
        vectors.extend(embed_for_collection(&state, &embedding_model, &name, dim, slice).await?);
    }

    let (chunks, documents) = with_collection(&handle, move |collection| {
        collection.rebuild_index(&vectors, &int_ids)?;
        let documents = collection.list_documents()?.len();
        Ok::<_, CoreError>((collection.len(), documents))
    })
    .await?
    .map_err(core_error)?;

    Ok(Json(serde_json::json!({
        "collection": name,
        "documents": documents,
        "chunks": chunks,
        "reindexed": true,
    }))
    .into_response())
}

/// DELETE /v1/collections/:name
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> RouteResult {
    let store = Arc::clone(&state.collections);
    run_blocking(move || lock(&store)?.delete(&name).map_err(core_error)).await?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

type CollectionHandle = Arc<Mutex<Collection>>;

/// Fetch a collection handle without holding the store lock.
fn collection_handle(state: &AppState, name: &str) -> RouteResult<CollectionHandle> {
    lock(&state.collections)?.get(name).cloned().ok_or_else(|| {
        route_error(
            StatusCode::NOT_FOUND,
            format!("collection '{}' not found", name),
        )
    })
}

/// A collection's `(dim, embedding_model)`, read under a brief lock so the
/// caller can embed (await) without holding it.
fn collection_params(handle: &CollectionHandle) -> RouteResult<(usize, String)> {
    let collection = lock(handle)?;
    Ok((collection.dim, collection.embedding_model.clone()))
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

/// One ingested document: its id and the chunk ids written for it.
struct IngestedDoc {
    document_id: String,
    chunk_ids: Vec<String>,
}

impl IngestedDoc {
    /// The per-document JSON shared by the single and batch ingest responses.
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "document_id": self.document_id,
            "chunks": self.chunk_ids.len(),
            "chunk_ids": self.chunk_ids,
        })
    }
}

/// The one ingest path: chunk every document, embed all chunks in ONE pass
/// (1024 documents must not become 1024 forward passes), then insert them
/// all-or-nothing with one turbovec insert, one SQLite transaction and one
/// flush on the blocking pool. Results come back in `docs` order.
async fn ingest_docs(
    state: &AppState,
    collection_name: &str,
    docs: Vec<Document>,
) -> RouteResult<Vec<IngestedDoc>> {
    let handle = collection_handle(state, collection_name)?;
    let (dim, embedding_model) = collection_params(&handle)?;

    let (chunk_tokens, chunk_overlap) = (
        state.config.retrieval.chunk_tokens,
        state.config.retrieval.chunk_overlap,
    );
    let mut chunked: Vec<(Document, Vec<Chunk>)> = docs
        .into_iter()
        .map(|doc| {
            let chunks = chunk_document(&doc, chunk_tokens, chunk_overlap);
            (doc, chunks)
        })
        .collect();

    // `embed_texts` takes `&[String]`: lend it the chunk texts by moving
    // them out and back instead of cloning the whole corpus.
    let texts: Vec<String> = chunked
        .iter_mut()
        .flat_map(|(_, chunks)| chunks.iter_mut())
        .map(|chunk| std::mem::take(&mut chunk.text))
        .collect();
    let embeddings =
        embed_for_collection(state, &embedding_model, collection_name, dim, &texts).await?;
    for (chunk, text) in chunked
        .iter_mut()
        .flat_map(|(_, chunks)| chunks.iter_mut())
        .zip(texts)
    {
        chunk.text = text;
    }

    let ingested = with_collection(&handle, move |collection| {
        let per_doc = {
            let refs: Vec<(&Document, &[Chunk])> = chunked
                .iter()
                .map(|(doc, chunks)| (doc, chunks.as_slice()))
                .collect();
            collection.add_documents_chunks(&refs, embeddings)?
        };
        let _ = collection.flush();
        Ok::<_, CoreError>(
            chunked
                .into_iter()
                .zip(per_doc)
                .map(|((doc, _), chunk_ids)| IngestedDoc {
                    document_id: doc.id,
                    chunk_ids,
                })
                .collect::<Vec<_>>(),
        )
    })
    .await?
    .map_err(core_error)?;

    let total_chunks: usize = ingested.iter().map(|d| d.chunk_ids.len()).sum();
    state
        .metrics
        .ingest_documents_total
        .fetch_add(ingested.len() as u64, std::sync::atomic::Ordering::Relaxed);
    state
        .metrics
        .ingest_chunks_total
        .fetch_add(total_chunks as u64, std::sync::atomic::Ordering::Relaxed);
    Ok(ingested)
}

/// POST /v1/collections/:name/documents — ingest a document (JSON or
/// multipart): a one-document [`ingest_docs`].
pub async fn ingest_document(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    req: axum::extract::Request,
) -> RouteResult {
    let doc = parse_ingest_request(req)
        .await
        .map_err(|(status, msg)| route_error(status, msg))?;
    let ingested = ingest_docs(&state, &collection_name, vec![doc]).await?;
    let doc = ingested.first().expect("one result per ingested document");
    Ok(Json(doc.to_json()).into_response())
}

/// Hard caps on one batch ingest request (docket M3). One filing is the
/// atomic unit: a 10-K at 220-word chunks is ~700 documents / ~1 MB, so
/// 1024 documents and 8 MiB keep a whole filing in one batch. The byte cap
/// is enforced by the route's `DefaultBodyLimit` (413 before the handler).
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
) -> RouteResult {
    let req: BatchIngestRequest = serde_json::from_slice(&body).map_err(|e| {
        route_error(
            StatusCode::BAD_REQUEST,
            format!("invalid request body: {e}"),
        )
    })?;
    if req.documents.len() > MAX_BATCH_DOCUMENTS {
        return Err(route_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "batch has {} documents; the limit is {}",
                req.documents.len(),
                MAX_BATCH_DOCUMENTS
            ),
        ));
    }

    let ingested = ingest_docs(&state, &collection_name, req.documents).await?;
    let total_chunks: usize = ingested.iter().map(|d| d.chunk_ids.len()).sum();
    let results: Vec<serde_json::Value> = ingested.iter().map(IngestedDoc::to_json).collect();
    Ok(Json(serde_json::json!({
        "documents": results.len(),
        "chunks": total_chunks,
        "results": results,
    }))
    .into_response())
}

/// GET /v1/collections/:name/documents — every stored document with its
/// chunk count and metadata, ordered by id, unpaginated (docket M4).
pub async fn list_documents(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> RouteResult {
    let handle = collection_handle(&state, &collection_name)?;
    let docs = with_collection(&handle, |c| c.list_documents())
        .await?
        .map_err(core_error)?;
    Ok(Json(docs).into_response())
}

/// DELETE /v1/collections/:name/documents/:id
pub async fn delete_document(
    State(state): State<Arc<AppState>>,
    Path((collection_name, doc_id)): Path<(String, String)>,
) -> RouteResult {
    let handle = collection_handle(&state, &collection_name)?;
    with_collection(&handle, move |collection| {
        collection.delete_document(&doc_id)?;
        let _ = collection.flush();
        Ok(())
    })
    .await?
    .map_err(|e: CoreError| route_error(StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Search one collection on the blocking pool and record the search latency
/// metric. The inner result is the collection's own verdict, left for the
/// caller to map (the search route and RAG chat map it differently).
async fn timed_search(
    state: &AppState,
    handle: &CollectionHandle,
    query_embedding: impl AsRef<[f32]> + Send + 'static,
    query: String,
    k: usize,
    mode: SearchMode,
    allowlist: Option<Vec<String>>,
) -> RouteResult<Result<Vec<ScoredChunk>, CoreError>> {
    let (searched, elapsed): (_, Duration) = with_collection(handle, move |collection| {
        let started = std::time::Instant::now();
        let searched = collection.search(
            query_embedding.as_ref(),
            &query,
            k,
            mode,
            allowlist.as_deref(),
        );
        (searched, started.elapsed())
    })
    .await?;
    state.metrics.observe_search(elapsed);
    Ok(searched)
}

/// POST /v1/collections/:name/search
pub async fn search_collection(
    State(state): State<Arc<AppState>>,
    Extension(ctx): Extension<RequestCtx>,
    Path(collection_name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> RouteResult {
    let handle = collection_handle(&state, &collection_name)?;
    let (dim, embedding_model) = collection_params(&handle)?;

    // Keyword mode never uses the query embedding (pure FTS5/bm25) — skip
    // the forward pass and any embedder failure entirely (Q13a).
    let query_embedding = if req.mode == SearchMode::Keyword {
        Vec::new()
    } else {
        embed_for_collection(
            &state,
            &embedding_model,
            &collection_name,
            dim,
            std::slice::from_ref(&req.query),
        )
        .await?
    };

    let mut results = timed_search(
        &state,
        &handle,
        query_embedding,
        req.query,
        req.k,
        req.mode,
        req.allowlist,
    )
    .await?
    .map_err(core_error)?;
    // Hooks may filter or re-rank the hits (`Hit` == `ScoredChunk`).
    state.hooks.on_retrieval(&ctx, &mut results);
    Ok(Json(results).into_response())
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
        Err(resp) => return resp.into_response(),
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
/// Each collection may use a different embedder, so the query is embedded
/// once per distinct `(embedding model, dim)` within the request and reused
/// for every collection that shares it.
async fn search_collections(state: &AppState, req: &RagChatRequest) -> RouteResult<Vec<Hit>> {
    let query = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let mut query_embeddings: HashMap<(String, usize), Arc<[f32]>> = HashMap::new();
    let mut all_hits = Vec::new();
    for name in &req.collections {
        let handle = collection_handle(state, name)?;
        let (dim, embedding_model) = collection_params(&handle)?;

        // RAG chat always searches dense/hybrid (server-side default mode),
        // so the query embedding is required.
        let key = (embedding_model, dim);
        let query_embedding = match query_embeddings.get(&key) {
            Some(embedding) => Arc::clone(embedding),
            None => {
                let embedding: Arc<[f32]> =
                    embed_for_collection(state, &key.0, name, dim, std::slice::from_ref(&query))
                        .await?
                        .into();
                query_embeddings.insert(key, Arc::clone(&embedding));
                embedding
            }
        };

        let results = timed_search(
            state,
            &handle,
            query_embedding,
            query.clone(),
            req.k,
            SearchMode::default(),
            None,
        )
        .await?
        .map_err(|e| route_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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
