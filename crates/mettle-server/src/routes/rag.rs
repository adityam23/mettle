use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json, Extension,
};
use mettle_core::hooks::{Hit, RequestCtx};
use mettle_core::retrieval::{
    chunk_document, Citation, Document, RagChatRequest, RagChatResponse, RagChatStreamChunk,
    SearchRequest,
};
use mettle_core::engine::{EngineRequest, EngineResponse, GenConstraint, RequestKind};
use mettle_core::stream::{
    engine_accumulate, engine_stream, Accumulated, FinishEvt, TokenEvt,
};
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

/// Embed `texts` with the collection's embedding model, falling back to
/// zero vectors when no engine (or no embedding model) is loaded so that
/// keyword search and ingest keep working on a fresh install.
async fn embed_or_fallback(
    state: &AppState,
    embedding_model: &str,
    dim: usize,
    texts: &[String],
) -> Result<Vec<f32>, (StatusCode, String)> {
    let alias = if embedding_model.is_empty() || embedding_model == "default" {
        state
            .registry
            .read()
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned".into()))?
            .default_embedding()
            .alias
            .clone()
    } else {
        embedding_model.to_string()
    };

    let Some(engine) = state.engine() else {
        tracing::warn!(
            "no engine loaded; using zero-vector embeddings (dense scores are meaningless)"
        );
        return Ok(vec![0.0f32; texts.len() * dim]);
    };

    match mettle_core::engine::embed_texts(engine.as_ref(), &alias, texts).await {
        Ok(vectors) => {
            let mut flat = Vec::with_capacity(texts.len() * dim);
            for v in vectors {
                if v.len() != dim {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(
                            "embedding model '{}' returned dim {}, collection expects {}",
                            alias,
                            v.len(),
                            dim
                        ),
                    ));
                }
                flat.extend_from_slice(&v);
            }
            Ok(flat)
        }
        Err(e) => {
            // The embedding model may simply not be resident; degrade rather
            // than fail ingest/search outright.
            tracing::warn!(
                "embedding via '{}' failed ({}); using zero-vector fallback",
                alias,
                e
            );
            Ok(vec![0.0f32; texts.len() * dim])
        }
    }
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

    match state.collections.lock() {
        Ok(mut store) => match store.create(&req.name, &embedding_model) {
            Ok(_) => StatusCode::CREATED.into_response(),
            Err(e) => api_error(core_error_status(&e), e.to_string()).into_response(),
        },
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response(),
    }
}

/// GET /v1/collections — list all collections.
pub async fn list_collections(
    State(state): State<Arc<AppState>>,
) -> axum::response::Response {
    match state.collections.lock() {
        Ok(store) => Json(store.list()).into_response(),
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned").into_response(),
    }
}

/// DELETE /v1/collections/:name
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> axum::response::Response {
    match state.collections.lock() {
        Ok(mut store) => match store.delete(&name) {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(e) => api_error(StatusCode::NOT_FOUND, e.to_string()).into_response(),
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
    let store = state
        .collections
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned".to_string()))?;
    store
        .get(name)
        .cloned()
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("collection '{}' not found", name),
            )
        })
}

/// Parse the ingest request body: JSON `Document`, or multipart with a file
/// field (text/Markdown) and optional `id` field (spec §6).
async fn parse_ingest_request(req: axum::extract::Request) -> Result<Document, (StatusCode, String)> {
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

    let embeddings = match embed_or_fallback(&state, &embedding_model, dim, &chunk_texts).await {
        Ok(e) => e,
        Err((status, msg)) => return api_error(status, msg).into_response(),
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

    let query_embedding = match embed_or_fallback(
        &state,
        &embedding_model,
        dim,
        std::slice::from_ref(&req.query),
    )
    .await
    {
        Ok(e) => e,
        Err((status, msg)) => return api_error(status, msg).into_response(),
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
        Err((status, msg)) => return api_error(status, msg).into_response(),
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
) -> Result<Vec<Hit>, (StatusCode, String)> {
    let query = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let mut all_hits = Vec::new();
    for name in &req.collections {
        let handle = collection_handle(state, name)?;

        let (dim, embedding_model) = {
            let coll = handle
                .lock()
                .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned".to_string()))?;
            (coll.dim, coll.embedding_model.clone())
        };

        let query_embedding =
            embed_or_fallback(state, &embedding_model, dim, std::slice::from_ref(&query)).await?;

        let coll = handle
            .lock()
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned".to_string()))?;

        let started = std::time::Instant::now();
        let results = coll
            .search(&query_embedding, &query, req.k, Default::default(), None)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
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
