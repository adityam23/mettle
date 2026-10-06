use std::sync::Arc;

use axum::{middleware, routing::delete, routing::get, routing::post, Router};

use crate::hooks_integration::{auth_middleware, request_hook_middleware};
use crate::state::AppState;

/// Build the complete axum router: Ollama-compatible `/api/*`, RAG
/// `/v1/collections` + `/v1/rag/chat`, OpenAI-compatible `/v1/*`
/// (chat/completions, embeddings, models), Anthropic `/v1/messages`, and
/// the OCR endpoint `/v1/ocr`.
pub fn build_router(state: Arc<AppState>) -> Router {
    let ollama_routes = Router::new()
        // Health + version
        .route("/", get(crate::routes::ollama_manage::health))
        .route("/api/version", get(crate::routes::ollama_manage::version))
        // Model listing
        .route("/api/tags", get(crate::routes::ollama_tags::tags_handler))
        .route("/api/ps", get(crate::routes::ollama_tags::ps_handler))
        // Model info
        .route("/api/show", post(crate::routes::ollama_show::show_handler))
        // Inference
        .route("/api/chat", post(crate::routes::ollama_chat::chat_handler))
        .route(
            "/api/generate",
            post(crate::routes::ollama_generate::generate_handler),
        )
        .route(
            "/api/embed",
            post(crate::routes::ollama_embed::embed_handler),
        )
        .route(
            "/api/embeddings",
            post(crate::routes::ollama_embed::embeddings_handler),
        )
        // Model management
        .route(
            "/api/pull",
            post(crate::routes::ollama_manage::pull_handler),
        )
        .route(
            "/api/delete",
            delete(crate::routes::ollama_manage::delete_handler),
        )
        .route(
            "/api/copy",
            post(crate::routes::ollama_manage::copy_handler),
        )
        .route(
            "/api/create",
            post(crate::routes::ollama_manage::create_handler),
        )
        .route(
            "/api/push",
            post(crate::routes::ollama_manage::push_handler),
        )
        // Observability
        .route(
            "/metrics",
            get(crate::routes::ollama_manage::metrics_handler),
        );

    let rag_routes = Router::new()
        .route(
            "/v1/collections",
            get(crate::routes::rag::list_collections).post(crate::routes::rag::create_collection),
        )
        .route(
            "/v1/collections/{name}",
            get(crate::routes::rag::get_collection).delete(crate::routes::rag::delete_collection),
        )
        .route(
            "/v1/collections/{name}/reindex",
            post(crate::routes::rag::reindex_collection),
        )
        .route(
            "/v1/collections/{name}/documents",
            post(crate::routes::rag::ingest_document).get(crate::routes::rag::list_documents),
        )
        .route(
            "/v1/collections/{name}/documents:batch",
            post(crate::routes::rag::ingest_documents_batch).layer(
                axum::extract::DefaultBodyLimit::max(crate::routes::rag::MAX_BATCH_BYTES),
            ),
        )
        .route(
            "/v1/collections/{name}/documents/{id}",
            delete(crate::routes::rag::delete_document),
        )
        .route(
            "/v1/collections/{name}/search",
            post(crate::routes::rag::search_collection),
        )
        .route("/v1/rag/chat", post(crate::routes::rag::rag_chat_handler));

    let engine_routes = Router::new()
        .route(
            "/engine/profiles",
            get(crate::routes::engine::profiles_handler),
        )
        .route(
            "/engine/profiles/active",
            axum::routing::put(crate::routes::engine::activate_handler),
        )
        .route("/engine/status", get(crate::routes::engine::status_handler))
        .route("/engine/ready", get(crate::routes::engine::ready_handler))
        .route("/engine/device", get(crate::routes::engine::device_handler))
        .route(
            "/engine/slots/{id}/load",
            post(crate::routes::engine::slot_load_handler),
        )
        .route(
            "/engine/slots/{id}/unload",
            post(crate::routes::engine::slot_unload_handler),
        )
        .route("/engine/events", get(crate::routes::engine::events_handler));

    let openai_routes = Router::new()
        .route(
            "/v1/chat/completions",
            post(crate::routes::openai::chat_completions_handler),
        )
        .route(
            "/v1/completions",
            post(crate::routes::openai::completions_handler),
        )
        .route(
            "/v1/embeddings",
            post(crate::routes::openai::embeddings_handler),
        )
        .route("/v1/rerank", post(crate::routes::openai::rerank_handler))
        .route("/v1/models", get(crate::routes::openai::models_handler))
        .route(
            "/v1/messages",
            post(crate::routes::anthropic::messages_handler),
        )
        .route("/v1/ocr", post(crate::routes::ocr::ocr_handler));

    ollama_routes
        .merge(rag_routes)
        .merge(openai_routes)
        .merge(engine_routes)
        // Hooks fire on every route; auth (outermost) runs before hooks.
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            request_hook_middleware,
        ))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state)
}
