use axum::{http::StatusCode, Json};

/// Uniform error response shape matching Ollama's `{"error": "message"}`.
#[derive(serde::Serialize)]
pub struct ApiError {
    pub error: String,
}

/// Build an `(StatusCode, Json<ApiError>)` tuple that implements `IntoResponse`.
pub fn api_error(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ApiError>) {
    (
        status,
        Json(ApiError {
            error: msg.into(),
        }),
    )
}

/// Convert a raw status code (from `mettle_core::stream`'s `(u16, String)`
/// errors) into an `api_error` response.
pub fn api_error_code(code: u16, msg: impl Into<String>) -> (StatusCode, Json<ApiError>) {
    api_error(
        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        msg,
    )
}
