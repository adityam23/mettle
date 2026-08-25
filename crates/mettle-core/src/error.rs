use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("model registry error: {0}")]
    Registry(String),

    #[error("retrieval error: {0}")]
    Retrieval(String),

    #[error("conflict: {0}")]
    Conflict(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    #[error("document not found: {0}")]
    DocumentNotFound(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("internal error: {0}")]
    Internal(String),
}
