//! OCR endpoint wire types (`POST /v1/ocr`).
//!
//! Contract: image in, text out. The consumer (agent-assistant) rasterizes
//! PDF pages itself and posts encoded images; mettle never touches PDFs.

use serde::{Deserialize, Serialize};

/// Default OCR instruction used when the request omits one. It is the
/// supported OCR model's (GLM-OCR) trained prompt for plain text; the model
/// also knows `Table Recognition:` and `Formula Recognition:`, which a
/// request can pass as its `instruction`.
pub const DEFAULT_OCR_INSTRUCTION: &str = "Text Recognition:";

#[derive(Debug, Clone, Deserialize)]
pub struct OcrRequest {
    /// Encoded image bytes (PNG/JPEG/BMP/GIF — stb_image formats), base64.
    pub image: String,
    /// Optional instruction override; defaults to [`DEFAULT_OCR_INSTRUCTION`].
    #[serde(default)]
    pub instruction: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OcrResponse {
    pub text: String,
    pub model: String,
}
