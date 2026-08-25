//! NDJSON streaming response helper.
//!
//! Ollama uses `Content-Type: application/x-ndjson` with one JSON object per
//! line terminated by `\n`. The final line has `done: true` with timing.

#![allow(dead_code)]

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use futures::{Stream, StreamExt};
use serde::Serialize;
use std::pin::Pin;

/// Wraps a `Stream` of serializable items as an NDJSON response.
///
/// Each item is serialized to JSON, followed by `\n`. Errors from the
/// inner stream are logged and terminate the stream.
pub struct NdJsonStream<S> {
    inner: Pin<Box<S>>,
}

impl<S, T, E> NdJsonStream<S>
where
    S: Stream<Item = Result<T, E>> + Send + 'static,
    T: Serialize + 'static,
    E: std::fmt::Display + 'static,
{
    pub fn new(stream: S) -> Self {
        Self {
            inner: Box::pin(stream),
        }
    }
}

impl<S, T, E> IntoResponse for NdJsonStream<S>
where
    S: Stream<Item = Result<T, E>> + Send + 'static,
    T: Serialize + 'static,
    E: std::fmt::Display + 'static,
{
    fn into_response(self) -> Response {
        let stream = self.into_stream();
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap()
    }
}

impl<S, T, E> NdJsonStream<S>
where
    S: Stream<Item = Result<T, E>> + Send + 'static,
    T: Serialize + 'static,
    E: std::fmt::Display + 'static,
{
    fn into_stream(
        self,
    ) -> impl Stream<Item = Result<axum::body::Bytes, std::convert::Infallible>> {
        self.inner.map(|item| {
            let bytes = match item {
                Ok(value) => {
                    let mut json = serde_json::to_vec(&value).unwrap_or_default();
                    json.push(b'\n');
                    axum::body::Bytes::from(json)
                }
                Err(e) => {
                    tracing::error!("stream error: {}", e);
                    let mut json =
                        serde_json::to_vec(&serde_json::json!({"error": e.to_string()}))
                            .unwrap_or_default();
                    json.push(b'\n');
                    axum::body::Bytes::from(json)
                }
            };
            Ok(bytes)
        })
    }
}

/// Server-sent-events response. Each item becomes one frame:
/// * `events: true` (Anthropic style) — `event: <"type" field>\ndata: <json>`;
/// * `events: false` (OpenAI style) — `data: <json>`.
/// A final `data: <done>` frame is appended when `done_marker` is set
/// (OpenAI's `[DONE]`).
pub struct SseStream<S> {
    inner: Pin<Box<S>>,
    events: bool,
    done_marker: Option<&'static str>,
}

impl<S, T, E> SseStream<S>
where
    S: Stream<Item = Result<T, E>> + Send + 'static,
    T: Serialize + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    pub fn new(stream: S, events: bool) -> Self {
        Self {
            inner: Box::pin(stream),
            events,
            done_marker: None,
        }
    }

    /// Append a final `data: <marker>` frame after the stream ends.
    pub fn with_done(mut self, marker: &'static str) -> Self {
        self.done_marker = Some(marker);
        self
    }
}

impl<S, T, E> IntoResponse for SseStream<S>
where
    S: Stream<Item = Result<T, E>> + Send + 'static,
    T: Serialize + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    fn into_response(self) -> Response {
        let SseStream {
            inner,
            events,
            done_marker,
        } = self;
        let stream = async_stream::stream! {
            for await item in inner {
                let bytes = match item {
                    Ok(value) => {
                        let mut frame = String::new();
                        if events {
                            if let Some(type_) = serde_json::to_value(&value)
                                .ok()
                                .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(str::to_string))
                            {
                                frame.push_str(&format!("event: {type_}\n"));
                            }
                        }
                        frame.push_str("data: ");
                        frame.push_str(&serde_json::to_string(&value).unwrap_or_default());
                        frame.push_str("\n\n");
                        axum::body::Bytes::from(frame)
                    }
                    Err(e) => {
                        tracing::error!("stream error: {}", e);
                        axum::body::Bytes::from(format!(
                            "data: {}\n\n",
                            serde_json::json!({"error": e.to_string()})
                        ))
                    }
                };
                yield Ok::<_, std::convert::Infallible>(bytes);
            }
            if let Some(marker) = done_marker {
                yield Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                    "data: {marker}\n\n"
                )));
            }
        };
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(Body::from_stream(stream))
            .unwrap()
    }
}
