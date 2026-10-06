//! Engine server library: axum router, state, hooks wiring, llama.cpp bridge.
//!
//! The `engine` binary (`main.rs`) is a thin wrapper around this crate;
//! integration tests exercise the router through these exports.

pub mod app;
pub mod events;
pub mod hooks_integration;
pub mod llama_bridge;
pub mod profiles;
pub mod response_stream;
pub mod routes;
pub mod state;
pub mod tune;
pub mod unavailable;
