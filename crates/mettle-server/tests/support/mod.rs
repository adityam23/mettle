//! Shared fixtures for the HTTP integration tests.

use std::sync::{Arc, Mutex};

use mettle_core::engine::{
    Engine, EngineError, EngineRequest, EngineResponse, ModelLoadSpec, RequestKind,
};
use tokio::sync::mpsc;

/// Deterministic in-process embedder: answers `RequestKind::Embedding` for
/// its resident aliases with a hashed bag-of-words vector of the alias's
/// dimension, so dense search has real (if crude) signal and no GGUF or
/// GPU is needed. `unload` simulates an embed slot going away.
pub struct FakeEmbedder {
    tx: mpsc::Sender<EngineRequest>,
    resident: Arc<Mutex<Vec<(String, usize)>>>,
}

impl FakeEmbedder {
    /// `models`: `(alias, dim)` pairs that start resident.
    pub fn new(models: &[(&str, usize)]) -> Arc<Self> {
        let resident: Arc<Mutex<Vec<(String, usize)>>> = Arc::new(Mutex::new(
            models.iter().map(|(a, d)| (a.to_string(), *d)).collect(),
        ));
        let (tx, mut rx) = mpsc::channel::<EngineRequest>(64);
        let worker_resident = Arc::clone(&resident);
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let dim = worker_resident
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|(a, _)| *a == req.model)
                    .map(|(_, d)| *d);
                let reply = match (&req.kind, dim) {
                    (RequestKind::Embedding { input }, Some(dim)) => {
                        EngineResponse::Embeddings(bag_of_words(input, dim))
                    }
                    (_, None) => EngineResponse::Error(format!("model {} is not loaded", req.model)),
                    _ => EngineResponse::Error("fake embedder only embeds".into()),
                };
                let _ = req.response.send(reply).await;
            }
        });
        Arc::new(Self { tx, resident })
    }

    pub fn unload(&self, alias: &str) {
        self.resident.lock().unwrap().retain(|(a, _)| a != alias);
    }
}

fn bag_of_words(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for word in text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
        let h = word
            .to_lowercase()
            .bytes()
            .fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
        v[(h % dim as u64) as usize] += 1.0;
    }
    v
}

impl Engine for FakeEmbedder {
    fn sender(&self) -> mpsc::Sender<EngineRequest> {
        self.tx.clone()
    }
    fn loaded(&self) -> Vec<String> {
        self.resident.lock().unwrap().iter().map(|(a, _)| a.clone()).collect()
    }
    fn load_model(&self, _spec: ModelLoadSpec) -> Result<(), EngineError> {
        Err(EngineError::Other("fake embedder cannot load models".into()))
    }
    fn remove_model(&self, alias: &str) -> Result<(), EngineError> {
        self.unload(alias);
        Ok(())
    }
}
