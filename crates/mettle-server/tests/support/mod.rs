//! Shared fixtures for the HTTP integration tests.

use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// `(alias, detail)` for a reranker whose per-call context "does not
    /// fit" — drives the §1.1 `rerank_context_vram` refusal.
    rerank_vram_refusal: Arc<Mutex<Option<(String, String)>>>,
    /// Embedding requests received (one per input text).
    embed_calls: Arc<AtomicUsize>,
}

impl FakeEmbedder {
    /// `models`: `(alias, dim)` pairs that start resident.
    pub fn new(models: &[(&str, usize)]) -> Arc<Self> {
        let resident: Arc<Mutex<Vec<(String, usize)>>> = Arc::new(Mutex::new(
            models.iter().map(|(a, d)| (a.to_string(), *d)).collect(),
        ));
        let (tx, mut rx) = mpsc::channel::<EngineRequest>(64);
        let worker_resident = Arc::clone(&resident);
        let refusal: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
        let worker_refusal = Arc::clone(&refusal);
        let embed_calls = Arc::new(AtomicUsize::new(0));
        let worker_embed_calls = Arc::clone(&embed_calls);
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let dim = worker_resident
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|(a, _)| *a == req.model)
                    .map(|(_, d)| *d);
                let reply = match &req.kind {
                    RequestKind::Rerank { documents, .. } => {
                        let refusing = worker_refusal
                            .lock()
                            .unwrap()
                            .clone()
                            .filter(|(a, _)| *a == req.model);
                        match refusing {
                            Some((_, detail)) => EngineResponse::ContextVram(detail),
                            None => {
                                EngineResponse::Scores((0..documents.len()).map(|_| 0.5).collect())
                            }
                        }
                    }
                    RequestKind::Embedding { input } => {
                        worker_embed_calls.fetch_add(1, Ordering::SeqCst);
                        match dim {
                            Some(dim) => EngineResponse::Embeddings(bag_of_words(input, dim)),
                            None => {
                                EngineResponse::Error(format!("model {} is not loaded", req.model))
                            }
                        }
                    }
                    _ => EngineResponse::Error("fake embedder only embeds".into()),
                };
                let _ = req.response.send(reply).await;
            }
        });
        Arc::new(Self {
            tx,
            resident,
            rerank_vram_refusal: refusal,
            embed_calls,
        })
    }

    /// Embedding requests received so far (one per input text).
    pub fn embed_calls(&self) -> usize {
        self.embed_calls.load(Ordering::SeqCst)
    }

    pub fn unload(&self, alias: &str) {
        self.resident.lock().unwrap().retain(|(a, _)| a != alias);
    }

    /// Make `alias`'s rerank calls fail with the §1.1 structured refusal.
    pub fn fail_rerank_context_vram(&self, alias: &str, detail: &str) {
        *self.rerank_vram_refusal.lock().unwrap() = Some((alias.to_string(), detail.to_string()));
    }
}

fn bag_of_words(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let h = word
            .to_lowercase()
            .bytes()
            .fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
            });
        v[(h % dim as u64) as usize] += 1.0;
    }
    v
}

impl Engine for FakeEmbedder {
    fn sender(&self) -> mpsc::Sender<EngineRequest> {
        self.tx.clone()
    }
    fn loaded(&self) -> Vec<String> {
        self.resident
            .lock()
            .unwrap()
            .iter()
            .map(|(a, _)| a.clone())
            .collect()
    }
    fn load_model(&self, _spec: ModelLoadSpec) -> Result<(), EngineError> {
        Err(EngineError::Other(
            "fake embedder cannot load models".into(),
        ))
    }
    fn remove_model(&self, alias: &str) -> Result<(), EngineError> {
        self.unload(alias);
        Ok(())
    }
}
