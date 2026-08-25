//! Benchmark harness (spec §9/§13): drives the engine's HTTP API and emits
//! reproducible JSON results — chat throughput and TTFT when a model is
//! loaded, search latency when a collection is given. The same harness is
//! used across releases so results are comparable.

use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct BenchOptions {
    pub url: String,
    pub model: String,
    pub collection: Option<String>,
    pub requests: usize,
    pub concurrency: usize,
    pub prompt: String,
}

#[derive(serde::Serialize, Default)]
struct LatencyStats {
    count: usize,
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
}

fn latency_stats(mut samples: Vec<Duration>) -> LatencyStats {
    if samples.is_empty() {
        return LatencyStats::default();
    }
    samples.sort();
    let ms = |d: &Duration| d.as_secs_f64() * 1000.0;
    let idx = |q: f64| ((samples.len() as f64 - 1.0) * q).round() as usize;
    LatencyStats {
        count: samples.len(),
        p50_ms: ms(&samples[idx(0.50)]),
        p95_ms: ms(&samples[idx(0.95)]),
        mean_ms: samples.iter().map(ms).sum::<f64>() / samples.len() as f64,
    }
}

/// One streaming chat request: returns (ttft, chunk count, wall time).
async fn chat_once(
    client: &reqwest::Client,
    opts: &BenchOptions,
) -> Result<(Duration, usize, Duration), String> {
    use futures::StreamExt;

    let started = Instant::now();
    let resp = client
        .post(format!("{}/api/chat", opts.url))
        .json(&mettle_core::ollama_types::ChatRequest {
            model: opts.model.clone(),
            messages: vec![mettle_core::ollama_types::ChatMessage {
                role: "user".into(),
                content: opts.prompt.clone(),
                images: None,
                tool_calls: None,
            }],
            tools: None,
            format: None,
            options: None,
            stream: Some(true),
            keep_alive: None,
        })
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {}: {}", status, body));
    }

    let mut ttft = None;
    let mut chunks = 0usize;
    let mut stream = resp.bytes_stream();
    while let Some(item) = stream.next().await {
        let bytes = item.map_err(|e| e.to_string())?;
        for line in String::from_utf8_lossy(&bytes).lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(err) = v.get("error") {
                    return Err(err.to_string());
                }
                chunks += 1;
                ttft.get_or_insert_with(|| started.elapsed());
            }
        }
    }

    Ok((
        ttft.ok_or("no chunks received")?,
        chunks,
        started.elapsed(),
    ))
}

async fn bench_chat(client: &reqwest::Client, opts: &BenchOptions) -> serde_json::Value {
    let started = Instant::now();
    let mut ttfts = Vec::new();
    let mut total_chunks = 0usize;
    let mut errors: Vec<String> = Vec::new();

    let mut remaining = opts.requests;
    while remaining > 0 {
        let batch = remaining.min(opts.concurrency);
        let futures: Vec<_> = (0..batch).map(|_| chat_once(client, opts)).collect();
        for result in futures::future::join_all(futures).await {
            match result {
                Ok((ttft, chunks, _wall)) => {
                    ttfts.push(ttft);
                    total_chunks += chunks;
                }
                Err(e) => errors.push(e),
            }
        }
        remaining -= batch;
    }

    let wall = started.elapsed().as_secs_f64();
    if ttfts.is_empty() {
        return serde_json::json!({
            "error": errors.first().cloned().unwrap_or_else(|| "no successful requests".into()),
            "failed_requests": errors.len(),
        });
    }

    serde_json::json!({
        "requests": opts.requests,
        "concurrency": opts.concurrency,
        "failed_requests": errors.len(),
        "ttft": latency_stats(ttfts),
        // One streamed chunk ≈ one token; the headline aggregate metric.
        "aggregate_tokens_per_second": total_chunks as f64 / wall,
        "wall_seconds": wall,
    })
}

async fn bench_search(
    client: &reqwest::Client,
    opts: &BenchOptions,
    collection: &str,
) -> serde_json::Value {
    let mut results = serde_json::Map::new();

    use mettle_core::retrieval::{SearchMode, SearchRequest};
    for mode in [SearchMode::Keyword, SearchMode::Hybrid, SearchMode::Dense] {
        let mut latencies = Vec::new();
        let mut error = None;
        for _ in 0..opts.requests {
            let started = Instant::now();
            let resp = client
                .post(format!("{}/v1/collections/{}/search", opts.url, collection))
                .json(&SearchRequest {
                    query: opts.prompt.clone(),
                    k: 10,
                    mode,
                    allowlist: None,
                    filter: None,
                })
                .send()
                .await;
            match resp {
                Ok(r) if r.status().is_success() => latencies.push(started.elapsed()),
                Ok(r) => {
                    error = Some(format!("HTTP {}", r.status()));
                    break;
                }
                Err(e) => {
                    error = Some(e.to_string());
                    break;
                }
            }
        }
        results.insert(
            mode.to_string(),
            match error {
                Some(e) => serde_json::json!({"error": e}),
                None => serde_json::to_value(latency_stats(latencies)).unwrap_or_default(),
            },
        );
    }

    serde_json::Value::Object(results)
}

pub async fn run(opts: BenchOptions, output: Option<String>) {
    let client = reqwest::Client::new();

    let chat = bench_chat(&client, &opts).await;
    let search = match &opts.collection {
        Some(c) => bench_search(&client, &opts, c).await,
        None => serde_json::json!({"skipped": "no --collection given"}),
    };

    let results = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "config": {
            "url": opts.url,
            "model": opts.model,
            "requests": opts.requests,
            "concurrency": opts.concurrency,
        },
        "chat": chat,
        "search": search,
    });

    let pretty = serde_json::to_string_pretty(&results).expect("results serialize");
    match output {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, &pretty) {
                eprintln!("failed to write bench results to {}: {}", path, e);
                std::process::exit(1);
            }
            eprintln!("bench results written to {}", path);
        }
        None => println!("{}", pretty),
    }
}
