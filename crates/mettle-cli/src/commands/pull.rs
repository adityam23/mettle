use mettle_core::ollama_types::{PullProgress, PullRequest};

pub async fn run(model: String, url: String) {
    let client = reqwest::Client::new();

    let resp = match client
        .post(format!("{}/api/pull", url))
        .json(&PullRequest {
            model,
            insecure: None,
            stream: Some(true),
        })
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to connect to engine: {}", e);
            std::process::exit(1);
        }
    };

    use futures::StreamExt;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<PullProgress>(line) {
                        Ok(p) => {
                            let pct = if let (Some(total), Some(completed)) =
                                (p.total, p.completed)
                            {
                                format!(" {:.1}%", completed as f64 / total as f64 * 100.0)
                            } else {
                                String::new()
                            };
                            eprintln!("{} {}", p.status, pct);
                        }
                        Err(_) => {
                            // Non-progress JSON — could be an error.
                            if let Ok(err) =
                                serde_json::from_str::<serde_json::Value>(line)
                            {
                                if err.get("error").is_some() {
                                    eprintln!("error: {}", err["error"]);
                                    std::process::exit(1);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("stream error: {}", e);
                break;
            }
        }
    }

    eprintln!("pull complete");
}
