use mettle_core::retrieval::{SearchMode, SearchRequest};

pub async fn run(collection: String, query: String, k: usize, url: String) {
    let client = reqwest::Client::new();
    let search_url = format!("{}/v1/collections/{}/search", url, collection);

    match client
        .post(&search_url)
        .json(&SearchRequest {
            query,
            k,
            mode: SearchMode::Hybrid,
            allowlist: None,
            filter: None,
        })
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.is_success() {
                // Pretty-print the results.
                match serde_json::from_str::<serde_json::Value>(&body) {
                    Ok(val) => println!("{}", serde_json::to_string_pretty(&val).unwrap_or(body)),
                    Err(_) => println!("{}", body),
                }
            } else {
                eprintln!("search failed ({}): {}", status, body);
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("search request failed: {}", e);
            std::process::exit(1);
        }
    }
}
