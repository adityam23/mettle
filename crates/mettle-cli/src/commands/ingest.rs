pub async fn run(collection: String, file: String, url: String) {
    let content = match std::fs::read_to_string(&file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to read {}: {}", file, e);
            std::process::exit(1);
        }
    };

    let doc_id = std::path::Path::new(&file)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "doc".into());

    let client = reqwest::Client::new();
    let ingest_url = format!("{}/v1/collections/{}/documents", url, collection);

    match client
        .post(&ingest_url)
        .json(&mettle_core::retrieval::Document {
            id: doc_id,
            content,
            metadata: Default::default(),
        })
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.is_success() {
                println!("{}", body);
            } else {
                eprintln!("ingest failed ({}): {}", status, body);
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("ingest request failed: {}", e);
            std::process::exit(1);
        }
    }
}
