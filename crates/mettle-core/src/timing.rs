/// ISO 8601 timestamp string for Ollama responses.
pub fn timestamp_iso8601() -> String {
    let now: chrono::DateTime<chrono::Utc> = chrono::Utc::now();
    now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
