mod commands;

use clap::{Parser, Subcommand};
use mettle_core::config::DEFAULT_OLLAMA_URL;

#[derive(Parser)]
#[command(name = "mettle", about = "A thin, Rust-native inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the engine server
    Serve {
        /// Path to config TOML file
        #[arg(short, long)]
        config: Option<String>,
    },
    /// Pull a model from Hugging Face
    Pull {
        /// Model name (e.g. gemma4, gemma4:e4b)
        model: String,
        /// Base URL of the engine server
        #[arg(short, long, default_value = DEFAULT_OLLAMA_URL)]
        url: String,
    },
    /// Ingest documents into a collection
    Ingest {
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Path to file to ingest
        #[arg(short, long)]
        file: String,
        /// Base URL of the engine server
        #[arg(short, long, default_value = DEFAULT_OLLAMA_URL)]
        url: String,
    },
    /// Search a collection
    Search {
        /// Collection name
        #[arg(short, long)]
        collection: String,
        /// Search query
        #[arg(short, long)]
        query: String,
        /// Number of results
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Base URL of the engine server
        #[arg(short, long, default_value = DEFAULT_OLLAMA_URL)]
        url: String,
    },
    /// Run benchmarks (chat throughput/TTFT and search latency), emitting JSON
    Bench {
        /// Path to output JSON results (stdout if omitted)
        #[arg(short, long)]
        output: Option<String>,
        /// Base URL of the engine server
        #[arg(short, long, default_value = DEFAULT_OLLAMA_URL)]
        url: String,
        /// Model to benchmark chat against
        #[arg(short, long, default_value = "gemma4")]
        model: String,
        /// Collection to benchmark search against (skipped if omitted)
        #[arg(short, long)]
        collection: Option<String>,
        /// Number of requests per benchmark section
        #[arg(short = 'n', long, default_value = "16")]
        requests: usize,
        /// Concurrent chat sessions
        #[arg(short = 'j', long, default_value = "4")]
        concurrency: usize,
        /// Prompt / query text
        #[arg(short, long, default_value = "Explain why the sky is blue in one paragraph.")]
        prompt: String,
    },
    /// Detect hardware and write a tuned config.toml
    Tune {
        /// Where to write the config (default: ~/.config/mettle/config.toml)
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Serve { config } => {
            commands::serve::run(config).await;
        }
        Command::Pull { model, url } => {
            commands::pull::run(model, url).await;
        }
        Command::Ingest {
            collection,
            file,
            url,
        } => {
            commands::ingest::run(collection, file, url).await;
        }
        Command::Search {
            collection,
            query,
            k,
            url,
        } => {
            commands::search::run(collection, query, k, url).await;
        }
        Command::Bench {
            output,
            url,
            model,
            collection,
            requests,
            concurrency,
            prompt,
        } => {
            commands::bench::run(
                commands::bench::BenchOptions {
                    url,
                    model,
                    collection,
                    requests,
                    concurrency,
                    prompt,
                },
                output,
            )
            .await;
        }
        Command::Tune { output } => {
            commands::tune::run(output);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn parse_serve_with_config() {
        let cli = Cli::try_parse_from(["mettle", "serve", "--config", "/tmp/c.toml"]).unwrap();
        assert!(matches!(cli.command, Command::Serve { config: Some(c) } if c == "/tmp/c.toml"));
    }

    #[test]
    fn parse_pull() {
        let cli = Cli::try_parse_from(["mettle", "pull", "gemma4:e4b"]).unwrap();
        assert!(
            matches!(cli.command, Command::Pull { model, url } if model == "gemma4:e4b" && url == DEFAULT_OLLAMA_URL)
        );
    }

    #[test]
    fn parse_search_defaults() {
        let cli =
            Cli::try_parse_from(["mettle", "search", "-c", "docs", "-q", "hello"]).unwrap();
        match cli.command {
            Command::Search { collection, query, k, url } => {
                assert_eq!(collection, "docs");
                assert_eq!(query, "hello");
                assert_eq!(k, 10);
                assert_eq!(url, DEFAULT_OLLAMA_URL);
            }
            _ => panic!("expected search"),
        }
    }

    #[test]
    fn missing_subcommand_is_an_error() {
        assert!(Cli::try_parse_from(["mettle"]).is_err());
    }
}
