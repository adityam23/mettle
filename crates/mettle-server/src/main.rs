use mettle_server::{app, llama_bridge, state, tune};

use std::net::SocketAddr;
use std::sync::Arc;

use mettle_core::config::RootConfig;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    // Never dump a ggml/abort backtrace (residency contract §1.6): a
    // supervised daemon's failures are structured (exit code + logs), and a
    // backtrace turns a clean exit into a crash the supervisor must parse.
    std::env::set_var("GGML_NO_BACKTRACE", "1");

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .init();

    let config_path = std::env::var("METTLE_CONFIG")
        .ok()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| mettle_core::config::config_dir().join("config.toml"));

    let config = match RootConfig::load_or_default(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::error!("failed to load config: {}", e);
            std::process::exit(1);
        }
    };

    // `mettled tune [output.toml]` — auto-tune and exit (spec §4).
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("tune") {
        let output = args.next().map(std::path::PathBuf::from);
        std::process::exit(tune::run(&config, output));
    }

    let app_state = Arc::new(state::AppState::new(config.clone()));

    // Load existing collections from disk on startup.
    if let Ok(mut store) = app_state.collections.lock() {
        match store.load_all() {
            Ok(n) => tracing::info!("loaded {} collections from disk", n),
            Err(e) => tracing::warn!("failed to load collections: {}", e),
        }
    }

    // The llama worker must exist before anything can load into it
    // (LlamaBackend is init-once-per-process). Boot creates it EMPTY — the
    // profiles contract has no implicit default profile: what loads is
    // decided by the persisted active profile, a JIT slot load, or
    // `/api/pull`. Inference routes answer 503 until then.
    {
        let cfg = llama_bridge::build_llama_config(&config);
        match tokio::task::spawn_blocking(move || mettle_llama::LlamaEngine::new(Vec::new(), cfg))
            .await
        {
            Ok(Ok(engine)) => {
                app_state.set_engine(Arc::new(engine) as mettle_server::state::SharedEngine);
            }
            Ok(Err(e)) => tracing::error!("engine worker failed to start: {e}"),
            Err(e) => tracing::error!("engine worker task panicked: {e}"),
        }
    }

    // Re-activate the persisted profile with degradation applied; without
    // one, the engine idles until a profile is activated.
    tokio::spawn(app_state.profiles.clone().boot_activate());
    tokio::spawn(app_state.profiles.clone().ttl_sweeper_loop());

    let addr = SocketAddr::new(
        config.server.host.parse().expect("invalid host"),
        config.server.port,
    );

    let router = app::build_router(app_state);

    // Exit 98 (EADDRINUSE) is the contract's "another engine holds the
    // port" signal (residency contract §1.6): docket maps it to `offline`
    // and does not respawn, rather than treating it as a crash.
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            tracing::error!("port {} is already in use: {e}", addr.port());
            eprintln!("mettled: port {} is already in use", addr.port());
            std::process::exit(98);
        }
        Err(e) => {
            tracing::error!("failed to bind {addr}: {e}");
            eprintln!("mettled: failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!(
        "engine listening on {} (Ollama /api/*, OpenAI /v1/*, Anthropic /v1/messages, OCR /v1/ocr)",
        addr
    );
    if let Err(e) = axum::serve(listener, router).await {
        tracing::error!("server error: {e}");
        std::process::exit(1);
    }
}
