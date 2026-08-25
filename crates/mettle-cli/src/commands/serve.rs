/// Delegates to the `mettled` server binary, passing the config path via
/// the METTLE_CONFIG environment variable.
pub async fn run(config_path: Option<String>) {
    let mut cmd = std::process::Command::new(
        std::env::current_exe()
            .unwrap_or_else(|_| "mettled".into())
            .with_file_name("mettled"),
    );
    if let Some(path) = &config_path {
        cmd.env("METTLE_CONFIG", path);
    }

    match cmd.status() {
        Ok(status) => {
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
        }
        Err(e) => {
            eprintln!(
                "failed to start mettled server: {}. \
                 Ensure the `mettled` server binary is installed.",
                e
            );
            std::process::exit(1);
        }
    }
}
