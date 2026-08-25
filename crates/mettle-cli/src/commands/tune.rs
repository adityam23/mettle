//! `mettle-cli tune` — delegates to the engine binary's `tune` mode:
//! VRAM detection (nvidia-smi) → per-model `n_gpu_layers` (full, partial,
//! or CPU) written to `config.toml`.

pub fn run(output: Option<String>) {
    let mut cmd = std::process::Command::new(
        std::env::current_exe()
            .unwrap_or_else(|_| "mettled".into())
            .with_file_name("mettled"),
    );
    cmd.arg("tune");
    if let Some(path) = &output {
        cmd.arg(path);
    }

    match cmd.status() {
        Ok(status) => {
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
        }
        Err(e) => {
            eprintln!(
                "failed to run mettled tune: {}. \
                 Ensure the `mettled` server binary is installed.",
                e
            );
            std::process::exit(1);
        }
    }
}
