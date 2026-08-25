//! Device abstraction for VRAM-aware profile planning
//! (agent-docs/profiles-contract.md).
//!
//! The planner only ever sees [`Device`] — never a backend API. CUDA
//! reports device-wide truth via `nvidia-smi` (other processes are
//! visible; the planner's safety margin absorbs them). A future Metal
//! backend reports budget-derived numbers (`recommendedMaxWorkingSetSize`
//! − own usage) through the same trait, plus a real [`Device::clear_cache`]
//! (MLX caches allocations; CUDA frees eagerly, so CUDA's is a no-op).
//!
//! `memory()` shells out (~50–100 ms): call it at activation/planning
//! time, never per request.

/// A memory snapshot in bytes. `free_bytes` is an estimate the planner
/// budgets against; the per-backend safety margin covers its coarseness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// One accelerator the planner can place model weights on.
pub trait Device: Send + Sync {
    /// Backend identifier (`"cuda"`, later `"metal"`).
    fn name(&self) -> &str;

    /// Current memory snapshot; `None` when the device is unreachable
    /// (no driver, no GPU). A `None` means GPU slots are unloadable —
    /// the GPU-or-refuse policy then refuses them.
    fn memory(&self) -> Option<DeviceMemory>;

    /// Release backend-held allocation caches before accounting. No-op on
    /// CUDA; mandatory on MLX/Metal where freed tensors stay cached.
    fn clear_cache(&self) {}
}

/// CUDA device probed through `nvidia-smi` (the same pattern as
/// `tune.rs`; no new link-time dependency on the CUDA driver API).
pub struct CudaDevice;

impl Device for CudaDevice {
    fn name(&self) -> &str {
        "cuda"
    }

    fn memory(&self) -> Option<DeviceMemory> {
        let out = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        // One "total, free" row (MiB) per GPU; plan against the GPU with
        // the most free memory (multi-GPU placement is deferred — this
        // matches the single-accelerator reality of every current host).
        let text = String::from_utf8_lossy(&out.stdout);
        let mut best: Option<DeviceMemory> = None;
        for line in text.lines() {
            let mut parts = line.split(',');
            let total_mib: u64 = parts.next()?.trim().parse().ok()?;
            let free_mib: u64 = parts.next()?.trim().parse().ok()?;
            if total_mib == 0 {
                continue;
            }
            let m = DeviceMemory {
                total_bytes: total_mib * 1024 * 1024,
                free_bytes: free_mib * 1024 * 1024,
            };
            if best.is_none_or(|b| m.free_bytes > b.free_bytes) {
                best = Some(m);
            }
        }
        best
    }
}

/// Probe for the first supported accelerator. CUDA today; Metal joins
/// behind the same trait when the llama.cpp Metal backend is wired up.
pub fn detect() -> Option<Box<dyn Device>> {
    let cuda = CudaDevice;
    cuda.memory().map(|_| Box::new(cuda) as Box<dyn Device>)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_memory_snapshot_is_consistent_when_present() {
        match CudaDevice.memory() {
            Some(m) => {
                assert!(m.total_bytes > 0);
                assert!(m.free_bytes <= m.total_bytes);
            }
            // No GPU / no driver on this host: the probe must fail soft.
            None => {}
        }
    }
}
