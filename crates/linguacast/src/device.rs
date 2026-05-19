use crate::cli;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Mps,
    Cuda,
    Cpu,
}

impl Device {
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Mps => "mps",
            Device::Cuda => "cuda",
            Device::Cpu => "cpu",
        }
    }
}

/// Resolve the device to use. Auto-detect prefers MPS on macOS-arm64, then
/// CUDA, then CPU. The actual availability check is done in the Python
/// sidecar; here we just translate the user's choice into a request hint.
pub fn resolve(arg: Option<&cli::Device>) -> Device {
    match arg {
        Some(cli::Device::Mps) => Device::Mps,
        Some(cli::Device::Cuda) => Device::Cuda,
        Some(cli::Device::Cpu) => Device::Cpu,
        None | Some(cli::Device::Auto) => autodetect(),
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn autodetect() -> Device {
    Device::Mps
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn autodetect() -> Device {
    // CUDA presence is verified by the sidecar; if torch.cuda.is_available()
    // is false it falls back to CPU and reports it back to us.
    Device::Cuda
}

#[cfg(not(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(target_os = "linux", target_arch = "x86_64")
)))]
fn autodetect() -> Device {
    Device::Cpu
}
