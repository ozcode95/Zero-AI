//! Hardware / OS probe. Cached to `system.json` so we don't pay the WMI cost
//! on every launch — re-probe is opt-in.
//!
//! GPU/VRAM detection is delegated to llmfit-core's `SystemSpecs::detect()`,
//! which already calls nvidia-smi, rocm-smi, wmic, and Vulkan internally.
//! This removes the need for the `wmi` and `windows` crate dependencies.

pub mod recommend;

use anyhow::Result;
use chrono::Utc;
use llmfit_core::hardware::{GpuBackend, SystemSpecs as LlmSpecs};
use serde::{Deserialize, Serialize};
use sysinfo::System;

use crate::paths;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuKind {
    Integrated,
    Discrete,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub vram_mb: Option<u64>,
    pub kind: GpuKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpuInfo {
    pub name: String,
    pub vendor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Specs {
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub cpu_brand: String,
    pub cpu_vendor: String,
    pub cpu_physical_cores: usize,
    pub cpu_logical_cores: usize,
    pub ram_total_mb: u64,
    pub gpus: Vec<GpuInfo>,
    pub npus: Vec<NpuInfo>,
    pub probed_at: String,
    /// Maximum CUDA version major supported by the installed NVIDIA driver
    /// (e.g. `13` for CUDA 13.x, `12` for CUDA 12.x). `None` when no NVIDIA
    /// GPU is detected or when `nvidia-smi` cannot be queried.
    /// `#[serde(default)]` ensures older cached `system.json` files without
    /// this field deserialise cleanly as `None`.
    #[serde(default)]
    pub cuda_version_major: Option<u32>,
}

pub fn load_cached() -> Option<Specs> {
    let p = paths::system_cache().ok()?;
    let s = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&s).ok()
}

pub fn save_cached(specs: &Specs) -> Result<()> {
    let p = paths::system_cache()?;
    std::fs::write(p, serde_json::to_vec_pretty(specs)?)?;
    Ok(())
}

/// True when the host has at least one [`GpuKind::Discrete`] GPU.
///
/// Used at startup to pick which local runtime to auto-provision:
/// dGPU → llama.cpp (the variant builds light up CUDA/SYCL/HIP
/// kernels), no dGPU → OVMS (whose OpenVINO backend gets the most
/// out of integrated Intel hardware and CPUs).
pub fn has_discrete_gpu(specs: &Specs) -> bool {
    specs
        .gpus
        .iter()
        .any(|g| matches!(g.kind, GpuKind::Discrete))
}

pub fn probe() -> Result<Specs> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpus = sys.cpus();
    let cpu_brand = cpus
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_default();
    let cpu_vendor = cpus
        .first()
        .map(|c| c.vendor_id().to_string())
        .unwrap_or_default();

    let logical_cores = cpus.len();
    let physical_cores = sys.physical_core_count().unwrap_or(logical_cores);

    // Delegate GPU/VRAM detection to llmfit-core. It already handles:
    //   - NVIDIA: nvidia-smi (accurate 64-bit VRAM, multi-GPU, unified memory)
    //   - AMD:    rocm-smi + sysfs fallback
    //   - Windows: wmic (no extra crate needed)
    //   - Intel Arc: sysfs
    //   - Apple Silicon: unified memory
    // `prefer_discrete_gpus()` inside llmfit filters out iGPUs when a dGPU
    // is present, so every entry here is effectively a discrete accelerator.
    // iGPUs (e.g. Intel UHD/Iris on a laptop that also has an NVIDIA dGPU) are
    // dropped by llmfit — re-add them via a WMI name scan so the system info
    // surface on the settings page lists every graphics adapter.
    let llm_specs = LlmSpecs::detect();
    let mut gpus: Vec<GpuInfo> = llm_specs
        .gpus
        .iter()
        .map(|g| GpuInfo {
            name: g.name.clone(),
            vendor: backend_to_vendor(g.backend, &g.name),
            vram_mb: g.vram_gb.map(|v| (v * 1024.0) as u64),
            kind: classify_gpu_kind(&g.name),
        })
        .collect();

    // Re-add integrated GPUs that llmfit dropped when a dGPU was present
    // (Windows-only — Linux/macOS paths differ and llmfit already keeps the
    // sole iGPU on iGPU-only hosts).
    #[cfg(target_os = "windows")]
    {
        let known: std::collections::HashSet<String> =
            gpus.iter().map(|g| g.name.clone()).collect();
        for name in enumerate_windows_gpu_names() {
            if known.contains(&name) {
                continue;
            }
            if is_integrated_gpu_name(&name) {
                gpus.push(GpuInfo {
                    name: name.clone(),
                    vendor: infer_vendor_from_name(&name),
                    vram_mb: None, // iGPUs share system RAM; no dedicated VRAM
                    kind: GpuKind::Integrated,
                });
            }
        }
    }

    // Detect the maximum CUDA version the NVIDIA driver supports. Used at
    // install time to choose between cuda-13.x and cuda-12.4 llama.cpp assets.
    let cuda_version_major = if llm_specs.gpus.iter().any(|g| g.backend == GpuBackend::Cuda) {
        detect_cuda_version_major()
    } else {
        None
    };

    Ok(Specs {
        os: System::name().unwrap_or_else(|| "unknown".into()),
        os_version: System::os_version().unwrap_or_else(|| "?".into()),
        arch: std::env::consts::ARCH.to_string(),
        cpu_brand,
        cpu_vendor,
        cpu_physical_cores: physical_cores,
        cpu_logical_cores: logical_cores,
        ram_total_mb: sys.total_memory() / 1024 / 1024,
        gpus,
        npus: Vec::new(), // Intel NPU (AI Boost) was detected via WMI; not available through llmfit
        probed_at: Utc::now().to_rfc3339(),
        cuda_version_major,
    })
}

/// Query `nvidia-smi` to find the maximum CUDA version the installed NVIDIA
/// driver supports. Returns the major version number (e.g. `13` for CUDA 13.5,
/// `12` for CUDA 12.4). Returns `None` when `nvidia-smi` is not available or
/// does not report a parseable CUDA version line.
fn detect_cuda_version_major() -> Option<u32> {
    let output = std::process::Command::new("nvidia-smi").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    // The nvidia-smi header contains a line like:
    //   "| NVIDIA-SMI 576.52   Driver Version: 576.52   CUDA Version: 13.5     |"
    for line in text.lines() {
        if let Some(pos) = line.find("CUDA Version:") {
            let rest = line[pos + "CUDA Version:".len()..].trim();
            // Take the first whitespace-delimited token (strips trailing pipe/spaces)
            let version_str = rest.split_whitespace().next().unwrap_or("");
            let major_str = version_str.split('.').next().unwrap_or("");
            if let Ok(major) = major_str.parse::<u32>() {
                return Some(major);
            }
        }
    }
    None
}

/// Derive a vendor string from llmfit's [`GpuBackend`] tag and the GPU name.
/// llmfit doesn't expose a separate vendor field, so we infer it.
/// Classify a GPU name as integrated vs discrete. Mirrors the heuristic in
/// llmfit-core's `is_integrated_gpu_name` (`src/hardware.rs`) so our `kind`
/// tag matches what llmfit would call the same adapter.
fn is_integrated_gpu_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    if lower.contains("(integrated)") {
        return true;
    }
    // Intel integrated: UHD, HD Graphics, Iris — but NOT Intel Arc (discrete).
    if lower.contains("intel")
        && !lower.contains("arc")
        && (lower.contains("uhd") || lower.contains("hd graphics") || lower.contains("iris"))
    {
        return true;
    }
    // AMD integrated (APU): "Radeon Graphics" / "Radeon(TM) Graphics" without
    // an RX/PRO/XT discrete designation.
    if (lower.contains("radeon graphics") || lower.contains("radeon(tm) graphics"))
        && !lower.contains("rx")
        && !lower.contains("pro")
        && !lower.contains("xt")
    {
        return true;
    }
    false
}

/// Classify a GPU as `Integrated` / `Discrete` / `Unknown` from its name.
/// Used for both llmfit-returned GPUs (so an iGPU-only host isn't mis-tagged
/// `Discrete`) and WMI-recovered names.
fn classify_gpu_kind(name: &str) -> GpuKind {
    if is_integrated_gpu_name(name) {
        GpuKind::Integrated
    } else if name.trim().is_empty() {
        GpuKind::Unknown
    } else {
        GpuKind::Discrete
    }
}

/// Infer a vendor string from a GPU name when the backend tag is unavailable
/// (recovered iGPUs from WMI). Matches `backend_to_vendor`'s Vulkan branch.
fn infer_vendor_from_name(name: &str) -> String {
    let n = name.to_lowercase();
    if n.contains("nvidia") {
        "NVIDIA".to_string()
    } else if n.contains("amd") || n.contains("radeon") {
        "Advanced Micro Devices".to_string()
    } else if n.contains("intel") {
        "Intel".to_string()
    } else {
        name.to_string()
    }
}

/// Enumerate every graphics adapter name Windows reports. Used to recover
/// iGPUs that llmfit-core drops on hosts that also have a dGPU.
///
/// Uses PowerShell + `Get-CimInstance` (`wmic` is deprecated / removed on
/// recent Windows 11 builds, and not reliably on PATH). Returns an empty list
/// when PowerShell is unavailable; in that case we fall back to llmfit's
/// GPU list alone.
#[cfg(target_os = "windows")]
fn enumerate_windows_gpu_names() -> Vec<String> {
    let output = match std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name",
        ])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };
    if !output.status.success() {
        return Vec::new();
    }
    let text = match String::from_utf8(output.stdout) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    // One adapter name per line, possibly with a trailing blank line.
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

fn backend_to_vendor(backend: GpuBackend, name: &str) -> String {
    let n = name.to_lowercase();
    match backend {
        GpuBackend::Cuda => "NVIDIA".to_string(),
        GpuBackend::Rocm => "Advanced Micro Devices".to_string(),
        GpuBackend::Vulkan => {
            if n.contains("nvidia") {
                "NVIDIA".to_string()
            } else if n.contains("amd") || n.contains("radeon") {
                "Advanced Micro Devices".to_string()
            } else if n.contains("intel") {
                "Intel".to_string()
            } else {
                name.to_string()
            }
        }
        GpuBackend::Sycl => "Intel".to_string(),
        GpuBackend::Metal => "Apple".to_string(),
        GpuBackend::Ascend => "Huawei".to_string(),
        GpuBackend::CpuX86 | GpuBackend::CpuArm => String::new(),
    }
}
