//! Map a probed [`Specs`] snapshot to the llama.cpp Windows release variant(s)
//! we should download for this host.
//!
//! Supported variants and their priority:
//!
//! | Variant slug | Asset substring           | Port | Selected when                       |
//! |--------------|---------------------------|------|--------------------------------------|
//! | `cuda`       | `cuda-12.4`               | 8081 | A discrete NVIDIA GPU is present     |
//! | `openvino`   | `openvino`                | 8082 | An Intel CPU / iGPU / dGPU is present|
//! | `hip-radeon` | `hip-radeon`              | 8083 | A discrete AMD GPU is present         |
//! | `cpu`        | `cpu-x64`                 | 8084 | Fallback when no accelerator fits     |
//!
//! A machine can have multiple applicable variants (e.g. Intel CPU+iGPU +
//! NVIDIA dGPU → both `cuda` and `openvino`). All applicable variants are
//! installed, and the highest-priority one is started by default. The user
//! can switch variants or run multiple simultaneously (each on its own port).
//!
//! SYCL is intentionally omitted: OpenVINO covers all Intel accelerators
//! (iGPU, CPU, Arc dGPU) while SYCL only targets discrete Intel GPUs. Vulkan
//! is omitted due to poor inference performance compared to the native
//! backends.

use crate::system::{GpuKind, Specs};
use serde::{Deserialize, Serialize};

/// Which precompiled llama.cpp Windows zip we want for this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LlamaVariant {
    /// NVIDIA CUDA 12.4 build. Asset name contains `cuda-12.4` and `x64`.
    Cuda,
    /// Intel OpenVINO build. Asset name contains `openvino` and `x64`.
    /// Works on Intel CPUs, Intel iGPUs, and Intel Arc dGPUs.
    OpenVino,
    /// AMD ROCm/HIP build for Radeon RX / Pro discrete GPUs.
    HipRadeon,
    /// Pure CPU build, used when no supported accelerator is detected.
    Cpu,
}

impl LlamaVariant {
    /// Short, stable identifier used in logs, telemetry, URLs, and the
    /// frontend status pill.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::OpenVino => "openvino",
            Self::HipRadeon => "hip-radeon",
            Self::Cpu => "cpu",
        }
    }

    /// Human-readable name for the UI.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Cuda => "NVIDIA CUDA",
            Self::OpenVino => "Intel OpenVINO",
            Self::HipRadeon => "AMD HIP/Radeon",
            Self::Cpu => "CPU",
        }
    }

    /// Fixed port per variant so multiple instances can run simultaneously
    /// without port conflicts. The active variant is exposed on its port,
    /// and the user can reach any other running variant on its own port.
    pub fn default_port(self) -> u16 {
        match self {
            Self::Cuda => 8081,
            Self::OpenVino => 8082,
            Self::HipRadeon => 8083,
            Self::Cpu => 8084,
        }
    }

    /// Database key for the `runtime_versions` table. Each variant gets its
    /// own row so installs, updates, and lifecycle are fully independent.
    pub fn runtime_name(self) -> String {
        format!("llama.cpp-{}", self.slug())
    }

    /// Directory name under `runtimes/llama.cpp/`. Each variant is extracted
    /// into its own subdirectory so they can coexist on disk.
    pub fn dir_name(self) -> &'static str {
        self.slug()
    }

    /// Priority for auto-start: lower number = start first. The active
    /// variant at startup is whichever installed variant has the lowest
    /// priority value.
    pub fn priority(self) -> u8 {
        match self {
            Self::Cuda => 0,
            Self::HipRadeon => 1,
            Self::OpenVino => 2,
            Self::Cpu => 3,
        }
    }

    /// Parse a slug back into a variant. Returns `None` for unknown slugs.
    pub fn from_slug(s: &str) -> Option<Self> {
        match s {
            "cuda" => Some(Self::Cuda),
            "openvino" => Some(Self::OpenVino),
            "hip-radeon" => Some(Self::HipRadeon),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    /// All variant enum values in priority order.
    pub fn all() -> &'static [LlamaVariant] {
        &[
            LlamaVariant::Cuda,
            LlamaVariant::OpenVino,
            LlamaVariant::HipRadeon,
            LlamaVariant::Cpu,
        ]
    }

    /// Asset matcher for the upstream release asset list. We look for the
    /// canonical filename shape `llama-<build>-bin-win-<flavour>-x64.zip`.
    ///
    /// The CUDA matcher specifically picks the `cuda-12.4` flavour over
    /// `cuda-13.3`: 12.4 has wider driver compatibility, and the perf gap
    /// between the two builds is negligible for consumer GPUs.
    ///
    /// We anchor on the `llama-` filename prefix so the matcher does
    /// **not** accept companion archives that share the same flavour
    /// suffix — most importantly `cudart-llama-bin-win-cuda-12.4-x64.zip`,
    /// which is the redistributable CUDA runtime DLL bundle and does
    /// not contain a `llama-server.exe`.
    pub fn matches_asset(self, name: &str) -> bool {
        let n = name.to_lowercase();
        if !n.starts_with("llama-")
            || !n.contains("-bin-win-")
            || !n.contains("x64")
            || !n.ends_with(".zip")
        {
            return false;
        }
        match self {
            Self::Cuda => n.contains("cuda-12.4"),
            Self::OpenVino => n.contains("-openvino-"),
            Self::HipRadeon => n.contains("hip-radeon"),
            Self::Cpu => n.contains("-cpu-"),
        }
    }

    /// Optional companion archive that must be extracted *on top of*
    /// the main install for the binary to run on a typical user
    /// machine.
    ///
    /// Today only the CUDA build needs one: `llama-server.exe` from
    /// `llama-<build>-bin-win-cuda-12.4-x64.zip` links against
    /// `cudart64_12.dll`, `cublas64_12.dll`, and `cublasLt64_12.dll`,
    /// none of which ship with consumer NVIDIA drivers. The upstream
    /// release pairs the main zip with `cudart-llama-bin-win-cuda-12.4-x64.zip`,
    /// which contains exactly those redistributable DLLs.
    ///
    /// OpenVINO, HIP, and CPU have no companion archives.
    pub fn matches_companion_asset(self, name: &str) -> bool {
        let n = name.to_lowercase();
        if !n.contains("-bin-win-") || !n.contains("x64") || !n.ends_with(".zip") {
            return false;
        }
        match self {
            Self::Cuda => n.starts_with("cudart-") && n.contains("cuda-12.4"),
            Self::OpenVino | Self::HipRadeon | Self::Cpu => false,
        }
    }
}

/// Determine all applicable variants for this host. Returns variants in
/// priority order (highest priority first). The machine will get all
/// matching accelerator builds so the user can switch between them or run
/// them simultaneously on different ports.
///
/// Logic:
/// - NVIDIA dGPU → `Cuda`
/// - AMD dGPU → `HipRadeon`
/// - Intel iGPU or Intel CPU → `OpenVino`
/// - If none of the above match → `Cpu` (universal fallback)
pub fn select_variants(specs: &Specs) -> Vec<LlamaVariant> {
    let mut has_nvidia = false;
    let mut has_amd = false;
    let mut has_intel = false;

    for gpu in &specs.gpus {
        let v = gpu.vendor.to_lowercase();
        let n = gpu.name.to_lowercase();
        match gpu.kind {
            GpuKind::Discrete => {
                if v.contains("nvidia") || n.contains("nvidia") {
                    has_nvidia = true;
                } else if v.contains("amd") || n.contains("radeon") || v.contains("advanced micro")
                {
                    has_amd = true;
                } else if v.contains("intel") || n.contains("arc") {
                    has_intel = true;
                }
            }
            GpuKind::Integrated => {
                if v.contains("intel")
                    || n.contains("uhd")
                    || n.contains("iris")
                    || n.contains("intel")
                {
                    has_intel = true;
                }
            }
            GpuKind::Unknown => {}
        }
    }

    // Any Intel CPU can benefit from OpenVINO's optimised kernels.
    let cpu_is_intel = specs.cpu_vendor.to_lowercase().contains("intel");
    if cpu_is_intel {
        has_intel = true;
    }

    let mut variants = Vec::new();

    if has_nvidia {
        variants.push(LlamaVariant::Cuda);
    }
    if has_amd {
        variants.push(LlamaVariant::HipRadeon);
    }
    if has_intel {
        variants.push(LlamaVariant::OpenVino);
    }

    // Fallback: if no accelerator matched, use the CPU build.
    if variants.is_empty() {
        variants.push(LlamaVariant::Cpu);
    }

    // Sort by priority (lowest number first = highest priority).
    variants.sort_by_key(|v| v.priority());
    variants.dedup();

    variants
}

/// Convenience: return the single best variant for a given specs snapshot.
/// Used when you only need one (e.g., legacy code paths or the install
/// fallback when multi-variant logic isn't needed).
pub fn select_variant(specs: &Specs) -> LlamaVariant {
    select_variants(specs)
        .into_iter()
        .next()
        .unwrap_or(LlamaVariant::Cpu)
}

/// Variants that can actually run on this host, in priority order.
///
/// Differs from [`select_variants`], which returns only the *preferred*
/// accelerator builds and falls back to CPU solely when nothing else
/// matches: here the CPU build is **always** included, because the pure
/// CPU build runs on every machine. The UI uses this to hide/disable the
/// accelerator variants a host can't use (e.g. CUDA on a machine with no
/// NVIDIA GPU) while still always offering the universal CPU build.
pub fn usable_variants(specs: &Specs) -> Vec<LlamaVariant> {
    let mut variants = select_variants(specs);
    if !variants.contains(&LlamaVariant::Cpu) {
        variants.push(LlamaVariant::Cpu);
    }
    variants.sort_by_key(|v| v.priority());
    variants.dedup();
    variants
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::GpuInfo;

    fn make_specs(gpus: Vec<GpuInfo>) -> Specs {
        Specs {
            os: "Windows".into(),
            os_version: "11".into(),
            arch: "x86_64".into(),
            cpu_brand: "test".into(),
            cpu_vendor: "test".into(),
            cpu_physical_cores: 8,
            cpu_logical_cores: 16,
            ram_total_mb: 32 * 1024,
            gpus,
            npus: Vec::new(),
            probed_at: "now".into(),
        }
    }

    fn gpu(name: &str, vendor: &str, kind: GpuKind) -> GpuInfo {
        GpuInfo {
            name: name.into(),
            vendor: vendor.into(),
            vram_mb: Some(8 * 1024),
            kind,
        }
    }

    fn specs_with_cpu_vendor(cpu_vendor: &str, gpus: Vec<GpuInfo>) -> Specs {
        let mut s = make_specs(gpus);
        s.cpu_vendor = cpu_vendor.into();
        s
    }

    // ─── select_variants ──────────────────────────────────────────────

    #[test]
    fn nvidia_dgpu_plus_intel_igpu() {
        // User's own machine: Intel CPU+iGPU + NVIDIA dGPU
        let s = specs_with_cpu_vendor(
            "GenuineIntel",
            vec![
                gpu("Intel(R) UHD Graphics", "Intel", GpuKind::Integrated),
                gpu("NVIDIA GeForce RTX 4070", "NVIDIA", GpuKind::Discrete),
            ],
        );
        let variants = select_variants(&s);
        assert_eq!(variants, vec![LlamaVariant::Cuda, LlamaVariant::OpenVino]);
    }

    #[test]
    fn nvidia_dgpu_only() {
        let s = make_specs(vec![gpu(
            "NVIDIA GeForce RTX 4070",
            "NVIDIA",
            GpuKind::Discrete,
        )]);
        let variants = select_variants(&s);
        assert_eq!(variants, vec![LlamaVariant::Cuda]);
    }

    #[test]
    fn amd_dgpu_plus_intel_igpu() {
        let s = specs_with_cpu_vendor(
            "GenuineIntel",
            vec![
                gpu("Intel(R) UHD Graphics", "Intel", GpuKind::Integrated),
                gpu(
                    "AMD Radeon RX 7900 XTX",
                    "Advanced Micro Devices",
                    GpuKind::Discrete,
                ),
            ],
        );
        let variants = select_variants(&s);
        assert_eq!(
            variants,
            vec![LlamaVariant::HipRadeon, LlamaVariant::OpenVino]
        );
    }

    #[test]
    fn intel_igpu_only() {
        let s = specs_with_cpu_vendor(
            "GenuineIntel",
            vec![gpu("Intel(R) UHD Graphics", "Intel", GpuKind::Integrated)],
        );
        let variants = select_variants(&s);
        assert_eq!(variants, vec![LlamaVariant::OpenVino]);
    }

    #[test]
    fn amd_dgpu_non_intel_cpu() {
        let s = specs_with_cpu_vendor(
            "AuthenticAMD",
            vec![gpu(
                "AMD Radeon RX 7900 XTX",
                "Advanced Micro Devices",
                GpuKind::Discrete,
            )],
        );
        let variants = select_variants(&s);
        assert_eq!(variants, vec![LlamaVariant::HipRadeon]);
    }

    #[test]
    fn no_accelerators_falls_back_to_cpu() {
        let s = make_specs(Vec::new());
        let variants = select_variants(&s);
        assert_eq!(variants, vec![LlamaVariant::Cpu]);
    }

    #[test]
    fn non_intel_cpu_no_dgpu_falls_back_to_cpu() {
        let s = specs_with_cpu_vendor("AuthenticAMD", Vec::new());
        let variants = select_variants(&s);
        assert_eq!(variants, vec![LlamaVariant::Cpu]);
    }

    #[test]
    fn intel_cpu_only_gets_openvino() {
        let s = specs_with_cpu_vendor("GenuineIntel", Vec::new());
        let variants = select_variants(&s);
        assert_eq!(variants, vec![LlamaVariant::OpenVino]);
    }

    #[test]
    fn intel_arc_dgpu_gets_openvino() {
        let s = make_specs(vec![gpu(
            "Intel(R) Arc(TM) A770 Graphics",
            "Intel",
            GpuKind::Discrete,
        )]);
        let variants = select_variants(&s);
        assert_eq!(variants, vec![LlamaVariant::OpenVino]);
    }

    // ─── select_variant (single best) ─────────────────────────────────

    #[test]
    fn picks_cuda_for_nvidia_discrete() {
        let s = make_specs(vec![gpu(
            "NVIDIA GeForce RTX 4070",
            "NVIDIA",
            GpuKind::Discrete,
        )]);
        assert_eq!(select_variant(&s), LlamaVariant::Cuda);
    }

    #[test]
    fn picks_hip_for_amd_radeon_discrete() {
        let s = make_specs(vec![gpu(
            "AMD Radeon RX 7900 XTX",
            "Advanced Micro Devices",
            GpuKind::Discrete,
        )]);
        assert_eq!(select_variant(&s), LlamaVariant::HipRadeon);
    }

    #[test]
    fn picks_openvino_for_intel_igpu() {
        let s = make_specs(vec![gpu(
            "Intel(R) UHD Graphics",
            "Intel",
            GpuKind::Integrated,
        )]);
        assert_eq!(select_variant(&s), LlamaVariant::OpenVino);
    }

    #[test]
    fn falls_back_to_cpu_when_nothing_matches() {
        let s = make_specs(Vec::new());
        assert_eq!(select_variant(&s), LlamaVariant::Cpu);
    }

    #[test]
    fn nvidia_wins_over_intel_igpu() {
        let s = make_specs(vec![
            gpu("Intel UHD Graphics", "Intel", GpuKind::Integrated),
            gpu("NVIDIA GeForce RTX 4060", "NVIDIA", GpuKind::Discrete),
        ]);
        assert_eq!(select_variant(&s), LlamaVariant::Cuda);
    }

    // ─── usable_variants (UI hide/disable set) ────────────────────────

    #[test]
    fn usable_always_includes_cpu_even_with_an_accelerator() {
        let s = make_specs(vec![gpu(
            "NVIDIA GeForce RTX 4070",
            "NVIDIA",
            GpuKind::Discrete,
        )]);
        let usable = usable_variants(&s);
        // CUDA is preferred and listed first; CPU is always usable too.
        assert_eq!(usable, vec![LlamaVariant::Cuda, LlamaVariant::Cpu]);
    }

    #[test]
    fn usable_excludes_accelerators_the_host_cannot_run() {
        // Pure CPU host (no GPUs, non-Intel CPU) → only the CPU build.
        let s = specs_with_cpu_vendor("AuthenticAMD", Vec::new());
        let usable = usable_variants(&s);
        assert_eq!(usable, vec![LlamaVariant::Cpu]);
        assert!(!usable.contains(&LlamaVariant::Cuda));
        assert!(!usable.contains(&LlamaVariant::HipRadeon));
        assert!(!usable.contains(&LlamaVariant::OpenVino));
    }

    #[test]
    fn usable_lists_intel_openvino_plus_cpu_in_priority_order() {
        let s = make_specs(vec![gpu(
            "Intel(R) UHD Graphics",
            "Intel",
            GpuKind::Integrated,
        )]);
        assert_eq!(
            usable_variants(&s),
            vec![LlamaVariant::OpenVino, LlamaVariant::Cpu]
        );
    }

    // ─── asset matchers ──────────────────────────────────────────────

    #[test]
    fn asset_matcher_picks_cuda_12_4_over_cuda_13_3() {
        assert!(LlamaVariant::Cuda.matches_asset("llama-b9724-bin-win-cuda-12.4-x64.zip"));
        assert!(!LlamaVariant::Cuda.matches_asset("llama-b9724-bin-win-cuda-13.3-x64.zip"));
    }

    #[test]
    fn asset_matcher_picks_openvino() {
        assert!(LlamaVariant::OpenVino.matches_asset("llama-b9733-bin-win-openvino-2026.2-x64.zip"));
        assert!(!LlamaVariant::OpenVino.matches_asset("llama-b9733-bin-win-cpu-x64.zip"));
    }

    #[test]
    fn asset_matcher_picks_hip_and_cpu() {
        assert!(LlamaVariant::HipRadeon.matches_asset("llama-b9724-bin-win-hip-radeon-x64.zip"));
        assert!(LlamaVariant::Cpu.matches_asset("llama-b9724-bin-win-cpu-x64.zip"));
    }

    #[test]
    fn asset_matcher_rejects_other_platforms() {
        assert!(!LlamaVariant::Cuda.matches_asset("llama-b9724-bin-linux-cuda-12.4-x64.tar.gz"));
        assert!(!LlamaVariant::Cpu.matches_asset("llama-b9724-bin-win-opencl-adreno-arm64.zip"));
    }

    #[test]
    fn asset_matcher_rejects_cudart_runtime_redistributable() {
        assert!(!LlamaVariant::Cuda.matches_asset("cudart-llama-bin-win-cuda-12.4-x64.zip"));
        assert!(LlamaVariant::Cuda.matches_asset("llama-b9732-bin-win-cuda-12.4-x64.zip"));
    }

    #[test]
    fn companion_matcher_picks_cudart_for_cuda_only() {
        assert!(
            LlamaVariant::Cuda.matches_companion_asset("cudart-llama-bin-win-cuda-12.4-x64.zip")
        );
        assert!(
            !LlamaVariant::Cuda.matches_companion_asset("llama-b9732-bin-win-cuda-12.4-x64.zip")
        );
        assert!(
            !LlamaVariant::Cuda.matches_companion_asset("cudart-llama-bin-win-cuda-13.3-x64.zip")
        );
        assert!(!LlamaVariant::OpenVino
            .matches_companion_asset("cudart-llama-bin-win-cuda-12.4-x64.zip"));
        assert!(!LlamaVariant::HipRadeon
            .matches_companion_asset("cudart-llama-bin-win-cuda-12.4-x64.zip"));
        assert!(
            !LlamaVariant::Cpu.matches_companion_asset("cudart-llama-bin-win-cuda-12.4-x64.zip")
        );
    }

    // ─── helpers ──────────────────────────────────────────────────────

    #[test]
    fn slug_and_port_and_runtime_name() {
        assert_eq!(LlamaVariant::Cuda.slug(), "cuda");
        assert_eq!(LlamaVariant::OpenVino.slug(), "openvino");
        assert_eq!(LlamaVariant::Cuda.default_port(), 8081);
        assert_eq!(LlamaVariant::OpenVino.default_port(), 8082);
        assert_eq!(LlamaVariant::HipRadeon.default_port(), 8083);
        assert_eq!(LlamaVariant::Cpu.default_port(), 8084);
        assert_eq!(LlamaVariant::Cuda.runtime_name(), "llama.cpp-cuda");
        assert_eq!(LlamaVariant::OpenVino.runtime_name(), "llama.cpp-openvino");
    }

    #[test]
    fn from_slug_roundtrip() {
        for &v in LlamaVariant::all() {
            assert_eq!(LlamaVariant::from_slug(v.slug()), Some(v));
        }
        assert_eq!(LlamaVariant::from_slug("unknown"), None);
    }

    #[test]
    fn priority_ordering() {
        assert!(LlamaVariant::Cuda.priority() < LlamaVariant::HipRadeon.priority());
        assert!(LlamaVariant::HipRadeon.priority() < LlamaVariant::OpenVino.priority());
        assert!(LlamaVariant::OpenVino.priority() < LlamaVariant::Cpu.priority());
    }
}
