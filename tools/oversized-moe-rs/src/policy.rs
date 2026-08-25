use crate::memory::MemoryInfo;
use crate::model::ModelInfo;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Default, Clone)]
pub struct MoePolicy {
    pub ready: bool,
    pub architecture_supported: bool,
    pub oversized: bool,

    pub oversubscription_ratio: f64,
    pub target_cache_fraction: f64,

    pub safety_reserve_bytes: u64,
    pub target_cache_budget_bytes: u64,
    pub cache_budget_bytes: u64,

    pub experts_per_tensor: u32,

    pub use_mmap: bool,
    pub disable_full_prefetch: bool,
    pub disable_repack: bool,

    pub memory_mode: String,
    pub reason: String,
}

fn supported_architecture(model: &ModelInfo) -> bool {
    model.architecture == "qwen3moe" || model.architecture == "qwen3next"
}

fn calibrated_cache_fraction(oversubscription_ratio: f64) -> f64 {
    (0.513 - 0.123 * oversubscription_ratio).clamp(0.166, 0.380)
}

pub fn make_moe_policy(model: &ModelInfo, memory: &MemoryInfo) -> MoePolicy {
    let mut policy = MoePolicy {
        use_mmap: true,
        ..Default::default()
    };

    if memory.physical_bytes == 0 {
        policy.reason = "physical RAM is unknown".to_owned();
        return policy;
    }

    policy.oversubscription_ratio = model.file_size as f64 / memory.physical_bytes as f64;

    policy.oversized = model.file_size > memory.physical_bytes;

    policy.memory_mode = if policy.oversized {
        "oversized"
    } else {
        "standard"
    }
    .to_owned();

    policy.architecture_supported = supported_architecture(model);

    if !model.sparse_moe {
        policy.reason = if model.expert_count == 0 && model.expert_tensor_count == 0 {
            "model is not a supported sparse MoE model"
        } else {
            "sparse MoE expert geometry is incomplete or unsupported"
        }
        .to_owned();

        return policy;
    }

    if !policy.architecture_supported {
        policy.reason = "sparse MoE architecture is not supported by MVP 0.1".to_owned();
        return policy;
    }

    if model.expert_count != 128 && model.expert_count != 512 {
        policy.reason = "ExpertResidency currently supports 128 or 512 experts".to_owned();
        return policy;
    }

    if model.lockable_bytes_per_quota == 0 {
        policy.reason = "cannot calculate expert residency geometry".to_owned();
        return policy;
    }

    if !policy.oversized {
        policy.ready = true;
        policy.reason = "model fits physical RAM; use normal engine defaults".to_owned();
        return policy;
    }

    policy.disable_full_prefetch = true;
    policy.disable_repack = true;

    policy.target_cache_fraction = calibrated_cache_fraction(policy.oversubscription_ratio);

    policy.target_cache_budget_bytes =
        (memory.physical_bytes as f64 * policy.target_cache_fraction) as u64;

    policy.safety_reserve_bytes = (4 * GIB).max(memory.physical_bytes / 4);

    let physical_budget_cap = memory
        .physical_bytes
        .saturating_sub(policy.safety_reserve_bytes);

    policy.cache_budget_bytes = policy.target_cache_budget_bytes.min(physical_budget_cap);

    if policy.cache_budget_bytes > 0 {
        let quota = policy.cache_budget_bytes / model.lockable_bytes_per_quota;

        policy.experts_per_tensor = quota.min(model.expert_count as u64) as u32;
    }

    policy.ready = true;

    policy.reason = if policy.experts_per_tensor == 0 {
        "oversized model recognized; residency disabled by RAM safety cap"
    } else {
        "oversized sparse-MoE policy available"
    }
    .to_owned();

    policy
}
