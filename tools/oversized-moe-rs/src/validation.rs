use crate::memory::MemoryInfo;
use crate::model::ModelInfo;
use crate::policy::MoePolicy;

use std::fs;
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[derive(Debug, Default)]
pub struct RuntimeValidation {
    pub ok: bool,
    pub errors: Vec<String>,
}

fn readable_file(path: &Path) -> bool {
    fs::File::open(path).is_ok() && fs::metadata(path).map(|m| m.is_file()).unwrap_or(false)
}

fn executable_file(path: &Path) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };

    if !meta.is_file() {
        return false;
    }

    #[cfg(unix)]
    {
        return meta.permissions().mode() & 0o111 != 0;
    }

    #[cfg(not(unix))]
    {
        true
    }
}

pub fn validate_runtime_policy(
    model: &ModelInfo,
    memory: &MemoryInfo,
    policy: &MoePolicy,
    engine_path: &Path,
) -> RuntimeValidation {
    let mut result = RuntimeValidation {
        ok: true,
        errors: Vec::new(),
    };

    let mut error = |message: String| {
        result.ok = false;
        result.errors.push(message);
    };

    if !policy.ready {
        error("memory policy is not ready".to_owned());
    }

    if !readable_file(Path::new(&model.path)) {
        error(format!("model file is not readable: {}", model.path));
    }

    if !executable_file(engine_path) {
        error(format!(
            "engine is not executable: {}",
            engine_path.display()
        ));
    }

    if memory.physical_bytes == 0 {
        error("physical RAM is zero".to_owned());
    }

    if policy.oversized {
        if policy.memory_mode != "oversized" {
            error("oversized model has inconsistent memory mode".to_owned());
        }

        if !model.sparse_moe {
            error("oversized policy applied to non-MoE model".to_owned());
        }

        if !policy.architecture_supported {
            error("oversized policy applied to unsupported architecture".to_owned());
        }

        if !policy.use_mmap {
            error("oversized mode requires mmap".to_owned());
        }

        if !policy.disable_full_prefetch {
            error("oversized mode requires full mmap prefetch disabled".to_owned());
        }

        if !policy.disable_repack {
            error("oversized mode requires repack disabled".to_owned());
        }

        if model.expert_count == 0 {
            error("oversized MoE model has zero experts".to_owned());
        }

        if model.lockable_bytes_per_quota == 0 {
            error("expert residency geometry is zero".to_owned());
        }

        if policy.experts_per_tensor > model.expert_count {
            error("expert residency quota exceeds expert count".to_owned());
        }

        if policy.safety_reserve_bytes > memory.physical_bytes {
            error("RAM safety reserve exceeds physical RAM".to_owned());
        }

        let physical_budget_cap = memory
            .physical_bytes
            .saturating_sub(policy.safety_reserve_bytes);

        if policy.cache_budget_bytes > physical_budget_cap {
            error("expert cache budget exceeds physical RAM budget".to_owned());
        }

        if policy.experts_per_tensor > 0 && model.lockable_bytes_per_quota > 0 {
            let required = model
                .lockable_bytes_per_quota
                .saturating_mul(policy.experts_per_tensor as u64);

            if required > policy.cache_budget_bytes {
                error("expert quota requires more memory than the cache budget".to_owned());
            }
        }
    } else {
        if policy.memory_mode != "standard" {
            error("standard model has inconsistent memory mode".to_owned());
        }

        if policy.experts_per_tensor != 0 {
            error("standard mode unexpectedly enables expert residency".to_owned());
        }

        if policy.disable_full_prefetch {
            error("standard mode unexpectedly disables full prefetch".to_owned());
        }

        if policy.disable_repack {
            error("standard mode unexpectedly disables repack".to_owned());
        }
    }

    result
}

pub fn print_runtime_validation(validation: &RuntimeValidation) {
    println!();
    println!("Runtime validation");
    println!("------------------");

    if validation.ok {
        println!("Runtime policy:     OK");
        return;
    }

    println!("Runtime policy:     FAILED");

    for message in &validation.errors {
        println!("  - {message}");
    }
}
