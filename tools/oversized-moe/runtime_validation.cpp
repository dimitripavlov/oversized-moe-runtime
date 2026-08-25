#include "runtime_validation.h"

#include <filesystem>
#include <ostream>
#include <string>

#include <unistd.h>

namespace {

bool readable_file(const std::string & path) {
    std::error_code ec;

    return
        std::filesystem::is_regular_file(path, ec) &&
        access(path.c_str(), R_OK) == 0;
}

bool executable_file(const std::string & path) {
    std::error_code ec;

    return
        std::filesystem::is_regular_file(path, ec) &&
        access(path.c_str(), X_OK) == 0;
}

} // namespace

RuntimeValidation validate_runtime_policy(
        const ModelInfo & model,
        const MemoryInfo & memory,
        const MoePolicy & policy,
        const std::string & engine_path) {
    RuntimeValidation result;

    const auto error =
        [&result](const std::string & message) {
            result.ok = false;
            result.errors.push_back(message);
        };

    if (!policy.ready) {
        error("memory policy is not ready");
    }

    if (!readable_file(model.path)) {
        error("model file is not readable: " + model.path);
    }

    if (!executable_file(engine_path)) {
        error("engine is not executable: " + engine_path);
    }

    if (memory.physical_bytes == 0) {
        error("physical RAM is zero");
    }

    if (policy.oversized) {
        if (policy.memory_mode != "oversized") {
            error("oversized model has inconsistent memory mode");
        }

        if (!model.sparse_moe) {
            error("oversized policy applied to non-MoE model");
        }

        if (!policy.architecture_supported) {
            error("oversized policy applied to unsupported architecture");
        }

        if (!policy.use_mmap) {
            error("oversized mode requires mmap");
        }

        if (!policy.disable_full_prefetch) {
            error("oversized mode requires full mmap prefetch disabled");
        }

        if (!policy.disable_repack) {
            error("oversized mode requires repack disabled");
        }

        if (model.expert_count == 0) {
            error("oversized MoE model has zero experts");
        }

        if (model.lockable_bytes_per_quota == 0) {
            error("expert residency geometry is zero");
        }

        if (policy.experts_per_tensor > model.expert_count) {
            error("expert residency quota exceeds expert count");
        }

        if (policy.safety_reserve_bytes >
            memory.physical_bytes) {
            error("RAM safety reserve exceeds physical RAM");
        }

        const std::uint64_t physical_budget_cap =
            memory.physical_bytes >
                    policy.safety_reserve_bytes
                ? memory.physical_bytes -
                    policy.safety_reserve_bytes
                : 0;

        if (policy.cache_budget_bytes >
            physical_budget_cap) {
            error("expert cache budget exceeds physical RAM budget");
        }

        if (policy.experts_per_tensor > 0 &&
            model.lockable_bytes_per_quota > 0) {
            const std::uint64_t required =
                model.lockable_bytes_per_quota *
                policy.experts_per_tensor;

            if (required >
                policy.cache_budget_bytes) {
                error(
                    "expert quota requires more memory "
                    "than the cache budget");
            }
        }
    } else {
        if (policy.memory_mode != "standard") {
            error("standard model has inconsistent memory mode");
        }

        if (policy.experts_per_tensor != 0) {
            error("standard mode unexpectedly enables expert residency");
        }

        if (policy.disable_full_prefetch) {
            error("standard mode unexpectedly disables full prefetch");
        }

        if (policy.disable_repack) {
            error("standard mode unexpectedly disables repack");
        }
    }

    return result;
}

void print_runtime_validation(
        const RuntimeValidation & validation,
        std::ostream & out) {
    out
        << "\nRuntime validation\n"
        << "------------------\n";

    if (validation.ok) {
        out << "Runtime policy:     OK\n";
        return;
    }

    out << "Runtime policy:     FAILED\n";

    for (const auto & message :
         validation.errors) {
        out
            << "  - "
            << message
            << '\n';
    }
}
