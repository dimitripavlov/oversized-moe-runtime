#include "moe_policy.h"

#include <algorithm>
#include <cstdint>

namespace {

constexpr std::uint64_t GiB =
    1024ULL * 1024ULL * 1024ULL;

bool supported_architecture(
        const ModelInfo & model) {
    // MVP 0.1 starts from architectures already validated
    // by the research branch.
    return
        model.architecture == "qwen3moe" ||
        model.architecture == "qwen3next";
}

double calibrated_cache_fraction(
        double oversubscription_ratio) {
    // MVP 0.1 heuristic.
    //
    // Calibrated from the two physically validated 16-GiB
    // operating points:
    //
    // Qwen3-30B-A3B (~1.08x oversized):
    //   capacity 48 needs about 6.02 GiB maximum lockable
    //   expert residency -> target about 38% of physical RAM.
    //
    // Qwen3-Next-80B-A3B (~2.82x oversized):
    //   capacity 32 needs about 2.65 GiB -> target about
    //   16.6% of physical RAM.
    //
    // More oversubscription preserves progressively more RAM
    // for dense/nonexpert weights, attention/DeltaNet state,
    // mmap page cache, KV/state and runtime buffers.
    const double fraction =
        0.513 -
        0.123 * oversubscription_ratio;

    return std::clamp(
        fraction,
        0.166,
        0.380);
}

} // namespace

MoePolicy make_moe_policy(
        const ModelInfo & model,
        const MemoryInfo & memory) {
    MoePolicy policy;

    if (memory.physical_bytes == 0) {
        policy.reason =
            "physical RAM is unknown";

        return policy;
    }

    policy.oversubscription_ratio =
        static_cast<double>(model.file_size) /
        static_cast<double>(memory.physical_bytes);

    policy.oversized =
        model.file_size >
        memory.physical_bytes;

    policy.memory_mode =
        policy.oversized
            ? "oversized"
            : "standard";

    policy.architecture_supported =
        supported_architecture(model);

    // Classify the model before applying architecture-specific
    // execution policy. Unsupported models must never inherit or
    // guess an ExpertResidency configuration.
    if (!model.sparse_moe) {
        if (model.expert_count == 0 &&
            model.expert_tensor_count == 0) {
            policy.reason =
                "model is not a supported sparse MoE model";
        } else {
            policy.reason =
                "sparse MoE expert geometry is incomplete "
                "or unsupported";
        }

        return policy;
    }

    if (!policy.architecture_supported) {
        policy.reason =
            "sparse MoE architecture is not supported "
            "by MVP 0.1";

        return policy;
    }

    if (model.expert_count != 128 &&
        model.expert_count != 512) {
        policy.reason =
            "ExpertResidency currently supports "
            "128 or 512 experts";

        return policy;
    }

    if (model.lockable_bytes_per_quota == 0) {
        policy.reason =
            "cannot calculate expert residency geometry";

        return policy;
    }

    policy.use_mmap = true;

    if (!policy.oversized) {
        policy.ready = true;

        policy.reason =
            "model fits physical RAM; "
            "use normal engine defaults";

        return policy;
    }

    policy.disable_full_prefetch = true;
    policy.disable_repack = true;

    policy.target_cache_fraction =
        calibrated_cache_fraction(
            policy.oversubscription_ratio);

    policy.target_cache_budget_bytes =
        static_cast<std::uint64_t>(
            static_cast<double>(
                memory.physical_bytes) *
            policy.target_cache_fraction);

    // Keep a hard global reserve in physical RAM for the OS,
    // dense/nonexpert weights, mmap page cache, KV/state and
    // runtime buffers.
    //
    // Do not use instantaneous "available RAM" as the residency
    // quota. With mmap-based oversized inference, reclaimable
    // file-backed pages are expected to move in and out of the
    // physical working set, making that startup estimate noisy
    // and overly conservative.
    policy.safety_reserve_bytes =
        std::max<std::uint64_t>(
            4ULL * GiB,
            memory.physical_bytes / 4);

    const std::uint64_t physical_budget_cap =
        memory.physical_bytes >
                policy.safety_reserve_bytes
            ? memory.physical_bytes -
                policy.safety_reserve_bytes
            : 0;

    policy.cache_budget_bytes =
        std::min(
            policy.target_cache_budget_bytes,
            physical_budget_cap);

    if (policy.cache_budget_bytes > 0) {
        const std::uint64_t quota =
            policy.cache_budget_bytes /
            model.lockable_bytes_per_quota;

        policy.experts_per_tensor =
            static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    quota,
                    model.expert_count));
    }

    policy.ready = true;

    if (policy.experts_per_tensor == 0) {
        policy.reason =
            "oversized model recognized; "
            "residency disabled by RAM safety cap";
    } else {
        policy.reason =
            "oversized sparse-MoE policy available";
    }

    return policy;
}
