#pragma once

#include "memory_budget.h"
#include "model_probe.h"

#include <cstdint>
#include <string>

struct MoePolicy {
    bool ready = false;
    bool architecture_supported = false;
    bool oversized = false;

    double oversubscription_ratio = 0.0;
    double target_cache_fraction = 0.0;

    std::uint64_t safety_reserve_bytes = 0;
    std::uint64_t target_cache_budget_bytes = 0;
    std::uint64_t cache_budget_bytes = 0;

    std::uint32_t experts_per_tensor = 0;

    bool use_mmap = true;
    bool disable_full_prefetch = false;
    bool disable_repack = false;

    std::string memory_mode;
    std::string reason;
};

MoePolicy make_moe_policy(
    const ModelInfo & model,
    const MemoryInfo & memory);
