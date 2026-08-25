#pragma once

#include "memory_budget.h"
#include "model_probe.h"
#include "moe_policy.h"

#include <iosfwd>
#include <string>
#include <vector>

struct RuntimeValidation {
    bool ok = true;
    std::vector<std::string> errors;
};

RuntimeValidation validate_runtime_policy(
    const ModelInfo & model,
    const MemoryInfo & memory,
    const MoePolicy & policy,
    const std::string & engine_path);

void print_runtime_validation(
    const RuntimeValidation & validation,
    std::ostream & out);
