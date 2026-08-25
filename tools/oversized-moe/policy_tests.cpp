#include "memory_budget.h"
#include "model_probe.h"
#include "moe_policy.h"
#include "runtime_validation.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

constexpr std::uint64_t KiB = 1024ull;
constexpr std::uint64_t MiB = 1024ull * KiB;
constexpr std::uint64_t GiB = 1024ull * MiB;

int failures = 0;

void check(
        bool condition,
        const std::string & message) {
    if (condition) {
        std::cout << "PASS: " << message << '\n';
        return;
    }

    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

bool contains_error(
        const RuntimeValidation & validation,
        const std::string & needle) {
    return std::any_of(
        validation.errors.begin(),
        validation.errors.end(),
        [&](const std::string & error) {
            return error.find(needle) != std::string::npos;
        });
}

MemoryInfo memory_16g() {
    MemoryInfo memory;

    memory.physical_bytes = 16ull * GiB;
    memory.available_bytes = 8ull * GiB;
    memory.page_size = 16ull * KiB;
    memory.available_source = "synthetic test";

    return memory;
}

ModelInfo qwen30_like(
        const std::string & path = {}) {
    ModelInfo model;

    model.path = path;
    model.name = "synthetic qwen3 30b";
    model.architecture = "qwen3moe";

    // Approximately the physically validated Qwen3-30B-A3B
    // Q4_K_M operating point.
    model.file_size = 18560000000ull;

    model.block_count = 48;
    model.embedding_length = 2048;

    model.expert_count = 128;
    model.expert_used_count = 8;
    model.expert_feed_forward_length = 768;
    model.expert_tensor_count = 144;

    model.raw_bytes_per_quota =
        static_cast<std::uint64_t>(130.78 * MiB);

    model.lockable_bytes_per_quota =
        static_cast<std::uint64_t>(128.53 * MiB);

    model.sparse_moe = true;

    return model;
}

ModelInfo qwen80_like(
        const std::string & path = {}) {
    ModelInfo model;

    model.path = path;
    model.name = "synthetic qwen3-next 80b";
    model.architecture = "qwen3next";

    // Exact GGUF size used during MVP validation.
    model.file_size = 48410988384ull;

    model.block_count = 48;
    model.embedding_length = 2048;

    model.expert_count = 512;
    model.expert_used_count = 10;
    model.expert_feed_forward_length = 512;
    model.expert_tensor_count = 144;

    model.raw_bytes_per_quota =
        static_cast<std::uint64_t>(87.19 * MiB);

    model.lockable_bytes_per_quota =
        static_cast<std::uint64_t>(84.94 * MiB);

    model.sparse_moe = true;

    return model;
}

void test_qwen30_policy() {
    std::cout << "\n[Qwen30-like oversized policy]\n";

    const MemoryInfo memory = memory_16g();
    const ModelInfo model = qwen30_like();

    const MoePolicy policy =
        make_moe_policy(model, memory);

    check(policy.ready,
          "policy is ready");

    check(policy.architecture_supported,
          "qwen3moe architecture is supported");

    check(policy.oversized,
          "model is classified as oversized");

    check(policy.memory_mode == "oversized",
          "memory mode is oversized");

    check(policy.experts_per_tensor == 48,
          "validated Qwen30 quota is 48 experts/tensor");

    check(policy.use_mmap,
          "mmap is enabled");

    check(policy.disable_full_prefetch,
          "full mmap prefetch is disabled");

    check(policy.disable_repack,
          "repack is disabled");

    check(policy.experts_per_tensor <= model.expert_count,
          "quota does not exceed expert count");

    const std::uint64_t footprint =
        static_cast<std::uint64_t>(
            policy.experts_per_tensor) *
        model.lockable_bytes_per_quota;

    check(footprint <= policy.cache_budget_bytes,
          "quota physical footprint fits cache budget");

    check(policy.cache_budget_bytes +
              policy.safety_reserve_bytes <=
          memory.physical_bytes,
          "cache budget preserves RAM safety reserve");
}

void test_qwen80_policy() {
    std::cout << "\n[Qwen80-like oversized policy]\n";

    const MemoryInfo memory = memory_16g();
    const ModelInfo model = qwen80_like();

    const MoePolicy policy =
        make_moe_policy(model, memory);

    check(policy.ready,
          "policy is ready");

    check(policy.architecture_supported,
          "qwen3next architecture is supported");

    check(policy.oversized,
          "model is classified as oversized");

    check(policy.memory_mode == "oversized",
          "memory mode is oversized");

    check(policy.experts_per_tensor == 32,
          "validated Qwen80 quota is 32 experts/tensor");

    check(policy.use_mmap,
          "mmap is enabled");

    check(policy.disable_full_prefetch,
          "full mmap prefetch is disabled");

    check(policy.disable_repack,
          "repack is disabled");

    check(policy.experts_per_tensor <= model.expert_count,
          "quota does not exceed expert count");

    const std::uint64_t footprint =
        static_cast<std::uint64_t>(
            policy.experts_per_tensor) *
        model.lockable_bytes_per_quota;

    check(footprint <= policy.cache_budget_bytes,
          "quota physical footprint fits cache budget");

    check(policy.cache_budget_bytes +
              policy.safety_reserve_bytes <=
          memory.physical_bytes,
          "cache budget preserves RAM safety reserve");
}

void test_standard_mode() {
    std::cout << "\n[Supported standard-memory policy]\n";

    const MemoryInfo memory = memory_16g();

    ModelInfo model = qwen30_like();

    // Fits comfortably in physical RAM.
    model.file_size = 8ull * GiB;

    const MoePolicy policy =
        make_moe_policy(model, memory);

    check(policy.ready,
          "standard-mode supported model is ready");

    check(policy.architecture_supported,
          "architecture remains supported");

    check(!policy.oversized,
          "model is not oversized");

    check(policy.memory_mode == "standard",
          "memory mode is standard");

    check(policy.experts_per_tensor == 0,
          "standard mode has zero residency quota");

    check(!policy.disable_full_prefetch,
          "standard mode does not disable full prefetch");

    check(!policy.disable_repack,
          "standard mode does not disable repack");
}

void test_unsupported_models() {
    std::cout << "\n[Unsupported-model policy]\n";

    const MemoryInfo memory = memory_16g();

    {
        ModelInfo dense;
        dense.name = "synthetic dense model";
        dense.architecture = "llama";
        dense.file_size = 32ull * GiB;
        dense.sparse_moe = false;

        const MoePolicy policy =
            make_moe_policy(dense, memory);

        check(!policy.ready,
              "dense model is refused");

        check(!policy.architecture_supported,
              "dense architecture is not marked supported");

        check(policy.experts_per_tensor == 0,
              "dense model gets no residency quota");
    }

    {
        ModelInfo unknown = qwen80_like();
        unknown.architecture = "future_moe";

        const MoePolicy policy =
            make_moe_policy(unknown, memory);

        check(!policy.ready,
              "unknown sparse MoE architecture is refused");

        check(!policy.architecture_supported,
              "unknown MoE architecture is not supported");

        check(policy.experts_per_tensor == 0,
              "unknown MoE gets no residency quota");
    }
}

void test_runtime_validation() {
    std::cout << "\n[Runtime validation]\n";

    const std::string model_path =
        "oversized-moe-policy-test-model.gguf";

    {
        std::ofstream out(
            model_path,
            std::ios::binary);

        out.put('\0');
    }

    const MemoryInfo memory = memory_16g();
    const ModelInfo model =
        qwen80_like(model_path);

    const MoePolicy valid_policy =
        make_moe_policy(model, memory);

    // /bin/sh is a small, known executable on the current
    // macOS/Linux product-validation platforms.
    const std::string engine_path = "/bin/sh";

    {
        const RuntimeValidation validation =
            validate_runtime_policy(
                model,
                memory,
                valid_policy,
                engine_path);

        check(validation.ok,
              "valid oversized policy passes validation");

        check(validation.errors.empty(),
              "valid policy has no validation errors");
    }

    {
        MoePolicy invalid = valid_policy;

        invalid.experts_per_tensor =
            model.expert_count + 1;

        const RuntimeValidation validation =
            validate_runtime_policy(
                model,
                memory,
                invalid,
                engine_path);

        check(!validation.ok,
              "quota greater than expert count is rejected");

        check(
            contains_error(
                validation,
                "expert residency quota exceeds expert count"),
            "quota overflow reports expected error");
    }

    {
        MoePolicy invalid = valid_policy;

        invalid.cache_budget_bytes =
            model.lockable_bytes_per_quota;

        invalid.experts_per_tensor = 32;

        const RuntimeValidation validation =
            validate_runtime_policy(
                model,
                memory,
                invalid,
                engine_path);

        check(!validation.ok,
              "quota footprint exceeding budget is rejected");

        check(
            contains_error(
                validation,
                "expert quota requires more memory"),
            "budget overflow reports expected error");
    }

    {
        ModelInfo standard_model = model;
        standard_model.file_size = 8ull * GiB;

        MoePolicy invalid =
            make_moe_policy(
                standard_model,
                memory);

        invalid.experts_per_tensor = 1;

        const RuntimeValidation validation =
            validate_runtime_policy(
                standard_model,
                memory,
                invalid,
                engine_path);

        check(!validation.ok,
              "nonzero standard-mode quota is rejected");
    }

    std::remove(model_path.c_str());
}

} // namespace

int main() {
    test_qwen30_policy();
    test_qwen80_policy();
    test_standard_mode();
    test_unsupported_models();
    test_runtime_validation();

    std::cout << '\n';

    if (failures != 0) {
        std::cerr
            << failures
            << " oversized-moe policy test(s) failed\n";

        return 1;
    }

    std::cout
        << "All oversized-moe policy tests passed\n";

    return 0;
}
