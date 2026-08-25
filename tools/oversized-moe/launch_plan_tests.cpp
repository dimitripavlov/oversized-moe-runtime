#include "launcher.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

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

bool contains_arg(
        const LaunchPlan & plan,
        const std::string & value) {
    return std::find(
               plan.argv.begin(),
               plan.argv.end(),
               value) != plan.argv.end();
}

bool contains_env_set(
        const LaunchPlan & plan,
        const std::string & key,
        const std::string & value) {
    return std::find(
               plan.env_set.begin(),
               plan.env_set.end(),
               std::make_pair(key, value)) !=
           plan.env_set.end();
}

bool contains_env_unset(
        const LaunchPlan & plan,
        const std::string & key) {
    return std::find(
               plan.env_unset.begin(),
               plan.env_unset.end(),
               key) != plan.env_unset.end();
}

ModelInfo synthetic_model() {
    ModelInfo model;

    model.path = "/models/qwen80.gguf";
    model.name = "synthetic qwen80";
    model.architecture = "qwen3next";
    model.sparse_moe = true;
    model.expert_count = 512;

    return model;
}

MoePolicy oversized_policy() {
    MoePolicy policy;

    policy.ready = true;
    policy.architecture_supported = true;
    policy.oversized = true;
    policy.memory_mode = "oversized";

    policy.use_mmap = true;
    policy.disable_full_prefetch = true;
    policy.disable_repack = true;

    policy.experts_per_tensor = 32;

    return policy;
}

MoePolicy standard_policy() {
    MoePolicy policy;

    policy.ready = true;
    policy.architecture_supported = true;
    policy.oversized = false;
    policy.memory_mode = "standard";

    policy.use_mmap = true;
    policy.disable_full_prefetch = false;
    policy.disable_repack = false;

    policy.experts_per_tensor = 0;

    return policy;
}

void test_oversized_completion_plan() {
    std::cout << "\n[Oversized completion launch plan]\n";

    const ModelInfo model = synthetic_model();
    const MoePolicy policy = oversized_policy();

    const std::vector<std::string> passthrough = {
        "-t", "4",
        "-n", "32",
        "-p", "Hello"
    };

    const LaunchPlan plan =
        make_completion_launch_plan(
            "/bin/llama-completion",
            model,
            policy,
            passthrough);

    check(
        plan.engine_path == "/bin/llama-completion",
        "completion engine path is preserved");

    check(
        plan.argv.size() >= 7,
        "completion argv contains runtime arguments");

    check(
        plan.argv[0] == "/bin/llama-completion",
        "argv[0] is completion engine");

    check(
        plan.argv[1] == "-m" &&
        plan.argv[2] == model.path,
        "runtime injects model path");

    check(
        contains_arg(plan, "-lm") &&
        contains_arg(plan, "mmap"),
        "oversized mode injects mmap load mode");

    check(
        contains_arg(plan, "--no-repack"),
        "oversized mode disables repack");

    check(
        contains_arg(plan, "-t") &&
        contains_arg(plan, "-n") &&
        contains_arg(plan, "-p"),
        "completion passthrough arguments are preserved");

    check(
        contains_env_set(
            plan,
            "GGML_EXPERT_RESIDENT_PER_TENSOR",
            "32"),
        "oversized mode sets expert residency quota");

    check(
        plan.env_unset.empty(),
        "oversized residency does not unset quota environment");
}

void test_oversized_server_plan() {
    std::cout << "\n[Oversized server launch plan]\n";

    const ModelInfo model = synthetic_model();
    const MoePolicy policy = oversized_policy();

    const std::vector<std::string> passthrough = {
        "--host", "127.0.0.1",
        "--port", "8080",
        "-c", "2048"
    };

    const LaunchPlan plan =
        make_server_launch_plan(
            "/bin/llama-server",
            model,
            policy,
            passthrough);

    check(
        plan.engine_path == "/bin/llama-server",
        "server engine path is preserved");

    check(
        plan.argv[0] == "/bin/llama-server",
        "argv[0] is server engine");

    check(
        plan.argv[1] == "-m" &&
        plan.argv[2] == model.path,
        "server plan injects model path");

    check(
        contains_arg(plan, "-lm") &&
        contains_arg(plan, "mmap"),
        "server oversized mode injects mmap");

    check(
        contains_arg(plan, "--no-repack"),
        "server oversized mode disables repack");

    check(
        contains_arg(plan, "--host") &&
        contains_arg(plan, "--port") &&
        contains_arg(plan, "-c"),
        "server passthrough arguments are preserved");

    check(
        contains_env_set(
            plan,
            "GGML_EXPERT_RESIDENT_PER_TENSOR",
            "32"),
        "server plan sets expert residency quota");
}

void test_standard_plan() {
    std::cout << "\n[Standard-memory launch plan]\n";

    const ModelInfo model = synthetic_model();
    const MoePolicy policy = standard_policy();

    const std::vector<std::string> passthrough = {
        "-t", "4",
        "-c", "512"
    };

    const LaunchPlan plan =
        make_completion_launch_plan(
            "/bin/llama-completion",
            model,
            policy,
            passthrough);

    check(
        plan.argv[1] == "-m" &&
        plan.argv[2] == model.path,
        "standard mode still injects model path");

    check(
        !contains_arg(plan, "-lm"),
        "standard mode does not force load mode");

    check(
        !contains_arg(plan, "--no-repack"),
        "standard mode does not disable repack");

    check(
        plan.env_set.empty(),
        "standard mode does not set residency quota");

    check(
        contains_env_unset(
            plan,
            "GGML_EXPERT_RESIDENT_PER_TENSOR"),
        "standard mode clears stale residency environment");

    check(
        contains_arg(plan, "-t") &&
        contains_arg(plan, "-c"),
        "standard-mode passthrough is preserved");
}

void test_zero_quota_clears_environment() {
    std::cout << "\n[Zero-quota environment safety]\n";

    const ModelInfo model = synthetic_model();

    MoePolicy policy = oversized_policy();
    policy.experts_per_tensor = 0;

    const LaunchPlan plan =
        make_completion_launch_plan(
            "/bin/llama-completion",
            model,
            policy,
            {});

    check(
        plan.env_set.empty(),
        "zero quota does not set residency environment");

    check(
        contains_env_unset(
            plan,
            "GGML_EXPERT_RESIDENT_PER_TENSOR"),
        "zero quota clears stale residency environment");
}

} // namespace

int main() {
    test_oversized_completion_plan();
    test_oversized_server_plan();
    test_standard_plan();
    test_zero_quota_clears_environment();

    std::cout << '\n';

    if (failures != 0) {
        std::cerr
            << failures
            << " oversized-moe launch-plan test(s) failed\n";

        return 1;
    }

    std::cout
        << "All oversized-moe launch-plan tests passed\n";

    return 0;
}
