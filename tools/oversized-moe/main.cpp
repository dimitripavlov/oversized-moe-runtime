#include "memory_budget.h"
#include "model_probe.h"
#include "moe_policy.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr double MiB =
    1024.0 * 1024.0;

constexpr double GiB =
    1024.0 * 1024.0 * 1024.0;

constexpr double GB =
    1000.0 * 1000.0 * 1000.0;

std::string gib(std::uint64_t bytes) {
    std::ostringstream out;

    out
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(bytes) / GiB
        << " GiB";

    return out.str();
}

std::string mib(std::uint64_t bytes) {
    std::ostringstream out;

    out
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(bytes) / MiB
        << " MiB";

    return out.str();
}

std::string model_size(
        std::uint64_t bytes) {
    std::ostringstream out;

    out
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(bytes) / GiB
        << " GiB"
        << "  ("
        << static_cast<double>(bytes) / GB
        << " GB)";

    return out.str();
}

void row(
        const std::string & key,
        const std::string & value) {
    std::cout
        << std::left
        << std::setw(31)
        << key
        << value
        << '\n';
}

void row(
        const std::string & key,
        std::uint64_t value) {
    row(key, std::to_string(value));
}

void section(
        const std::string & name) {
    std::cout
        << '\n'
        << name
        << '\n'
        << std::string(name.size(), '-')
        << '\n';
}

std::string yes_no(bool value) {
    return value ? "yes" : "no";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr
            << "usage: "
            << argv[0]
            << " MODEL.gguf\n";

        return 1;
    }

    try {
        const MemoryInfo memory =
            probe_memory();

        const ModelInfo model =
            probe_model(
                argv[1],
                memory.page_size);

        const MoePolicy policy =
            make_moe_policy(
                model,
                memory);

        section("Model");

        row("path:", model.path);

        if (!model.name.empty()) {
            row("name:", model.name);
        }

        row(
            "file size:",
            model_size(model.file_size));

        row(
            "architecture:",
            model.architecture);

        row(
            "sparse MoE:",
            yes_no(model.sparse_moe));

        row("layers:", model.block_count);

        row(
            "embedding:",
            model.embedding_length);

        row(
            "experts/layer:",
            model.expert_count);

        row(
            "active experts:",
            model.expert_used_count);

        row(
            "expert FFN:",
            model.expert_feed_forward_length);

        row(
            "expert tensors:",
            model.expert_tensor_count);

        if (model.expert_tensor_count > 0) {
            row(
                "raw bytes/quota:",
                mib(model.raw_bytes_per_quota));

            row(
                "lockable bytes/quota:",
                mib(
                    model.lockable_bytes_per_quota));
        }

        section("Memory");

        row(
            "physical RAM:",
            gib(memory.physical_bytes));

        if (memory.available_bytes > 0) {
            row(
                "available RAM estimate:",
                gib(memory.available_bytes));
        } else {
            row(
                "available RAM estimate:",
                "unknown");
        }

        row(
            "available RAM source:",
            memory.available_source);

        row(
            "VM page size:",
            std::to_string(memory.page_size) +
            " bytes");

        {
            std::ostringstream ratio;

            ratio
                << std::fixed
                << std::setprecision(2)
                << policy.oversubscription_ratio
                << "x";

            row(
                "oversubscription:",
                ratio.str());
        }

        row(
            "memory mode:",
            policy.memory_mode);

        section("Expert residency");

        row(
            "architecture supported:",
            yes_no(
                policy.architecture_supported));

        if (policy.oversized &&
            policy.architecture_supported &&
            model.sparse_moe) {
            std::ostringstream fraction;

            fraction
                << std::fixed
                << std::setprecision(1)
                << policy.target_cache_fraction * 100.0
                << "% physical RAM";

            row(
                "target budget fraction:",
                fraction.str());

            row(
                "target cache budget:",
                gib(
                    policy.target_cache_budget_bytes));

            row(
                "RAM safety reserve:",
                gib(
                    policy.safety_reserve_bytes));

            row(
                "safe cache budget:",
                gib(
                    policy.cache_budget_bytes));

            row(
                "estimated quota:",
                policy.experts_per_tensor);

            if (policy.experts_per_tensor > 0) {
                row(
                    "runtime flag:",
                    "--expert-residency-per-tensor " +
                    std::to_string(
                        policy.experts_per_tensor));
            } else {
                row(
                    "runtime flag:",
                    "residency disabled");
            }
        } else {
            row(
                "estimated quota:",
                policy.experts_per_tensor);
        }

        section("Engine policy");

        row(
            "mmap:",
            yes_no(policy.use_mmap));

        row(
            "full mmap prefetch:",
            policy.disable_full_prefetch
                ? "disabled"
                : "engine default");

        row(
            "repack:",
            policy.disable_repack
                ? "disabled"
                : "engine default");

        section("Launch");

        row(
            "ready:",
            yes_no(policy.ready));

        row(
            "reason:",
            policy.reason);

        if (policy.ready &&
            policy.oversized) {
            std::cout
                << '\n'
                << "planned llama.cpp flags:\n"
                << "  -lm mmap\n";

            if (policy.disable_full_prefetch) {
                std::cout
                    << "  --no-mmap-prefetch\n";
            }

            if (policy.disable_repack) {
                std::cout
                    << "  --no-repack\n";
            }

            if (policy.experts_per_tensor > 0) {
                std::cout
                    << "  --expert-residency-per-tensor "
                    << policy.experts_per_tensor
                    << '\n';
            }
        }

        std::cout << '\n';

        return policy.ready ? 0 : 2;
    }
    catch (const std::exception & e) {
        std::cerr
            << "oversized-moe-probe: error: "
            << e.what()
            << '\n';

        return 1;
    }
}
