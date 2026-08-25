#include "launcher.h"
#include "memory_budget.h"
#include "model_probe.h"
#include "moe_policy.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double GiB =
    1024.0 * 1024.0 * 1024.0;

std::string gib(
        std::uint64_t bytes) {
    std::ostringstream out;

    out
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(bytes) / GiB
        << " GiB";

    return out.str();
}

void usage(
        const char * program) {
    std::cout
        << "usage:\n"
        << "  "
        << program
        << " [--dry-run] MODEL.gguf "
           "[LLAMA-COMPLETION-ARGS...]\n"
        << "\n"
        << "examples:\n"
        << "  "
        << program
        << " --dry-run model.gguf "
           "-c 512 -n 64 -p \"Hello\"\n"
        << "\n"
        << "  "
        << program
        << " model.gguf "
           "-c 512 -n 64 -p \"Hello\"\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 1;
        }

        bool dry_run = false;
        int index = 1;

        if (std::string(argv[index]) ==
            "--help") {
            usage(argv[0]);
            return 0;
        }

        if (std::string(argv[index]) ==
            "--dry-run") {
            dry_run = true;
            ++index;
        }

        if (index >= argc) {
            usage(argv[0]);
            return 1;
        }

        const std::string model_path =
            argv[index++];

        std::vector<std::string>
            passthrough_args;

        for (; index < argc; ++index) {
            passthrough_args.emplace_back(
                argv[index]);
        }

        const MemoryInfo memory =
            probe_memory();

        const ModelInfo model =
            probe_model(
                model_path,
                memory.page_size);

        const MoePolicy policy =
            make_moe_policy(
                model,
                memory);

        if (!policy.ready) {
            std::cerr
                << "oversized-moe-run: "
                << policy.reason
                << '\n';

            return 2;
        }

        const std::string engine =
            find_llama_completion(
                argv[0]);

        const LaunchPlan plan =
            make_completion_launch_plan(
                engine,
                model,
                policy,
                passthrough_args);

        std::cout
            << "Oversized MoE runtime\n"
            << "---------------------\n";

        std::cout
            << "Model:              "
            << (
                model.name.empty()
                    ? model.path
                    : model.name)
            << '\n';

        std::cout
            << "Architecture:       "
            << model.architecture
            << '\n';

        std::cout
            << "Model size:         "
            << gib(model.file_size)
            << '\n';

        std::cout
            << "Physical RAM:       "
            << gib(memory.physical_bytes)
            << '\n';

        std::cout
            << "Memory mode:        "
            << policy.memory_mode
            << '\n';

        {
            std::ostringstream ratio;

            ratio
                << std::fixed
                << std::setprecision(2)
                << policy.oversubscription_ratio
                << 'x';

            std::cout
                << "Oversubscription:   "
                << ratio.str()
                << '\n';
        }

        if (policy.oversized) {
            std::cout
                << "Expert budget:      "
                << gib(
                    policy.cache_budget_bytes)
                << '\n';

            std::cout
                << "Expert quota:       "
                << policy.experts_per_tensor
                << "/tensor\n";
        }

        std::cout
            << "Engine:             "
            << engine
            << '\n';

        print_launch_plan(plan);

        if (dry_run) {
            std::cout
                << "\nDry run only; "
                   "llama-completion was not started.\n";

            return 0;
        }

        std::cout
            << "\nStarting llama-completion...\n"
            << std::flush;

        execute_launch_plan(plan);
    }
    catch (const std::exception & e) {
        std::cerr
            << "oversized-moe-run: error: "
            << e.what()
            << '\n';

        return 1;
    }
}
