#include "runtime_frontend.h"

#include "launcher.h"
#include "memory_budget.h"
#include "model_probe.h"
#include "moe_policy.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double GiB =
    1024.0 * 1024.0 * 1024.0;

bool is_server(RuntimeFrontendMode mode) {
    return mode == RuntimeFrontendMode::server;
}

const char * runtime_name(RuntimeFrontendMode mode) {
    return is_server(mode)
        ? "oversized-moe-serve"
        : "oversized-moe-run";
}

const char * engine_name(RuntimeFrontendMode mode) {
    return is_server(mode)
        ? "llama-server"
        : "llama-completion";
}

std::string gib(std::uint64_t bytes) {
    std::ostringstream out;

    out
        << std::fixed
        << std::setprecision(2)
        << static_cast<double>(bytes) / GiB
        << " GiB";

    return out.str();
}

void usage(
        RuntimeFrontendMode mode,
        const char * program) {
    std::cout
        << "usage:\n"
        << "  "
        << program
        << " [--dry-run] MODEL.gguf ";

    if (is_server(mode)) {
        std::cout << "[LLAMA-SERVER-ARGS...]\n\n"
                  << "example:\n"
                  << "  "
                  << program
                  << " --dry-run model.gguf "
                     "--host 127.0.0.1 --port 8080 "
                     "-t 4 -c 2048\n";
    } else {
        std::cout << "[LLAMA-COMPLETION-ARGS...]\n\n"
                  << "example:\n"
                  << "  "
                  << program
                  << " --dry-run model.gguf "
                     "-c 512 -n 64 -p \"Hello\"\n";
    }
}

std::string find_engine(
        RuntimeFrontendMode mode,
        const std::string & argv0) {
    return is_server(mode)
        ? find_llama_server(argv0)
        : find_llama_completion(argv0);
}

LaunchPlan make_plan(
        RuntimeFrontendMode mode,
        const std::string & engine,
        const ModelInfo & model,
        const MoePolicy & policy,
        const std::vector<std::string> & args) {
    return is_server(mode)
        ? make_server_launch_plan(
              engine,
              model,
              policy,
              args)
        : make_completion_launch_plan(
              engine,
              model,
              policy,
              args);
}

void print_summary(
        RuntimeFrontendMode mode,
        const ModelInfo & model,
        const MemoryInfo & memory,
        const MoePolicy & policy,
        const std::string & engine) {
    if (is_server(mode)) {
        std::cout
            << "Oversized MoE server runtime\n"
            << "----------------------------\n";
    } else {
        std::cout
            << "Oversized MoE runtime\n"
            << "---------------------\n";
    }

    std::cout
        << "Model:              "
        << (model.name.empty()
                ? model.path
                : model.name)
        << '\n'
        << "Architecture:       "
        << model.architecture
        << '\n'
        << "Model size:         "
        << gib(model.file_size)
        << '\n'
        << "Physical RAM:       "
        << gib(memory.physical_bytes)
        << '\n'
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
            << gib(policy.cache_budget_bytes)
            << '\n'
            << "Expert quota:       "
            << policy.experts_per_tensor
            << "/tensor\n";
    }

    std::cout
        << "Engine:             "
        << engine
        << '\n';
}

} // namespace

int oversized_moe_runtime_main(
        RuntimeFrontendMode mode,
        int argc,
        char ** argv) {
    try {
        if (argc < 2) {
            usage(mode, argv[0]);
            return 1;
        }

        bool dry_run = false;
        int index = 1;

        const std::string first(argv[index]);

        if (first == "--help" ||
            first == "-h") {
            usage(mode, argv[0]);
            return 0;
        }

        if (first == "--dry-run") {
            dry_run = true;
            ++index;
        }

        if (index >= argc) {
            usage(mode, argv[0]);
            return 1;
        }

        const std::string model_path =
            argv[index++];

        std::vector<std::string> passthrough_args;

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
                << runtime_name(mode)
                << ": "
                << policy.reason
                << '\n';

            return 2;
        }

        const std::string engine =
            find_engine(
                mode,
                argv[0]);

        const LaunchPlan plan =
            make_plan(
                mode,
                engine,
                model,
                policy,
                passthrough_args);

        print_summary(
            mode,
            model,
            memory,
            policy,
            engine);

        print_launch_plan(plan);

        if (dry_run) {
            std::cout
                << "\nDry run only; "
                << engine_name(mode)
                << " was not started.\n";

            return 0;
        }

        std::cout
            << "\nStarting "
            << engine_name(mode)
            << "...\n"
            << std::flush;

        execute_launch_plan(plan);
    }
    catch (const std::exception & e) {
        std::cerr
            << runtime_name(mode)
            << ": error: "
            << e.what()
            << '\n';

        return 1;
    }
}
