#include "launcher.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

bool executable_file(
        const std::filesystem::path & path) {
    std::error_code ec;

    if (!std::filesystem::is_regular_file(path, ec)) {
        return false;
    }

    return access(path.c_str(), X_OK) == 0;
}

std::filesystem::path resolve_from_path(
        const std::string & program) {
    const char * path_env =
        std::getenv("PATH");

    if (path_env == nullptr) {
        return {};
    }

    std::string path_list(path_env);
    std::size_t begin = 0;

    while (begin <= path_list.size()) {
        const std::size_t end =
            path_list.find(':', begin);

        const std::string dir =
            end == std::string::npos
                ? path_list.substr(begin)
                : path_list.substr(
                    begin,
                    end - begin);

        const std::filesystem::path candidate =
            (dir.empty()
                ? std::filesystem::current_path()
                : std::filesystem::path(dir))
            / program;

        if (executable_file(candidate)) {
            return
                std::filesystem::absolute(candidate)
                .lexically_normal();
        }

        if (end == std::string::npos) {
            break;
        }

        begin = end + 1;
    }

    return {};
}

std::filesystem::path resolve_launcher(
        const std::string & argv0) {
    const std::filesystem::path raw(argv0);

    if (raw.has_parent_path()) {
        return
            std::filesystem::absolute(raw)
            .lexically_normal();
    }

    const auto from_path =
        resolve_from_path(argv0);

    if (!from_path.empty()) {
        return from_path;
    }

    return
        std::filesystem::absolute(raw)
        .lexically_normal();
}

std::string find_engine(
        const std::string & launcher_argv0,
        const char * override_env,
        const std::string & engine_name) {
    if (const char * override_path =
            std::getenv(override_env);
        override_path != nullptr &&
        *override_path != '\0') {

        const std::filesystem::path candidate(
            override_path);

        if (!executable_file(candidate)) {
            throw std::runtime_error(
                std::string(override_env) +
                " is not executable: " +
                candidate.string());
        }

        return
            std::filesystem::absolute(candidate)
            .lexically_normal()
            .string();
    }

    const auto launcher =
        resolve_launcher(launcher_argv0);

    const auto sibling =
        launcher.parent_path() /
        engine_name;

    if (executable_file(sibling)) {
        return sibling.string();
    }

    const auto repo_build_candidate =
        std::filesystem::current_path() /
        "build/bin" /
        engine_name;

    if (executable_file(repo_build_candidate)) {
        return
            std::filesystem::absolute(
                repo_build_candidate)
            .lexically_normal()
            .string();
    }

    const auto from_path =
        resolve_from_path(engine_name);

    if (!from_path.empty()) {
        return from_path.string();
    }

    throw std::runtime_error(
        "cannot find " +
        engine_name +
        "; expected it next to the launcher "
        "or set " +
        override_env);
}

std::string shell_quote(
        const std::string & value) {
    if (value.empty()) {
        return "''";
    }

    bool simple = true;

    for (const unsigned char c : value) {
        const bool ok =
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '_' ||
            c == '-' ||
            c == '.' ||
            c == '/' ||
            c == ':' ||
            c == '=';

        if (!ok) {
            simple = false;
            break;
        }
    }

    if (simple) {
        return value;
    }

    std::string out = "'";

    for (const char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }

    out += '\'';

    return out;
}

bool starts_with(
        const std::string & value,
        const std::string & prefix) {
    return
        value.size() >= prefix.size() &&
        value.compare(
            0,
            prefix.size(),
            prefix) == 0;
}

LaunchPlan make_engine_launch_plan(
        const std::string & engine_path,
        const ModelInfo & model,
        const MoePolicy & policy,
        const std::vector<std::string> &
            passthrough_args) {
    validate_passthrough_args(
        passthrough_args);

    LaunchPlan plan;

    plan.engine_path = engine_path;

    plan.argv.push_back(engine_path);

    plan.argv.push_back("-m");
    plan.argv.push_back(model.path);

    if (policy.oversized) {
        if (policy.use_mmap) {
            plan.argv.push_back("-lm");
            plan.argv.push_back("mmap");
        }

        if (policy.disable_repack) {
            plan.argv.push_back("--no-repack");
        }
    }

    for (const auto & arg : passthrough_args) {
        plan.argv.push_back(arg);
    }

    if (policy.oversized &&
        policy.experts_per_tensor > 0) {
        plan.env_set.emplace_back(
            "GGML_EXPERT_RESIDENT_PER_TENSOR",
            std::to_string(
                policy.experts_per_tensor));
    } else {
        // Never inherit a stale research setting when the
        // automatic policy has decided not to use residency.
        plan.env_unset.push_back(
            "GGML_EXPERT_RESIDENT_PER_TENSOR");
    }

    return plan;
}

} // namespace

std::string find_llama_completion(
        const std::string & launcher_argv0) {
    return find_engine(
        launcher_argv0,
        "LLAMA_COMPLETION_BIN",
        "llama-completion");
}

std::string find_llama_server(
        const std::string & launcher_argv0) {
    return find_engine(
        launcher_argv0,
        "LLAMA_SERVER_BIN",
        "llama-server");
}

void validate_passthrough_args(
            const std::vector<std::string> & args) {
        for (const auto & arg : args) {
            if (arg == "-m" ||
                arg == "--model" ||
                starts_with(arg, "--model=")) {
                throw std::runtime_error(
                    "do not pass -m/--model: "
                    "the model is controlled by "
                    "the oversized MoE runtime");
            }

            if (arg == "-lm" ||
                arg == "--load-mode" ||
                starts_with(arg, "--load-mode=")) {
                throw std::runtime_error(
                    "do not pass -lm/--load-mode: "
                    "model load mode is controlled by "
                    "the oversized MoE runtime");
            }

            if (arg == "--mmap" ||
                arg == "--no-mmap") {
                throw std::runtime_error(
                    "do not pass --mmap/--no-mmap: "
                    "mmap policy is controlled by "
                    "the oversized MoE runtime");
            }

            if (arg == "--repack" ||
                arg == "--no-repack") {
                throw std::runtime_error(
                    "do not pass --repack/--no-repack: "
                    "repack policy is controlled by "
                    "the oversized MoE runtime");
            }

            if (arg == "--mlock") {
                throw std::runtime_error(
                    "do not pass --mlock: "
                    "memory locking policy is controlled by "
                    "the oversized MoE runtime");
            }
        }
    }

LaunchPlan make_completion_launch_plan(
        const std::string & engine_path,
        const ModelInfo & model,
        const MoePolicy & policy,
        const std::vector<std::string> &
            passthrough_args) {
    return make_engine_launch_plan(
        engine_path,
        model,
        policy,
        passthrough_args);
}

LaunchPlan make_server_launch_plan(
        const std::string & engine_path,
        const ModelInfo & model,
        const MoePolicy & policy,
        const std::vector<std::string> &
            passthrough_args) {
    return make_engine_launch_plan(
        engine_path,
        model,
        policy,
        passthrough_args);
}

void print_launch_plan(
        const LaunchPlan & plan) {
    std::cout
        << "\nEnvironment\n"
        << "-----------\n";

    if (plan.env_set.empty() &&
        plan.env_unset.empty()) {
        std::cout << "(unchanged)\n";
    }

    for (const auto & [key, value] :
         plan.env_set) {
        std::cout
            << key
            << '='
            << value
            << '\n';
    }

    for (const auto & key :
         plan.env_unset) {
        std::cout
            << "unset "
            << key
            << '\n';
    }

    std::cout
        << "\nCommand\n"
        << "-------\n";

    for (std::size_t i = 0;
         i < plan.argv.size();
         ++i) {
        if (i != 0) {
            std::cout << ' ';
        }

        std::cout
            << shell_quote(plan.argv[i]);
    }

    std::cout << '\n';
}

[[noreturn]]
void execute_launch_plan(
        const LaunchPlan & plan) {
    for (const auto & key :
         plan.env_unset) {
        if (unsetenv(key.c_str()) != 0) {
            throw std::runtime_error(
                "unsetenv(" +
                key +
                ") failed");
        }
    }

    for (const auto & [key, value] :
         plan.env_set) {
        if (setenv(
                key.c_str(),
                value.c_str(),
                1) != 0) {
            throw std::runtime_error(
                "setenv(" +
                key +
                ") failed");
        }
    }

    std::vector<char *> argv;

    argv.reserve(
        plan.argv.size() + 1);

    for (const auto & arg : plan.argv) {
        argv.push_back(
            const_cast<char *>(
                arg.c_str()));
    }

    argv.push_back(nullptr);

    execv(
        plan.engine_path.c_str(),
        argv.data());

    const int error = errno;

    throw std::runtime_error(
        "execv failed: " +
        std::string(
            std::strerror(error)));
}
