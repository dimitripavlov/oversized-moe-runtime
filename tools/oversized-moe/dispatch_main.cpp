#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

bool executable_file(const std::filesystem::path & path) {
    std::error_code ec;

    if (!std::filesystem::is_regular_file(path, ec)) {
        return false;
    }

    return access(path.c_str(), X_OK) == 0;
}

std::filesystem::path launcher_path(const char * argv0) {
    return std::filesystem::absolute(
        std::filesystem::path(argv0)
    ).lexically_normal();
}

void usage(const char * program) {
    std::cout
        << "Oversized sparse-MoE runtime\n"
        << "\n"
        << "usage:\n"
        << "  " << program
        << " probe MODEL.gguf\n"
        << "\n"
        << "  " << program
        << " run [--dry-run] MODEL.gguf "
           "[LLAMA-COMPLETION-ARGS...]\n"
        << "\n"
        << "  " << program
        << " serve [--dry-run] MODEL.gguf "
           "[LLAMA-SERVER-ARGS...]\n"
        << "\n"
        << "examples:\n"
        << "  " << program
        << " probe model.gguf\n"
        << "\n"
        << "  " << program
        << " run model.gguf "
           "-t 4 -c 512 -n 64 -p \"Hello\"\n"
        << "\n"
        << "  " << program
        << " serve model.gguf "
           "--host 127.0.0.1 --port 8080 "
           "-t 4 -c 2048\n";
}

std::string binary_for_command(const std::string & command) {
    if (command == "probe") {
        return "oversized-moe-probe";
    }

    if (command == "run") {
        return "oversized-moe-run";
    }

    if (command == "serve") {
        return "oversized-moe-serve";
    }

    return {};
}

[[noreturn]]
void dispatch(
        const char * argv0,
        const std::string & command,
        int argc,
        char ** argv) {
    const std::string binary =
        binary_for_command(command);

    if (binary.empty()) {
        throw std::runtime_error(
            "unknown command: " + command);
    }

    const std::filesystem::path self =
        launcher_path(argv0);

    const std::filesystem::path target =
        self.parent_path() / binary;

    if (!executable_file(target)) {
        throw std::runtime_error(
            "required runtime component not found or "
            "not executable: " +
            target.string());
    }

    std::vector<std::string> child_storage;

    child_storage.reserve(
        static_cast<std::size_t>(argc));

    child_storage.push_back(
        target.string());

    // argv[0] = oversized-moe
    // argv[1] = subcommand
    // argv[2...] is passed unchanged to the child.
    for (int i = 2; i < argc; ++i) {
        child_storage.emplace_back(argv[i]);
    }

    std::vector<char *> child_argv;

    child_argv.reserve(
        child_storage.size() + 1);

    for (auto & arg : child_storage) {
        child_argv.push_back(arg.data());
    }

    child_argv.push_back(nullptr);

    execv(
        target.c_str(),
        child_argv.data());

    const int error = errno;

    throw std::runtime_error(
        "execv(" +
        target.string() +
        ") failed: " +
        std::strerror(error));
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 1;
        }

        const std::string command(argv[1]);

        if (command == "--help" ||
            command == "-h" ||
            command == "help") {
            usage(argv[0]);
            return 0;
        }

        if (command != "probe" &&
            command != "run" &&
            command != "serve") {
            std::cerr
                << "oversized-moe: unknown command: "
                << command
                << "\n\n";

            usage(argv[0]);

            return 1;
        }

        dispatch(
            argv[0],
            command,
            argc,
            argv);
    }
    catch (const std::exception & e) {
        std::cerr
            << "oversized-moe: error: "
            << e.what()
            << '\n';

        return 1;
    }
}
