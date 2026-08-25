#include "launcher.h"

#include <exception>
#include <iostream>
#include <string>
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

void expect_rejected(
        const std::vector<std::string> & args,
        const std::string & description) {
    try {
        validate_passthrough_args(args);

        check(false, description + " is rejected");
    }
    catch (const std::exception &) {
        check(true, description + " is rejected");
    }
}

void expect_allowed(
        const std::vector<std::string> & args,
        const std::string & description) {
    try {
        validate_passthrough_args(args);

        check(true, description + " is allowed");
    }
    catch (const std::exception & e) {
        std::cerr
            << "unexpected exception: "
            << e.what()
            << '\n';

        check(false, description + " is allowed");
    }
}

void test_runtime_owned_options() {
    std::cout << "\n[Runtime-owned options]\n";

    expect_rejected(
        {"-m", "other.gguf"},
        "-m");

    expect_rejected(
        {"--model", "other.gguf"},
        "--model");

    expect_rejected(
        {"-lm", "mmap"},
        "-lm");

    expect_rejected(
        {"--load-mode", "mmap"},
        "--load-mode");

    expect_rejected(
        {"--load-mode=mmap"},
        "--load-mode=...");

    expect_rejected(
        {"--mmap"},
        "--mmap");

    expect_rejected(
        {"--no-mmap"},
        "--no-mmap");

    expect_rejected(
        {"--repack"},
        "--repack");

    expect_rejected(
        {"--no-repack"},
        "--no-repack");

    expect_rejected(
        {"--mlock"},
        "--mlock");
}

void test_completion_passthrough() {
    std::cout << "\n[Completion passthrough]\n";

    expect_allowed(
        {
            "-t", "4",
            "-tb", "4",
            "-c", "512",
            "-n", "32",
            "--temp", "0",
            "-p", "Hello"
        },
        "normal completion arguments");
}

void test_server_passthrough() {
    std::cout << "\n[Server passthrough]\n";

    expect_allowed(
        {
            "--host", "127.0.0.1",
            "--port", "8080",
            "-t", "4",
            "-c", "2048"
        },
        "normal server arguments");
}

void test_passthrough_forward_compatibility() {
    std::cout << "\n[Passthrough forward compatibility]\n";

    expect_allowed(
        {},
        "empty argument list");

    // The runtime owns only its memory-policy options.
    // Other llama.cpp options should remain transparent
    // passthrough so the wrapper does not need to mirror
    // the complete engine CLI.
    expect_allowed(
        {
            "--future-llama-option",
            "value"
        },
        "unknown non-runtime option");
}

} // namespace

int main() {
    test_runtime_owned_options();
    test_completion_passthrough();
    test_server_passthrough();
    test_passthrough_forward_compatibility();

    std::cout << '\n';

    if (failures != 0) {
        std::cerr
            << failures
            << " oversized-moe CLI argument test(s) failed\n";

        return 1;
    }

    std::cout
        << "All oversized-moe CLI argument tests passed\n";

    return 0;
}
