#pragma once

enum class RuntimeFrontendMode {
    completion,
    server,
};

int oversized_moe_runtime_main(
    RuntimeFrontendMode mode,
    int argc,
    char ** argv);
