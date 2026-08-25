#include "runtime_frontend.h"

int main(int argc, char ** argv) {
    return oversized_moe_runtime_main(
        RuntimeFrontendMode::completion,
        argc,
        argv);
}
