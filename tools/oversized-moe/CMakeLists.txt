set(PROBE_TARGET oversized-moe-probe)

add_executable(${PROBE_TARGET}
    main.cpp
    model_probe.cpp
    memory_budget.cpp
    moe_policy.cpp
)

target_link_libraries(
    ${PROBE_TARGET}
    PRIVATE
    ggml
)

target_compile_features(
    ${PROBE_TARGET}
    PRIVATE
    cxx_std_17
)


set(RUN_TARGET oversized-moe-run)

add_executable(${RUN_TARGET}
    run_main.cpp
    launcher.cpp
    model_probe.cpp
    memory_budget.cpp
    moe_policy.cpp
)

target_link_libraries(
    ${RUN_TARGET}
    PRIVATE
    ggml
)

target_compile_features(
    ${RUN_TARGET}
    PRIVATE
    cxx_std_17
)


set(SERVE_TARGET oversized-moe-serve)

add_executable(${SERVE_TARGET}
    serve_main.cpp
    launcher.cpp
    model_probe.cpp
    memory_budget.cpp
    moe_policy.cpp
)

target_link_libraries(
    ${SERVE_TARGET}
    PRIVATE
    ggml
)

target_compile_features(
    ${SERVE_TARGET}
    PRIVATE
    cxx_std_17
)


set(CLI_TARGET oversized-moe)

add_executable(${CLI_TARGET}
    dispatch_main.cpp
)

target_compile_features(
    ${CLI_TARGET}
    PRIVATE
    cxx_std_17
)
