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
