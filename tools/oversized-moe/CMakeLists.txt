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
    runtime_frontend.cpp
    runtime_validation.cpp
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
    runtime_frontend.cpp
    runtime_validation.cpp
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


if (LLAMA_BUILD_TESTS)
    set(POLICY_TEST_TARGET oversized-moe-policy-tests)

    add_executable(
        ${POLICY_TEST_TARGET}
        policy_tests.cpp
        runtime_validation.cpp
        moe_policy.cpp
    )

    target_compile_features(
        ${POLICY_TEST_TARGET}
        PRIVATE
        cxx_std_17
    )

    add_test(
        NAME oversized-moe-policy
        COMMAND ${POLICY_TEST_TARGET}
    )
endif()


if (LLAMA_BUILD_TESTS)
    set(CLI_ARGS_TEST_TARGET oversized-moe-cli-args-tests)

    add_executable(
        ${CLI_ARGS_TEST_TARGET}
        cli_args_tests.cpp
        launcher.cpp
    )

    target_compile_features(
        ${CLI_ARGS_TEST_TARGET}
        PRIVATE
        cxx_std_17
    )

    add_test(
        NAME oversized-moe-cli-args
        COMMAND ${CLI_ARGS_TEST_TARGET}
    )
endif()
