set(TARGET oversized-moe-probe)

add_executable(${TARGET}
    main.cpp
    model_probe.cpp
    memory_budget.cpp
    moe_policy.cpp
)

target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
