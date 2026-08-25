#pragma once

#include "model_probe.h"
#include "moe_policy.h"

#include <string>
#include <utility>
#include <vector>

struct LaunchPlan {
    std::string engine_path;

    std::vector<std::string> argv;

    std::vector<std::pair<std::string, std::string>>
        env_set;

    std::vector<std::string> env_unset;
};

std::string find_llama_completion(
    const std::string & launcher_argv0);

std::string find_llama_server(
    const std::string & launcher_argv0);

void validate_passthrough_args(
    const std::vector<std::string> & args);

LaunchPlan make_completion_launch_plan(
    const std::string & engine_path,
    const ModelInfo & model,
    const MoePolicy & policy,
    const std::vector<std::string> & passthrough_args);

LaunchPlan make_server_launch_plan(
    const std::string & engine_path,
    const ModelInfo & model,
    const MoePolicy & policy,
    const std::vector<std::string> & passthrough_args);

void print_launch_plan(
    const LaunchPlan & plan);

[[noreturn]]
void execute_launch_plan(
    const LaunchPlan & plan);
