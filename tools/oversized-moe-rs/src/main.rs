mod gguf;
mod launch;
mod memory;
mod model;
mod policy;
mod validation;

use crate::launch::{
    execute_launch_plan, find_llama_completion, make_completion_launch_plan, print_launch_plan,
};
use crate::memory::probe_memory;
use crate::model::{ModelInfo, probe_model};
use crate::policy::{MoePolicy, make_moe_policy};
use crate::validation::{print_runtime_validation, validate_runtime_policy};

use std::env;
use std::fmt::Display;
use std::process::ExitCode;

fn row(label: &str, value: impl Display) {
    println!("{label:<31}{value}");
}

fn section(name: &str) {
    println!();
    println!("{name}");
    println!("{}", "-".repeat(name.len()));
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn mib(bytes: u64) -> String {
    format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn model_size(bytes: u64) -> String {
    format!(
        "{:.2} GiB  ({:.2} GB)",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        bytes as f64 / 1_000_000_000.0
    )
}

fn usage(program: &str) {
    println!("Oversized sparse-MoE runtime");
    println!();
    println!("usage:");
    println!("  {program} probe MODEL.gguf");
    println!();
    println!(
        "  {program} run [--dry-run] MODEL.gguf \
[LLAMA-COMPLETION-ARGS...]"
    );
}

fn probe(path: &str) -> Result<i32, Box<dyn std::error::Error>> {
    let memory = probe_memory()?;
    let model = probe_model(path, memory.page_size)?;
    let policy = make_moe_policy(&model, &memory);

    section("Model");

    row("path:", &model.path);

    if !model.name.is_empty() {
        row("name:", &model.name);
    }

    row("file size:", model_size(model.file_size));
    row("architecture:", &model.architecture);
    row("sparse MoE:", yes_no(model.sparse_moe));
    row("layers:", model.block_count);
    row("embedding:", model.embedding_length);
    row("experts/layer:", model.expert_count);
    row("active experts:", model.expert_used_count);
    row("expert FFN:", model.expert_feed_forward_length);
    row("expert tensors:", model.expert_tensor_count);

    if model.expert_tensor_count > 0 {
        row("raw bytes/quota:", mib(model.raw_bytes_per_quota));
        row("lockable bytes/quota:", mib(model.lockable_bytes_per_quota));
    }

    section("Memory");

    row("physical RAM:", gib(memory.physical_bytes));

    if memory.available_bytes > 0 {
        row("available RAM estimate:", gib(memory.available_bytes));
    } else {
        row("available RAM estimate:", "unknown");
    }

    row("available RAM source:", &memory.available_source);

    row("VM page size:", format!("{} bytes", memory.page_size));

    row(
        "oversubscription:",
        format!("{:.2}x", policy.oversubscription_ratio),
    );

    row("memory mode:", &policy.memory_mode);

    section("Expert residency");

    row(
        "architecture supported:",
        yes_no(policy.architecture_supported),
    );

    if policy.oversized && policy.architecture_supported && model.sparse_moe {
        row(
            "target budget fraction:",
            format!("{:.1}% physical RAM", policy.target_cache_fraction * 100.0),
        );

        row(
            "target cache budget:",
            gib(policy.target_cache_budget_bytes),
        );

        row("RAM safety reserve:", gib(policy.safety_reserve_bytes));

        row("safe cache budget:", gib(policy.cache_budget_bytes));

        row("estimated quota:", policy.experts_per_tensor);

        if policy.experts_per_tensor > 0 {
            row(
                "runtime flag:",
                format!(
                    "--expert-residency-per-tensor {}",
                    policy.experts_per_tensor
                ),
            );
        } else {
            row("runtime flag:", "residency disabled");
        }
    } else {
        row("estimated quota:", policy.experts_per_tensor);
    }

    section("Engine policy");

    row("mmap:", yes_no(policy.use_mmap));

    row(
        "full mmap prefetch:",
        if policy.disable_full_prefetch {
            "disabled"
        } else {
            "engine default"
        },
    );

    row(
        "repack:",
        if policy.disable_repack {
            "disabled"
        } else {
            "engine default"
        },
    );

    section("Launch");

    row("ready:", yes_no(policy.ready));
    row("reason:", &policy.reason);

    if policy.ready && policy.oversized {
        println!();
        println!("planned llama.cpp flags:");
        println!("  -lm mmap");

        if policy.disable_full_prefetch {
            println!("  --no-mmap-prefetch");
        }

        if policy.disable_repack {
            println!("  --no-repack");
        }

        if policy.experts_per_tensor > 0 {
            println!(
                "  --expert-residency-per-tensor {}",
                policy.experts_per_tensor
            );
        }
    }

    println!();

    Ok(if policy.ready { 0 } else { 2 })
}

fn print_run_summary(
    model: &ModelInfo,
    physical_bytes: u64,
    policy: &MoePolicy,
    engine: &std::path::Path,
) {
    println!("Oversized MoE runtime");
    println!("---------------------");

    println!(
        "Model:              {}",
        if model.name.is_empty() {
            &model.path
        } else {
            &model.name
        }
    );
    println!("Architecture:       {}", model.architecture);
    println!("Model size:         {}", gib(model.file_size));
    println!("Physical RAM:       {}", gib(physical_bytes));
    println!("Memory mode:        {}", policy.memory_mode);

    if policy.oversized {
        println!("Oversubscription:   {:.2}x", policy.oversubscription_ratio);
        println!("Expert budget:      {}", gib(policy.cache_budget_bytes));
        println!("Expert quota:       {}/tensor", policy.experts_per_tensor);
    }

    println!("Engine:             {}", engine.display());
}

fn run(program: &str, args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    let mut index = 0usize;
    let mut dry_run = false;

    if args.first().is_some_and(|a| a == "--dry-run") {
        dry_run = true;
        index += 1;
    }

    let Some(model_path) = args.get(index) else {
        return Err("run requires MODEL.gguf".into());
    };

    index += 1;

    let passthrough_args = &args[index..];

    let memory = probe_memory()?;
    let model = probe_model(model_path, memory.page_size)?;
    let policy = make_moe_policy(&model, &memory);

    if !policy.ready {
        eprintln!("oversized-moe-rs: {}", policy.reason);
        return Ok(2);
    }

    let engine = find_llama_completion(program)?;

    let plan = make_completion_launch_plan(engine.clone(), &model, &policy, passthrough_args)?;

    let validation = validate_runtime_policy(&model, &memory, &policy, &engine);

    if !validation.ok {
        print_runtime_validation(&validation);
        return Ok(3);
    }

    print_run_summary(&model, memory.physical_bytes, &policy, &engine);

    print_runtime_validation(&validation);
    print_launch_plan(&plan);

    if dry_run {
        println!();
        println!("Dry run only; llama-completion was not started.");
        return Ok(0);
    }

    println!();
    println!("Starting llama-completion...");

    use std::io::Write;
    std::io::stdout().flush()?;

    let error = execute_launch_plan(&plan);

    Err(format!("exec failed: {error}").into())
}

fn real_main() -> Result<i32, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let program = args
        .first()
        .map(String::as_str)
        .unwrap_or("oversized-moe-rs");

    let Some(command) = args.get(1) else {
        usage(program);
        return Ok(1);
    };

    match command.as_str() {
        "probe" => {
            if args.len() != 3 {
                usage(program);
                return Ok(1);
            }

            probe(&args[2])
        }

        "run" => run(program, &args[2..]),

        "--help" | "-h" | "help" => {
            usage(program);
            Ok(0)
        }

        _ => {
            eprintln!("oversized-moe-rs: unknown command: {command}");
            eprintln!();
            usage(program);
            Ok(1)
        }
    }
}

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => ExitCode::from(code as u8),

        Err(e) => {
            eprintln!("oversized-moe-rs: error: {e}");
            ExitCode::from(1)
        }
    }
}
