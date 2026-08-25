mod gguf;
mod memory;
mod model;
mod policy;

use crate::memory::probe_memory;
use crate::model::probe_model;
use crate::policy::make_moe_policy;

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

fn model_size(bytes: u64) -> String {
    format!(
        "{:.2} GiB  ({:.2} GB)",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        bytes as f64 / 1_000_000_000.0
    )
}

fn mib(bytes: u64) -> String {
    format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn usage(program: &str) {
    eprintln!("usage: {program} probe MODEL.gguf");
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

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 || args[1] != "probe" {
        usage(
            args.first()
                .map(String::as_str)
                .unwrap_or("oversized-moe-rs"),
        );
        return ExitCode::from(1);
    }

    match probe(&args[2]) {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("oversized-moe-rs: error: {e}");
            ExitCode::from(1)
        }
    }
}
