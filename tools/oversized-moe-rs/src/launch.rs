use crate::model::ModelInfo;
use crate::policy::MoePolicy;

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[derive(Debug, Clone)]
pub struct LaunchPlan {
    pub engine_path: PathBuf,
    pub argv: Vec<String>,
    pub env_set: Vec<(String, String)>,
    pub env_unset: Vec<String>,
}

fn executable_file(path: &Path) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };

    if !meta.is_file() {
        return false;
    }

    #[cfg(unix)]
    {
        return meta.permissions().mode() & 0o111 != 0;
    }

    #[cfg(not(unix))]
    {
        true
    }
}

fn absolute(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    let path = path.as_ref();

    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn resolve_from_path(program: &str) -> Option<PathBuf> {
    let path_env = env::var_os("PATH")?;

    for dir in env::split_paths(&path_env) {
        let candidate = if dir.as_os_str().is_empty() {
            env::current_dir().ok()?.join(program)
        } else {
            dir.join(program)
        };

        if executable_file(&candidate) {
            return absolute(candidate).ok();
        }
    }

    None
}

fn resolve_launcher(argv0: &str) -> io::Result<PathBuf> {
    let raw = Path::new(argv0);

    if raw.parent().is_some_and(|p| !p.as_os_str().is_empty()) {
        return absolute(raw);
    }

    if let Some(found) = resolve_from_path(argv0) {
        return Ok(found);
    }

    absolute(raw)
}

fn find_engine(launcher_argv0: &str, override_env: &str, engine_name: &str) -> io::Result<PathBuf> {
    if let Ok(override_path) = env::var(override_env) {
        if !override_path.is_empty() {
            let candidate = PathBuf::from(&override_path);

            if !executable_file(&candidate) {
                return Err(io::Error::other(format!(
                    "{override_env} is not executable: {}",
                    candidate.display()
                )));
            }

            return absolute(candidate);
        }
    }

    let launcher = resolve_launcher(launcher_argv0)?;

    if let Some(parent) = launcher.parent() {
        let sibling = parent.join(engine_name);

        if executable_file(&sibling) {
            return Ok(sibling);
        }
    }

    let repo_build_candidate = env::current_dir()?.join("build/bin").join(engine_name);

    if executable_file(&repo_build_candidate) {
        return absolute(repo_build_candidate);
    }

    if let Some(found) = resolve_from_path(engine_name) {
        return Ok(found);
    }

    Err(io::Error::other(format!(
        "cannot find {engine_name}; expected it next to the launcher, \
in build/bin, on PATH, or via {override_env}"
    )))
}

pub fn find_llama_completion(launcher_argv0: &str) -> io::Result<PathBuf> {
    find_engine(launcher_argv0, "LLAMA_COMPLETION_BIN", "llama-completion")
}

fn starts_with(arg: &str, prefix: &str) -> bool {
    arg.starts_with(prefix)
}

pub fn validate_passthrough_args(args: &[String]) -> io::Result<()> {
    for arg in args {
        if arg == "-m" || arg == "--model" || starts_with(arg, "--model=") {
            return Err(io::Error::other(
                "do not pass -m/--model: the model is controlled by \
the oversized MoE runtime",
            ));
        }

        if arg == "-lm" || arg == "--load-mode" || starts_with(arg, "--load-mode=") {
            return Err(io::Error::other(
                "do not pass -lm/--load-mode: model load mode is \
controlled by the oversized MoE runtime",
            ));
        }

        if arg == "--mmap" || arg == "--no-mmap" {
            return Err(io::Error::other(
                "do not pass --mmap/--no-mmap: mmap policy is \
controlled by the oversized MoE runtime",
            ));
        }

        if arg == "--mmap-prefetch" || arg == "--no-mmap-prefetch" {
            return Err(io::Error::other(
                "do not pass --mmap-prefetch/--no-mmap-prefetch: \
mmap prefetch policy is controlled by the oversized MoE runtime",
            ));
        }

        if arg == "--expert-residency-per-tensor"
            || starts_with(arg, "--expert-residency-per-tensor=")
        {
            return Err(io::Error::other(
                "do not pass --expert-residency-per-tensor: expert \
residency policy is controlled by the oversized MoE runtime",
            ));
        }

        if arg == "--repack" || arg == "--no-repack" {
            return Err(io::Error::other(
                "do not pass --repack/--no-repack: repack policy is \
controlled by the oversized MoE runtime",
            ));
        }

        if arg == "--mlock" {
            return Err(io::Error::other(
                "do not pass --mlock: memory locking policy is \
controlled by the oversized MoE runtime",
            ));
        }
    }

    Ok(())
}

pub fn make_completion_launch_plan(
    engine_path: PathBuf,
    model: &ModelInfo,
    policy: &MoePolicy,
    passthrough_args: &[String],
) -> io::Result<LaunchPlan> {
    validate_passthrough_args(passthrough_args)?;

    let engine = engine_path.to_string_lossy().into_owned();

    let mut plan = LaunchPlan {
        engine_path,
        argv: vec![engine, "-m".to_owned(), model.path.clone()],
        env_set: Vec::new(),
        env_unset: Vec::new(),
    };

    if policy.oversized {
        if policy.use_mmap {
            plan.argv.push("-lm".to_owned());
            plan.argv.push("mmap".to_owned());
        }

        plan.argv.push("--no-mmap-prefetch".to_owned());

        if policy.disable_repack {
            plan.argv.push("--no-repack".to_owned());
        }
    }

    if policy.oversized && policy.experts_per_tensor > 0 {
        plan.argv.push("--expert-residency-per-tensor".to_owned());
        plan.argv.push(policy.experts_per_tensor.to_string());
    }

    plan.argv.extend_from_slice(passthrough_args);

    plan.env_unset
        .push("GGML_EXPERT_RESIDENT_PER_TENSOR".to_owned());
    plan.env_unset
        .push("LLAMA_ARG_EXPERT_RESIDENCY_PER_TENSOR".to_owned());

    Ok(plan)
}

fn shell_quote(arg: &str) -> String {
    if !arg.is_empty()
        && arg.bytes().all(|c| {
            c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.' | b'/' | b':' | b'=' | b'+')
        })
    {
        return arg.to_owned();
    }

    format!("'{}'", arg.replace('\'', "'\\''"))
}

pub fn print_launch_plan(plan: &LaunchPlan) {
    println!();
    println!("Environment");
    println!("-----------");

    if plan.env_set.is_empty() && plan.env_unset.is_empty() {
        println!("(unchanged)");
    }

    for (key, value) in &plan.env_set {
        println!("{key}={value}");
    }

    for key in &plan.env_unset {
        println!("unset {key}");
    }

    println!();
    println!("Command");
    println!("-------");

    println!(
        "{}",
        plan.argv
            .iter()
            .map(|v| shell_quote(v))
            .collect::<Vec<_>>()
            .join(" ")
    );
}

pub fn execute_launch_plan(plan: &LaunchPlan) -> io::Error {
    let mut command = Command::new(&plan.engine_path);

    command.args(&plan.argv[1..]);

    for key in &plan.env_unset {
        command.env_remove(key);
    }

    for (key, value) in &plan.env_set {
        command.env(key, value);
    }

    #[cfg(unix)]
    {
        command.exec()
    }

    #[cfg(not(unix))]
    {
        match command.status() {
            Ok(status) => io::Error::other(format!("engine exited with status {status}")),
            Err(e) => e,
        }
    }
}
