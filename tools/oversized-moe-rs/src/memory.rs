#[cfg(target_os = "linux")]
use std::fs;

use std::io;
use std::process::Command;

#[derive(Debug, Default, Clone)]
pub struct MemoryInfo {
    pub physical_bytes: u64,
    pub available_bytes: u64,
    pub page_size: u64,
    pub available_source: String,
}

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::other(msg.into())
}

fn command_u64(program: &str, args: &[&str]) -> io::Result<u64> {
    let output = Command::new(program).args(args).output()?;

    if !output.status.success() {
        return Err(err(format!("{program} failed")));
    }

    let text = String::from_utf8_lossy(&output.stdout);

    text.trim()
        .parse::<u64>()
        .map_err(|_| err(format!("cannot parse output from {program}")))
}

#[cfg(target_os = "macos")]
fn probe_platform(page_size: u64) -> io::Result<MemoryInfo> {
    let physical_bytes = command_u64("sysctl", &["-n", "hw.memsize"])?;

    let output = Command::new("vm_stat").output()?;

    let mut free = 0u64;
    let mut inactive = 0u64;
    let mut speculative = 0u64;

    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout);

        for line in text.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };

            let value = value
                .trim()
                .trim_end_matches('.')
                .parse::<u64>()
                .unwrap_or(0);

            match key.trim() {
                "Pages free" => free = value,
                "Pages inactive" => inactive = value,
                "Pages speculative" => speculative = value,
                _ => {}
            }
        }
    }

    let pages = free.saturating_add(inactive).saturating_add(speculative);

    Ok(MemoryInfo {
        physical_bytes,
        available_bytes: pages.saturating_mul(page_size),
        page_size,
        available_source: if pages > 0 {
            "free + inactive + speculative VM pages".to_owned()
        } else {
            "unavailable".to_owned()
        },
    })
}

#[cfg(target_os = "linux")]
fn probe_platform(page_size: u64) -> io::Result<MemoryInfo> {
    let text = fs::read_to_string("/proc/meminfo")?;

    let mut total_kib = 0u64;
    let mut available_kib = 0u64;

    for line in text.lines() {
        if let Some(value) = line.strip_prefix("MemTotal:") {
            total_kib = value
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
        } else if let Some(value) = line.strip_prefix("MemAvailable:") {
            available_kib = value
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
        }
    }

    Ok(MemoryInfo {
        physical_bytes: total_kib * 1024,
        available_bytes: available_kib * 1024,
        page_size,
        available_source: if available_kib > 0 {
            "/proc/meminfo MemAvailable".to_owned()
        } else {
            "unavailable".to_owned()
        },
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_platform(page_size: u64) -> io::Result<MemoryInfo> {
    let physical_pages = command_u64("getconf", &["_PHYS_PAGES"])?;

    let available_pages = command_u64("getconf", &["_AVPHYS_PAGES"]).unwrap_or(0);

    Ok(MemoryInfo {
        physical_bytes: physical_pages.saturating_mul(page_size),
        available_bytes: available_pages.saturating_mul(page_size),
        page_size,
        available_source: if available_pages > 0 {
            "_SC_AVPHYS_PAGES".to_owned()
        } else {
            "unavailable".to_owned()
        },
    })
}

pub fn probe_memory() -> io::Result<MemoryInfo> {
    let page_size = command_u64("getconf", &["PAGESIZE"])?;

    if page_size == 0 {
        return Err(err("failed to determine VM page size"));
    }

    let info = probe_platform(page_size)?;

    if info.physical_bytes == 0 {
        return Err(err("physical RAM probe returned zero"));
    }

    Ok(info)
}
