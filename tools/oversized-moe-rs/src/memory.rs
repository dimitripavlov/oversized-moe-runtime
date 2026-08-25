use std::io;

#[cfg(target_os = "linux")]
use std::fs;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
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

#[cfg(any(target_os = "macos", target_os = "linux"))]
unsafe extern "C" {
    fn getpagesize() -> std::ffi::c_int;
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn page_size() -> io::Result<u64> {
    let value = unsafe { getpagesize() };

    if value <= 0 {
        return Err(err("failed to determine VM page size"));
    }

    Ok(value as u64)
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Default)]
struct VmStatistics64 {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,

    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,

    purgeable_count: u32,
    speculative_count: u32,

    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,

    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,

    total_uncompressed_pages_in_compressor: u64,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> std::ffi::c_int;

    fn mach_host_self() -> u32;

    fn host_statistics64(
        host: u32,
        flavor: std::ffi::c_int,
        host_info: *mut std::ffi::c_int,
        host_info_count: *mut u32,
    ) -> std::ffi::c_int;
}

#[cfg(target_os = "macos")]
fn physical_memory() -> io::Result<u64> {
    let mut value = 0u64;
    let mut size = std::mem::size_of::<u64>();

    let rc = unsafe {
        sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut value as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };

    if rc != 0 || size != std::mem::size_of::<u64>() {
        return Err(err("sysctlbyname(hw.memsize) failed"));
    }

    Ok(value)
}

#[cfg(target_os = "macos")]
fn available_memory(page_size: u64) -> (u64, String) {
    const HOST_VM_INFO64: std::ffi::c_int = 4;

    let mut stats = VmStatistics64::default();

    let mut count =
        (std::mem::size_of::<VmStatistics64>() / std::mem::size_of::<std::ffi::c_int>()) as u32;

    let rc = unsafe {
        host_statistics64(
            mach_host_self(),
            HOST_VM_INFO64,
            (&mut stats as *mut VmStatistics64).cast::<std::ffi::c_int>(),
            &mut count,
        )
    };

    if rc != 0 {
        return (0, "unavailable".to_owned());
    }

    let pages = u64::from(stats.free_count)
        .saturating_add(u64::from(stats.inactive_count))
        .saturating_add(u64::from(stats.speculative_count));

    (
        pages.saturating_mul(page_size),
        "free + inactive + speculative VM pages".to_owned(),
    )
}

#[cfg(target_os = "macos")]
fn probe_platform(page_size: u64) -> io::Result<MemoryInfo> {
    let physical_bytes = physical_memory()?;

    let (available_bytes, available_source) = available_memory(page_size);

    Ok(MemoryInfo {
        physical_bytes,
        available_bytes,
        page_size,
        available_source,
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
        physical_bytes: total_kib.saturating_mul(1024),
        available_bytes: available_kib.saturating_mul(1024),
        page_size,
        available_source: if available_kib > 0 {
            "/proc/meminfo MemAvailable".to_owned()
        } else {
            "unavailable".to_owned()
        },
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
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

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn page_size() -> io::Result<u64> {
    let value = command_u64("getconf", &["PAGESIZE"])?;

    if value == 0 {
        return Err(err("failed to determine VM page size"));
    }

    Ok(value)
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
    let page_size = page_size()?;
    let info = probe_platform(page_size)?;

    if info.physical_bytes == 0 {
        return Err(err("physical RAM probe returned zero"));
    }

    Ok(info)
}
