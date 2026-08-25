#include "memory_budget.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <unistd.h>

#if defined(__APPLE__)

#include <mach/mach.h>
#include <sys/sysctl.h>

#elif defined(__linux__)

#include <sys/sysinfo.h>

#endif

namespace {

#if defined(__linux__)

std::uint64_t linux_mem_available() {
    std::ifstream in("/proc/meminfo");

    if (!in) {
        return 0;
    }

    std::string line;

    while (std::getline(in, line)) {
        if (line.rfind("MemAvailable:", 0) != 0) {
            continue;
        }

        std::istringstream iss(line);

        std::string key;
        std::uint64_t kib = 0;
        std::string unit;

        iss >> key >> kib >> unit;

        if (!iss) {
            return 0;
        }

        return kib * 1024ULL;
    }

    return 0;
}

#endif

} // namespace

MemoryInfo probe_memory() {
    MemoryInfo info;

    const long page_size =
        sysconf(_SC_PAGESIZE);

    if (page_size <= 0) {
        throw std::runtime_error(
            "failed to determine VM page size");
    }

    info.page_size =
        static_cast<std::uint64_t>(page_size);

#if defined(__APPLE__)

    {
        std::uint64_t physical = 0;
        size_t len = sizeof(physical);

        if (sysctlbyname(
                "hw.memsize",
                &physical,
                &len,
                nullptr,
                0) != 0) {
            throw std::runtime_error(
                "sysctlbyname(hw.memsize) failed");
        }

        info.physical_bytes = physical;
    }

    {
        const mach_port_t host =
            mach_host_self();

        vm_statistics64_data_t vm_stats {};

        mach_msg_type_number_t count =
            HOST_VM_INFO64_COUNT;

        const kern_return_t kr =
            host_statistics64(
                host,
                HOST_VM_INFO64,
                reinterpret_cast<host_info64_t>(
                    &vm_stats),
                &count);

        if (kr == KERN_SUCCESS) {
            const std::uint64_t pages =
                static_cast<std::uint64_t>(
                    vm_stats.free_count) +
                static_cast<std::uint64_t>(
                    vm_stats.inactive_count) +
                static_cast<std::uint64_t>(
                    vm_stats.speculative_count);

            info.available_bytes =
                pages * info.page_size;

            info.available_source =
                "free + inactive + speculative VM pages";
        } else {
            info.available_source =
                "unavailable";
        }

        mach_port_deallocate(
            mach_task_self(),
            host);
    }

#elif defined(__linux__)

    {
        struct sysinfo si {};

        if (sysinfo(&si) != 0) {
            throw std::runtime_error(
                "sysinfo() failed");
        }

        info.physical_bytes =
            static_cast<std::uint64_t>(
                si.totalram) *
            static_cast<std::uint64_t>(
                si.mem_unit);
    }

    info.available_bytes =
        linux_mem_available();

    info.available_source =
        info.available_bytes > 0
            ? "/proc/meminfo MemAvailable"
            : "unavailable";

#else

    const long phys_pages =
        sysconf(_SC_PHYS_PAGES);

    if (phys_pages <= 0) {
        throw std::runtime_error(
            "failed to determine physical RAM");
    }

    info.physical_bytes =
        static_cast<std::uint64_t>(
            phys_pages) *
        info.page_size;

    #ifdef _SC_AVPHYS_PAGES

    const long avail_pages =
        sysconf(_SC_AVPHYS_PAGES);

    if (avail_pages > 0) {
        info.available_bytes =
            static_cast<std::uint64_t>(
                avail_pages) *
            info.page_size;
    }

    info.available_source =
        "_SC_AVPHYS_PAGES";

    #else

    info.available_source =
        "unavailable";

    #endif

#endif

    if (info.physical_bytes == 0) {
        throw std::runtime_error(
            "physical RAM probe returned zero");
    }

    return info;
}
