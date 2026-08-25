#pragma once

#include <cstdint>
#include <string>

struct MemoryInfo {
    std::uint64_t physical_bytes = 0;

    // Reclaimable/available RAM estimate.
    std::uint64_t available_bytes = 0;

    std::uint64_t page_size = 0;

    std::string available_source;
};

MemoryInfo probe_memory();
