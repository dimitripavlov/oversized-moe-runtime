#pragma once

#include <cstdint>
#include <string>

struct ModelInfo {
    std::string path;
    std::string name;
    std::string architecture;

    std::uint64_t file_size = 0;

    std::uint32_t block_count = 0;
    std::uint32_t embedding_length = 0;

    std::uint32_t expert_count = 0;
    std::uint32_t expert_used_count = 0;
    std::uint32_t expert_feed_forward_length = 0;

    std::uint32_t expert_tensor_count = 0;

    // Approximate physical bytes needed to raise
    // ExpertResidency capacity by one expert/tensor.
    std::uint64_t raw_bytes_per_quota = 0;

    // Same quantity using the actual residency rule:
    // only complete VM pages fully contained in an expert
    // slice are counted.
    std::uint64_t lockable_bytes_per_quota = 0;

    bool sparse_moe = false;
};

ModelInfo probe_model(
    const std::string & path,
    std::uint64_t page_size);
