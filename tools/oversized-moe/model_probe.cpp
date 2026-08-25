#include "model_probe.h"

#include "gguf.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using gguf_ptr =
    std::unique_ptr<gguf_context, decltype(&gguf_free)>;

std::optional<std::string> get_string(
        const gguf_context * ctx,
        const std::string & key) {
    const int64_t id =
        gguf_find_key(ctx, key.c_str());

    if (id < 0) {
        return std::nullopt;
    }

    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return std::nullopt;
    }

    const char * value =
        gguf_get_val_str(ctx, id);

    if (value == nullptr) {
        return std::nullopt;
    }

    return std::string(value);
}

std::optional<std::uint64_t> get_uint(
        const gguf_context * ctx,
        const std::string & key) {
    const int64_t id =
        gguf_find_key(ctx, key.c_str());

    if (id < 0) {
        return std::nullopt;
    }

    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:
            return gguf_get_val_u8(ctx, id);

        case GGUF_TYPE_UINT16:
            return gguf_get_val_u16(ctx, id);

        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(ctx, id);

        case GGUF_TYPE_UINT64:
            return gguf_get_val_u64(ctx, id);

        case GGUF_TYPE_INT8: {
            const auto v = gguf_get_val_i8(ctx, id);
            if (v >= 0) {
                return static_cast<std::uint64_t>(v);
            }
            return std::nullopt;
        }

        case GGUF_TYPE_INT16: {
            const auto v = gguf_get_val_i16(ctx, id);
            if (v >= 0) {
                return static_cast<std::uint64_t>(v);
            }
            return std::nullopt;
        }

        case GGUF_TYPE_INT32: {
            const auto v = gguf_get_val_i32(ctx, id);
            if (v >= 0) {
                return static_cast<std::uint64_t>(v);
            }
            return std::nullopt;
        }

        case GGUF_TYPE_INT64: {
            const auto v = gguf_get_val_i64(ctx, id);
            if (v >= 0) {
                return static_cast<std::uint64_t>(v);
            }
            return std::nullopt;
        }

        default:
            return std::nullopt;
    }
}

std::uint32_t get_u32_or_zero(
        const gguf_context * ctx,
        const std::string & key) {
    const auto value = get_uint(ctx, key);

    if (!value.has_value()) {
        return 0;
    }

    if (*value >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "GGUF value is too large: " + key);
    }

    return static_cast<std::uint32_t>(*value);
}

bool is_expert_weight_tensor(
        const std::string & name) {
    // Known llama.cpp MoE tensor names include:
    //
    // blk.N.ffn_gate_exps.weight
    // blk.N.ffn_up_exps.weight
    // blk.N.ffn_down_exps.weight
    //
    // This also accepts a fused *_exps.weight tensor.
    return
        name.find(".ffn_") != std::string::npos &&
        name.find("_exps.weight") != std::string::npos;
}

std::uint64_t align_up(
        std::uint64_t value,
        std::uint64_t alignment) {
    return
        ((value + alignment - 1) / alignment) *
        alignment;
}

std::uint64_t align_down(
        std::uint64_t value,
        std::uint64_t alignment) {
    return
        (value / alignment) *
        alignment;
}

} // namespace

ModelInfo probe_model(
        const std::string & path,
        std::uint64_t page_size) {
    if (page_size == 0) {
        throw std::runtime_error(
            "system page size is zero");
    }

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "model file does not exist: " + path);
    }

    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "model path is not a regular file: " + path);
    }

    ModelInfo info;

    info.path = path;
    info.file_size =
        std::filesystem::file_size(path);

    gguf_init_params params = {
        /* no_alloc = */ true,
        /* ctx      = */ nullptr,
    };

    gguf_context * raw =
        gguf_init_from_file(path.c_str(), params);

    if (raw == nullptr) {
        throw std::runtime_error(
            "failed to open GGUF: " + path);
    }

    gguf_ptr ctx(raw, &gguf_free);

    const auto architecture =
        get_string(ctx.get(), "general.architecture");

    if (!architecture.has_value() ||
        architecture->empty()) {
        throw std::runtime_error(
            "GGUF has no general.architecture");
    }

    info.architecture = *architecture;

    if (const auto name =
            get_string(ctx.get(), "general.name");
        name.has_value()) {
        info.name = *name;
    }

    const std::string prefix =
        info.architecture + ".";

    info.block_count =
        get_u32_or_zero(
            ctx.get(),
            prefix + "block_count");

    info.embedding_length =
        get_u32_or_zero(
            ctx.get(),
            prefix + "embedding_length");

    info.expert_count =
        get_u32_or_zero(
            ctx.get(),
            prefix + "expert_count");

    info.expert_used_count =
        get_u32_or_zero(
            ctx.get(),
            prefix + "expert_used_count");

    info.expert_feed_forward_length =
        get_u32_or_zero(
            ctx.get(),
            prefix + "expert_feed_forward_length");

    if (info.expert_count == 0) {
        return info;
    }

    const std::uint64_t data_offset =
        static_cast<std::uint64_t>(
            gguf_get_data_offset(ctx.get()));

    const int64_t n_tensors =
        gguf_get_n_tensors(ctx.get());

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name_c =
            gguf_get_tensor_name(ctx.get(), i);

        if (name_c == nullptr) {
            continue;
        }

        const std::string name(name_c);

        if (!is_expert_weight_tensor(name)) {
            continue;
        }

        const int64_t * ne =
            gguf_get_tensor_ne(ctx.get(), i);

        if (ne == nullptr) {
            throw std::runtime_error(
                "cannot read tensor shape: " + name);
        }

        // Qwen3-MoE / Qwen3-Next expert tensors use
        // dimension 2 as the expert dimension.
        if (ne[2] !=
            static_cast<int64_t>(info.expert_count)) {
            throw std::runtime_error(
                "unsupported expert tensor layout: " +
                name +
                " has ne[2]=" +
                std::to_string(ne[2]) +
                ", expected " +
                std::to_string(info.expert_count));
        }

        const std::uint64_t tensor_bytes =
            static_cast<std::uint64_t>(
                gguf_get_tensor_size(ctx.get(), i));

        if (tensor_bytes == 0 ||
            tensor_bytes % info.expert_count != 0) {
            throw std::runtime_error(
                "tensor size not divisible by expert count: " +
                name);
        }

        const std::uint64_t slice_bytes =
            tensor_bytes / info.expert_count;

        const std::uint64_t tensor_file_offset =
            data_offset +
            static_cast<std::uint64_t>(
                gguf_get_tensor_offset(ctx.get(), i));

        std::uint64_t lockable_total = 0;

        for (std::uint32_t expert = 0;
             expert < info.expert_count;
             ++expert) {
            const std::uint64_t slice_begin =
                tensor_file_offset +
                static_cast<std::uint64_t>(expert) *
                slice_bytes;

            const std::uint64_t slice_end =
                slice_begin + slice_bytes;

            const std::uint64_t lock_begin =
                align_up(
                    slice_begin,
                    page_size);

            const std::uint64_t lock_end =
                align_down(
                    slice_end,
                    page_size);

            if (lock_end > lock_begin) {
                lockable_total +=
                    lock_end - lock_begin;
            }
        }

        const std::uint64_t avg_lockable_slice =
            lockable_total /
            info.expert_count;

        info.raw_bytes_per_quota +=
            slice_bytes;

        info.lockable_bytes_per_quota +=
            avg_lockable_slice;

        ++info.expert_tensor_count;
    }

    info.sparse_moe =
        info.expert_count > 0 &&
        info.expert_used_count > 0 &&
        info.expert_tensor_count > 0 &&
        info.lockable_bytes_per_quota > 0;

    return info;
}
