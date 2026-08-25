use crate::gguf::{read_gguf, tensor_nbytes};

use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Default, Clone)]
pub struct ModelInfo {
    pub path: String,
    pub name: String,
    pub architecture: String,

    pub file_size: u64,

    pub block_count: u32,
    pub embedding_length: u32,

    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_feed_forward_length: u32,

    pub expert_tensor_count: u32,

    pub raw_bytes_per_quota: u64,
    pub lockable_bytes_per_quota: u64,

    pub sparse_moe: bool,
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn u32_or_zero(value: Option<u64>, key: &str) -> io::Result<u32> {
    match value {
        None => Ok(0),
        Some(v) => u32::try_from(v).map_err(|_| invalid(format!("GGUF value is too large: {key}"))),
    }
}

fn is_expert_weight_tensor(name: &str) -> bool {
    name.contains(".ffn_") && name.contains("_exps.weight")
}

fn align_up(value: u64, alignment: u64) -> io::Result<u64> {
    let x = value
        .checked_add(alignment - 1)
        .ok_or_else(|| invalid("alignment overflow"))?;

    Ok((x / alignment) * alignment)
}

fn align_down(value: u64, alignment: u64) -> u64 {
    (value / alignment) * alignment
}

pub fn probe_model(path_text: &str, page_size: u64) -> io::Result<ModelInfo> {
    if page_size == 0 {
        return Err(invalid("system page size is zero"));
    }

    let path = Path::new(path_text);

    let metadata = fs::metadata(path)
        .map_err(|_| invalid(format!("model file does not exist: {path_text}")))?;

    if !metadata.is_file() {
        return Err(invalid(format!(
            "model path is not a regular file: {path_text}"
        )));
    }

    let gguf = read_gguf(path)?;

    let architecture = gguf
        .string("general.architecture")
        .filter(|v| !v.is_empty())
        .ok_or_else(|| invalid("GGUF has no general.architecture"))?
        .to_owned();

    let prefix = format!("{architecture}.");

    let mut info = ModelInfo {
        path: path_text.to_owned(),
        name: gguf.string("general.name").unwrap_or("").to_owned(),
        architecture,
        file_size: metadata.len(),
        ..Default::default()
    };

    info.block_count = u32_or_zero(
        gguf.uint(&(prefix.clone() + "block_count")),
        &(prefix.clone() + "block_count"),
    )?;

    info.embedding_length = u32_or_zero(
        gguf.uint(&(prefix.clone() + "embedding_length")),
        &(prefix.clone() + "embedding_length"),
    )?;

    info.expert_count = u32_or_zero(
        gguf.uint(&(prefix.clone() + "expert_count")),
        &(prefix.clone() + "expert_count"),
    )?;

    info.expert_used_count = u32_or_zero(
        gguf.uint(&(prefix.clone() + "expert_used_count")),
        &(prefix.clone() + "expert_used_count"),
    )?;

    info.expert_feed_forward_length = u32_or_zero(
        gguf.uint(&(prefix.clone() + "expert_feed_forward_length")),
        &(prefix + "expert_feed_forward_length"),
    )?;

    if info.expert_count == 0 {
        return Ok(info);
    }

    for tensor in &gguf.tensors {
        if !is_expert_weight_tensor(&tensor.name) {
            continue;
        }

        let ne2 = tensor.ne.get(2).copied().unwrap_or(1);

        if ne2 != info.expert_count as u64 {
            return Err(invalid(format!(
                "unsupported expert tensor layout: {} has ne[2]={}, expected {}",
                tensor.name, ne2, info.expert_count
            )));
        }

        let tensor_bytes = tensor_nbytes(tensor)?;

        if tensor_bytes == 0 || tensor_bytes % info.expert_count as u64 != 0 {
            return Err(invalid(format!(
                "tensor size not divisible by expert count: {}",
                tensor.name
            )));
        }

        let slice_bytes = tensor_bytes / info.expert_count as u64;

        let tensor_file_offset = gguf
            .data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| invalid("tensor file offset overflow"))?;

        let mut lockable_total = 0u64;

        for expert in 0..info.expert_count as u64 {
            let slice_begin = tensor_file_offset
                .checked_add(
                    expert
                        .checked_mul(slice_bytes)
                        .ok_or_else(|| invalid("expert slice offset overflow"))?,
                )
                .ok_or_else(|| invalid("expert slice offset overflow"))?;

            let slice_end = slice_begin
                .checked_add(slice_bytes)
                .ok_or_else(|| invalid("expert slice size overflow"))?;

            let lock_begin = align_up(slice_begin, page_size)?;

            let lock_end = align_down(slice_end, page_size);

            if lock_end > lock_begin {
                lockable_total = lockable_total
                    .checked_add(lock_end - lock_begin)
                    .ok_or_else(|| invalid("lockable byte count overflow"))?;
            }
        }

        let avg_lockable_slice = lockable_total / info.expert_count as u64;

        info.raw_bytes_per_quota = info
            .raw_bytes_per_quota
            .checked_add(slice_bytes)
            .ok_or_else(|| invalid("raw quota byte count overflow"))?;

        info.lockable_bytes_per_quota = info
            .lockable_bytes_per_quota
            .checked_add(avg_lockable_slice)
            .ok_or_else(|| invalid("lockable quota byte count overflow"))?;

        info.expert_tensor_count += 1;
    }

    info.sparse_moe = info.expert_count > 0
        && info.expert_used_count > 0
        && info.expert_tensor_count > 0
        && info.lockable_bytes_per_quota > 0;

    Ok(info)
}
