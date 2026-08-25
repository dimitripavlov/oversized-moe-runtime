use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ne: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    strings: HashMap<String, String>,
    uints: HashMap<String, u64>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
}

impl GgufFile {
    pub fn string(&self, key: &str) -> Option<&str> {
        self.strings.get(key).map(String::as_str)
    }

    pub fn uint(&self, key: &str) -> Option<u64> {
        self.uints.get(key).copied()
    }
}

struct Reader {
    file: BufReader<File>,
}

impl Reader {
    fn open(path: &Path) -> io::Result<Self> {
        Ok(Self {
            file: BufReader::with_capacity(256 * 1024, File::open(path)?),
        })
    }

    fn pos(&mut self) -> io::Result<u64> {
        self.file.stream_position()
    }

    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }

    fn i8(&mut self) -> io::Result<i8> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }

    fn i16(&mut self) -> io::Result<i16> {
        Ok(i16::from_le_bytes(self.bytes()?))
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.bytes()?))
    }

    fn skip(&mut self, mut n: u64) -> io::Result<()> {
        while n > 0 {
            let available = self.file.fill_buf()?;

            if available.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected EOF while skipping GGUF value",
                ));
            }

            let take = n.min(available.len() as u64) as usize;

            self.file.consume(take);
            n -= take as u64;
        }

        Ok(())
    }

    fn string(&mut self) -> io::Result<String> {
        let len = self.u64()?;

        let len = usize::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "GGUF string is too large"))?;

        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;

        String::from_utf8(buf)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "GGUF string is not UTF-8"))
    }

    fn skip_string(&mut self) -> io::Result<()> {
        let len = self.u64()?;
        self.skip(len)
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn align_up(value: u64, alignment: u64) -> io::Result<u64> {
    if alignment == 0 {
        return Err(invalid("GGUF alignment is zero"));
    }

    let add = alignment - 1;

    value
        .checked_add(add)
        .map(|v| (v / alignment) * alignment)
        .ok_or_else(|| invalid("GGUF alignment overflow"))
}

fn fixed_value_size(typ: u32) -> Option<u64> {
    match typ {
        0 | 1 | 7 => Some(1),    // uint8, int8, bool
        2 | 3 => Some(2),        // uint16, int16
        4 | 5 | 6 => Some(4),    // uint32, int32, float32
        10 | 11 | 12 => Some(8), // uint64, int64, float64
        _ => None,
    }
}

fn skip_array(reader: &mut Reader, element_type: u32, count: u64) -> io::Result<()> {
    if element_type == 8 {
        for _ in 0..count {
            reader.skip_string()?;
        }
        return Ok(());
    }

    if element_type == 9 {
        return Err(invalid("nested GGUF arrays are unsupported"));
    }

    let width = fixed_value_size(element_type)
        .ok_or_else(|| invalid(format!("unknown GGUF array element type {element_type}")))?;

    let bytes = width
        .checked_mul(count)
        .ok_or_else(|| invalid("GGUF array size overflow"))?;

    reader.skip(bytes)
}

pub fn read_gguf(path: &Path) -> io::Result<GgufFile> {
    let mut r = Reader::open(path)?;

    if &r.bytes::<4>()? != GGUF_MAGIC {
        return Err(invalid("not a GGUF file"));
    }

    let version = r.u32()?;
    if version < 2 || version > 3 {
        return Err(invalid(format!("unsupported GGUF version {version}")));
    }

    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut strings = HashMap::new();
    let mut uints = HashMap::new();

    for _ in 0..kv_count {
        let key = r.string()?;
        let typ = r.u32()?;

        match typ {
            0 => {
                uints.insert(key, r.u8()? as u64);
            }
            1 => {
                let v = r.i8()?;
                if v >= 0 {
                    uints.insert(key, v as u64);
                }
            }
            2 => {
                uints.insert(key, r.u16()? as u64);
            }
            3 => {
                let v = r.i16()?;
                if v >= 0 {
                    uints.insert(key, v as u64);
                }
            }
            4 => {
                uints.insert(key, r.u32()? as u64);
            }
            5 => {
                let v = r.i32()?;
                if v >= 0 {
                    uints.insert(key, v as u64);
                }
            }
            6 => {
                r.skip(4)?;
            }
            7 => {
                r.skip(1)?;
            }
            8 => {
                strings.insert(key, r.string()?);
            }
            9 => {
                let element_type = r.u32()?;
                let count = r.u64()?;
                skip_array(&mut r, element_type, count)?;
            }
            10 => {
                uints.insert(key, r.u64()?);
            }
            11 => {
                let v = r.i64()?;
                if v >= 0 {
                    uints.insert(key, v as u64);
                }
            }
            12 => {
                r.skip(8)?;
            }
            other => {
                return Err(invalid(format!("unknown GGUF metadata type {other}")));
            }
        }
    }

    let mut tensors = Vec::with_capacity(
        usize::try_from(tensor_count).map_err(|_| invalid("too many GGUF tensors"))?,
    );

    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()?;

        if n_dims == 0 || n_dims > 8 {
            return Err(invalid(format!(
                "unsupported tensor dimension count {n_dims}: {name}"
            )));
        }

        let mut ne = Vec::with_capacity(n_dims as usize);

        for _ in 0..n_dims {
            ne.push(r.u64()?);
        }

        let ggml_type = r.u32()?;
        let offset = r.u64()?;

        tensors.push(TensorInfo {
            name,
            ne,
            ggml_type,
            offset,
        });
    }

    let alignment = uints
        .get("general.alignment")
        .copied()
        .unwrap_or(DEFAULT_ALIGNMENT);

    let data_offset = align_up(r.pos()?, alignment)?;

    Ok(GgufFile {
        strings,
        uints,
        tensors,
        data_offset,
    })
}

pub fn tensor_nbytes(tensor: &TensorInfo) -> io::Result<u64> {
    let (block_size, type_size): (u64, u64) = match tensor.ggml_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 40),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        30 => (1, 2),     // BF16
        other => {
            return Err(invalid(format!(
                "unsupported GGML tensor type {other}: {}",
                tensor.name
            )));
        }
    };

    let elements = tensor
        .ne
        .iter()
        .try_fold(1u64, |acc, &v| acc.checked_mul(v))
        .ok_or_else(|| invalid(format!("tensor element count overflow: {}", tensor.name)))?;

    if elements % block_size != 0 {
        return Err(invalid(format!(
            "tensor element count is not divisible by block size: {}",
            tensor.name
        )));
    }

    (elements / block_size)
        .checked_mul(type_size)
        .ok_or_else(|| invalid(format!("tensor byte size overflow: {}", tensor.name)))
}
