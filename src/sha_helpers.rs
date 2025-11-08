use sha2::{Digest, Sha256};
use byteorder::{LittleEndian, WriteBytesExt};
use bitcoin::hashes::{sha256d, Hash};
use bitcoin::hash_types::TxMerkleNode;
use bitcoin::blockdata::transaction::Transaction;
use bitcoin::blockdata::block::BlockHeader;
use bitcoin::consensus::{deserialize, encode::serialize};
use serde_json::Value;
use std::str::FromStr;
use std::convert::TryInto;
use std::io::Write;
use std::time::Instant;
use rayon::prelude::*;
use hex;

// ----------------- Double SHA256 -----------------
pub fn double_sha256_bytes(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    let second = Sha256::digest(&first);
    let mut out = [0u8; 32];
    out.copy_from_slice(&second);
    out
}

// ----------------- Target Conversion -----------------
pub fn target_from_bits(bits: u32) -> [u8; 32] {
    let mut target = [0u8; 32];
    let exponent = ((bits >> 24) & 0xff) as i32;
    let mantissa = bits & 0x007fffff;

    if exponent <= 3 {
        let value = (mantissa >> (8 * (3 - exponent))) as u32;
        let bytes = value.to_be_bytes();
        target[28..32].copy_from_slice(&bytes);
    } else if exponent <= 32 {
        let offset = (exponent - 3) as usize;
        let mant_bytes = mantissa.to_be_bytes();
        if offset + 3 <= 32 {
            target[offset..offset + 3].copy_from_slice(&mant_bytes[1..4]);
        }
    }
    target
}

// ----------------- Hash vs Target -----------------
pub fn hash_le_target(hash_be: &[u8; 32], target_be: &[u8; 32]) -> bool {
    hash_be <= target_be
}

// ----------------- Serialize Block Header -----------------
pub fn serialize_block_header_bytes(header: &BlockHeader) -> [u8; 80] {
    let bytes = serialize(header);
    let mut out = [0u8; 80];
    out[..bytes.len().min(80)].copy_from_slice(&bytes[..bytes.len().min(80)]);
    out
}

// ----------------- Digest Meets Target -----------------
pub fn digest_meets_target(digest_words: &[u32; 8], bits: u32) -> bool {
    let target = target_from_bits(bits);
    let mut hash_bytes = [0u8; 32];
    for (i, word) in digest_words.iter().enumerate() {
        hash_bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    hash_le_target(&hash_bytes, &target)
}

// ----------------- GPU Buffer Helpers -----------------
pub fn aligned_u32_buffer(device: &metal::Device, count: usize, nibble_threads: bool) -> metal::Buffer {
    let scale = if nibble_threads { 16 } else { 1 };
    device.new_buffer((count * 4 * scale) as u64, metal::MTLResourceOptions::StorageModeShared)
}

pub fn aligned_f32_buffer(device: &metal::Device, count: usize, nibble_threads: bool) -> metal::Buffer {
    let scale = if nibble_threads { 16 } else { 1 };
    device.new_buffer((count * 4 * scale) as u64, metal::MTLResourceOptions::StorageModeShared)
}

/// Create a 16-bit aligned Metal buffer for `ushort` data.
pub fn aligned_ushort_buffer(device: &metal::Device, count: usize, _nibble_threads: bool) -> metal::Buffer {
    device.new_buffer(
        (count * std::mem::size_of::<u16>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

// ----------------- SHA256 Midstate & Schedule -----------------
pub fn compute_midstate(header_words: &[u32; 19]) -> [u32; 8] {
    let mut first_block = vec![];
    for &w in header_words.iter().take(16) {
        first_block.write_u32::<LittleEndian>(w).unwrap();
    }
    let first_hash = Sha256::digest(&first_block);
    let second_hash = Sha256::digest(&first_hash);
    let mut midstate = [0u32; 8];
    for (i, chunk) in second_hash.chunks(4).enumerate() {
        midstate[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    midstate
}

pub fn precompute_schedule(header_words: &[u32; 19]) -> [u32; 64] {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = header_words[i].to_le();
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    w
}

// ----------------- Sliding Entropy -----------------
pub fn sliding_entropy(data: &[u8], window_bits: usize) -> Vec<f32> {
    let window_bytes = (window_bits + 7) / 8;
    let mut entropies = Vec::with_capacity(data.len().saturating_sub(window_bytes) + 1);
    for chunk in data.windows(window_bytes) {
        let mut counts = [0usize; 256];
        for &b in chunk { counts[b as usize] += 1; }
        let mut entropy = 0.0;
        let len = chunk.len() as f32;
        for &c in counts.iter() {
            if c == 0 { continue; }
            let p = c as f32 / len;
            entropy -= p * p.log2();
        }
        entropies.push(entropy);
    }
    entropies
}

pub fn score_root(root: &TxMerkleNode) -> f32 {
    sliding_entropy(root.as_ref(), 8).iter().copied().sum()
}

// ----------------- Merkle Root Computation -----------------
pub fn merkle_root(tx_hashes: Vec<sha256d::Hash>) -> TxMerkleNode {
    if tx_hashes.is_empty() {
        return TxMerkleNode::from_inner([0u8; 32]);
    }

    let mut hashes = tx_hashes;
    while hashes.len() > 1 {
        let mut new_hashes = vec![];
        for i in (0..hashes.len()).step_by(2) {
            let left = hashes[i];
            let right = if i + 1 < hashes.len() { hashes[i + 1] } else { hashes[i] };
            let concat = [left.as_ref(), right.as_ref()].concat();
            new_hashes.push(sha256d::Hash::hash(&concat));
        }
        hashes = new_hashes;
    }
    TxMerkleNode::from_inner(hashes[0].into_inner())
}

// ----------------- Candidate Merkle Roots -----------------
pub fn candidate_merkle_roots(template: &Value, num_candidates: u32) -> Vec<TxMerkleNode> {
    let mut roots = Vec::new();
    for extra_nonce in 0..num_candidates {
        let mut coinbase = crate::coinbase::build_coinbase_from_template_with_height(template);
        crate::coinbase::patch_insert_nonce_into_coinbase(&mut coinbase, extra_nonce);

        let mut txs: Vec<Transaction> = vec![deserialize(
            &hex::decode(
                coinbase["result"]["transactions"][0]["data"].as_str().unwrap_or("")
            ).unwrap()
        ).unwrap()];

        if let Some(tx_arr) = template["result"]["transactions"].as_array() {
            for tx_json in tx_arr {
                if let Ok(raw_tx) = hex::decode(tx_json["data"].as_str().unwrap_or("")) {
                    txs.push(deserialize(&raw_tx).unwrap());
                }
            }
        }

        let tx_hashes: Vec<sha256d::Hash> = txs
            .iter()
            .map(|tx| sha256d::Hash::from_slice(&tx.txid().as_ref()).unwrap())
            .collect();
        let root = merkle_root(tx_hashes);
        roots.push(root);
    }
    roots
}

// ----------------- Precompute Schedule with Coinbase Nonce -----------------
pub fn precompute_schedule_with_nonce(header_words: &[u32; 19], coinbase: &Value) -> [u32; 64] {
    let mut header_bytes = Vec::with_capacity(76);
    for w in header_words.iter().take(16) {
        header_bytes.extend_from_slice(&w.to_le_bytes());
    }

    if let Some(tx) = coinbase["result"]["transactions"].get(0) {
        if let Some(vin) = tx["vin"].get(0) {
            if let Some(script_hex) = vin["coinbase"].as_str() {
                if let Ok(script_bytes) = hex::decode(script_hex) {
                    header_bytes.extend_from_slice(&script_bytes);
                }
            }
        }
    }

    while header_bytes.len() < 64 { header_bytes.push(0); }

    let mut schedule = [0u32; 64];
    for (i, chunk) in header_bytes.chunks(4).enumerate().take(16) {
        let mut buf = [0u8; 4];
        buf[..chunk.len()].copy_from_slice(chunk);
        schedule[i] = u32::from_le_bytes(buf);
    }

    for i in 16..64 {
        let s0 = schedule[i - 15].rotate_right(7) ^ schedule[i - 15].rotate_right(18) ^ (schedule[i - 15] >> 3);
        let s1 = schedule[i - 2].rotate_right(17) ^ schedule[i - 2].rotate_right(19) ^ (schedule[i - 2] >> 10);
        schedule[i] = schedule[i - 16].wrapping_add(s0).wrapping_add(schedule[i - 7]).wrapping_add(s1);
    }

    schedule
}

// ----------------- Midstate with Coinbase Nonce -----------------
pub fn compute_midstate_with_nonce(header_words: &[u32; 19], coinbase: &Value) -> [u32; 8] {
    let mut header_bytes = Vec::with_capacity(64 + 32);
    for &w in header_words.iter().take(16) { header_bytes.write_u32::<LittleEndian>(w).unwrap(); }

    if let Some(tx) = coinbase["result"]["transactions"].get(0) {
        if let Some(vin) = tx["vin"].get(0) {
            if let Some(script_hex) = vin["coinbase"].as_str() {
                if let Ok(script_bytes) = hex::decode(script_hex) {
                    header_bytes.extend_from_slice(&script_bytes);
                }
            }
        }
    }

    let first_hash = Sha256::digest(&header_bytes);
    let second_hash = Sha256::digest(&first_hash);

    let mut midstate = [0u32; 8];
    for (i, chunk) in second_hash.chunks(4).enumerate() {
        midstate[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    midstate
}

// ----------------- Block Header Preparation -----------------
pub fn prepare_block_header(template: &Value) -> [u32; 19] {
    let mut header_words = [0u32; 19];
    header_words[0] = template["result"]["version"].as_u64().unwrap_or(4) as u32;

    let prev_hash_str = template["result"]["previousblockhash"].as_str().unwrap_or("00");
    let prev_hash = sha256d::Hash::from_str(prev_hash_str).unwrap_or_else(|_| sha256d::Hash::hash(&[0u8; 32]));

    for i in 0..8 {
        if i < 2 {
            header_words[1 + i] = u32::from_le_bytes(prev_hash[i*4..i*4+4].try_into().unwrap());
        } else {
            header_words[3 + i - 2] = 0;
        }
    }

    if let Some(merkle) = template["result"]["merkleroot"].as_str() {
        let merkle_bytes = hex::decode(merkle).unwrap_or_default();
        for (i, chunk) in merkle_bytes.chunks(4).enumerate().take(8) {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);
            header_words[3 + i] = u32::from_le_bytes(buf);
        }
    }

    header_words[18] = template["result"]["curtime"].as_u64().unwrap_or(0) as u32 + 31;
    header_words
}
