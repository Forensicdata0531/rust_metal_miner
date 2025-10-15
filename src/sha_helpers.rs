use sha2::{Digest, Sha256};
use byteorder::{LittleEndian, WriteBytesExt};
use bitcoin::hashes::{sha256d, Hash};
use bitcoin::hash_types::TxMerkleNode;
use bitcoin::blockdata::transaction::Transaction;
use bitcoin::consensus::deserialize;
use serde_json::Value;
use std::str::FromStr;
use hex;
use rayon::prelude::*;

// ----------------- Merkle Root Helper -----------------
pub fn merkle_root(mut nodes: Vec<TxMerkleNode>) -> sha256d::Hash {
    if nodes.is_empty() {
        return sha256d::Hash::default();
    }
    while nodes.len() > 1 {
        let mut next = vec![];
        for i in (0..nodes.len()).step_by(2) {
            let left = nodes[i];
            let right = if i + 1 < nodes.len() { nodes[i + 1] } else { nodes[i] };
            let mut data = left.as_ref().to_vec();
            data.extend_from_slice(right.as_ref());
            next.push(TxMerkleNode::from(sha256d::Hash::hash(&data)));
        }
        nodes = next;
    }
    nodes[0].into()
}

pub fn digest_meets_target(digest: &[u32; 8], bits: u32) -> bool {
    let exponent = ((bits >> 24) & 0xFF) as usize;
    let mantissa = bits & 0x007FFFFF;

    let mut target = [0u8; 32];
    if exponent >= 3 && exponent <= 32 {
        let shift = exponent - 3;
        target[shift..shift + 3].copy_from_slice(&mantissa.to_be_bytes()[1..4]);
    }

    let mut digest_be = [0u8; 32];
    for i in 0..8 {
        digest_be[i * 4..i * 4 + 4].copy_from_slice(&digest[i].to_be_bytes());
    }
    digest_be.reverse();
    digest_be <= target
}

/// Create a u32 buffer, optionally scaling for nibble-per-thread layout
pub fn aligned_u32_buffer(device: &metal::Device, count: usize, nibble_threads: bool) -> metal::Buffer {
    // If each lane has 16 threads (1 per nibble), scale the count
    let scale = if nibble_threads { 16 } else { 1 };
    device.new_buffer((count * 4 * scale) as u64, metal::MTLResourceOptions::StorageModeShared)
}

/// Create a f32 buffer, optionally scaling for nibble-per-thread layout
pub fn aligned_f32_buffer(device: &metal::Device, count: usize, nibble_threads: bool) -> metal::Buffer {
    let scale = if nibble_threads { 16 } else { 1 };
    device.new_buffer((count * 4 * scale) as u64, metal::MTLResourceOptions::StorageModeShared)
}

// ==========================================================
// Optimized GPU Midstate + Schedule Precomputation
// ==========================================================
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

// Sliding-window Shannon entropy
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

// Score a candidate merkle root by summing sliding-window entropies
pub fn score_root(root: &TxMerkleNode) -> f32 {
    let hash_bytes = root.as_ref();
    sliding_entropy(hash_bytes, 8).iter().copied().sum()
}

// Build candidate merkle roots with different extra nonces
pub fn candidate_merkle_roots(template: &serde_json::Value, num_candidates: u32) -> Vec<TxMerkleNode> {
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

        let root: TxMerkleNode = merkle_root(
            txs.iter().map(|tx| TxMerkleNode::from(sha256d::Hash::from_inner(tx.txid().into_inner()))).collect()
        ).into();

        roots.push(root);
    }
    roots
}

pub fn precompute_schedule_with_nonce(header_words: &[u32; 19], coinbase: &serde_json::Value) -> [u32; 64] {
    use sha2::{Digest, Sha256};

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

    while header_bytes.len() < 64 {
        header_bytes.push(0);
    }

    let mut schedule = [0u32; 64];
    for (i, chunk) in header_bytes.chunks(4).enumerate().take(16) {
        let mut buf = [0u8; 4];
        buf[..chunk.len()].copy_from_slice(chunk);
        schedule[i] = u32::from_le_bytes(buf);
    }

    for i in 16..64 {
        let s0 = schedule[i - 15].rotate_right(7)
            ^ schedule[i - 15].rotate_right(18)
            ^ (schedule[i - 15] >> 3);
        let s1 = schedule[i - 2].rotate_right(17)
            ^ schedule[i - 2].rotate_right(19)
            ^ (schedule[i - 2] >> 10);
        schedule[i] = schedule[i - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[i - 7])
            .wrapping_add(s1);
    }

    schedule
}

pub fn compute_midstate_with_nonce(header_words: &[u32; 19], coinbase: &serde_json::Value) -> [u32; 8] {
    let mut header_bytes = Vec::with_capacity(64 + 32);
    for &w in header_words.iter().take(16) {
        header_bytes.write_u32::<LittleEndian>(w).unwrap();
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

    let first_hash = Sha256::digest(&header_bytes);
    let second_hash = Sha256::digest(&first_hash);

    let mut midstate = [0u32; 8];
    for (i, chunk) in second_hash.chunks(4).enumerate() {
        midstate[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }

    midstate
}

// ==========================================================
// Block Header Preparation
// ==========================================================
pub fn prepare_block_header(template: &serde_json::Value) -> [u32; 19] {

    let mut header_words = [0u32; 19];
    header_words[0] = template["result"]["version"].as_u64().unwrap_or(4) as u32;

    let prev_hash_str = template["result"]["previousblockhash"].as_str().unwrap_or("00");
    let prev_hash = sha256d::Hash::from_str(prev_hash_str)
        .unwrap_or_else(|_| sha256d::Hash::hash(&[0u8; 32]));

    header_words[1..3].copy_from_slice(&[
        u32::from_le_bytes(prev_hash[0..4].try_into().unwrap()),
        u32::from_le_bytes(prev_hash[4..8].try_into().unwrap()),
    ]);

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
