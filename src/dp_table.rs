use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use metal::Buffer;
use crate::{LANES, NONCES_PER_THREAD, MinerMetrics};
use crate::sha_helpers::{aligned_u32_buffer, aligned_f32_buffer};

// ----------------- Distinguished Point -----------------
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DistinguishedPoint {
    pub value: [u8; 32],
    pub seed: u64,
    pub steps: u64,
}

impl DistinguishedPoint {
    /// Only one implementation
    pub fn is_distinguished(&self, dp_bits: usize) -> bool {
        let leading = (self.value[0] as u32) << 16
            | (self.value[1] as u32) << 8
            | (self.value[2] as u32);
        leading >> (24 - dp_bits.min(24)) == 0
    }
}

// ----------------- DPTable with duplicate detection -----------------
#[derive(Default)]
pub struct DPTable {
    table: HashMap<[u8; 32], DistinguishedPoint>,
}

impl DPTable {
    pub fn new() -> Self {
        Self { table: HashMap::new() }
    }

    pub fn insert_and_check(&mut self, dp: DistinguishedPoint) -> bool {
        if self.table.contains_key(&dp.value) {
         // println!("⚠️ Duplicate DP detected: {:?}", dp);
            true
        } else {
            self.table.insert(dp.value, dp.clone());
         // println!("✅ New DP inserted: {:?}", dp);
            false
        }
    }

    /// Only one update_from_digest
    pub fn update_from_digest(&mut self, slice: &[u32]) -> bool {
        assert!(slice.len() >= 8, "Digest slice must have at least 8 u32 words");
        let mut value = [0u8; 32];
        for (i, &word) in slice.iter().take(8).enumerate() {
            value[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let dp = DistinguishedPoint { value, seed: 0, steps: 0 };
        self.insert_and_check(dp)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn contains(&self, dp: &DistinguishedPoint) -> bool {
        self.table.contains_key(&dp.value)
    }
}

// ----------------- DP Table Async Update -----------------
pub async fn update_dp_table_from_gpu_async(
    dp_table: &Arc<RwLock<DPTable>>,
    posterior_slice: &[f32],
    nibble_slice: &[f32],
    fwht_slice: &[f32],
    cs_slice: &[f32],
    shannon_slice: &[f32],
    digest_slice: &[u32],
    metrics_tx: &tokio::sync::mpsc::UnboundedSender<MinerMetrics>,
) {
    let mut avg_post = vec![0.0f32; LANES];
    let mut avg_fwht = vec![0.0f32; LANES];
    let mut avg_cs = vec![0.0f32; LANES];

    for lane in 0..LANES {
        for nonce in 0..NONCES_PER_THREAD {
            let idx = lane * NONCES_PER_THREAD + nonce;
            avg_post[lane] += posterior_slice[idx];
            avg_fwht[lane] += fwht_slice[idx];
            avg_cs[lane] += cs_slice[idx];
        }
        avg_post[lane] /= NONCES_PER_THREAD as f32;
        avg_fwht[lane] /= NONCES_PER_THREAD as f32;
        avg_cs[lane] /= NONCES_PER_THREAD as f32;
    }

    {
        let mut dp = dp_table.write().await;
        for lane in 0..LANES {
            let start = lane * NONCES_PER_THREAD * 8;
            let slice = &digest_slice[start..start + 8];
            dp.update_from_digest(slice);
        }
    }

    let nibble_tree = {
        let mut tree = [[0u32; 4]; 4];
        for (i, row) in tree.iter_mut().enumerate() {
            let v = (avg_post.get(i).copied().unwrap_or(0.0) * 255.0) as u32;
            row.copy_from_slice(&[v, v.wrapping_add(1), v.wrapping_add(2), v.wrapping_add(3)]);
        }
        tree
    };

    let metrics = MinerMetrics {
        mask: 0.0,
        prune: 0.0,
        gain: 0.0,
        avg_post,
        avg_fwht,
        avg_cs,
        nibble_tree,
        hashrate_mhs: 0.0,
        total_hashes: 0,
        timestamp: Instant::now(),
        last_hashrate: 0.0,
        last_gpu_time: 0.0,
        last_cycle_time: 0.0,
        last_debug_flags: vec![],
    };
    let _ = metrics_tx.send(metrics);
}
