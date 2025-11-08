// src/dp_table.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;
use metal::{Device, CommandQueue, ComputePipelineState, MTLSize, Buffer};
use crate::constants::{LANES, NONCES_PER_THREAD};
use crate::MinerMetrics;

#[derive(Clone, Debug)]
pub struct DistinguishedPoint {
    pub value: [u8; 32],
    pub seed: u64,
    pub steps: u64,
    pub probability: f32,
}

impl DistinguishedPoint {
    pub fn is_distinguished(&self, dp_bits: usize) -> bool {
        let leading = (self.value[0] as u32) << 16
            | (self.value[1] as u32) << 8
            | (self.value[2] as u32);
        leading >> (24 - dp_bits.min(24)) == 0
    }
}

#[derive(Default)]
pub struct DPTable {
    table: HashMap<[u8; 32], DistinguishedPoint>,
    max_entries: usize,
}

impl DPTable {
    pub fn new(max_entries: usize) -> Self {
        Self { table: HashMap::new(), max_entries }
    }

    pub fn insert_and_check(&mut self, dp: DistinguishedPoint) -> bool {
        if self.table.contains_key(&dp.value) {
            true
        } else {
            if self.table.len() >= self.max_entries {
                if let Some((lowest_key, _)) = self.table.iter()
                    .min_by(|a, b| a.1.probability.partial_cmp(&b.1.probability).unwrap())
                    .map(|(k, v)| (k.clone(), v.clone()))
                {
                    self.table.remove(&lowest_key);
                }
            }
            self.table.insert(dp.value, dp);
            false
        }
    }

    pub fn update_from_digest(&mut self, slice: &[u32], probability: f32) -> bool {
        assert!(slice.len() >= 8);
        let mut value = [0u8; 32];
        for (i, &word) in slice.iter().take(8).enumerate() {
            value[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let dp = DistinguishedPoint { value, seed: 0, steps: 0, probability };
        self.insert_and_check(dp)
    }

    pub fn len(&self) -> usize { self.table.len() }
}

#[derive(Clone, Debug)]
pub struct CandidateDP {
    pub value: [u8; 32],
    pub seed: u64,
    pub steps: u64,
    pub probability: f32,
}

#[derive(Clone)]
struct HeapEntry {
    probability: f32,
    lane: usize,
    dp: CandidateDP,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.probability.partial_cmp(&other.probability).unwrap().reverse()
    }
}
impl PartialOrd for HeapEntry { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
impl PartialEq for HeapEntry { fn eq(&self, other: &Self) -> bool { self.probability == other.probability } }
impl Eq for HeapEntry {}

fn candidate_probability(
    lane: usize,
    avg_post: &[f32],
    avg_fwht: &[f32],
    avg_cs: &[f32],
    shannon_slice: &[f32],
) -> f32 {
    0.4 * avg_post.get(lane).copied().unwrap_or(0.0)
        + 0.3 * avg_fwht.get(lane).copied().unwrap_or(0.0)
        + 0.2 * avg_cs.get(lane).copied().unwrap_or(0.0)
        + 0.1 * shannon_slice.get(lane).copied().unwrap_or(0.0)
}

fn gpu_submit_lane(device: &metal::Device, lane: usize, candidates: &[CandidateDP], lane_buffers: &Vec<Buffer>) {
    let buffer = &lane_buffers[lane];
    unsafe {
        let ptr = buffer.contents() as *mut u8;
        for (i, candidate) in candidates.iter().enumerate() {
            std::ptr::copy_nonoverlapping(candidate.value.as_ptr(), ptr.add(i * 32), 32);
        }
    }
}

/// ==================== Async DP Table Update with Real Submission ====================
pub async fn update_dp_table_from_gpu_async(
    dp_table: &Arc<RwLock<DPTable>>,
    posterior_slice: &[f32],
    nibble_slice: &[f32],
    fwht_slice: &[f32],
    cs_slice: &[f32],
    shannon_slice: &[f32],
    digest_slice: &[u32],
    metrics_tx: &tokio::sync::mpsc::UnboundedSender<MinerMetrics>,
    device: &Device,
    lane_buffers: &Vec<Buffer>,
) {
    const BASE_TOP_N: usize = 2;

    let mut avg_post = vec![0.0f32; LANES];
    let mut avg_fwht = vec![0.0f32; LANES];
    let mut avg_cs = vec![0.0f32; LANES];

    // Compute lane averages
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

    // Adaptive threshold
    let dp_threshold = {
        let dp = dp_table.read().await;
        0.2 + (dp.len() as f32 / 10_000.0).min(0.5)
    };

    // Lane-wise top-N queues
    let mut lane_queues: Vec<Vec<HeapEntry>> = vec![Vec::new(); LANES];

    {
        let mut dp = dp_table.write().await;
        for lane in 0..LANES {
            let entropy = shannon_slice.get(lane).copied().unwrap_or(0.0);
            if entropy < 0.15 { continue; }

            let top_n = BASE_TOP_N + ((entropy * 10.0) as usize);
            let prob = candidate_probability(lane, &avg_post, &avg_fwht, &avg_cs, shannon_slice);
            if prob < dp_threshold { continue; }

            let start = lane * NONCES_PER_THREAD * 8;
            let slice = &digest_slice[start..start + 8];
            dp.update_from_digest(slice, prob);

            let mut value = [0u8; 32];
            for (i, &word) in slice.iter().take(8).enumerate() {
                value[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }

            lane_queues[lane].push(HeapEntry {
                probability: prob,
                lane,
                dp: CandidateDP { value, seed: 0, steps: 0, probability: prob },
            });

            lane_queues[lane].sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
            if lane_queues[lane].len() > top_n {
                lane_queues[lane].truncate(top_n);
            }
        }
    }

    // Real-time lane reordering: flatten all top candidates into global queue
    let mut global_queue: Vec<HeapEntry> = lane_queues.iter().flatten().cloned().collect();
    global_queue.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

    // Batch submit top candidates per lane to GPU
    let mut lane_batch: Vec<Vec<CandidateDP>> = vec![Vec::new(); LANES];
    for entry in global_queue {
        lane_batch[entry.lane].push(entry.dp);
    }

    for lane in 0..LANES {
        if !lane_batch[lane].is_empty() {
            gpu_submit_lane(device, lane, &lane_batch[lane], lane_buffers);
        }
    }

    // Metrics update
    let mut nibble_tree = [[0u32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let post_val = (avg_post.get(i % LANES).copied().unwrap_or(0.0) * 255.0) as u32;
            let fwht_val = (avg_fwht.get(j % LANES).copied().unwrap_or(0.0) * 255.0) as u32;
            nibble_tree[i][j] = post_val.wrapping_add(fwht_val);
        }
    }

    let metrics = MinerMetrics {
        mask: 0.0,
        prune: dp_threshold,
        gain: 0.0,
        entanglement: 0.0,
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
        adaptive_factor: 0.0,
    };
    let _ = metrics_tx.send(metrics);
}
