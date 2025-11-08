// src/adaptive.rs

use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;
use metal::{Device, CommandQueue, ComputePipelineState, MTLSize, Buffer};
use tokio::sync::mpsc::UnboundedSender;

// ✅ Correct imports for constants
use crate::constants::{
    LANES,
    NONCES_PER_THREAD,
    GATE_LUT_SIZE,
    CHAOS_LUT_SIZE,
    DEFAULT_GATE_LUT,
    DEFAULT_CHAOS_LUT,
};

// ✅ Single, authoritative MinerMetrics definition
#[derive(Clone, Debug)]
pub struct MinerMetrics {
    pub mask: f32,
    pub prune: f32,
    pub gain: f32,
    pub entanglement: f32,
    pub avg_post: Vec<f32>,
    pub avg_fwht: Vec<f32>,
    pub adaptive_factor: f32,
    pub avg_cs: Vec<f32>,
    pub nibble_tree: [[u32; 4]; 4],
    pub hashrate_mhs: f32,
    pub total_hashes: u64,
    pub timestamp: Instant,
    pub last_hashrate: f32,
    pub last_gpu_time: f64,
    pub last_cycle_time: f64,
    pub last_debug_flags: Vec<u32>,
}

impl Default for MinerMetrics {
    fn default() -> Self {
        MinerMetrics {
            mask: 0.0,
            prune: 0.0,
            gain: 0.0,
            entanglement: 0.0,
            avg_post: vec![],
            avg_fwht: vec![],
            adaptive_factor: 0.0,
            avg_cs: vec![],
            nibble_tree: [[0; 4]; 4],
            hashrate_mhs: 0.0,
            total_hashes: 0,
            timestamp: Instant::now(),
            last_hashrate: 0.0,
            last_gpu_time: 0.0,
            last_cycle_time: 0.0,
            last_debug_flags: vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub enum UiMessage {
    Status(String),
    Metrics(MinerMetrics),
}

// ----------------- Adaptive Feedback Loop -----------------
pub fn spawn_adaptive_feedback(
    lane_posteriors_bufs: Vec<Arc<RwLock<Buffer>>>,
    cs_bufs: Vec<Arc<RwLock<Buffer>>>,
    fwht_bufs: Vec<Arc<RwLock<Buffer>>>,
    adaptive_params_buf: Arc<RwLock<Buffer>>,
    metrics_tx: UnboundedSender<MinerMetrics>,
) {
    let buf = adaptive_params_buf.clone();
    tokio::task::spawn_local(async move {
        let mut current_params = [0.08_f32, 0.25_f32, 0.5_f32];
        let base_alpha = 0.2_f32;
        let mut last_update = Instant::now();

        loop {
            if lane_posteriors_bufs.is_empty() || cs_bufs.is_empty() || fwht_bufs.is_empty() {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }

            let idx = (Instant::now().elapsed().as_secs() as usize) % lane_posteriors_bufs.len();

            let post_lock = lane_posteriors_bufs[idx].read().await;
            let cs_lock = cs_bufs[idx].read().await;
            let fwht_lock = fwht_bufs[idx].read().await;

            let mean_post;
            let mean_cs;
            let mean_fwht;
            let entanglement_coeff;

            unsafe {
                let posterior = std::slice::from_raw_parts(post_lock.contents() as *const f32, 4096);
                let cs = std::slice::from_raw_parts(cs_lock.contents() as *const f32, 4096);
                let fwht = std::slice::from_raw_parts(fwht_lock.contents() as *const f32, 4096);

                mean_post = posterior.iter().copied().sum::<f32>() / posterior.len() as f32;
                mean_cs = cs.iter().copied().sum::<f32>() / cs.len() as f32;
                mean_fwht = fwht.iter().copied().sum::<f32>() / fwht.len() as f32;

                let hamming_corr = (mean_post - mean_cs).abs();
                let coherence = mean_fwht * (1.0 - hamming_corr);
                entanglement_coeff = 1.0 - (-coherence).exp();
            }

            current_params[0] = (1.0 - base_alpha) * current_params[0] + base_alpha * mean_post;
            current_params[1] = (1.0 - base_alpha) * current_params[1] + base_alpha * (mean_cs + mean_fwht * 0.1);
            current_params[2] = (1.0 - base_alpha) * current_params[2] + base_alpha * (mean_fwht + mean_cs * 0.1);
            for p in current_params.iter_mut() { *p = p.clamp(0.001, 1.0); }

            let params_lock = buf.write().await;
            unsafe {
                params_lock.contents().copy_from(
                    current_params.as_ptr() as *const std::ffi::c_void,
                    std::mem::size_of::<f32>() * current_params.len()
                );
            }

            if last_update.elapsed().as_secs_f32() > 1.0 {
                let metrics = MinerMetrics {
                    mask: current_params[0],
                    prune: current_params[1],
                    gain: current_params[2],
                    entanglement: entanglement_coeff,
                    avg_post: vec![mean_post],
                    avg_fwht: vec![mean_fwht],
                    avg_cs: vec![mean_cs],
                    nibble_tree: [[0; 4]; 4],
                    hashrate_mhs: 0.0,
                    total_hashes: 0,
                    timestamp: Instant::now(),
                    last_hashrate: 0.0,
                    last_gpu_time: 0.0,
                    last_cycle_time: 0.0,
                    last_debug_flags: vec![],
                    adaptive_factor: entanglement_coeff,
                };
                let _ = metrics_tx.send(metrics);
                last_update = Instant::now();
            }

            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    });
}

// ----------------- GPU Pruning Pass -----------------
pub fn dispatch_pruning_pass(
    queue: &CommandQueue,
    prune_pipeline: &ComputePipelineState,
    fwht_buf: &Buffer,
    cs_buf: &Buffer,
    nibble_buf: &Buffer,
    digest_buf: &Buffer,
    posterior_buf: &Buffer,
    nibble_probs_buf: &Buffer,
    adaptive_params_buf: &Buffer,
    metrics_tx: &tokio::sync::mpsc::UnboundedSender<MinerMetrics>,
) {
    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(prune_pipeline);

    encoder.set_buffer(0, Some(fwht_buf), 0);
    encoder.set_buffer(1, Some(cs_buf), 0);
    encoder.set_buffer(2, Some(nibble_buf), 0);

    let threads_per_group = prune_pipeline.thread_execution_width() as u64 * 256;
    let total_threads = digest_buf.length() as u64 / 8;
    let threadgroup_count = MTLSize {
        width: (total_threads + threads_per_group - 1) / threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(
        threadgroup_count,
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    unsafe {
        let digest_slice = std::slice::from_raw_parts(digest_buf.contents() as *const u32, LANES * NONCES_PER_THREAD * 8);
        let posterior_slice = std::slice::from_raw_parts(posterior_buf.contents() as *const f32, LANES * NONCES_PER_THREAD);
        let fwht_slice = std::slice::from_raw_parts(fwht_buf.contents() as *const f32, LANES * NONCES_PER_THREAD);
        let cs_slice = std::slice::from_raw_parts(cs_buf.contents() as *const f32, LANES * NONCES_PER_THREAD);
        let nibble_slice = std::slice::from_raw_parts(nibble_probs_buf.contents() as *const f32, LANES * NONCES_PER_THREAD * 16);
        let adaptive_slice = std::slice::from_raw_parts_mut(adaptive_params_buf.contents() as *mut f32, 4);

        let avg_post = posterior_slice.iter().copied().sum::<f32>() / posterior_slice.len() as f32;
        let avg_fwht = fwht_slice.iter().copied().sum::<f32>() / fwht_slice.len() as f32;
        let avg_cs = cs_slice.iter().copied().sum::<f32>() / cs_slice.len() as f32;
        let gpu_feedback = adaptive_slice[3];

        for i in 0..3 {
            adaptive_slice[i] = (adaptive_slice[i] * 0.95 + gpu_feedback * 0.05).clamp(0.01, 1.0);
        }

        let hamming_corr = (avg_post - avg_cs).abs();
        let coherence = avg_fwht * (1.0 - hamming_corr);
        let entanglement_coeff = 1.0 - (-coherence).exp();

        let metrics = MinerMetrics {
            mask: adaptive_slice[0],
            prune: adaptive_slice[1],
            gain: adaptive_slice[2],
            entanglement: entanglement_coeff,
            avg_post: vec![avg_post],
            avg_fwht: vec![avg_fwht],
            avg_cs: vec![avg_cs],
            nibble_tree: [[0; 4]; 4],
            hashrate_mhs: 0.0,
            total_hashes: 0,
            timestamp: Instant::now(),
            last_hashrate: 0.0,
            last_gpu_time: 0.0,
            last_cycle_time: 0.0,
            last_debug_flags: vec![],
            adaptive_factor: entanglement_coeff,
        };
        let _ = metrics_tx.send(metrics);

        let avg_nibble = nibble_slice.iter().copied().sum::<f32>() / nibble_slice.len() as f32;
        println!("🌿 GPU pruning pass complete — avg nibble weight = {:.6}", avg_nibble);
    }
}
