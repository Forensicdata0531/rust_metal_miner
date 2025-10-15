use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tokio::task;
use tokio::sync::RwLock;
use reqwest::Client;
use metal::*;
use itertools::izip;
use std::sync::Mutex;
use rayon::prelude::*;
use std::f32;
use std::str::FromStr;

use sha2::{Digest, Sha256};
use byteorder::{LittleEndian, WriteBytesExt};
use block::ConcreteBlock;
mod dp_table;
use dp_table::*;
mod adaptive;
use adaptive::*;
mod constants;
use constants::*;
mod ui;
use ui::*;
mod coinbase;
use coinbase::*;
mod rpc;
use rpc::*;
mod sha_helpers;
use sha_helpers::*;
mod mitm;
use mitm::*;
use bitcoin::consensus::deserialize;
use bitcoin_hashes::sha256d;
use crate::constants::{LANES, NONCES_PER_THREAD};

// ----------------- SHA / Crypto -----------------
use generic_array::GenericArray;
use sha2::compress256;

// ----------------- Adaptive Masks -----------------
struct AdaptiveMasks {
    pub mask16: u32,
}
impl AdaptiveMasks {
    fn new() -> Self {
        Self { mask16: 0xFFFF_FFFF }
    }
    fn update_from_telemetry(&mut self, weights: &[f32], lanes: usize) {
        let avg = weights.iter().copied().sum::<f32>() / lanes as f32;
        if avg > 0.05 {
            self.mask16 = (self.mask16 >> 1) | 1;
        } else if avg < 0.01 {
            self.mask16 = (self.mask16 << 1) | 1;
        }
    }
}

// ----------------- Helpers -----------------
fn read_cookie(datadir: &str) -> String {
    let mut path = PathBuf::from(datadir);
    path.push(".cookie");
    fs::read_to_string(path)
        .unwrap_or_default()
        .trim()
        .to_string()
}

// ----------------- Main Async (Nibble-per-thread version) -----------------
async fn async_main() {
    use std::sync::atomic::Ordering;
    use std::time::Instant;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use metal::*;
    use crate::constants::*;

    // ---------------- Metrics + UI Channels ----------------
    let metrics = Arc::new(RwLock::new(MinerMetrics::default()));
    let (metrics_tx, mut metrics_rx) = tokio::sync::mpsc::unbounded_channel::<MinerMetrics>();
    let (ui_tx, ui_rx) = tokio::sync::mpsc::unbounded_channel::<UiMessage>();

    // ---------------- TUI Task ----------------
    let ui_metrics = metrics.clone();
    tokio::spawn(async move {
        if let Err(e) = run_ui(ui_metrics, ui_rx).await {
            eprintln!("UI exited: {:?}", e);
        }
    });

    // ---------------- Metrics Receiver ----------------
    let metrics_clone = metrics.clone();
    let ui_tx_clone = ui_tx.clone();
    tokio::spawn(async move {
        while let Some(updated) = metrics_rx.recv().await {
            let mut m = metrics_clone.write().await;
            *m = updated.clone();
            let _ = ui_tx_clone.send(UiMessage::Metrics(updated));
        }
    });

    // ---------------- Load RPC Cookie ----------------
    let cookie_path = PathBuf::from(format!(
        "{}/Library/Application Support/Bitcoin-Pruned/.cookie",
        std::env::var("HOME").unwrap()
    ));
    let cookie = std::fs::read_to_string(&cookie_path)
        .expect("⚠️ Bitcoin cookie not found. Make sure bitcoind is running.");
    let mut parts = cookie.trim().splitn(2, ':');
    let rpc_user = parts.next().unwrap_or("__cookie__").to_string();
    let rpc_pass = parts.next().unwrap_or("").to_string();

    // ---------------- Metal Device & GPU Buffers ----------------
    let device = Device::system_default().expect("❌ No Metal device found");
    let command_queue = device.new_command_queue();

    let metallib_path = format!("{}/shaders/kernels.metallib", env!("CARGO_MANIFEST_DIR"));
    let library = device.new_library_with_file(&metallib_path)
        .expect("❌ Failed to load Metal library");

    let fused_fn = library.get_function("fused_sha256d_fwht_cs", None)
        .expect("❌ Kernel not found");
    let fused_pipeline = device.new_compute_pipeline_state_with_function(&fused_fn)
        .expect("❌ Failed to create compute pipeline");

    // ---------------- GPU Buffers ----------------
    const NONCES_PER_NIBBLE: usize = 32; // 32 nonces per nibble
    let total_threads = LANES * NIBBLES;

    let digest_buf = Arc::new(aligned_u32_buffer(&device, total_threads * NONCES_PER_NIBBLE * 8, false));
    let posterior_buf = Arc::new(aligned_f32_buffer(&device, total_threads * NONCES_PER_NIBBLE, false));
    let fwht_buf = Arc::new(aligned_f32_buffer(&device, total_threads * NONCES_PER_NIBBLE, false));
    let cs_buf = Arc::new(aligned_f32_buffer(&device, total_threads * NONCES_PER_NIBBLE, false));
    let nibble_probs_buf = Arc::new(aligned_f32_buffer(&device, total_threads * NONCES_PER_NIBBLE * NIBBLES, true));
    let midstate_buf = Arc::new(aligned_u32_buffer(&device, LANES * 8, false));
    let schedule_buf = Arc::new(aligned_u32_buffer(&device, LANES * 16, false));

    let start_nonce_buf = Arc::new(aligned_u32_buffer(&device, LANES, false));
    let adaptive_params_buf = Arc::new(aligned_f32_buffer(&device, 4, false));
    let debug_flags_buf = Arc::new(aligned_u32_buffer(&device, total_threads * NONCES_PER_NIBBLE, false));
    let shannon_entropy_buf = Arc::new(aligned_f32_buffer(&device, total_threads * NONCES_PER_NIBBLE, false));
    let mitm_states_buf = Arc::new(aligned_u32_buffer(&device, total_threads * NONCES_PER_NIBBLE * MITM_STATE_U32_WORDS, false));
    let adaptive_feedback_buf = Arc::new(aligned_f32_buffer(&device, total_threads * NONCES_PER_NIBBLE, false));
    let submit_mask_buf = Arc::new(aligned_u32_buffer(&device, total_threads * NONCES_PER_NIBBLE, false));

    let digest_out_len_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);
    unsafe { *(digest_out_len_buf.contents() as *mut u32) = (total_threads * NONCES_PER_NIBBLE) as u32; }

    let nibble_probs_len_buf = device.new_buffer(std::mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);
    unsafe { *(nibble_probs_len_buf.contents() as *mut u32) = (total_threads * NONCES_PER_NIBBLE * NIBBLES) as u32; }

    let nonce_base = Arc::new(AtomicU32::new(0));

    // ---------------- Initialize adaptive params ----------------
    unsafe {
        let params_ptr = adaptive_params_buf.contents() as *mut f32;
        params_ptr.copy_from_nonoverlapping([0.999999, 0.95, 0.2, 0.0].as_ptr(), 4);
    }

    spawn_adaptive_feedback(
        vec![Arc::new(RwLock::new((*posterior_buf).clone()))],
        vec![Arc::new(RwLock::new((*cs_buf).clone()))],
        vec![Arc::new(RwLock::new((*fwht_buf).clone()))],
        Arc::new(RwLock::new((*adaptive_params_buf).clone())),
        metrics_tx.clone(),
    );

    // ---------------- Main Mining Loop ----------------
    loop {
        let start_time = Instant::now();

        // Fetch block template
        let template = match fetch_block_template(&Client::new(), "http://127.0.0.1:8332", &rpc_user, &rpc_pass).await {
            Some(t) => t,
            None => {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }
        };

        // Compute midstate and schedule
        let header_words = prepare_block_header(&template);
        let midstate = compute_midstate_with_nonce(&header_words, &template);
        let schedule = precompute_schedule_with_nonce(&header_words, &template);

        // Upload buffers & initialize lane-dependent memory
        unsafe {
            std::ptr::copy_nonoverlapping(midstate.as_ptr(), midstate_buf.contents() as *mut u32, midstate.len());
            std::ptr::copy_nonoverlapping(schedule.as_ptr(), schedule_buf.contents() as *mut u32, schedule.len());

            let start_ptr = start_nonce_buf.contents() as *mut u32;
            let posterior_ptr = posterior_buf.contents() as *mut f32;
            let fwht_ptr = fwht_buf.contents() as *mut f32;
            let cs_ptr = cs_buf.contents() as *mut f32;
            let shannon_ptr = shannon_entropy_buf.contents() as *mut f32;
            let mitm_ptr = mitm_states_buf.contents() as *mut u32;
            let feedback_ptr = adaptive_feedback_buf.contents() as *mut f32;
            let debug_ptr = debug_flags_buf.contents() as *mut u32;
            let submit_ptr = submit_mask_buf.contents() as *mut u32;

            for lane in 0..LANES {
                *start_ptr.add(lane) = nonce_base.fetch_add(NONCES_PER_NIBBLE as u32 * NIBBLES as u32, Ordering::Relaxed);

                for nibble in 0..NIBBLES {
                    let base = lane * NIBBLES * NONCES_PER_NIBBLE + nibble * NONCES_PER_NIBBLE;
                    for i in 0..NONCES_PER_NIBBLE {
                        let idx = base + i;
                        *posterior_ptr.add(idx) = 1.0;
                        *fwht_ptr.add(idx) = 0.0;
                        *cs_ptr.add(idx) = 0.0;
                        *shannon_ptr.add(idx) = 0.0;
                        *feedback_ptr.add(idx) = 0.0;
                        *debug_ptr.add(idx) = 0;
                        *submit_ptr.add(idx) = 0;

                        let mitm_base = idx * MITM_STATE_U32_WORDS;
                        for w in 0..MITM_STATE_U32_WORDS {
                            *mitm_ptr.add(mitm_base + w) = 0;
                        }
                    }
                }
            }
        }

        // ---------------- GPU Dispatch ----------------
        let cmd_buf = command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&fused_pipeline);

        encoder.set_buffer(0, Some(&*midstate_buf), 0);
        encoder.set_buffer(1, Some(&*schedule_buf), 0);
        encoder.set_buffer(2, Some(&*start_nonce_buf), 0);
        encoder.set_buffer(3, Some(&*digest_buf), 0);
        encoder.set_buffer(4, Some(&*posterior_buf), 0);
        encoder.set_buffer(5, Some(&*mitm_states_buf), 0);
        encoder.set_buffer(6, Some(&*fwht_buf), 0);
        encoder.set_buffer(7, Some(&*cs_buf), 0);
        encoder.set_buffer(8, Some(&*nibble_probs_buf), 0);
        encoder.set_buffer(9, Some(&*adaptive_params_buf), 0);
        encoder.set_buffer(10, Some(&*debug_flags_buf), 0);
        encoder.set_buffer(11, Some(&digest_out_len_buf), 0);
        encoder.set_buffer(12, Some(&nibble_probs_len_buf), 0);
        encoder.set_buffer(13, Some(&*adaptive_feedback_buf), 0);
        encoder.set_buffer(14, Some(&*shannon_entropy_buf), 0);
        encoder.set_buffer(15, Some(&*submit_mask_buf), 0);

        let threads_per_group = MTLSize { width: NIBBLES as u64, height: 1, depth: 1 };
        let grid_size = MTLSize { width: NIBBLES as u64, height: LANES as u64, depth: NONCES_PER_NIBBLE as u64 };

        encoder.dispatch_threads(grid_size, threads_per_group);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // ---------------- Synchronize ----------------
        let sync_buf = command_queue.new_command_buffer();
        let blit = sync_buf.new_blit_command_encoder();
        blit.synchronize_resource(&*digest_buf);
        blit.synchronize_resource(&*posterior_buf);
        blit.synchronize_resource(&*fwht_buf);
        blit.synchronize_resource(&*cs_buf);
        blit.synchronize_resource(&*nibble_probs_buf);
        blit.synchronize_resource(&*submit_mask_buf);
        blit.synchronize_resource(&*shannon_entropy_buf);
        blit.end_encoding();
        sync_buf.commit();
        sync_buf.wait_until_completed();

        // ---------------- Telemetry (fully aggregated) ----------------
        unsafe {
            let posterior_slice = std::slice::from_raw_parts(posterior_buf.contents() as *const f32, total_threads * NONCES_PER_NIBBLE);
            let fwht_slice = std::slice::from_raw_parts(fwht_buf.contents() as *const f32, total_threads * NONCES_PER_NIBBLE);
            let cs_slice = std::slice::from_raw_parts(cs_buf.contents() as *const f32, total_threads * NONCES_PER_NIBBLE);

            let elapsed = start_time.elapsed().as_secs_f64();
            let hashrate = (total_threads * NONCES_PER_NIBBLE) as f32 / elapsed.max(1e-6) as f32;

            let mut m = metrics.write().await;
            m.last_hashrate = hashrate;
            m.last_gpu_time = elapsed * 1000.0;
            m.last_cycle_time = elapsed * 1000.0;
            metrics_tx.send(m.clone()).ok();

            let lane_size = NIBBLES * NONCES_PER_NIBBLE;
            for lane in 0..LANES {
                let base = lane * lane_size;
                let avg_fwht: f32 = fwht_slice[base..base + lane_size].iter().sum::<f32>() / lane_size as f32;
                let avg_post: f32 = posterior_slice[base..base + lane_size].iter().sum::<f32>() / lane_size as f32;
                let avg_cs: f32 = cs_slice[base..base + lane_size].iter().sum::<f32>() / lane_size as f32;

                println!("Lane {:2}: FWHT={:.3e}  POST={:.6}  CS={:.3}", lane, avg_fwht, avg_post, avg_cs);
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

// ----------------- Main -----------------
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let local = tokio::task::LocalSet::new();
    local.run_until(async_main()).await;
}
