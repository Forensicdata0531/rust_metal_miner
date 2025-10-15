use bitcoin::hashes::{sha256d, Hash};
use metal::*;
use std::ffi::c_void;
use std::mem;

#[test]
fn fused_sha256_fwht_cs() {
    const LANES: usize = 4;
    const LANES_STATE_WORDS: usize = 8;
    const ALIGNED_NUM_THREADS: usize = 1;

    // ---------------- Step 1: Create deterministic header ----------------
    let mut header = [0u32; 19];
    for i in 0..19 {
        header[i] = (0x01020304u32).wrapping_mul(i as u32 + 1);
    }

    // ---------------- Step 2: Compute CPU double SHA256 matching GPU kernel ----------------
    let start_nonce: u32 = 1;
    let mut header_with_nonce = header;
    // Inject nonce into the 4th word (index 3) to match GPU kernel (W[3])
    header_with_nonce[3] ^= start_nonce;
    let first_hash_input: Vec<u8> = header_with_nonce
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let first_hash = sha256d::Hash::hash(&first_hash_input);

    // Second SHA256 (double hash)
    let cpu_hash = sha256d::Hash::hash(first_hash.as_ref());
    println!("CPU reference double SHA256 = {}", hex::encode(cpu_hash));

    // ---------------- Step 3: Setup Metal ----------------
    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    let library = device
        .new_library_with_file("shaders/kernels.metallib")
        .expect("Failed to load Metal library");
    let func = library
        .get_function("fused_sha256_fwht_cs", None)
        .expect("Missing kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .expect("Pipeline creation failed");

    // ---------------- Step 4: Prepare midstate and schedule ----------------
    use generic_array::GenericArray;
    use sha2::compress256;

    let mut midstate = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];

    // Build an 80-byte block from header_with_nonce's first 16 words (little-endian)
    let mut block = [0u8; 64];
    for i in 0..16 {
        block[i * 4..i * 4 + 4].copy_from_slice(&header_with_nonce[i].to_le_bytes());
    }
    compress256(&mut midstate, &[GenericArray::clone_from_slice(&block)]);

    let midstate_buf = device.new_buffer_with_data(
        midstate.as_ptr() as *const c_void,
        (midstate.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Prepare schedule: first 16 words are header_with_nonce words (already little-endian u32 words)
    let mut schedule = [0u32; 64];
    for i in 0..16 {
        schedule[i] = header_with_nonce[i];
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

    let schedule_buf = device.new_buffer_with_data(
        schedule.as_ptr() as *const c_void,
        (schedule.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let start_nonce_buf = device.new_buffer_with_data(
        &start_nonce as *const u32 as *const c_void,
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let digest_buf = device.new_buffer(
        (ALIGNED_NUM_THREADS * LANES_STATE_WORDS * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_digest_buf = device.new_buffer(
        (ALIGNED_NUM_THREADS * LANES_STATE_WORDS * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let active_mask_buf = device.new_buffer(
        (ALIGNED_NUM_THREADS * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lane_posteriors_buf = device.new_buffer(
        (ALIGNED_NUM_THREADS * LANES * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let adaptive_params_buf =
        device.new_buffer((3 * 4) as u64, MTLResourceOptions::StorageModeShared);
    let aligned_num_threads_buf = device.new_buffer_with_data(
        &(ALIGNED_NUM_THREADS as u32) as *const u32 as *const c_void,
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // ---------------- Step 5: Dispatch GPU kernel ----------------
    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(&midstate_buf), 0);
    encoder.set_buffer(1, Some(&schedule_buf), 0);
    encoder.set_buffer(2, Some(&start_nonce_buf), 0);
    encoder.set_buffer(3, Some(&digest_buf), 0);
    encoder.set_buffer(4, Some(&partial_digest_buf), 0);
    encoder.set_buffer(5, Some(&active_mask_buf), 0);
    encoder.set_buffer(6, Some(&lane_posteriors_buf), 0);
    encoder.set_buffer(7, Some(&adaptive_params_buf), 0);
    encoder.set_buffer(8, Some(&aligned_num_threads_buf), 0);

    let threadgroup_size = MTLSize {
        width: pipeline.thread_execution_width() as u64,
        height: 1,
        depth: 1,
    };
    let threadgroup_count = MTLSize {
        width: (ALIGNED_NUM_THREADS as u64 + threadgroup_size.width - 1) / threadgroup_size.width,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // ---------------- Step 6: Read GPU output ----------------
    let digest_slice = unsafe {
        std::slice::from_raw_parts(
            digest_buf.contents() as *const u32,
            ALIGNED_NUM_THREADS * LANES_STATE_WORDS,
        )
    };
    let gpu_bytes: Vec<u8> = digest_slice
        .iter()
        .flat_map(|w| w.to_be_bytes()) // read as big-endian to match CPU SHA256d
        .take(32)
        .collect();
    println!("GPU output digest = {}", hex::encode(&gpu_bytes));

    // ---------------- Step 7: Compare with CPU ----------------
    assert_eq!(&gpu_bytes[..], cpu_hash.as_ref(), "❌ GPU hash mismatch");

    println!("✅ GPU and CPU double SHA256 match perfectly!");
}
