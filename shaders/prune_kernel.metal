#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

inline simd_float4 amplitude_bias(simd_uint4 state, simd_uint4 mask) {
    simd_uint4 diff = state ^ mask;
    simd_float4 dist = simd_float4(popcount(diff.x), popcount(diff.y), popcount(diff.z), popcount(diff.w));
    return exp2(-0.05f * dist);
}

kernel void weighting_stage(
    device uint* __restrict partial_digest  [[buffer(0)]],
    device const uint* __restrict target_words [[buffer(1)]],
    device atomic_uint* __restrict hit_counter [[buffer(2)]],
    device uint* __restrict results [[buffer(3)]],
    device uint* __restrict telemetry [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    constexpr uint LANES = 4;
    constexpr uint TG_CAP = 16;
    threadgroup atomic_uint tg_hit_count[1];
    threadgroup uint tg_results[TG_CAP*5];

    if(tid == 0) atomic_store_explicit(&tg_hit_count[0], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load partial digest from stage 1
    simd_uint4 digest = simd_uint4(
        partial_digest[tid*LANES + 0],
        partial_digest[tid*LANES + 1],
        partial_digest[tid*LANES + 2],
        partial_digest[tid*LANES + 3]
    );

    // Compute amplitude bias
    simd_uint4 mask = simd_uint4(target_words[0]);
    simd_float4 amplitudes = amplitude_bias(digest, mask);

    // Normalize
    float inv_norm = 1.0f / max(amplitudes.x + amplitudes.y + amplitudes.z + amplitudes.w, 1e-6f);
    amplitudes *= inv_norm;

    // Select best lane
    float maxw = amplitudes.x;
    uint best_lane = 0;
    if(amplitudes.y > maxw){ maxw = amplitudes.y; best_lane = 1; }
    if(amplitudes.z > maxw){ maxw = amplitudes.z; best_lane = 2; }
    if(amplitudes.w > maxw){ maxw = amplitudes.w; best_lane = 3; }

    uint best_nonce = tid*LANES + best_lane;
    uint digest_val = digest[best_lane];

    // Threadgroup-safe write
    if(digest_val < target_words[7]){
        uint local_idx = atomic_fetch_add_explicit(&tg_hit_count[0], 1u, memory_order_relaxed);
        if(local_idx < TG_CAP){
            tg_results[local_idx*2 + 0] = best_nonce;
            tg_results[local_idx*2 + 1] = digest_val;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(tid == 0){
        uint count = atomic_load_explicit(&tg_hit_count[0], memory_order_relaxed);
        for(uint i = 0; i < count; i++){
            uint idx = atomic_fetch_add_explicit(hit_counter, 1u, memory_order_relaxed);
            results[idx*2 + 0] = tg_results[i*2 + 0];
            results[idx*2 + 1] = tg_results[i*2 + 1];
        }
        atomic_store_explicit(&tg_hit_count[0], 0u, memory_order_relaxed);
    }
}
