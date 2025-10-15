#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

constant uint TILE_SIZE = 256;
constant uint LANES = 4;
constant uint NIBBLES = 8;
constant float BASE_PRUNE = 0.93f;

inline uint get_nibble(uint x, uint idx) { return (x >> (idx*4)) & 0xF; }
inline uint nibble_hw(uint nib) { return popcount(nib); }
inline uint blend(uint a, uint b, float f) { return uint(float(a)*(1.0f-f)+float(b)*f); }

kernel void fwht_parallel_stage(
    device uint* partial_digest,
    device uint* telemetry,
    device float* lane_posteriors,
    device atomic_uint* active_mask,
    device float3* adaptive_params_buf,
    constant uint& aligned_num_threads,
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
){
    uint global_idx = tgid*TILE_SIZE + tid;
    if(global_idx >= aligned_num_threads) return;

    threadgroup simd_uint4 tile[TILE_SIZE];
    threadgroup atomic_uint tg_mean_accum;
    threadgroup atomic_uint tg_var_accum;
    threadgroup atomic_uint tg_count_accum;

    if(tid==0){
        atomic_store_explicit(&tg_mean_accum,0u,memory_order_relaxed);
        atomic_store_explicit(&tg_var_accum,0u,memory_order_relaxed);
        atomic_store_explicit(&tg_count_accum,0u,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float3 adaptive_params = adaptive_params_buf[0];
    float prune_threshold = adaptive_params.x;
    float weight_mix = adaptive_params.y;
    float diffusion_mix = adaptive_params.z;

    simd_uint4 x;
    for(uint l=0;l<LANES;l++)
        x[l] = partial_digest[global_idx*LANES + l];
    tile[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint ROUNDS = 4;
    for(uint r=0;r<ROUNDS;r++){
        uint partner = tid^(1u<<(r&3));
        if(partner<TILE_SIZE){
            simd_uint4 a = tile[tid];
            simd_uint4 b = tile[partner];
            simd_uint4 mixed;
            for(uint l=0;l<LANES;l++){
                uint mask_bits = atomic_load_explicit(&active_mask[global_idx], memory_order_relaxed);
                if((mask_bits&(1<<l))==0){ mixed[l]=a[l]; continue; }
                uint add=a[l]+b[l];
                uint sub=a[l]-b[l];
                uint blended = blend(a[l], (add^sub), diffusion_mix);
                mixed[l] = metal::rotate(blended, ((l+r)*7)&31);
            }
            tile[tid]=mixed;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float lane_weight[LANES];
    for(uint l=0;l<LANES;l++){
        lane_weight[l]=0.0f;
        uint mask_bits = atomic_load_explicit(&active_mask[global_idx], memory_order_relaxed);
        if((mask_bits&(1<<l))!=0){
            for(uint n=0;n<NIBBLES;n++){
                uint nib=get_nibble(tile[tid][l], n);
                float amp=exp2(-0.085f*float(nibble_hw(nib)));
                telemetry[global_idx*LANES*NIBBLES+l*NIBBLES+n]=uint(clamp(amp,0.0f,1.0f)*65535.0f);
                lane_weight[l]+=amp;
            }
        }
    }

    float total=0.0f; for(uint l=0;l<LANES;l++) total+=lane_weight[l];
    total=fmax(total,1e-6f);

    float dynamic_prune=clamp(BASE_PRUNE+0.05f*(1.0f-total/float(LANES)), prune_threshold*0.8f, 0.995f);

    for(uint l=0;l<LANES;l++){
        float posterior=lane_weight[l]/total;
        lane_posteriors[global_idx*LANES+l] = lane_posteriors[global_idx*LANES+l]*(1.0f-weight_mix)+posterior*weight_mix;
        if(posterior<dynamic_prune){
            uint val=tile[tid][l];
            uint salt=(uint)(posterior*1e6f)^0x5A5A5A5Au;
            tile[tid][l]=metal::rotate(val^salt,(l*7+tid)&31);
            uint mask_bits=atomic_load_explicit(&active_mask[global_idx], memory_order_relaxed);
            atomic_store_explicit(&active_mask[global_idx], mask_bits & ~(1<<l), memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint l=0;l<LANES;l++)
        partial_digest[global_idx*LANES+l]=tile[tid][l];
}
