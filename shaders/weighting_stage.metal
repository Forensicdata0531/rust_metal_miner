#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

inline uint get_nibble(uint x,uint idx){return (x>>(idx*4))&0xF;}
inline uint nibble_hw(uint nib){return popcount(nib);}

kernel void weighting_stage(
    device uint* partial_digest,
    device const uint* target_words,
    device atomic_uint* hit_counter,
    device uint* telemetry_nibbles,
    device float* lane_posteriors,
    device const uint* history_digest,
    constant float3& adaptive_params,
    device uint* active_mask,
    constant uint& aligned_num_threads,
    uint tid [[thread_position_in_grid]]
){
    constexpr uint LANES=4;
    constexpr uint NIBBLES=8;
    constexpr uint HISTORY_COUNT=100;
    constexpr float k=0.075f;

    if(tid>=aligned_num_threads) return;
    float PRUNE_THRESHOLD=adaptive_params[0];
    float HISTORY_WEIGHT=adaptive_params[1];
    float DIFFUSION_MIX=adaptive_params[2];

    simd_uint4 digest;
    for(uint l=0;l<LANES;l++)
        digest[l]=partial_digest[tid*LANES+l];

    uint active=active_mask[tid];

    float amplitudes[LANES*NIBBLES];
    for(uint l=0;l<LANES;l++){
        if((active&(1<<l))==0){
            for(uint n=0;n<NIBBLES;n++) amplitudes[l*NIBBLES+n]=0.0f;
            continue;
        }
        for(uint n=0;n<NIBBLES;n++){
            uint nib=get_nibble(digest[l],n);
            uint target_nib=get_nibble(target_words[l%8],n);
            float amp=exp2(-k*float(nibble_hw(nib^target_nib)));
            amplitudes[l*NIBBLES+n]=amp;
            telemetry_nibbles[tid*LANES*NIBBLES+l*NIBBLES+n]=nib;
        }
    }

    float lane_weight[LANES];
    for(uint l=0;l<LANES;l++){
        lane_weight[l]=0.0f;
        for(uint n=0;n<NIBBLES;n++) lane_weight[l]+=amplitudes[l*NIBBLES+n];
    }

    float posterior[LANES],sum_post=0.0f;
    for(uint l=0;l<LANES;l++){
        if((active&(1<<l))==0){posterior[l]=0.0f; continue;}
        posterior[l]=lane_weight[l]; sum_post+=posterior[l];
    }
    sum_post=fmax(sum_post,1e-6f);
    for(uint l=0;l<LANES;l++){posterior[l]/=sum_post; lane_posteriors[tid*LANES+l]=posterior[l];}

    uint new_active=active; uint hits=0;
    for(uint l=0;l<LANES;l++){
        if((active&(1<<l))==0) continue;
        if(posterior[l]<PRUNE_THRESHOLD){digest[l]=0; new_active&=~(1<<l);}
        else hits++;
        partial_digest[tid*LANES+l]=digest[l];
    }
    active_mask[tid]=new_active;
    if(hits>0) atomic_fetch_add_explicit(hit_counter,hits,memory_order_relaxed);
}
