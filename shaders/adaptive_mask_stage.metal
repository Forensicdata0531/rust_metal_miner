#include <metal_stdlib>
using namespace metal;

kernel void adaptive_mask_stage(
    device atomic_uint* active_mask         [[buffer(0)]],
    device const float* lane_posteriors     [[buffer(1)]],
    constant float3& adaptive_params        [[buffer(2)]],
    constant uint& aligned_num_threads      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
){
    constexpr uint LANES = 4u;
    if(tid >= aligned_num_threads) return;

    const uint base = tid * LANES;

    // Compute average posterior for this thread's lanes
    float sum_post = 0.0f;
    for(uint l=0; l<LANES; ++l) sum_post += lane_posteriors[base+l];
    float avg_post = max(sum_post / float(LANES), 1e-8f);

    // Load current lane mask
    uint cur_mask = atomic_load_explicit(&active_mask[tid], memory_order_relaxed);
    uint new_mask = cur_mask;

    // Base pruning threshold from adaptive parameters
    float prune_base  = adaptive_params.x;
    float weight_scale= adaptive_params.y;
    float diffusion  = adaptive_params.z;

    // Dynamically compute prune threshold for this thread
    float thread_threshold = clamp(prune_base * (0.8f + weight_scale * avg_post), 0.0f, 0.999f);

    // Iterate over lanes
    for(uint l=0; l<LANES; ++l){
        float posterior = lane_posteriors[base+l];

        // Diffusion offset per lane
        float diff_offset = ((float(l)/float(max(1u, LANES-1u))) - 0.5f) * (diffusion * 0.1f);
        float effective_thresh = thread_threshold * (1.0f + diff_offset);

        // Additional adaptive multiplier for low average posterior
        float adapt_mult = 1.0f;
        if(avg_post < 0.2f) adapt_mult += (0.2f - avg_post) * 0.5f;
        effective_thresh *= adapt_mult;

        // Prune or keep
        bool keep = posterior > effective_thresh;
        if(keep) new_mask |= (1u << l);
        else       new_mask &= ~(1u << l);
    }

    // Store updated lane mask
    atomic_store_explicit(&active_mask[tid], new_mask, memory_order_relaxed);
}
