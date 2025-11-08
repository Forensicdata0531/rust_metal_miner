#include <metal_stdlib>
using namespace metal;
using namespace simd;

constant uint K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4d7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

// ------------------- Utility -------------------
inline uint rot_r(uint x, uint n) { return (x >> n) | (x << (32 - n)); }

inline ushort gate_bitmask(uint nib) {
    return ushort(((nib & 0x1) << 15) | ((nib & 0x2) << 13) | ((nib & 0x4) << 11) | ((nib & 0x8) << 9) | 0x7FFF);
}

inline void atomic_min_fixed(device atomic_uint* addr, uint val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    while (old_val > val &&
           !atomic_compare_exchange_weak_explicit(addr, &old_val, val,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {}
}

// ------------------- 8-Round SHA -------------------
inline void sha256_round8(
    thread uint &a, thread uint &b, thread uint &c, thread uint &d,
    thread uint &e, thread uint &f, thread uint &g, thread uint &h,
    thread uint K_local[8], thread uint W_local[8]
) {
    uint A = a, B = b, C = c, D = d;
    uint E = e, F = f, G = g, H = h;

    #pragma unroll
    for (uint i = 0; i < 8; i++) {
        uint Σ0 = rot_r(A,2) ^ rot_r(A,13) ^ rot_r(A,22);
        uint Σ1 = rot_r(E,6) ^ rot_r(E,11) ^ rot_r(E,25);
        uint ch = (E & F) ^ ((~E) & G);
        uint maj = (A & B) ^ (A & C) ^ (B & C);
        uint T1 = (H + Σ1 + ch + K_local[i] + W_local[i]) & 0xFFFFFFFF;
        uint T2 = (Σ0 + maj + T1) & 0xFFFFFFFF;

        H = G; G = F; F = E; E = (D + T1) & 0xFFFFFFFF;
        D = C; C = B; B = A; A = T2;
    }

    a = A; b = B; c = C; d = D;
    e = E; f = F; g = G; h = H;
}

// ------------------- Main Kernel -------------------
kernel void fused_sha256d_fwht_cs(
    constant uint* midstates               [[buffer(0)]],
    constant uint* preexp_schedule         [[buffer(1)]],
    device const uint* nonce_start         [[buffer(2)]],
    device uint* digest_out                [[buffer(3)]],
    device const ushort* posterior_in      [[buffer(4)]],
    device ushort* posterior_out           [[buffer(20)]],
    device uint* mitm_states               [[buffer(5)]],
    device ushort* fwht_out                [[buffer(6)]],
    device ushort* cs_out                  [[buffer(7)]],
    device ushort* nibble_probs            [[buffer(8)]],
    device uint* adaptive_params           [[buffer(9)]],
    device uint* debug_flags               [[buffer(10)]],
    device uint* submit_mask               [[buffer(15)]],
    constant uint& digest_out_len          [[buffer(11)]],
    constant uint& nibble_probs_len        [[buffer(12)]],
    device ushort* adaptive_feedback       [[buffer(13)]],
    device ushort* shannon_entropy_buf     [[buffer(14)]],
    device ushort* hamming_buf             [[buffer(16)]],
    device ushort* monte_buf               [[buffer(17)]],
    constant uint& iteration_counter       [[buffer(18)]],
    constant uint& lane_count              [[buffer(19)]],
    uint tid                               [[thread_position_in_grid]],
    threadgroup ushort* tg_lane_min        [[threadgroup(0)]],
    device atomic_uint* global_lane_min_int [[buffer(21)]]
) {
    constexpr uint NONCES_PER_THREAD = 32;
    constexpr uint NIBBLES = 16;
    constexpr uint SHA_ROUNDS = 64;

    uint lane = tid / NIBBLES;
    uint nibble_idx = tid % NIBBLES;
    ushort mask_threshold = adaptive_params[0];

    uint a = midstates[lane*8+0], b = midstates[lane*8+1], c = midstates[lane*8+2], d = midstates[lane*8+3];
    uint e = midstates[lane*8+4], f = midstates[lane*8+5], g = midstates[lane*8+6], h = midstates[lane*8+7];

    if (nibble_idx == 0) tg_lane_min[lane] = 0xFFFF;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint nonce_iter = 0; nonce_iter < NONCES_PER_THREAD; nonce_iter++) {
        uint base_idx = lane*(NIBBLES*NONCES_PER_THREAD) + nibble_idx*NONCES_PER_THREAD + nonce_iter;
        ushort pred_post = posterior_in[base_idx];
        if (pred_post < mask_threshold) {
            submit_mask[base_idx]=0u; debug_flags[base_idx]=3u; posterior_out[base_idx]=pred_post;
            continue;
        }

        uint w[16];
        for (uint i=0;i<16;i++) w[i]=preexp_schedule[lane*64+i];

        // --- Create plain local arrays for 8-round slices ---
        uint K_local[8] = { K[0], K[1], K[2], K[3], K[4], K[5], K[6], K[7] };
        uint W_local[8] = { w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7] };

        uint aa=a, bb=b, cc=c, dd=d, ee=e, ff=f, gg=g, hh=h;
        sha256_round8(aa, bb, cc, dd, ee, ff, gg, hh, K_local, W_local);

        a+=aa; b+=bb; c+=cc; d+=dd;
        e+=ee; f+=ff; g+=gg; h+=hh;

        // ------------------ FWHT / entropy / nibble ------------------
        ushort4 state0, state1, state2, state3;
        #pragma unroll
        for (uint i=0;i<4;i++) {
            uint nib0 = ((i==0?a:a>>4)&0xF);
            uint nib1 = ((i==0?c:c>>4)&0xF);
            uint nib2 = ((i==0?e:e>>4)&0xF);
            uint nib3 = ((i==0?g:g>>4)&0xF);

            state0[i] = gate_bitmask(nib0);
            state1[i] = gate_bitmask(nib1);
            state2[i] = gate_bitmask(nib2);
            state3[i] = gate_bitmask(nib3);

            fwht_out[base_idx*16+i]      = ushort(nib0<<12);
            fwht_out[base_idx*16+i+4]    = ushort(nib1<<12);
            fwht_out[base_idx*16+i+8]    = ushort(nib2<<12);
            fwht_out[base_idx*16+i+12]   = ushort(nib3<<12);

            cs_out[base_idx*16+i]        = ushort(abs(int(nib0)-8)<<12);
            cs_out[base_idx*16+i+4]      = ushort(abs(int(nib1)-8)<<12);
            cs_out[base_idx*16+i+8]      = ushort(abs(int(nib2)-8)<<12);
            cs_out[base_idx*16+i+12]     = ushort(abs(int(nib3)-8)<<12);

            nibble_probs[base_idx*16+i]      = state0[i];
            nibble_probs[base_idx*16+i+4]    = state1[i];
            nibble_probs[base_idx*16+i+8]    = state2[i];
            nibble_probs[base_idx*16+i+12]   = state3[i];
        }

        ushort post_mean = (state0[0]+state0[1]+state0[2]+state0[3]+
                            state1[0]+state1[1]+state1[2]+state1[3]+
                            state2[0]+state2[1]+state2[2]+state2[3]+
                            state3[0]+state3[1]+state3[2]+state3[3])>>4;
        if (post_mean < tg_lane_min[lane]) tg_lane_min[lane]=post_mean;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid<lane_count)
            atomic_min_fixed(global_lane_min_int, uint(tg_lane_min[tid]));
        threadgroup_barrier(mem_flags::mem_device);

        ushort min_entropy = ushort(atomic_load_explicit(&global_lane_min_int[0], memory_order_relaxed));
        ushort combined_post = post_mean>min_entropy?min_entropy:post_mean;
        posterior_out[base_idx]=combined_post;
        adaptive_feedback[base_idx]=combined_post;
        if (combined_post < mask_threshold) continue;

        digest_out[base_idx*8+0]=a; digest_out[base_idx*8+1]=b;
        digest_out[base_idx*8+2]=c; digest_out[base_idx*8+3]=d;
        digest_out[base_idx*8+4]=e; digest_out[base_idx*8+5]=f;
        digest_out[base_idx*8+6]=g; digest_out[base_idx*8+7]=h;

        debug_flags[base_idx]=0u; submit_mask[base_idx]=1u;
    }
}
