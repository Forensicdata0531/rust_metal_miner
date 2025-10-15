# 1 "shaders/miner_kernel.metal"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 656 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
#pragma clang module import metal_types /* clang -E: implicit import for #include "metal_types" */
# 1 "shaders/miner_kernel.metal" 2
#pragma clang module import metal_stdlib /* clang -E: implicit import for #include <metal_stdlib> */
# 1 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/simd.h" 1 3
# 17 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/simd.h" 3
# 1 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/matrix_types.h" 1 3
# 29 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/matrix_types.h" 3
#pragma clang module import metal_matrix /* clang -E: implicit import for #include <metal_matrix> */

using matrix_half2x2 = metal::half2x2;
using matrix_half3x2 = metal::half3x2;
using matrix_half4x2 = metal::half4x2;

using matrix_half2x3 = metal::half2x3;
using matrix_half3x3 = metal::half3x3;
using matrix_half4x3 = metal::half4x3;

using matrix_half2x4 = metal::half2x4;
using matrix_half3x4 = metal::half3x4;
using matrix_half4x4 = metal::half4x4;

using matrix_float2x2 = metal::float2x2;
using matrix_float3x2 = metal::float3x2;
using matrix_float4x2 = metal::float4x2;

using matrix_float2x3 = metal::float2x3;
using matrix_float3x3 = metal::float3x3;
using matrix_float4x3 = metal::float4x3;

using matrix_float2x4 = metal::float2x4;
using matrix_float3x4 = metal::float3x4;
using matrix_float4x4 = metal::float4x4;
# 70 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/matrix_types.h" 3
using simd_half2x2 = metal::half2x2;
using simd_half3x2 = metal::half3x2;
using simd_half4x2 = metal::half4x2;

using simd_half2x3 = metal::half2x3;
using simd_half3x3 = metal::half3x3;
using simd_half4x3 = metal::half4x3;

using simd_half2x4 = metal::half2x4;
using simd_half3x4 = metal::half3x4;
using simd_half4x4 = metal::half4x4;

using simd_float2x2 = metal::float2x2;
using simd_float3x2 = metal::float3x2;
using simd_float4x2 = metal::float4x2;

using simd_float2x3 = metal::float2x3;
using simd_float3x3 = metal::float3x3;
using simd_float4x3 = metal::float4x3;

using simd_float2x4 = metal::float2x4;
using simd_float3x4 = metal::float3x4;
using simd_float4x4 = metal::float4x4;
# 110 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/matrix_types.h" 3
namespace simd = metal;
# 18 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/simd.h" 2 3
# 1 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/packed.h" 1 3
# 16 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/packed.h" 3
namespace simd = metal;
# 19 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/simd.h" 2 3
# 1 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/vector_types.h" 1 3
# 13 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/vector_types.h" 3
namespace simd = metal;
# 20 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/32023/lib/clang/32023.404/include/metal/simd/simd.h" 2 3
# 3 "shaders/miner_kernel.metal" 2
using namespace metal;


constant uint K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};


inline simd_uint4 rotr(simd_uint4 x, uint n) {
    return (x >> n) | (x << (32u - n));
}


inline simd_uint4 csa_add(simd_uint4 x, simd_uint4 y, simd_uint4 z) {
    simd_uint4 sum = x ^ y ^ z;
    simd_uint4 carry = ((x & y) | (x & z) | (y & z)) << 1;
    return sum + carry;
}


inline simd_float4 amplitude_bias(simd_uint4 state, simd_uint4 mask) {
    simd_uint4 diff = state ^ mask;
    simd_float4 dist = simd_float4(popcount(diff.x), popcount(diff.y), popcount(diff.z), popcount(diff.w));
    return exp2(-0.05f * dist);
}


kernel void miner_kernel_batched(
    device const simd_uint4* __restrict midstates [[buffer(0)]],
    device const simd_uint4* __restrict preexp_schedule [[buffer(1)]],
    device const uint* __restrict start_nonce [[buffer(2)]],
    device const uint* __restrict target_words [[buffer(3)]],
    device atomic_uint* __restrict hit_counter [[buffer(4)]],
    device uint* __restrict results [[buffer(5)]],
    device uint* __restrict telemetry [[buffer(6)]],
    device const uint* __restrict mask16_ptr [[buffer(7)]],
    device const uint* __restrict mask32_ptr [[buffer(8)]],
    device const uint* __restrict mask48_ptr [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    constexpr uint LANES = 4;
    constexpr uint TG_CAP = 128;


    threadgroup atomic_uint tg_hit_count[1];
    threadgroup uint tg_results[TG_CAP*5];


    if(tid == 0) atomic_store_explicit(&tg_hit_count[0], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_nonce = start_nonce[0] + tid * LANES;
    simd_float4 amplitudes = simd_float4(1.0f);
    simd_uint4 active_mask = simd_uint4(0xFFFFFFFF);

    simd_uint4 a = midstates[0];
    simd_uint4 b = midstates[1];
    simd_uint4 c = midstates[2];
    simd_uint4 d = midstates[3];
    simd_uint4 e = midstates[4];
    simd_uint4 f = midstates[5];
    simd_uint4 g = midstates[6];
    simd_uint4 h = midstates[7];

    simd_uint4 nonce = simd_uint4(base_nonce, base_nonce+1, base_nonce+2, base_nonce+3);
    simd_uint4 mask = simd_uint4(target_words[0]);
    uint mask16 = mask16_ptr[0];
    uint mask32 = mask32_ptr[0];
    uint mask48 = mask48_ptr[0];


    for(uint i = 0; i < 64; i++){
        simd_uint4 W = preexp_schedule[i] ^ nonce;

        simd_uint4 S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        simd_uint4 S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        simd_uint4 ch = (e & f) ^ ((~e) & g);
        simd_uint4 maj = (a & b) ^ (a & c) ^ (b & c);

        simd_uint4 temp1 = csa_add(h, S1, csa_add(ch, simd_uint4(K[i]), W));
        simd_uint4 temp2 = S0 + maj;
        simd_uint4 new_a = temp1 + temp2;

        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = new_a;

        simd_float4 bias = amplitude_bias(a, mask);
        amplitudes = select(amplitudes, amplitudes * bias, active_mask != simd_uint4(0));


        if(i==16){
            simd_uint4 partial = h & mask16;
            simd_uint4 partial_target = simd_uint4(target_words[7] & mask16);
            active_mask &= select(simd_uint4(0), simd_uint4(0xFFFFFFFF), partial < partial_target);
        }
        if(i==32){
            simd_uint4 partial = h & mask32;
            simd_uint4 partial_target = simd_uint4(target_words[7] & mask32);
            active_mask &= select(simd_uint4(0), simd_uint4(0xFFFFFFFF), partial < partial_target);
        }
        if(i==48){
            simd_uint4 partial = h & mask48;
            simd_uint4 partial_target = simd_uint4(target_words[7] & mask48);
            active_mask &= select(simd_uint4(0), simd_uint4(0xFFFFFFFF), partial < partial_target);
        }

        if(all(active_mask == simd_uint4(0))) break;
    }


    simd_uint4 telem = simd_uint4((active_mask.x != 0) ? 1 : 0,
                                  (active_mask.y != 0) ? 1 : 0,
                                  (active_mask.z != 0) ? 1 : 0,
                                  (active_mask.w != 0) ? 1 : 0);
    telemetry[tid*LANES + 0] = telem.x;
    telemetry[tid*LANES + 1] = telem.y;
    telemetry[tid*LANES + 2] = telem.z;
    telemetry[tid*LANES + 3] = telem.w;


    float inv_norm = 1.0f / max(amplitudes.x + amplitudes.y + amplitudes.z + amplitudes.w, 1e-6f);
    amplitudes *= inv_norm;


    float maxw = amplitudes.x;
    uint best_lane = 0;
    if(amplitudes.y > maxw){ maxw = amplitudes.y; best_lane = 1; }
    if(amplitudes.z > maxw){ maxw = amplitudes.z; best_lane = 2; }
    if(amplitudes.w > maxw){ maxw = amplitudes.w; best_lane = 3; }

    uint best_nonce = base_nonce + best_lane;
    uint digest_a = a[best_lane];
    uint digest_e = e[best_lane];
    uint digest_h = h[best_lane];


    if(digest_h < target_words[7]){
        uint local_idx = atomic_fetch_add_explicit(&tg_hit_count[0], 1u, memory_order_relaxed);
        if(local_idx < TG_CAP){
            tg_results[local_idx*5 + 0] = best_nonce;
            tg_results[local_idx*5 + 1] = digest_a;
            tg_results[local_idx*5 + 2] = digest_e;
            tg_results[local_idx*5 + 3] = digest_h;
            tg_results[local_idx*5 + 4] = as_type<uint>(maxw);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);


    if(tid == 0){
        uint count = atomic_load_explicit(&tg_hit_count[0], memory_order_relaxed);
        for(uint i = 0; i < count; i++){
            uint idx = atomic_fetch_add_explicit(hit_counter, 1u, memory_order_relaxed);
            if(idx*5 + 4 < TG_CAP*5){
                device uint* res_ptr = results + idx*5;
                res_ptr[0] = tg_results[i*5 + 0];
                res_ptr[1] = tg_results[i*5 + 1];
                res_ptr[2] = tg_results[i*5 + 2];
                res_ptr[3] = tg_results[i*5 + 3];
                res_ptr[4] = tg_results[i*5 + 4];
            }
        }
        atomic_store_explicit(&tg_hit_count[0], 0u, memory_order_relaxed);
    }
}
