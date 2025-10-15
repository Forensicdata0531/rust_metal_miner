#include <metal_stdlib>
#include <simd/simd.h>
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

// Stage 1: SHA-256 per-lane computation
kernel void sha256_stage(
    device const simd_uint4* __restrict midstates        [[buffer(0)]],
    device const simd_uint4* __restrict preexp_schedule [[buffer(1)]],
    device const uint* __restrict start_nonce           [[buffer(2)]],
    device uint* __restrict partial_digest              [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    constexpr uint LANES = 4;
    uint base_nonce = start_nonce[0] + tid * LANES;

    // Load initial state
    simd_uint4 a = midstates[0];
    simd_uint4 b = midstates[1];
    simd_uint4 c = midstates[2];
    simd_uint4 d = midstates[3];
    simd_uint4 e = midstates[4];
    simd_uint4 f = midstates[5];
    simd_uint4 g = midstates[6];
    simd_uint4 h = midstates[7];

    simd_uint4 nonce = simd_uint4(base_nonce, base_nonce+1, base_nonce+2, base_nonce+3);

    // SHA-256 rounds (broken down for compiler safety)
    for (uint i = 0; i < 64; i++) {
        simd_uint4 W = preexp_schedule[i] ^ nonce;

        simd_uint4 S1a = rotr(e, 6);
        simd_uint4 S1b = rotr(e, 11);
        simd_uint4 S1c = rotr(e, 25);
        simd_uint4 S1 = S1a ^ S1b ^ S1c;

        simd_uint4 S0a = rotr(a, 2);
        simd_uint4 S0b = rotr(a, 13);
        simd_uint4 S0c = rotr(a, 22);
        simd_uint4 S0 = S0a ^ S0b ^ S0c;

        simd_uint4 ch = (e & f) ^ ((~e) & g);
        simd_uint4 maj = (a & b) ^ (a & c) ^ (b & c);

        simd_uint4 temp1_part = csa_add(ch, simd_uint4(K[i]), W);
        simd_uint4 temp1 = csa_add(h, S1, temp1_part);
        simd_uint4 temp2 = S0 + maj;
        simd_uint4 new_a = temp1 + temp2;

        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = new_a;
    }

    // Store partial digest
    partial_digest[tid*LANES + 0] = a[0];
    partial_digest[tid*LANES + 1] = a[1];
    partial_digest[tid*LANES + 2] = a[2];
    partial_digest[tid*LANES + 3] = a[3];
}
