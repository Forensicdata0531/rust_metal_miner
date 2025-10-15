#include <metal_stdlib>
using namespace metal;
using namespace simd;

constant uint K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4d7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

// ------------------- Constants -------------------
constant float INV_U32_MAX = 1.0f / 4294967295.0f;

// Move round targets to top-level constant (avoids per-kernel reallocation)
constant float ROUND_TARGETS[3] = { 1e-6f, 1e-8f, 1e-10f };

// ------------------- Helpers -------------------
inline uint rotr32(uint x, uint n) { return (x >> n) | (x << (32 - n)); }

inline uint popcount_fwht16(uint x) {
    ushort4 v = ushort4((x >> 0) & 0xF, (x >> 4) & 0xF, (x >> 8) & 0xF, (x >> 12) & 0xF);
    v ^= (v >> 1); v ^= (v >> 2); v ^= (v >> 4);
    return popcount(uint(v[0] + v[1] + v[2] + v[3]));
}

inline uint fast_thread_entropy(uint seed) {
    seed ^= seed >> 13;
    seed ^= seed << 17;
    seed ^= seed >> 5;
    return seed;
}

// SHA-256 scalar round
inline void sha256_round_macro(thread uint4 *ab, thread uint4 *ef, uint k, uint w) {
    uint a = (*ab).x, b = (*ab).y, c = (*ab).z, d = (*ab).w;
    uint e = (*ef).x, f = (*ef).y, g = (*ef).z, h = (*ef).w;

    uint S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
    uint ch = (e & f) ^ ((~e) & g);
    uint temp1 = h + S1 + ch + k + w;
    uint S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
    uint maj = (a & b) ^ (a & c) ^ (b & c);
    uint temp2 = S0 + maj;

    h = g; g = f; f = e; e = d + temp1;
    d = c; c = b; b = a; a = temp1 + temp2;

    *ab = uint4(a,b,c,d);
    *ef = uint4(e,f,g,h);
}

// ==================== LUT SPLIT FIX ====================
// Gate LUT split into 4 parts of 64 floats each (256 total)
constant float gate_lut_0[64] = {
    1.000000e-12f, 1.174618e-12f, 1.379615e-12f, 1.621810e-12f, 1.906807e-12f, 2.242917e-12f, 2.640062e-12f, 3.110457e-12f,
    3.669713e-12f, 4.336810e-12f, 5.133317e-12f, 6.083176e-12f, 7.213512e-12f, 8.555492e-12f, 1.013365e-11f, 1.200331e-11f,
    1.422985e-11f, 1.684892e-11f, 1.991799e-11f, 2.350657e-11f, 2.770711e-11f, 3.263477e-11f, 3.841815e-11f, 4.520975e-11f,
    5.319600e-11f, 6.258628e-11f, 7.362430e-11f, 8.659463e-11f, 1.017857e-10f, 1.196321e-10f, 1.405736e-10f, 1.651772e-10f,
    1.940917e-10f, 2.281964e-10f, 2.685973e-10f, 3.166584e-10f, 3.741838e-10f, 4.432065e-10f, 5.261469e-10f, 6.257416e-10f,
    7.451200e-10f, 8.880039e-10f, 1.059905e-09f, 1.264541e-09f, 1.509733e-09f, 1.802037e-09f, 2.152717e-09f, 2.574560e-09f,
    3.082221e-09f, 3.693442e-09f, 4.427107e-09f, 5.306138e-09f, 6.357573e-09f, 7.613903e-09f, 9.112486e-09f, 1.090760e-08f,
    1.305966e-08f, 1.563335e-08f, 1.871437e-08f, 2.240950e-08f, 2.684031e-08f, 3.215560e-08f, 3.853177e-08f, 4.616693e-08f
};

constant float gate_lut_1[64] = {
    5.529428e-08f, 6.620789e-08f, 7.923526e-08f, 9.476078e-08f, 1.132831e-07f, 1.354116e-07f, 1.618374e-07f, 1.935752e-07f,
    2.315460e-07f, 2.768542e-07f, 3.308406e-07f, 3.949943e-07f, 4.710086e-07f, 5.609657e-07f, 6.672855e-07f, 7.928008e-07f,
    9.418082e-07f, 1.117713e-06f, 1.326847e-06f, 1.574705e-06f, 1.867145e-06f, 2.211300e-06f, 2.615789e-06f, 3.090773e-06f,
    3.648883e-06f, 4.306831e-06f, 5.084234e-06f, 6.003998e-06f, 7.093016e-06f, 8.382828e-06f, 9.910388e-06f, 1.170838e-05f,
    1.383413e-05f, 1.635710e-05f, 1.936949e-05f, 2.299562e-05f, 2.738424e-05f, 3.271202e-05f, 3.918860e-05f, 4.705597e-05f,
    5.660979e-05f, 6.819997e-05f, 8.226571e-05f, 9.933242e-05f, 1.199158e-04f, 1.447747e-04f, 1.747732e-04f, 2.108055e-04f,
    2.540229e-04f, 3.058586e-04f, 3.681995e-04f, 4.433253e-04f, 5.339698e-04f, 6.432987e-04f, 7.753152e-04f, 9.351195e-04f,
    1.128116e-03f, 1.362149e-03f, 1.643632e-03f, 1.983596e-03f, 2.390715e-03f, 2.876072e-03f, 3.455918e-03f, 4.149804e-03f
};

constant float gate_lut_2[64] = {
    4.992492e-03f, 6.005051e-03f, 7.229560e-03f, 8.704643e-03f, 1.047995e-02f, 1.261671e-02f, 1.517832e-02f, 1.823422e-02f,
    2.186692e-02f, 2.617404e-02f, 3.127056e-02f, 3.729171e-02f, 4.439670e-02f, 5.277343e-02f, 6.264377e-02f, 7.427009e-02f,
    8.796285e-02f, 1.041094e-01f, 1.232173e-01f, 1.458196e-01f, 1.725043e-01f, 2.039904e-01f, 2.411518e-01f, 2.850522e-01f,
    3.369789e-01f, 3.984840e-01f, 4.714300e-01f, 5.580491e-01f, 6.610078e-01f, 7.834865e-01f, 9.292729e-01f, 1.102857e+00f,
    1.309078e+00f, 1.554190e+00f, 1.845483e+00f, 2.191186e+00f, 2.600612e+00f, 3.084327e+00f, 3.654333e+00f, 4.324251e+00f,
    5.109550e+00f, 6.028810e+00f, 7.103010e+00f, 8.356898e+00f, 9.819421e+00f, 1.152440e+01f, 1.350688e+01f, 1.580951e+01f,
    1.847690e+01f, 2.156117e+01f, 2.512288e+01f, 2.923230e+01f, 3.397074e+01f, 3.943181e+01f, 4.572289e+01f, 5.296689e+01f,
    6.130438e+01f, 7.089576e+01f, 8.192382e+01f, 9.459655e+01f, 1.091512e+02f, 1.258875e+02f, 1.451013e+02f, 1.671997e+02f
};

constant float gate_lut_3[64] = {
    1.926448e+02f, 2.219942e+02f, 2.558335e+02f, 2.948108e+02f, 3.396480e+02f, 3.911556e+02f, 4.502484e+02f, 5.179617e+02f,
    5.954666e+02f, 6.840863e+02f, 7.853122e+02f, 9.008217e+02f, 1.032507e+03f, 1.182537e+03f, 1.353016e+03f, 1.546879e+03f,
    1.767300e+03f, 2.017763e+03f, 2.302091e+03f, 2.624470e+03f, 2.989510e+03f, 3.402282e+03f, 3.868351e+03f, 4.393826e+03f,
    4.985405e+03f, 5.650438e+03f, 6.397005e+03f, 7.233981e+03f, 8.171109e+03f, 9.219091e+03f, 1.038975e+04f, 1.169855e+04f,
    1.315928e+04f, 1.478930e+04f, 1.660749e+04f, 1.863451e+04f, 2.089298e+04f, 2.340755e+04f, 2.620505e+04f, 2.931468e+04f,
    3.276837e+04f, 3.660100e+04f, 4.085064e+04f, 4.555869e+04f, 5.077019e+04f, 5.653394e+04f, 6.290292e+04f, 6.993462e+04f,
    7.769138e+04f, 8.624088e+04f, 9.565650e+04f, 1.060179e+05f, 1.174434e+05f, 1.300414e+05f, 1.439086e+05f, 1.591505e+05f,
    1.758820e+05f, 1.942279e+05f, 2.143227e+05f, 2.363108e+05f, 2.603468e+05f, 2.865950e+05f, 3.152305e+05f, 3.464399e+05f
};

// ==================== CHAOS LUT (Split 10x100) ====================
constant float chaos_lut_0[100] = {
    0.015625f, 0.015635f, 0.015645f, 0.015655f, 0.015665f, 0.015675f, 0.015685f, 0.015695f, 0.015705f, 0.015715f,
    0.015725f, 0.015735f, 0.015745f, 0.015755f, 0.015765f, 0.015775f, 0.015785f, 0.015795f, 0.015805f, 0.015815f,
    0.015825f, 0.015835f, 0.015845f, 0.015855f, 0.015865f, 0.015875f, 0.015885f, 0.015895f, 0.015905f, 0.015915f,
    0.015925f, 0.015935f, 0.015945f, 0.015955f, 0.015965f, 0.015975f, 0.015985f, 0.015995f, 0.016005f, 0.016015f,
    0.016025f, 0.016035f, 0.016045f, 0.016055f, 0.016065f, 0.016075f, 0.016085f, 0.016095f, 0.016105f, 0.016115f,
    0.016125f, 0.016135f, 0.016145f, 0.016155f, 0.016165f, 0.016175f, 0.016185f, 0.016195f, 0.016205f, 0.016215f,
    0.016225f, 0.016235f, 0.016245f, 0.016255f, 0.016265f, 0.016275f, 0.016285f, 0.016295f, 0.016305f, 0.016315f,
    0.016325f, 0.016335f, 0.016345f, 0.016355f, 0.016365f, 0.016375f, 0.016385f, 0.016395f, 0.016405f, 0.016415f,
    0.016425f, 0.016435f, 0.016445f, 0.016455f, 0.016465f, 0.016475f, 0.016485f, 0.016495f, 0.016505f, 0.016515f,
    0.016525f, 0.016535f, 0.016545f, 0.016555f, 0.016565f, 0.016575f, 0.016585f, 0.016595f, 0.016605f, 0.016615f,
};

constant float chaos_lut_1[100] = {
    0.016625f, 0.016635f, 0.016645f, 0.016655f, 0.016665f, 0.016675f, 0.016685f, 0.016695f, 0.016705f, 0.016715f,
    0.016725f, 0.016735f, 0.016745f, 0.016755f, 0.016765f, 0.016775f, 0.016785f, 0.016795f, 0.016805f, 0.016815f,
    0.016825f, 0.016835f, 0.016845f, 0.016855f, 0.016865f, 0.016875f, 0.016885f, 0.016895f, 0.016905f, 0.016915f,
    0.016925f, 0.016935f, 0.016945f, 0.016955f, 0.016965f, 0.016975f, 0.016985f, 0.016995f, 0.017005f, 0.017015f,
    0.017025f, 0.017035f, 0.017045f, 0.017055f, 0.017065f, 0.017075f, 0.017085f, 0.017095f, 0.017105f, 0.017115f,
    0.017125f, 0.017135f, 0.017145f, 0.017155f, 0.017165f, 0.017175f, 0.017185f, 0.017195f, 0.017205f, 0.017215f,
    0.017225f, 0.017235f, 0.017245f, 0.017255f, 0.017265f, 0.017275f, 0.017285f, 0.017295f, 0.017305f, 0.017315f,
    0.017325f, 0.017335f, 0.017345f, 0.017355f, 0.017365f, 0.017375f, 0.017385f, 0.017395f, 0.017405f, 0.017415f,
    0.017425f, 0.017435f, 0.017445f, 0.017455f, 0.017465f, 0.017475f, 0.017485f, 0.017495f, 0.017505f, 0.017515f,
    0.017525f, 0.017535f, 0.017545f, 0.017555f, 0.017565f, 0.017575f, 0.017585f, 0.017595f, 0.017605f, 0.017615f,
};

constant float chaos_lut_2[100] = {
    0.017625f, 0.017635f, 0.017645f, 0.017655f, 0.017665f, 0.017675f, 0.017685f, 0.017695f, 0.017705f, 0.017715f,
    0.017725f, 0.017735f, 0.017745f, 0.017755f, 0.017765f, 0.017775f, 0.017785f, 0.017795f, 0.017805f, 0.017815f,
    0.017825f, 0.017835f, 0.017845f, 0.017855f, 0.017865f, 0.017875f, 0.017885f, 0.017895f, 0.017905f, 0.017915f,
    0.017925f, 0.017935f, 0.017945f, 0.017955f, 0.017965f, 0.017975f, 0.017985f, 0.017995f, 0.018005f, 0.018015f,
    0.018025f, 0.018035f, 0.018045f, 0.018055f, 0.018065f, 0.018075f, 0.018085f, 0.018095f, 0.018105f, 0.018115f,
    0.018125f, 0.018135f, 0.018145f, 0.018155f, 0.018165f, 0.018175f, 0.018185f, 0.018195f, 0.018205f, 0.018215f,
    0.018225f, 0.018235f, 0.018245f, 0.018255f, 0.018265f, 0.018275f, 0.018285f, 0.018295f, 0.018305f, 0.018315f,
    0.018325f, 0.018335f, 0.018345f, 0.018355f, 0.018365f, 0.018375f, 0.018385f, 0.018395f, 0.018405f, 0.018415f,
    0.018425f, 0.018435f, 0.018445f, 0.018455f, 0.018465f, 0.018475f, 0.018485f, 0.018495f, 0.018505f, 0.018515f,
    0.018525f, 0.018535f, 0.018545f, 0.018555f, 0.018565f, 0.018575f, 0.018585f, 0.018595f, 0.018605f, 0.018615f,
};

constant float chaos_lut_3[100] = {
    0.018625f, 0.018635f, 0.018645f, 0.018655f, 0.018665f, 0.018675f, 0.018685f, 0.018695f, 0.018705f, 0.018715f,
    0.018725f, 0.018735f, 0.018745f, 0.018755f, 0.018765f, 0.018775f, 0.018785f, 0.018795f, 0.018805f, 0.018815f,
    0.018825f, 0.018835f, 0.018845f, 0.018855f, 0.018865f, 0.018875f, 0.018885f, 0.018895f, 0.018905f, 0.018915f,
    0.018925f, 0.018935f, 0.018945f, 0.018955f, 0.018965f, 0.018975f, 0.018985f, 0.018995f, 0.019005f, 0.019015f,
    0.019025f, 0.019035f, 0.019045f, 0.019055f, 0.019065f, 0.019075f, 0.019085f, 0.019095f, 0.019105f, 0.019115f,
    0.019125f, 0.019135f, 0.019145f, 0.019155f, 0.019165f, 0.019175f, 0.019185f, 0.019195f, 0.019205f, 0.019215f,
    0.019225f, 0.019235f, 0.019245f, 0.019255f, 0.019265f, 0.019275f, 0.019285f, 0.019295f, 0.019305f, 0.019315f,
    0.019325f, 0.019335f, 0.019345f, 0.019355f, 0.019365f, 0.019375f, 0.019385f, 0.019395f, 0.019405f, 0.019415f,
    0.019425f, 0.019435f, 0.019445f, 0.019455f, 0.019465f, 0.019475f, 0.019485f, 0.019495f, 0.019505f, 0.019515f,
    0.019525f, 0.019535f, 0.019545f, 0.019555f, 0.019565f, 0.019575f, 0.019585f, 0.019595f, 0.019605f, 0.019615f,
};

constant float chaos_lut_4[100] = {
    0.019625f, 0.019635f, 0.019645f, 0.019655f, 0.019665f, 0.019675f, 0.019685f, 0.019695f, 0.019705f, 0.019715f,
    0.019725f, 0.019735f, 0.019745f, 0.019755f, 0.019765f, 0.019775f, 0.019785f, 0.019795f, 0.019805f, 0.019815f,
    0.019825f, 0.019835f, 0.019845f, 0.019855f, 0.019865f, 0.019875f, 0.019885f, 0.019895f, 0.019905f, 0.019915f,
    0.019925f, 0.019935f, 0.019945f, 0.019955f, 0.019965f, 0.019975f, 0.019985f, 0.019995f, 0.020005f, 0.020015f,
    0.020025f, 0.020035f, 0.020045f, 0.020055f, 0.020065f, 0.020075f, 0.020085f, 0.020095f, 0.020105f, 0.020115f,
    0.020125f, 0.020135f, 0.020145f, 0.020155f, 0.020165f, 0.020175f, 0.020185f, 0.020195f, 0.020205f, 0.020215f,
    0.020225f, 0.020235f, 0.020245f, 0.020255f, 0.020265f, 0.020275f, 0.020285f, 0.020295f, 0.020305f, 0.020315f,
    0.020325f, 0.020335f, 0.020345f, 0.020355f, 0.020365f, 0.020375f, 0.020385f, 0.020395f, 0.020405f, 0.020415f,
    0.020425f, 0.020435f, 0.020445f, 0.020455f, 0.020465f, 0.020475f, 0.020485f, 0.020495f, 0.020505f, 0.020515f,
    0.020525f, 0.020535f, 0.020545f, 0.020555f, 0.020565f, 0.020575f, 0.020585f, 0.020595f, 0.020605f, 0.020615f,
};

constant float chaos_lut_5[100] = {
    0.020625f, 0.020635f, 0.020645f, 0.020655f, 0.020665f, 0.020675f, 0.020685f, 0.020695f, 0.020705f, 0.020715f,
    0.020725f, 0.020735f, 0.020745f, 0.020755f, 0.020765f, 0.020775f, 0.020785f, 0.020795f, 0.020805f, 0.020815f,
    0.020825f, 0.020835f, 0.020845f, 0.020855f, 0.020865f, 0.020875f, 0.020885f, 0.020895f, 0.020905f, 0.020915f,
    0.020925f, 0.020935f, 0.020945f, 0.020955f, 0.020965f, 0.020975f, 0.020985f, 0.020995f, 0.021005f, 0.021015f,
    0.021025f, 0.021035f, 0.021045f, 0.021055f, 0.021065f, 0.021075f, 0.021085f, 0.021095f, 0.021105f, 0.021115f,
    0.021125f, 0.021135f, 0.021145f, 0.021155f, 0.021165f, 0.021175f, 0.021185f, 0.021195f, 0.021205f, 0.021215f,
    0.021225f, 0.021235f, 0.021245f, 0.021255f, 0.021265f, 0.021275f, 0.021285f, 0.021295f, 0.021305f, 0.021315f,
    0.021325f, 0.021335f, 0.021345f, 0.021355f, 0.021365f, 0.021375f, 0.021385f, 0.021395f, 0.021405f, 0.021415f,
    0.021425f, 0.021435f, 0.021445f, 0.021455f, 0.021465f, 0.021475f, 0.021485f, 0.021495f, 0.021505f, 0.021515f,
    0.021525f, 0.021535f, 0.021545f, 0.021555f, 0.021565f, 0.021575f, 0.021585f, 0.021595f, 0.021605f, 0.021615f,
};

constant float chaos_lut_6[100] = {
    0.021625f, 0.021635f, 0.021645f, 0.021655f, 0.021665f, 0.021675f, 0.021685f, 0.021695f, 0.021705f, 0.021715f,
    0.021725f, 0.021735f, 0.021745f, 0.021755f, 0.021765f, 0.021775f, 0.021785f, 0.021795f, 0.021805f, 0.021815f,
    0.021825f, 0.021835f, 0.021845f, 0.021855f, 0.021865f, 0.021875f, 0.021885f, 0.021895f, 0.021905f, 0.021915f,
    0.021925f, 0.021935f, 0.021945f, 0.021955f, 0.021965f, 0.021975f, 0.021985f, 0.021995f, 0.022005f, 0.022015f,
    0.022025f, 0.022035f, 0.022045f, 0.022055f, 0.022065f, 0.022075f, 0.022085f, 0.022095f, 0.022105f, 0.022115f,
    0.022125f, 0.022135f, 0.022145f, 0.022155f, 0.022165f, 0.022175f, 0.022185f, 0.022195f, 0.022205f, 0.022215f,
    0.022225f, 0.022235f, 0.022245f, 0.022255f, 0.022265f, 0.022275f, 0.022285f, 0.022295f, 0.022305f, 0.022315f,
    0.022325f, 0.022335f, 0.022345f, 0.022355f, 0.022365f, 0.022375f, 0.022385f, 0.022395f, 0.022405f, 0.022415f,
    0.022425f, 0.022435f, 0.022445f, 0.022455f, 0.022465f, 0.022475f, 0.022485f, 0.022495f, 0.022505f, 0.022515f,
    0.022525f, 0.022535f, 0.022545f, 0.022555f, 0.022565f, 0.022575f, 0.022585f, 0.022595f, 0.022605f, 0.022615f,
};

constant float chaos_lut_7[100] = {
    0.022625f, 0.022635f, 0.022645f, 0.022655f, 0.022665f, 0.022675f, 0.022685f, 0.022695f, 0.022705f, 0.022715f,
    0.022725f, 0.022735f, 0.022745f, 0.022755f, 0.022765f, 0.022775f, 0.022785f, 0.022795f, 0.022805f, 0.022815f,
    0.022825f, 0.022835f, 0.022845f, 0.022855f, 0.022865f, 0.022875f, 0.022885f, 0.022895f, 0.022905f, 0.022915f,
    0.022925f, 0.022935f, 0.022945f, 0.022955f, 0.022965f, 0.022975f, 0.022985f, 0.022995f, 0.023005f, 0.023015f,
    0.023025f, 0.023035f, 0.023045f, 0.023055f, 0.023065f, 0.023075f, 0.023085f, 0.023095f, 0.023105f, 0.023115f,
    0.023125f, 0.023135f, 0.023145f, 0.023155f, 0.023165f, 0.023175f, 0.023185f, 0.023195f, 0.023205f, 0.023215f,
    0.023225f, 0.023235f, 0.023245f, 0.023255f, 0.023265f, 0.023275f, 0.023285f, 0.023295f, 0.023305f, 0.023315f,
    0.023325f, 0.023335f, 0.023345f, 0.023355f, 0.023365f, 0.023375f, 0.023385f, 0.023395f, 0.023405f, 0.023415f,
    0.023425f, 0.023435f, 0.023445f, 0.023455f, 0.023465f, 0.023475f, 0.023485f, 0.023495f, 0.023505f, 0.023515f,
    0.023525f, 0.023535f, 0.023545f, 0.023555f, 0.023565f, 0.023575f, 0.023585f, 0.023595f, 0.023605f, 0.023615f,
};

constant float chaos_lut_8[100] = {
    0.023625f, 0.023635f, 0.023645f, 0.023655f, 0.023665f, 0.023675f, 0.023685f, 0.023695f, 0.023705f, 0.023715f,
    0.023725f, 0.023735f, 0.023745f, 0.023755f, 0.023765f, 0.023775f, 0.023785f, 0.023795f, 0.023805f, 0.023815f,
    0.023825f, 0.023835f, 0.023845f, 0.023855f, 0.023865f, 0.023875f, 0.023885f, 0.023895f, 0.023905f, 0.023915f,
    0.023925f, 0.023935f, 0.023945f, 0.023955f, 0.023965f, 0.023975f, 0.023985f, 0.023995f, 0.024005f, 0.024015f,
    0.024025f, 0.024035f, 0.024045f, 0.024055f, 0.024065f, 0.024075f, 0.024085f, 0.024095f, 0.024105f, 0.024115f,
    0.024125f, 0.024135f, 0.024145f, 0.024155f, 0.024165f, 0.024175f, 0.024185f, 0.024195f, 0.024205f, 0.024215f,
    0.024225f, 0.024235f, 0.024245f, 0.024255f, 0.024265f, 0.024275f, 0.024285f, 0.024295f, 0.024305f, 0.024315f,
    0.024325f, 0.024335f, 0.024345f, 0.024355f, 0.024365f, 0.024375f, 0.024385f, 0.024395f, 0.024405f, 0.024415f,
    0.024425f, 0.024435f, 0.024445f, 0.024455f, 0.024465f, 0.024475f, 0.024485f, 0.024495f, 0.024505f, 0.024515f,
    0.024525f, 0.024535f, 0.024545f, 0.024555f, 0.024565f, 0.024575f, 0.024585f, 0.024595f, 0.024605f, 0.024615f,
};

constant float chaos_lut_9[100] = {
    0.024625f, 0.024635f, 0.024645f, 0.024655f, 0.024665f, 0.024675f, 0.024685f, 0.024695f, 0.024705f, 0.024715f,
    0.024725f, 0.024735f, 0.024745f, 0.024755f, 0.024765f, 0.024775f, 0.024785f, 0.024795f, 0.024805f, 0.024815f,
    0.024825f, 0.024835f, 0.024845f, 0.024855f, 0.024865f, 0.024875f, 0.024885f, 0.024895f, 0.024905f, 0.024915f,
    0.024925f, 0.024935f, 0.024945f, 0.024955f, 0.024965f, 0.024975f, 0.024985f, 0.024995f, 0.025005f, 0.025015f,
    0.025025f, 0.025035f, 0.025045f, 0.025055f, 0.025065f, 0.025075f, 0.025085f, 0.025095f, 0.025105f, 0.025115f,
    0.025125f, 0.025135f, 0.025145f, 0.025155f, 0.025165f, 0.025175f, 0.025185f, 0.025195f, 0.025205f, 0.025215f,
    0.025225f, 0.025235f, 0.025245f, 0.025255f, 0.025265f, 0.025275f, 0.025285f, 0.025295f, 0.025305f, 0.025315f,
    0.025325f, 0.025335f, 0.025345f, 0.025355f, 0.025365f, 0.025375f, 0.025385f, 0.025395f, 0.025405f, 0.025415f,
    0.025425f, 0.025435f, 0.025445f, 0.025455f, 0.025465f, 0.025475f, 0.025485f, 0.025495f, 0.025505f, 0.025515f,
    0.025525f, 0.025535f, 0.025545f, 0.025555f, 0.025565f, 0.025575f, 0.025585f, 0.025595f, 0.025605f, 0.025615f,
};

// ------------------- CSA/CLA helpers (fully branchless) -------------------

// CSA for thread-local uints
inline void csa(thread uint* sum, thread uint* carry, uint x, uint y, uint z) {
    uint xy = x & y;
    uint xz = x & z;
    uint yz = y & z;
    *sum = x ^ y ^ z;
    *carry = (xy | xz | yz) << 1;
}

// CSA for threadgroup-shared uints
inline void csa(threadgroup uint* sum, threadgroup uint* carry, uint x, uint y, uint z) {
    uint xy = x & y;
    uint xz = x & z;
    uint yz = y & z;
    *sum = x ^ y ^ z;
    *carry = (xy | xz | yz) << 1;
}

// Mixed address-space CSA (threadgroup sum, thread-local carry)
inline void csa(threadgroup uint* sum, thread uint* carry, uint x, uint y, uint z) {
    uint xy = x & y;
    uint xz = x & z;
    uint yz = y & z;
    *sum = x ^ y ^ z;
    *carry = (xy | xz | yz) << 1;
}

// Fully branchless 32-bit CLA addition
inline uint cla_add(uint x, uint y) {
    uint sum = x ^ y;
    uint carry = (x & y) << 1;

    // Perform 32 iterations, propagating carry each time
    #pragma unroll
    for (uint i = 0; i < 32; i++) {
        uint new_sum = sum ^ carry;
        uint new_carry = (sum & carry) << 1;
        sum = new_sum;
        carry = new_carry;
    }

    return sum;
}

// ------------------- Unified Gate LUT Accessor -------------------
inline float gate_from_lut(float x) {
    uint idx = min(uint(clamp(x, 0.0f, 0.999f) * 255.0f), 255u);
    if (idx < 64u) return gate_lut_0[idx];
    else if (idx < 128u) return gate_lut_1[idx - 64u];
    else if (idx < 192u) return gate_lut_2[idx - 128u];
    else return gate_lut_3[idx - 192u];
}

// ------------------- Unified Chaos LUT Accessor -------------------
inline float chaos_from_lut(uint idx) {
    idx = min(idx, 999u);
    if (idx < 100u) return chaos_lut_0[idx];
    else if (idx < 200u) return chaos_lut_1[idx - 100u];
    else if (idx < 300u) return chaos_lut_2[idx - 200u];
    else if (idx < 400u) return chaos_lut_3[idx - 300u];
    else if (idx < 500u) return chaos_lut_4[idx - 400u];
    else if (idx < 600u) return chaos_lut_5[idx - 500u];
    else if (idx < 700u) return chaos_lut_6[idx - 600u];
    else if (idx < 800u) return chaos_lut_7[idx - 700u];
    else if (idx < 900u) return chaos_lut_8[idx - 800u];
    else return chaos_lut_9[idx - 900u];
}

kernel void fused_sha256d_fwht_cs(
    constant uint* midstates               [[buffer(0)]],
    constant uint* preexp_schedule         [[buffer(1)]],
    device const uint* nonce_start         [[buffer(2)]],
    device uint* digest_out                [[buffer(3)]],
    device float* posterior_out            [[buffer(4)]],
    device uint* mitm_states               [[buffer(5)]],
    device float* fwht_out                 [[buffer(6)]],
    device float* cs_out                   [[buffer(7)]],
    device float* nibble_probs             [[buffer(8)]],
    device float* adaptive_params          [[buffer(9)]],
    device uint* debug_flags               [[buffer(10)]],
    device uint* submit_mask               [[buffer(15)]],
    constant uint& digest_out_len          [[buffer(11)]],
    constant uint& nibble_probs_len        [[buffer(12)]],
    device float* adaptive_feedback        [[buffer(13)]],
    device float* shannon_entropy_buf      [[buffer(14)]],
    uint tid                               [[thread_position_in_grid]]
)
{
    constexpr uint NONCES_PER_THREAD = 32;
    constexpr uint NIBBLES = 16;

    const uint max_digest_idx = digest_out_len / 8;
    const uint max_nibble_idx = nibble_probs_len / NIBBLES;

    float mask_threshold     = adaptive_params[0];
    float prune_intensity    = adaptive_params[1];
    float adaptive_gain_base = adaptive_params[2];

    uint lane       = tid / NIBBLES;
    uint nibble_idx = tid % NIBBLES;
    uint base_nonce = nonce_start[lane];

    // Load midstate
    uint4 ab = uint4(midstates[lane*8 + 0], midstates[lane*8 + 1], midstates[lane*8 + 2], midstates[lane*8 + 3]);
    uint4 ef = uint4(midstates[lane*8 + 4], midstates[lane*8 + 5], midstates[lane*8 + 6], midstates[lane*8 + 7]);

    // Use precomputed schedule directly
    uint w_schedule[64];
    for(uint i=0;i<64;i++) w_schedule[i] = preexp_schedule[lane*64 + i]; // full 64-word schedule precomputed

    for(uint nonce_iter=0; nonce_iter<NONCES_PER_THREAD; nonce_iter++){
        uint nonce = base_nonce + nonce_iter;
        uint base_idx = lane * (NIBBLES * NONCES_PER_THREAD) + nibble_idx * NONCES_PER_THREAD + nonce_iter;
        if(base_idx >= max_digest_idx || base_idx >= max_nibble_idx) continue;

        // --- SHA-256 second pass
        uint4 ab2 = ab;
        uint4 ef2 = ef;
        bool pruned_second = false;

        #pragma unroll 64
        for(uint i=0;i<64;i++){
            sha256_round_macro(&ab2,&ef2,K[i],w_schedule[i]);
            if(i==15||i==31||i==47){
                float u2 = float(ab2.x ^ ef2.w)/4294967296.0f;
                if(u2 >= ROUND_TARGETS[(i==15)?0:(i==31)?1:2]) { pruned_second=true; break; }
            }
        }

        if(!pruned_second){ ab ^= ab2; ef ^= ef2; }

        // --- Extract 16 nibbles vectorized
        uint4 nib0 = uint4(ab.x &0xF, (ab.x>>4)&0xF, ab.y &0xF, (ab.y>>4)&0xF);
        uint4 nib1 = uint4(ab.z &0xF, (ab.z>>4)&0xF, ab.w &0xF, (ab.w>>4)&0xF);
        uint4 nib2 = uint4(ef.x &0xF, (ef.x>>4)&0xF, ef.y &0xF, (ef.y>>4)&0xF);
        uint4 nib3 = uint4(ef.z &0xF, (ef.z>>4)&0xF, ef.w &0xF, (ef.w>>4)&0xF);

        // --- Vectorized Monte Carlo/Bayesian update
        float4 gates0 = float4(gate_from_lut(float(nib0.x)), gate_from_lut(float(nib0.y)),
                               gate_from_lut(float(nib0.z)), gate_from_lut(float(nib0.w)));
        float4 gates1 = float4(gate_from_lut(float(nib1.x)), gate_from_lut(float(nib1.y)),
                               gate_from_lut(float(nib1.z)), gate_from_lut(float(nib1.w)));
        float4 gates2 = float4(gate_from_lut(float(nib2.x)), gate_from_lut(float(nib2.y)),
                               gate_from_lut(float(nib2.z)), gate_from_lut(float(nib2.w)));
        float4 gates3 = float4(gate_from_lut(float(nib3.x)), gate_from_lut(float(nib3.y)),
                               gate_from_lut(float(nib3.z)), gate_from_lut(float(nib3.w)));

        float entropy_scale = chaos_from_lut(fast_thread_entropy(tid + nonce)%1000) * adaptive_gain_base;

        float4 prob0 = clamp(float4(1.0f/16.0f) * mix(1.0f, prune_intensity, 1.0f-gates0) * entropy_scale, 1e-12f, 1.0f);
        float4 prob1 = clamp(float4(1.0f/16.0f) * mix(1.0f, prune_intensity, 1.0f-gates1) * entropy_scale, 1e-12f, 1.0f);
        float4 prob2 = clamp(float4(1.0f/16.0f) * mix(1.0f, prune_intensity, 1.0f-gates2) * entropy_scale, 1e-12f, 1.0f);
        float4 prob3 = clamp(float4(1.0f/16.0f) * mix(1.0f, prune_intensity, 1.0f-gates3) * entropy_scale, 1e-12f, 1.0f);

        // --- FWHT and CS vectorized
        float4 fwht0 = float4(nib0) - 8.0f;  float4 fwht1 = float4(nib1) - 8.0f;
        float4 fwht2 = float4(nib2) - 8.0f;  float4 fwht3 = float4(nib3) - 8.0f;

        float4 cs0 = abs(float4(nib0)) / 16.0f;
        float4 cs1 = abs(float4(nib1)) / 16.0f;
        float4 cs2 = abs(float4(nib2)) / 16.0f;
        float4 cs3 = abs(float4(nib3)) / 16.0f;

        // --- Posterior vector
        float4 post0 = prob0;
        float4 post1 = prob1;
        float4 post2 = prob2;
        float4 post3 = prob3;

        // --- Single fused write for all 16 elements
        for(uint i=0;i<4;i++){
            uint idx16 = base_idx*16 + i;
            fwht_out[idx16]      = fwht0[i];  cs_out[idx16]      = cs0[i];  nibble_probs[idx16]      = prob0[i];  posterior_out[idx16] = post0[i];
            fwht_out[idx16+4]    = fwht1[i];  cs_out[idx16+4]    = cs1[i];  nibble_probs[idx16+4]    = prob1[i];  posterior_out[idx16+4] = post1[i];
            fwht_out[idx16+8]    = fwht2[i];  cs_out[idx16+8]    = cs2[i];  nibble_probs[idx16+8]    = prob2[i];  posterior_out[idx16+8] = post2[i];
            fwht_out[idx16+12]   = fwht3[i];  cs_out[idx16+12]   = cs3[i];  nibble_probs[idx16+12]   = prob3[i];  posterior_out[idx16+12] = post3[i];
        }

        // --- Digest outputs
        digest_out[base_idx*8+0]=ab.x; digest_out[base_idx*8+1]=ab.y;
        digest_out[base_idx*8+2]=ab.z; digest_out[base_idx*8+3]=ab.w;
        digest_out[base_idx*8+4]=ef.x; digest_out[base_idx*8+5]=ef.y;
        digest_out[base_idx*8+6]=ef.z; digest_out[base_idx*8+7]=ef.w;

        // --- Feedback, debug, submit, entropy
        adaptive_feedback[base_idx] = (post0.x + post0.y + post0.z + post0.w +
                               post1.x + post1.y + post1.z + post1.w +
                               post2.x + post2.y + post2.z + post2.w +
                               post3.x + post3.y + post3.z + post3.w) / 16.0f;
        debug_flags[base_idx] = pruned_second ? 2u : 0u;
        submit_mask[base_idx] = pruned_second ? 0u : 1u;
        shannon_entropy_buf[base_idx] = gates0[nibble_idx<4? nibble_idx :
                                                nibble_idx<8? nibble_idx-4 :
                                                nibble_idx<12? nibble_idx-8 : nibble_idx-12];
    }
}
