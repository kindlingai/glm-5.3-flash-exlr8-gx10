// exlr8 MoE: the routed and shared experts of one exlr8 v2 MoE layer on one tensor-parallel rank, prefill and
// decode, on GB10 (sm_121a). The trellis bytes are read as the release stores them.
//
// Per call (M tokens, R routes per token: top-k routed plus, optionally, the shared expert as one more route):
//   rot    x_rot = fp16(had128(x * signs_in)) once per token. Every expert of a layer shares the input signs; its own
//          input scale is one value per 128-channel block, so x * suh = signs * blk, and the block scale moves to
//          the FC1 accumulators.
//   plan   one block: counts routes per expert, sorts them by expert and writes the FC1 and FC2 work lists. Grids
//          are sized for the worst case and blocks past the list exit, so a call captures in a CUDA graph.
//   fc1    item = (expert, row block, one owned 128-channel unit, K range). Per 128-wide K stage the item's x_rot
//          rows are staged as fp16 A tiles. Each warp owns one 16-column gate tile and one up tile: it loads the
//          trellis words coalesced, decodes each 16x16 tile once into mma B fragments and multiplies it against
//          every 16-row block of the item. Decode items split K; the last split to arrive sums the partials in
//          split order and runs the epilogue: had128 * svh on gate and up, the swiglu clamp, silu(g) * u, * down
//          suh, had128, stored as fp16 h[route][I].
//   fc2    item = (expert, row block, 128 * NTW output columns), K = I over the owned units. Epilogue: down block
//          scale * routing weight into a per-route row; the last of a token's routes to finish a 128-column chunk
//          sums them in route order, applies had128 and the output signs, and writes the output.
//
// Numerics follow exllamav3's EXL3 linear: fp16 activations, fp32 accumulation, GEMM outputs rounded to fp16, the
// fused activation in fp16. Codebook values are rounded to E4M3, as the format requires.
//
// Derived from exl3rate (kindlingai/exl3rate). The trellis decode, Hadamard butterfly and window extraction follow
// exllamav3 (quant/exl3_dq.cuh, codebook.cuh, hadamard_inner.cuh, exl3_gemv_kernel.cuh; MIT, Copyright (c) 2025
// Turboderp, https://github.com/turboderp-org/exllamav3). The licence is in LICENSE-exllamav3 beside this file.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef EXLR8_H
#define EXLR8_H 4096                         // hidden size
#endif
#ifndef EXLR8_E
#define EXLR8_E 288                          // routed experts; index E is the shared expert
#endif
#ifndef EXLR8_UPR
#define EXLR8_UPR 4                          // 128-channel units of each expert on this rank
#endif
#ifndef EXLR8_TOPK
#define EXLR8_TOPK 8
#endif
#ifndef EXLR8_DECPF
#define EXLR8_DECPF 2                        // decode FC1 prefetches weight stages this far ahead into L2
#endif
#ifndef EXLR8_FC2NTW
#define EXLR8_FC2NTW 2                       // prefill FC2 output columns per block, in 128s
#endif
#ifndef EXLR8_FP8A
#define EXLR8_FP8A 1                         // prefill FC1 in FP8 (m16n8k32 e4m3): x_rot as E4M3, one scale per token.
#endif                                       //   The decoded weights are E4M3 values already, so only x is rounded.
#ifndef EXLR8_FP8_MIN_M
#define EXLR8_FP8_MIN_M 1024                 // calls of fewer tokens keep fp16 activations everywhere, so a request's
#endif                                       //   arithmetic at decode does not depend on what else is in the batch
#ifndef EXLR8_FP8H
#define EXLR8_FP8H 1                         // prefill FC2 in FP8: FC1 writes h as E4M3 with a scale per row and
#endif                                       //   128-channel chunk, FC2 multiplies on the FP8 mma
#ifndef EXLR8_Y8
#define EXLR8_Y8 1                           // per-route FC2 rows as int8, one scale per 128-column chunk: half the
#endif                                       //   bytes through the combine
#ifndef EXLR8_MAXMB
#define EXLR8_MAXMB 8                        // 16-row blocks per prefill item, at most (4 or 8)
#endif
#ifndef EXLR8_FC2MB
#define EXLR8_FC2MB 4                        // 16-row blocks per prefill FC2 item, at most (4 or 8)
#endif
#ifndef EXLR8_PIPE
#define EXLR8_PIPE 1                         // FP8 prefill FC1 streams x_rot and the weight words through a 3-stage
#endif                                       //   cp.async pipeline in shared memory (0: register prefetch)
#ifndef EXLR8_PIPE_MB
#define EXLR8_PIPE_MB 8                      // items of at least this many 16-row blocks take the pipeline
#endif
#ifndef EXLR8_MB4_MIN_M
#define EXLR8_MB4_MIN_M 4096                 // FP8 calls of this many tokens split FC1 into 64-row items, two blocks
#endif                                       // an SM: twice the warps to hide latency, each weight decoded twice
#ifndef EXLR8_FC1X
#define EXLR8_FC1X 0                         // timing only, wrong output, FP8 prefill FC1: 2 skips the weight decode
#endif                                       //   (raw words as B), 3 also the multiplies
#ifndef EXLR8_FC2X
#define EXLR8_FC2X 0                         // timing only, wrong output, prefill FC2: 1 skips the per-route rows and
#endif                                       //   the combine, 2 also the weight decode, 3 also the multiplies
#ifndef FC1_DEC_BLOCKS
#define FC1_DEC_BLOCKS 2                     // decode FC1 blocks per SM the compiler must fit
#endif

namespace exlr8 {

constexpr int H = EXLR8_H;
constexpr int E = EXLR8_E;
constexpr int NE = E + 1;                    // with the shared expert
constexpr int UPR = EXLR8_UPR;
constexpr int I = UPR * 128;                 // expert intermediate channels on this rank
constexpr int TOPK = EXLR8_TOPK;
constexpr int RMAX = TOPK + 1;               // routes per token, at most
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int PLAN_THREADS = (NE + 31) / 32 * 32;
constexpr int PLAN_WARPS = PLAN_THREADS / 32;
constexpr int KSTAGE = 128;                  // K per stage: one Hadamard block
constexpr int MAXMB = EXLR8_MAXMB;           // 16-row blocks per item, at most
constexpr int APAD = 8;                      // A tile row padding (halves), against bank conflicts
constexpr int ASTRIDE = KSTAGE + APAD;
constexpr int HSTRIDE = I + APAD;            // h row stride (halves) when a block holds all of K
constexpr int HB = H / 128;                  // 128-channel blocks on the hidden side
constexpr float R_SCALE = 0.088388347648f;   // 1/sqrt(128)
static_assert(H % 128 == 0 && UPR >= 1, "geometry");

struct FragB { half2 v[2]; };                // mma m16n8k16 B fragment
struct FragA { half2 v[4]; };
struct FragC { float v[4]; };

// ------------------------------------------------------------------------------------------------
// Trellis decode (exllamav3's mcg codebook, then E4M3 rounding)

union h2u { uint32_t u; half2 h; };

// Two weights as E4M3 bytes (low byte x0's).
__device__ __forceinline__ uint32_t decode_pair8(uint32_t x0, uint32_t x1)
{
    x0 *= 0xCBAC1FEDu;
    x1 *= 0xCBAC1FEDu;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    h2u a, b;
    a.u = x0;
    b.u = x1;
    const half2 v = __hadd2(__lows2half2(a.h, b.h), __highs2half2(a.h, b.h));
    return __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(v), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ half2 decode_pair(uint32_t x0, uint32_t x1)
{
    return __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) decode_pair8(x0, x1), __NV_E4M3));
}

__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
                                        uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    f0.v[0] = decode_pair(w0 & 0xffff, w1 & 0xffff);
    f0.v[1] = decode_pair(w2 & 0xffff, w3 & 0xffff);
    f1.v[0] = decode_pair(w4 & 0xffff, w5 & 0xffff);
    f1.v[1] = decode_pair(w6 & 0xffff, w7 & 0xffff);
}

// decode8 for the FP8 multiplies: each n8 fragment's 4 weights as E4M3 bytes, the layout to_e4m3x4 gives.
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
                                        uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, uint32_t& f0, uint32_t& f1)
{
    f0 = decode_pair8(w0 & 0xffff, w1 & 0xffff) | decode_pair8(w2 & 0xffff, w3 & 0xffff) << 16;
    f1 = decode_pair8(w4 & 0xffff, w5 & 0xffff) | decode_pair8(w6 & 0xffff, w7 & 0xffff) << 16;
}

__device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int shift)
{
    return (uint32_t) ((((uint64_t) a << 32) | (uint64_t) b) >> shift);
}

// Lane l's 8 weights of a 16x16 tile are the 16-bit windows ending at stream bits (8l + j + 1) * K, j = 0..7,
// wrapping at the tile's 256 K bits. Words are 32 stream bits, MSB first. The lane fetches the word holding its
// last window (b) and the one before it (a) from the warp's coalesced load.
template <int K>
struct TileLane
{
    int src_a, src_b, s2;
    __device__ __forceinline__ TileLane(int lane)
    {
        if constexpr (K == 4)
        {
            src_a = (lane + 31) & 31;
            src_b = lane;
            s2 = 0;
        }
        else if constexpr (K == 2)
        {
            // Windows end at 16 (l & 1) + 2 + 2j bits into word l / 2.
            src_a = ((lane >> 1) + 15) & 15;
            src_b = lane >> 1;
            s2 = 16 - 16 * (lane & 1);
        }
        else
        {
            const int t_offset = lane << 3;
            const int b1 = (t_offset + 257) * 3;
            const int b2 = b1 + 21;
            const int i0 = (b1 - 16) / 32;
            const int i2 = (b2 - 1) / 32;
            s2 = (i2 + 1) * 32 - b2;
            src_a = i0 % 24;
            src_b = i2 % 24;
        }
    }
};

// Decode one tile from the warp's coalesced load (lane l holds word l; lanes past the tile's 8K words hold
// anything) into this lane's two n8 B fragments.
template <int K, typename F>
__device__ __forceinline__ void decode_tile(uint32_t word, const TileLane<K>& tl, F& f0, F& f1)
{
    const uint32_t a = __shfl_sync(0xffffffffu, word, tl.src_a);
    const uint32_t b = __shfl_sync(0xffffffffu, word, tl.src_b);
    if constexpr (K == 4)
    {
        uint32_t s;
        asm("shf.r.wrap.b32 %0, %1, %2, 20;" : "=r"(s) : "r"(b), "r"(a));
        decode8(s >> 8, s >> 4, s, b >> 16, b >> 12, b >> 8, b >> 4, b, f0, f1);
    }
    else if constexpr (K == 2)
    {
        const uint32_t c = fshift(b, a, tl.s2);
        decode8(c >> 14, c >> 12, c >> 10, c >> 8, c >> 6, c >> 4, c >> 2, c, f0, f1);
    }
    else
    {
        const uint32_t w7 = fshift(b, a, tl.s2);
        const uint32_t w3 = fshift(b, a, tl.s2 + 12);
        decode8(w3 >> 9, w3 >> 6, w3 >> 3, w3, w7 >> 9, w7 >> 6, w7 >> 3, w7, f0, f1);
    }
}

__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }

// 8 bytes global -> shared without registers; n = 0 writes zeros.
__device__ __forceinline__ void cp_async8(void* dst, const void* src, int n)
{
    const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, int n)
{
    const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// 16 bytes global -> shared, bypassing L1; n = 0 writes zeros.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int n)
{
    const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(n));
}

__device__ __forceinline__ void mma(const FragA& a, const FragB& b, FragC& c)
{
    const uint32_t* aa = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c.v[0]), "+f"(c.v[1]), "+f"(c.v[2]), "+f"(c.v[3])
        : "r"(aa[0]), "r"(aa[1]), "r"(aa[2]), "r"(aa[3]), "r"(bb[0]), "r"(bb[1]));
}

// FP8 prefill. An m16n8k32 e4m3 B fragment wants k = 4t..4t+3 and 16+4t.. for lane%4 = t, but an fp16 m16n8k16
// fragment holds k = 2t, 2t+1, 2t+8, 2t+9. Rather than move values between lanes, each 32-wide K block is permuted:
// logical k 4t+j of the FP8 multiply is physical k {2t, 2t+1, 2t+8, 2t+9}[j] of tile 0, and 16+4t+j the same of
// tile 1. The weights then convert in place, and x_rot is stored in that logical order (fp8_pos).
struct FragA8 { uint32_t v[4]; };
struct FragB8 { uint32_t v[2]; };

__host__ __device__ constexpr int fp8_pos(int k)           // logical position of physical k within its 32-block
{
    return (k & ~31) + (k & 16) + 4 * ((k & 7) >> 1) + 2 * ((k >> 3) & 1) + (k & 1);
}

__device__ __forceinline__ void mma8(const FragA8& a, const FragB8& b, FragC& c)
{
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c.v[0]), "+f"(c.v[1]), "+f"(c.v[2]), "+f"(c.v[3])
        : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]), "r"(b.v[0]), "r"(b.v[1]));
}

// 128-point Hadamard across a warp, 4 consecutive values per lane (exllamav3's butterfly), * 1/sqrt(128)
__device__ __forceinline__ void had128(float& h0, float& h1, float& h2, float& h3, int lane)
{
    const float s0 = h0 + h1, d0 = h0 - h1, s1 = h2 + h3, d1 = h2 - h3;
    h0 = s0 + s1;
    h1 = d0 + d1;
    h2 = s0 - s1;
    h3 = d0 - d1;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        const float p0 = __shfl_xor_sync(0xffffffffu, h0, i);
        const float p1 = __shfl_xor_sync(0xffffffffu, h1, i);
        const float p2 = __shfl_xor_sync(0xffffffffu, h2, i);
        const float p3 = __shfl_xor_sync(0xffffffffu, h3, i);
        const bool hi = lane & i;
        h0 = (hi ? -h0 : h0) + p0;
        h1 = (hi ? -h1 : h1) + p1;
        h2 = (hi ? -h2 : h2) + p2;
        h3 = (hi ? -h3 : h3) + p3;
    }
    h0 *= R_SCALE;
    h1 *= R_SCALE;
    h2 *= R_SCALE;
    h3 *= R_SCALE;
}

__device__ __forceinline__ float rh(float v) { return __half2float(__float2half_rn(v)); }  // round to fp16

__device__ __forceinline__ float silu_h(float x)   // exllamav3's fp16 silu
{
    const half hx = __float2half_rn(x);
    const half one = __float2half_rn(1.0f);
    const half r = hrcp(__hadd(one, hexp(__hneg(hx))));
    return __half2float(__hmul(hx, r));
}

// ------------------------------------------------------------------------------------------------
// Parameters

struct Params
{
    // Inputs
    const __nv_bfloat16* x;        // [M, H]
    const int64_t* ids;            // [M, TOPK] routed expert ids (-1: no route)
    const float* rw;               // [M, TOPK] routing weights
    int M;
    int R;                         // routes per token: TOPK, or TOPK + 1 with the shared expert last
    float routed_scale;            // multiplies the routed weights (the shared route's weight is 1)
    float swiglu_limit;            // gate <= limit, |up| <= limit (inf: none)
    // Weights (see release.MoeSlab)
    const uint8_t* slab;
    const int64_t* off;            // [NE, 3, UPR] unit byte offsets in slab
    const int* bits;               // [NE, 3] width per expert and projection
    const half* signs;             // [2, H] input and output signs
    const float* blk;              // [NE, 3, HB] hidden-side block scales, signed
    const half* ch;                // [NE, 3, I] owned units' channel values (gate/up svh, down suh)
    // Plan (written by the plan kernel)
    int* count;                    // [NE] routes per expert
    int* order;                    // [M * R] routes sorted by expert
    int4* fc1_items;               // decode items (one 16-row block): (expert, first sorted route, rows,
                                   //   split group << 12 | 1 << 8 | split << 4 | splits)
    int4* fc1p_items;              // prefill items (2 or 4 16-row blocks): (expert, first, rows, mb << 8 | 1)
    int4* fc2_items;               // decode items: (expert, first, rows, 1)
    int4* fc2p_items;              // prefill items: (expert, first, rows, mb)
    int* n_items;                  // [4] FC1 decode, FC1 prefill, FC2 decode, FC2 prefill item counts
    // Scratch
    half* xr;                      // [M, H] x_rot
    uint8_t* xr8;                  // [M, H] x_rot as E4M3 in fp8_pos order (EXLR8_FP8A)
    float* xr8s;                   // [M] its per-token scales
    int pre_rot;                   // xr8 / xr8s come rotated from the caller (the SP gather); no fp16 x_rot
    int rows_out;                  // FC2 leaves rotated int8 rows for the caller's finalize; no combine
    float* fc1_part;               // [split groups, UPR, splits, 16, 256] partial gate/up sums
    int* fc1_ctr;                  // [split groups, UPR] split arrival counters
    half* h;                       // [M * R, I] FC1 output, rows in sorted-route order
    uint8_t* h8;                   // EXLR8_FP8H: prefill items' h as E4M3 [M * R, I], fp8_pos order within each 32,
                                   //   then [M * R, UPR] float scales
    half* y;                       // [M * R, H] per-route down output, rows indexed by route (EXLR8_Y8: int8
                                   //   rows, then [M * R, HB] float scales)
    int* y_ctr;                    // [M, HB] route arrival counters
    void* out;                     // [M, H], fp32 or bf16 (out_bf16)
    int out_bf16;
    int fp8;                       // M >= EXLR8_FP8_MIN_M: prefill items take the FP8 paths (FP8A, FP8H)
    int item_mb;                   // 16-row blocks per prefill item, at most: MAXMB, or 4 from EXLR8_MB4_MIN_M
    int ks_max;                    // K splits of a decode item, at most
    int slots;                     // decode FC1 blocks resident at once (SMs x blocks per SM)
};

__device__ __forceinline__ int route_expert(const Params& p, int r)
{
    const int m = r / p.R, s = r % p.R;
    if (s == TOPK) return E;
    const int e = (int) p.ids[m * TOPK + s];
    return e < 0 ? 0 : e;          // no route: expert 0 at weight 0 adds nothing
}

__device__ __forceinline__ float route_weight(const Params& p, int r)
{
    const int m = r / p.R, s = r % p.R;
    if (s == TOPK) return 1.f;
    return p.ids[m * TOPK + s] < 0 ? 0.f : p.rw[m * TOPK + s] * p.routed_scale;
}

__device__ __forceinline__ const uint32_t* unit_words(const Params& p, int e, int proj, int j)
{
    return (const uint32_t*) (p.slab + p.off[(e * 3 + proj) * UPR + j]);
}

// ------------------------------------------------------------------------------------------------
// x_rot, once per token: one warp per (token, 128-block).

__global__ void __launch_bounds__(THREADS) rot_kernel(Params p)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int unit = blockIdx.x * WARPS + warp;
    if (unit >= p.M * HB) return;
    const int m = unit / HB, k0 = (unit % HB) * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const float x = __bfloat162float(p.x[(size_t) m * H + k0 + j]);
        v[j] = __half2float(p.signs[k0 + j]) < 0.f ? -x : x;
    }
    had128(v[0], v[1], v[2], v[3], lane);
    half* o = p.xr + (size_t) m * H + k0;
    *((half2*) o) = __floats2half2_rn(v[0], v[1]);
    *((half2*) (o + 2)) = __floats2half2_rn(v[2], v[3]);
}

// x_rot as fp16 and, for the FP8 prefill, as E4M3 with one scale per token (amax / 448): one block per token.
__global__ void __launch_bounds__(THREADS) rot8_kernel(Params p)
{
    constexpr int CPW = HB / WARPS;                      // 128-blocks per warp
    static_assert(HB % WARPS == 0, "hidden blocks per warp");
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, m = blockIdx.x;
    __shared__ float wmax[WARPS];
    float v[CPW][4];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < CPW; ++i)
    {
        const int k0 = (warp + i * WARPS) * 128 + lane * 4;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float x = __bfloat162float(p.x[(size_t) m * H + k0 + j]);
            v[i][j] = __half2float(p.signs[k0 + j]) < 0.f ? -x : x;
        }
        had128(v[i][0], v[i][1], v[i][2], v[i][3], lane);
        if (p.xr)
        {
            half* o = p.xr + (size_t) m * H + k0;
            *((half2*) o) = __floats2half2_rn(v[i][0], v[i][1]);
            *((half2*) (o + 2)) = __floats2half2_rn(v[i][2], v[i][3]);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            v[i][j] = rh(v[i][j]);
            amax = fmaxf(amax, fabsf(v[i][j]));
        }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) wmax[warp] = amax;
    __syncthreads();
    amax = 0.f;
    #pragma unroll
    for (int w = 0; w < WARPS; ++w) amax = fmaxf(amax, wmax[w]);
    const float scale = amax > 0.f ? amax / 448.f : 1.f, inv = 1.f / scale;
    uint8_t* o8 = p.xr8 + (size_t) m * H;
    #pragma unroll
    for (int i = 0; i < CPW; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int k = (warp + i * WARPS) * 128 + lane * 4 + j;
            o8[fp8_pos(k)] = __nv_cvt_float_to_fp8(v[i][j] * inv, __NV_SATFINITE, __NV_E4M3);
        }
    if (threadIdx.x == 0) p.xr8s[m] = scale;
}

// ------------------------------------------------------------------------------------------------
// Plan: one block of PLAN_THREADS threads, thread e owning expert e.

__device__ __forceinline__ int block_scan(int v, int* sh, int& total)
{
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        const int n = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += n;
    }
    if (lane == 31) sh[warp] = x;
    __syncthreads();
    int base = 0;
    total = 0;
    #pragma unroll
    for (int w = 0; w < PLAN_WARPS; ++w)
    {
        base += w < warp ? sh[w] : 0;
        total += sh[w];
    }
    __syncthreads();
    return base + x - v;
}

__global__ void __launch_bounds__(PLAN_THREADS) plan_kernel(Params p)
{
    __shared__ int cnt[PLAN_THREADS];
    __shared__ int fill[PLAN_THREADS];
    __shared__ int starts[PLAN_THREADS];
    __shared__ int scan[PLAN_WARPS];
    const int e = threadIdx.x;
    const int routes = p.M * p.R;
    cnt[e] = 0;
    fill[e] = 0;
    __syncthreads();
    for (int r = e; r < routes; r += PLAN_THREADS) atomicAdd(&cnt[route_expert(p, r)], 1);
    __syncthreads();
    const int c = e < NE ? cnt[e] : 0;
    int n_routes;
    const int start = block_scan(c, scan, n_routes);
    starts[e] = start;
    if (e < NE) p.count[e] = c;
    __syncthreads();
    // Placement inside an expert by arrival: every row is computed on its own, so row order inside an
    // expert changes no result.
    for (int r = e; r < routes; r += PLAN_THREADS)
    {
        const int x = route_expert(p, r);
        p.order[starts[x] + atomicAdd(&fill[x], 1)] = r;
    }
    // Row blocks of this expert. Up to 16 rows: one decode item (it splits K). More: the rows spread evenly
    // over ceil(c / 64) prefill blocks, so no block is a small tail that re-reads the expert's weights.
    // Calls of up to 16 tokens have no prefill list (pre_blocks_max), since an expert takes at most one route per
    // token. Rows without routes break that: their id -1 routes all land on expert 0, which can then pass 16
    // routes, so there it takes one decode item per 16 routes. FP8 calls have no decode list: every row of the call
    // then takes the same FP8 arithmetic, and the decode launches go away.
    const bool small = p.M <= 16;
    const int dec_rows = p.fp8 ? 0 : 16;
    const int n_dec = c == 0 || c > dec_rows ? (small ? (c + 15) / 16 : 0) : 1;
    const int n_pre = c > dec_rows && !small ? (c + p.item_mb * 16 - 1) / (p.item_mb * 16) : 0;
    // FC2 takes at most 64 rows an item (its accumulators for more spill at two blocks per SM), so an FC1 item of
    // more than 64 rows becomes two FC2 items.
    constexpr int FC2ROWS = EXLR8_FC2MB * 16;
    int n_pre2 = 0;
    for (int b = 0; b < n_pre; ++b) n_pre2 += (b + 1) * c / n_pre - b * c / n_pre > FC2ROWS ? 2 : 1;
    int total_dec, total_pre, total_pre2;
    const int dec0 = block_scan(n_dec, scan, total_dec);
    const int pre0 = block_scan(n_pre, scan, total_pre);
    int pre2 = block_scan(n_pre2, scan, total_pre2);
    // Split count: the one whose blocks fill the resident slots best (whole waves), each extra split costing 2%
    // for its partial-sum traffic. Every thread computes the same answer.
    const int tiles = total_dec * UPR;
    int ks = 1;
    if (tiles)
    {
        float best = -1.f;
        for (int k = 1; k <= p.ks_max; ++k)
        {
            const int blocks = tiles * k;
            const int waves = (blocks + p.slots - 1) / p.slots;
            const float score = (float) blocks / (float) (waves * p.slots) - 0.02f * (k - 1);
            if (score > best + 1e-6f) { best = score; ks = k; }
        }
    }
    for (int b = 0; b < n_pre; ++b)
    {
        const int r0 = b * c / n_pre;
        const int rows = (b + 1) * c / n_pre - r0;
        const int mbp = rows <= 32 ? 2 : rows <= 64 || p.item_mb == 4 ? 4 : 8;
        p.fc1p_items[pre0 + b] = make_int4(e, start + r0, rows, (mbp << 8) | 1);
        auto fc2_mb = [](int n) { return n <= 32 ? 2 : n <= 64 ? 4 : 8; };
        if (rows > FC2ROWS)
        {
            const int h = rows / 2;
            p.fc2p_items[pre2++] = make_int4(e, start + r0, h, fc2_mb(h));
            p.fc2p_items[pre2++] = make_int4(e, start + r0 + h, rows - h, fc2_mb(rows - h));
        }
        else
            p.fc2p_items[pre2++] = make_int4(e, start + r0, rows, fc2_mb(rows));
    }
    for (int d = 0; d < n_dec; ++d)
    {
        const int g = dec0 + d, rows = min(16, c - 16 * d);
        for (int k = 0; k < ks; ++k)
            p.fc1_items[g * ks + k] = make_int4(e, start + 16 * d, rows, (g << 12) | (1 << 8) | (k << 4) | ks);
        p.fc2_items[g] = make_int4(e, start + 16 * d, rows, 1);
    }
    if (e == 0)
    {
        p.n_items[0] = total_dec * ks;
        p.n_items[1] = total_pre;
        p.n_items[2] = total_dec;
        p.n_items[3] = total_pre2;
    }
}

// ------------------------------------------------------------------------------------------------
// FC1 helpers

// One K stage of the item's A operand in registers: warp w holds rows w, w + 8, ... (RPW = MB * 2 rows), each
// lane 4 consecutive x_rot values per row.
template <int MB>
struct XStage
{
    static constexpr int RPW = MB * 16 / WARPS;
    uint2 x[RPW];
};

template <int MB>
__device__ __forceinline__ void load_x(const Params& p, const int* tok, int k0, int lane, XStage<MB>& xs)
{
    #pragma unroll
    for (int i = 0; i < XStage<MB>::RPW; ++i)
    {
        if (tok[i] < 0) { xs.x[i] = make_uint2(0u, 0u); continue; }
        if (!p.pre_rot) { xs.x[i] = *((const uint2*) (p.xr + (size_t) tok[i] * H + k0 + lane * 4)); continue; }
        // Only the caller's E4M3 x_rot exists: these 4 channels' bytes sit at their fp8_pos positions.
        const uint8_t* row = p.xr8 + (size_t) tok[i] * H;
        const float sc = p.xr8s[tok[i]];
        half v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) row[fp8_pos(k0 + lane * 4 + q)], __NV_E4M3);
            v[q] = __float2half(__half2float(__half(hr)) * sc);
        }
        xs.x[i] = make_uint2(*((uint32_t*) &v[0]), *((uint32_t*) &v[2]));
    }
}

template <int MB>
__device__ __forceinline__ void store_x(const XStage<MB>& xs, half* A, int warp, int lane)
{
    #pragma unroll
    for (int i = 0; i < XStage<MB>::RPW; ++i)
        *((uint2*) (A + (warp + i * WARPS) * ASTRIDE + lane * 4)) = xs.x[i];
}

// A fragment of 16-row block mbk, k16 step kk, from an fp16 tile with row stride `stride`, in one ldmatrix.x4.
__device__ __forceinline__ FragA load_a(const half* As, int mbk, int kk, int lane, int stride = ASTRIDE)
{
    const int r = mbk * 16 + (lane & 15);
    const int c = kk * 16 + (lane >> 4) * 8;
    const uint32_t addr = (uint32_t) __cvta_generic_to_shared(As + r * stride + c);
    FragA a;
    uint32_t* v = reinterpret_cast<uint32_t*>(&a);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
    return a;
}

constexpr int ASTRIDE8 = KSTAGE + 16;    // FP8 A tile row stride (bytes), against bank conflicts

// FP8 A fragment of 16-row block mbk, 32-wide K step kk: the same ldmatrix.x4 as load_a, reading byte pairs as b16.
__device__ __forceinline__ FragA8 load_a8(const uint8_t* As, int mbk, int kk, int lane)
{
    const int r = mbk * 16 + (lane & 15);
    const int c = kk * 32 + (lane >> 4) * 16;
    const uint32_t addr = (uint32_t) __cvta_generic_to_shared(As + r * ASTRIDE8 + c);
    FragA8 a;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a.v[0]), "=r"(a.v[1]), "=r"(a.v[2]), "=r"(a.v[3]) : "r"(addr));
    return a;
}

// FP8 prefill stage of x_rot: 4 bytes per lane per row, already in fp8_pos order.
template <int MB>
struct XStage8
{
    static constexpr int RPW = MB * 16 / WARPS;
    uint32_t x[RPW];
};

template <int MB>
__device__ __forceinline__ void load_x8(const Params& p, const int* tok, int k0, int lane, XStage8<MB>& xs)
{
    #pragma unroll
    for (int i = 0; i < XStage8<MB>::RPW; ++i)
        xs.x[i] = tok[i] < 0 ? 0u : *((const uint32_t*) (p.xr8 + (size_t) tok[i] * H + k0 + lane * 4));
}

template <int MB>
__device__ __forceinline__ void store_x8(const XStage8<MB>& xs, uint8_t* A, int warp, int lane)
{
    #pragma unroll
    for (int i = 0; i < XStage8<MB>::RPW; ++i)
        *((uint32_t*) (A + (warp + i * WARPS) * ASTRIDE8 + lane * 4)) = xs.x[i];
}

// After stage st the accumulators hold (sum so far) / blk[st]: move them to the next stage's scale, or after the
// item's last stage to 1. x_rot carries no block scale, so this is where gate's and up's own scales enter.
template <int MB>
__device__ __forceinline__ void rescale(FragC (&acc)[2][MB][2], const float* gb, const float* ub, int kb, bool last)
{
    const float rg = gb[kb] / (last ? 1.f : gb[kb + 1]);
    const float ru = ub[kb] / (last ? 1.f : ub[kb + 1]);
    #pragma unroll
    for (int mbk = 0; mbk < MB; ++mbk)
        #pragma unroll
        for (int f = 0; f < 2; ++f)
            #pragma unroll
            for (int q = 0; q < 4; ++q) { acc[0][mbk][f].v[q] *= rg; acc[1][mbk][f].v[q] *= ru; }
}

// The stage's 8 k16 rows of this warp's gate and up tiles. Within a unit, tile (k16, c) is at word (8 k16 + c) TW.
template <int K>
__device__ __forceinline__ void load_w(const uint32_t* gt, const uint32_t* ut, int k0, int warp, int lane, bool loads,
                                       uint32_t (&wg)[KSTAGE / 16], uint32_t (&wu)[KSTAGE / 16])
{
    constexpr int TW = 8 * K;
    #pragma unroll
    for (int kk = 0; kk < KSTAGE / 16; ++kk)
    {
        const size_t off = (size_t) ((k0 / 16 + kk) * 8 + warp) * TW + lane;
        wg[kk] = loads ? __ldcs(gt + off) : 0u;
        wu[kk] = loads ? __ldcs(ut + off) : 0u;
    }
}

// Prefill items: A double-buffered, one barrier per stage, x one stage ahead in registers, weights WD ahead.
template <int K, int MB, int WD>
__device__ __forceinline__ void fc1_gemm(const Params& p, const int* tok, int e, int j, int k_begin, int k_end,
                                         half* A, int abuf, FragC (&acc)[2][MB][2], int warp, int lane)
{
    const uint32_t* gt = unit_words(p, e, 0, j);
    const uint32_t* ut = unit_words(p, e, 1, j);
    const float* gb = p.blk + (e * 3 + 0) * HB;
    const float* ub = p.blk + (e * 3 + 1) * HB;
    const TileLane<K> tl(lane);
    const bool loads = lane < 8 * K;
    const int nst = (k_end - k_begin) / KSTAGE;
    uint32_t wg[WD][KSTAGE / 16], wu[WD][KSTAGE / 16];
    #pragma unroll
    for (int d = 0; d < WD; ++d)
        if (d < nst) load_w<K>(gt, ut, k_begin + d * KSTAGE, warp, lane, loads, wg[d], wu[d]);
    XStage<MB> xs;
    load_x<MB>(p, tok, k_begin, lane, xs);
    store_x<MB>(xs, A, warp, lane);
    if (nst > 1) load_x<MB>(p, tok, k_begin + KSTAGE, lane, xs);
    __syncthreads();
    for (int st = 0; st < nst; ++st)
    {
        const half* Ac = A + (st & 1) * abuf;
        #pragma unroll
        for (int kk = 0; kk < KSTAGE / 16; ++kk)
        {
            FragB g0, g1, u0, u1;
            decode_tile<K>(wg[0][kk], tl, g0, g1);
            decode_tile<K>(wu[0][kk], tl, u0, u1);
            #pragma unroll
            for (int mbk = 0; mbk < MB; ++mbk)
            {
                const FragA a = load_a(Ac, mbk, kk, lane);
                mma(a, g0, acc[0][mbk][0]);
                mma(a, g1, acc[0][mbk][1]);
                mma(a, u0, acc[1][mbk][0]);
                mma(a, u1, acc[1][mbk][1]);
            }
        }
        rescale<MB>(acc, gb, ub, k_begin / KSTAGE + st, st + 1 == nst);
        if (st + 1 < nst)
        {
            store_x<MB>(xs, A + ((st + 1) & 1) * abuf, warp, lane);
            if (st + 2 < nst) load_x<MB>(p, tok, k_begin + (st + 2) * KSTAGE, lane, xs);
        }
        #pragma unroll
        for (int d = 0; d + 1 < WD; ++d)
            #pragma unroll
            for (int kk = 0; kk < KSTAGE / 16; ++kk) { wg[d][kk] = wg[d + 1][kk]; wu[d][kk] = wu[d + 1][kk]; }
        if (st + WD < nst) load_w<K>(gt, ut, k_begin + (st + WD) * KSTAGE, warp, lane, loads, wg[WD - 1], wu[WD - 1]);
        __syncthreads();
    }
}

// FP8 prefill items: as fc1_gemm, with x_rot as E4M3 and the decoded weights converted to E4M3 (exact: the codebook
// values are E4M3 already). The token scales multiply the accumulators in the epilogue.
template <int K, int MB, int WD>
__device__ __forceinline__ void fc1_gemm8(const Params& p, const int* tok, int e, int j, int k_begin, int k_end,
                                          uint8_t* A, int abuf, FragC (&acc)[2][MB][2], int warp, int lane)
{
    const uint32_t* gt = unit_words(p, e, 0, j);
    const uint32_t* ut = unit_words(p, e, 1, j);
    const float* gb = p.blk + (e * 3 + 0) * HB;
    const float* ub = p.blk + (e * 3 + 1) * HB;
    const TileLane<K> tl(lane);
    const bool loads = lane < 8 * K;
    const int nst = (k_end - k_begin) / KSTAGE;
    uint32_t wg[WD][KSTAGE / 16], wu[WD][KSTAGE / 16];
    #pragma unroll
    for (int d = 0; d < WD; ++d)
        if (d < nst) load_w<K>(gt, ut, k_begin + d * KSTAGE, warp, lane, loads, wg[d], wu[d]);
    XStage8<MB> xs;
    load_x8<MB>(p, tok, k_begin, lane, xs);
    store_x8<MB>(xs, A, warp, lane);
    if (nst > 1) load_x8<MB>(p, tok, k_begin + KSTAGE, lane, xs);
    __syncthreads();
    for (int st = 0; st < nst; ++st)
    {
        const uint8_t* Ac = A + (st & 1) * abuf;
        #pragma unroll
        for (int kk = 0; kk < KSTAGE / 32; ++kk)
        {
            FragB8 bg0, bg1, bu0, bu1;
            if (EXLR8_FC1X >= 2)
            {
                bg0 = FragB8{{wg[0][2 * kk], wg[0][2 * kk + 1]}};
                bg1 = bg0;
                bu0 = FragB8{{wu[0][2 * kk], wu[0][2 * kk + 1]}};
                bu1 = bu0;
            }
            else
            {
                uint32_t g0, g1, h0, h1, u0, u1, v0, v1;
                decode_tile<K>(wg[0][2 * kk], tl, g0, g1);
                decode_tile<K>(wg[0][2 * kk + 1], tl, h0, h1);
                decode_tile<K>(wu[0][2 * kk], tl, u0, u1);
                decode_tile<K>(wu[0][2 * kk + 1], tl, v0, v1);
                bg0 = FragB8{{g0, h0}};
                bg1 = FragB8{{g1, h1}};
                bu0 = FragB8{{u0, v0}};
                bu1 = FragB8{{u1, v1}};
            }
            #pragma unroll
            for (int mbk = 0; mbk < MB; ++mbk)
            {
                const FragA8 a = load_a8(Ac, mbk, kk, lane);
                if (EXLR8_FC1X >= 3)
                {
                    acc[0][mbk][0].v[0] += __uint_as_float(a.v[0] ^ bg0.v[0] ^ bu0.v[1]);
                    continue;
                }
                mma8(a, bg0, acc[0][mbk][0]);
                mma8(a, bg1, acc[0][mbk][1]);
                mma8(a, bu0, acc[1][mbk][0]);
                mma8(a, bu1, acc[1][mbk][1]);
            }
        }
        rescale<MB>(acc, gb, ub, k_begin / KSTAGE + st, st + 1 == nst);
        if (st + 1 < nst)
        {
            store_x8<MB>(xs, A + ((st + 1) & 1) * abuf, warp, lane);
            if (st + 2 < nst) load_x8<MB>(p, tok, k_begin + (st + 2) * KSTAGE, lane, xs);
        }
        #pragma unroll
        for (int d = 0; d + 1 < WD; ++d)
            #pragma unroll
            for (int kk = 0; kk < KSTAGE / 16; ++kk) { wg[d][kk] = wg[d + 1][kk]; wu[d][kk] = wu[d + 1][kk]; }
        if (st + WD < nst) load_w<K>(gt, ut, k_begin + (st + WD) * KSTAGE, warp, lane, loads, wg[WD - 1], wu[WD - 1]);
        __syncthreads();
    }
}

// The pipelined FP8 prefill (EXLR8_PIPE). Each stage holds the item's E4M3 x_rot rows, 16-byte chunks XOR-swizzled by
// row so ldmatrix's eight rows hit eight different banks, and the gate and up words of the stage's 64 tiles, which
// the v2 layout stores contiguously: tiles (k16, c) of a unit are at (8 k16 + c) TW words.
// 128-row items take 3 stages at one block per SM; 64-row items 2 stages, so two blocks fit an SM.
constexpr int PW_BYTES = 64 * 8 * 4 * 4;                       // one projection's stage at K4, the widest
template <int MB>
struct Pipe
{
    static constexpr int STAGES = MB >= 8 ? 3 : 2;
    static constexpr int A_BYTES = MB * 16 * KSTAGE;
    static constexpr int STAGE_BYTES = A_BYTES + 2 * PW_BYTES;
    static constexpr int BYTES = STAGES * STAGE_BYTES;
};

__device__ __forceinline__ FragA8 load_a8_swz(const uint8_t* As, int mbk, int kk, int lane)
{
    const int r = mbk * 16 + (lane & 15);
    const int chunk = kk * 2 + (lane >> 4);
    const uint32_t addr = (uint32_t) __cvta_generic_to_shared(As + r * KSTAGE + ((chunk ^ (r & 7)) << 4));
    FragA8 a;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a.v[0]), "=r"(a.v[1]), "=r"(a.v[2]), "=r"(a.v[3]) : "r"(addr));
    return a;
}

template <int K, int MB>
__device__ __forceinline__ void fc1_gemm8_pipe(const Params& p, const int* tokS, int e, int j, uint8_t* smem,
                                               FragC (&acc)[2][MB][2], int warp, int lane)
{
    constexpr int TW = 8 * K;
    constexpr int WSTAGE = 64 * TW * 4;                        // bytes of one projection's stage
    constexpr int NST = H / KSTAGE;
    constexpr int PSTAGES = Pipe<MB>::STAGES, PA_BYTES = Pipe<MB>::A_BYTES, PSTAGE_BYTES = Pipe<MB>::STAGE_BYTES;
    const uint8_t* gt = (const uint8_t*) unit_words(p, e, 0, j);
    const uint8_t* ut = (const uint8_t*) unit_words(p, e, 1, j);
    const float* gb = p.blk + (e * 3 + 0) * HB;
    const float* ub = p.blk + (e * 3 + 1) * HB;
    const TileLane<K> tl(lane);
    auto issue = [&](int st)
    {
        uint8_t* A = smem + (st % PSTAGES) * PSTAGE_BYTES;
        uint8_t* W = A + PA_BYTES;
        const int k0 = st * KSTAGE;
        for (int i = threadIdx.x; i < MB * 16 * 8; i += THREADS)
        {
            const int r = i >> 3, c = i & 7, t = tokS[r];
            cp_async16(A + r * KSTAGE + ((c ^ (r & 7)) << 4), p.xr8 + (size_t) (t < 0 ? 0 : t) * H + k0 + c * 16,
                       t < 0 ? 0 : 16);
        }
        for (int i = threadIdx.x; i < 2 * WSTAGE / 16; i += THREADS)
        {
            const int pj = i / (WSTAGE / 16), o = (i % (WSTAGE / 16)) * 16;
            cp_async16(W + pj * PW_BYTES + o, (pj ? ut : gt) + (size_t) st * WSTAGE + o, 16);
        }
        cp_async_commit();
    };
    for (int st = 0; st < PSTAGES - 1; ++st) issue(st);
    for (int st = 0; st < NST; ++st)
    {
        if (PSTAGES > 2 && st + 1 < NST) cp_async_wait<PSTAGES - 2>();
        else cp_async_wait<0>();
        __syncthreads();                       // stage st landed for every thread; the oldest buffer is free
        if (st + PSTAGES - 1 < NST) issue(st + PSTAGES - 1);
        const uint8_t* A = smem + (st % PSTAGES) * PSTAGE_BYTES;
        const uint32_t* Wg = (const uint32_t*) (A + PA_BYTES);
        const uint32_t* Wu = (const uint32_t*) (A + PA_BYTES + PW_BYTES);
        #pragma unroll
        for (int kk = 0; kk < KSTAGE / 32; ++kk)
        {
            const int t0 = ((2 * kk) * 8 + warp) * TW + lane, t1 = ((2 * kk + 1) * 8 + warp) * TW + lane;
            FragB8 bg0, bg1, bu0, bu1;
            if constexpr (EXLR8_FC1X >= 2)
            {
                bg0 = FragB8{{Wg[t0], Wg[t1]}};
                bg1 = FragB8{{Wg[t1], Wg[t0]}};
                bu0 = FragB8{{Wu[t0], Wu[t1]}};
                bu1 = FragB8{{Wu[t1], Wu[t0]}};
            }
            else
            {
                uint32_t g0, g1, h0, h1, u0, u1, v0, v1;
                decode_tile<K>(Wg[t0], tl, g0, g1);
                decode_tile<K>(Wg[t1], tl, h0, h1);
                decode_tile<K>(Wu[t0], tl, u0, u1);
                decode_tile<K>(Wu[t1], tl, v0, v1);
                bg0 = FragB8{{g0, h0}};
                bg1 = FragB8{{g1, h1}};
                bu0 = FragB8{{u0, v0}};
                bu1 = FragB8{{u1, v1}};
            }
            #pragma unroll
            for (int mbk = 0; mbk < MB; ++mbk)
            {
                const FragA8 a = load_a8_swz(A, mbk, kk, lane);
                if constexpr (EXLR8_FC1X >= 3)
                {
                    acc[0][mbk][0].v[0] += __uint_as_float(a.v[0] ^ bg0.v[0] ^ bu0.v[1] ^ bg1.v[0] ^ bu1.v[1]);
                    continue;
                }
                mma8(a, bg0, acc[0][mbk][0]);
                mma8(a, bg1, acc[0][mbk][1]);
                mma8(a, bu0, acc[1][mbk][0]);
                mma8(a, bu1, acc[1][mbk][1]);
            }
        }
        rescale<MB>(acc, gb, ub, st, st + 1 == NST);
    }
    __syncthreads();                           // the epilogue's buffer overlays the stages
}

// Decode items (one 16-row block): two barriers per stage and an L2 prefetch EXLR8_DECPF stages ahead.
template <int K>
__device__ __forceinline__ void fc1_gemm_dec(const Params& p, const int* tok, int e, int j, int k_begin, int k_end,
                                             half* A, FragC (&acc)[2][1][2], int warp, int lane)
{
    constexpr int TW = 8 * K;
    const uint32_t* gt = unit_words(p, e, 0, j);
    const uint32_t* ut = unit_words(p, e, 1, j);
    const float* gb = p.blk + (e * 3 + 0) * HB;
    const float* ub = p.blk + (e * 3 + 1) * HB;
    const TileLane<K> tl(lane);
    const bool loads = lane < TW;
    auto prefetch_stage = [&](int k0)
    {
        if (k0 >= k_end || !loads || (lane & 7)) return;   // one lane per 32-byte sector
        #pragma unroll
        for (int kk = 0; kk < KSTAGE / 16; ++kk)
        {
            const size_t off = (size_t) ((k0 / 16 + kk) * 8 + warp) * TW + lane;
            prefetch_l2(gt + off);
            prefetch_l2(ut + off);
        }
    };
    uint32_t wg[KSTAGE / 16], wu[KSTAGE / 16];
    XStage<1> xs;
    load_w<K>(gt, ut, k_begin, warp, lane, loads, wg, wu);
    for (int d = 2; d <= EXLR8_DECPF; ++d) prefetch_stage(k_begin + d * KSTAGE);
    load_x<1>(p, tok, k_begin, lane, xs);
    for (int k0 = k_begin; k0 < k_end; k0 += KSTAGE)
    {
        uint32_t ng[KSTAGE / 16], nu[KSTAGE / 16];
        XStage<1> nx;
        if (EXLR8_DECPF > 1) prefetch_stage(k0 + (EXLR8_DECPF + 1) * KSTAGE);
        if (k0 + KSTAGE < k_end)
        {
            load_w<K>(gt, ut, k0 + KSTAGE, warp, lane, loads, ng, nu);
            load_x<1>(p, tok, k0 + KSTAGE, lane, nx);
        }
        __syncthreads();
        store_x<1>(xs, A, warp, lane);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KSTAGE / 16; ++kk)
        {
            FragB g0, g1, u0, u1;
            decode_tile<K>(wg[kk], tl, g0, g1);
            decode_tile<K>(wu[kk], tl, u0, u1);
            const FragA a = load_a(A, 0, kk, lane);
            mma(a, g0, acc[0][0][0]);
            mma(a, g1, acc[0][0][1]);
            mma(a, u0, acc[1][0][0]);
            mma(a, u1, acc[1][0][1]);
        }
        rescale<1>(acc, gb, ub, k0 / KSTAGE, k0 + KSTAGE >= k_end);
        if (k0 + KSTAGE < k_end)               // ng, nu and nx are unset after the last stage
        {
            #pragma unroll
            for (int kk = 0; kk < KSTAGE / 16; ++kk) { wg[kk] = ng[kk]; wu[kk] = nu[kk]; }
            xs = nx;
        }
    }
}

// FC1 epilogue for one row: gate and up sums for one unit's 128 channels (fp32 in shared memory) -> h row.
__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ void fc1_epilogue_row(const Params& p, int e, int j, const T* G, const T* U,
                                                 half* hrow, int lane, uint8_t* h8row = nullptr, float* h8scale = nullptr)
{
    const half* gsvh = p.ch + (size_t) (e * 3 + 0) * I + j * 128 + lane * 4;
    const half* usvh = p.ch + (size_t) (e * 3 + 1) * I + j * 128 + lane * 4;
    const half* dsuh = p.ch + (size_t) (e * 3 + 2) * I + j * 128 + lane * 4;
    float g[4], u[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        g[q] = rh(to_f(G[lane * 4 + q]));
        u[q] = rh(to_f(U[lane * 4 + q]));
    }
    had128(g[0], g[1], g[2], g[3], lane);
    had128(u[0], u[1], u[2], u[3], lane);
    const float lim = p.swiglu_limit;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        const float gs = fminf(rh(rh(g[q]) * __half2float(gsvh[q])), lim);
        const float us = fminf(fmaxf(rh(rh(u[q]) * __half2float(usvh[q])), -lim), lim);
        const float a = rh(silu_h(gs) * us);
        v[q] = rh(a * __half2float(dsuh[q]));
    }
    had128(v[0], v[1], v[2], v[3], lane);
    if (h8row)
    {
        float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        #pragma unroll
        for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float scale = amax > 0.f ? amax / 448.f : 1.f, inv = 1.f / scale;
        uint32_t q[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) q[k] = __nv_cvt_float_to_fp8(v[k] * inv, __NV_SATFINITE, __NV_E4M3);
        // Lane l writes logical bytes 4l..4l+3 (fp8_pos order) as one word: physical k = 16 (l / 4) + 2 (l % 4) + {0, 1}
        // and the same + 8, held as byte pairs by lanes 4 (l / 4) + (l % 4) / 2 and that + 2.
        const uint32_t p01 = q[0] | (q[1] << 8), p23 = q[2] | (q[3] << 8);
        const int src = 4 * (lane >> 2) + ((lane & 3) >> 1);
        const uint32_t a01 = __shfl_sync(0xffffffffu, p01, src), a23 = __shfl_sync(0xffffffffu, p23, src);
        const uint32_t b01 = __shfl_sync(0xffffffffu, p01, src + 2), b23 = __shfl_sync(0xffffffffu, p23, src + 2);
        const bool odd = lane & 1;
        *((uint32_t*) (h8row + j * 128 + lane * 4)) = (odd ? a23 : a01) | ((odd ? b23 : b01) << 16);
        if (lane == 0) h8scale[j] = scale;
        return;
    }
    *((half2*) (hrow + j * 128 + lane * 4)) = __floats2half2_rn(v[0], v[1]);
    *((half2*) (hrow + j * 128 + lane * 4 + 2)) = __floats2half2_rn(v[2], v[3]);
}

template <int K, int MB, int MBMAX>
__device__ void fc1_item(const Params& p, int4 it, int j, char* smem)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int e = it.x, first = it.y, rows = it.z;
    const int ksi = (it.w >> 4) & 15, ks = it.w & 15;
    static_assert(MB <= MBMAX, "item larger than the launch's tiles");
    const int k_begin = (ksi * HB / ks) * KSTAGE, k_end = ((ksi + 1) * HB / ks) * KSTAGE;
    constexpr int ABUF = MBMAX * 16 * ASTRIDE;
    constexpr int ABUF8 = MBMAX * 16 * ASTRIDE8;
    half* A = (half*) smem;                                // two buffers of ABUF
    // [MB*16][256] gate | up, over the A tiles: fp32 for decode items (their split-K partials sum in it), fp16 for
    // prefill items, whose epilogue rounds to fp16 first anyway (and 8 row blocks of fp32 would not fit).
    float* red = (float*) smem;
    half* redh = (half*) smem;
    __shared__ int last;

    int tok[XStage<MB>::RPW];
    #pragma unroll
    for (int i = 0; i < XStage<MB>::RPW; ++i)
    {
        const int r = warp + i * WARPS;
        tok[i] = r < rows ? p.order[first + r] / p.R : -1;
    }
    FragC acc[2][MB][2];
    #pragma unroll
    for (int a = 0; a < 2; ++a)
        #pragma unroll
        for (int b = 0; b < MB; ++b)
            #pragma unroll
            for (int c = 0; c < 2; ++c)
                acc[a][b][c] = FragC{{0.f, 0.f, 0.f, 0.f}};
    const bool f8 = EXLR8_FP8A && MB > 1 && p.fp8;
    if constexpr (MB == 1) fc1_gemm_dec<K>(p, tok, e, j, k_begin, k_end, A, acc, warp, lane);
    else if (EXLR8_FP8A && EXLR8_PIPE && MB >= (MBMAX == 4 ? 4 : EXLR8_PIPE_MB) && f8)
    {
        // The row tokens follow the stages in dynamic shared memory (static arrays would repeat per instantiation).
        int* tokS = (int*) (smem + Pipe<MB>::BYTES);
        for (int r = threadIdx.x; r < MB * 16; r += THREADS) tokS[r] = r < rows ? p.order[first + r] / p.R : -1;
        __syncthreads();
        fc1_gemm8_pipe<K, MB>(p, tokS, e, j, (uint8_t*) smem, acc, warp, lane);
    }
    else if (EXLR8_FP8A && f8) fc1_gemm8<K, MB, 2>(p, tok, e, j, k_begin, k_end, (uint8_t*) smem, ABUF8, acc, warp, lane);
    else fc1_gemm<K, MB, 2>(p, tok, e, j, k_begin, k_end, A, ABUF, acc, warp, lane);
    // FP8 prefill: each row's token scale.
    float rs[MB][2];
    #pragma unroll
    for (int mbk = 0; mbk < MB; ++mbk)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
            const int r = mbk * 16 + (lane >> 2) + hh * 8;
            rs[mbk][hh] = f8 && r < rows ? p.xr8s[p.order[first + r] / p.R] : 1.f;
        }

    // Accumulators -> red[row][proj * 128 + col]. C fragment: c0, c1 at row lane/4, cols 2(lane%4) + 0/1;
    // c2, c3 at row lane/4 + 8.
    __syncthreads();
    #pragma unroll
    for (int pj = 0; pj < 2; ++pj)
        #pragma unroll
        for (int mbk = 0; mbk < MB; ++mbk)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int r = mbk * 16 + (lane >> 2);
                const int c = pj * 128 + warp * 16 + f * 8 + 2 * (lane & 3);
                const float v0 = acc[pj][mbk][f].v[0] * rs[mbk][0], v1 = acc[pj][mbk][f].v[1] * rs[mbk][0];
                const float v2 = acc[pj][mbk][f].v[2] * rs[mbk][1], v3 = acc[pj][mbk][f].v[3] * rs[mbk][1];
                if constexpr (MB == 1)
                {
                    red[r * 256 + c] = v0;
                    red[r * 256 + c + 1] = v1;
                    red[(r + 8) * 256 + c] = v2;
                    red[(r + 8) * 256 + c + 1] = v3;
                }
                else
                {
                    *((half2*) (redh + r * 256 + c)) = __floats2half2_rn(v0, v1);
                    *((half2*) (redh + (r + 8) * 256 + c)) = __floats2half2_rn(v2, v3);
                }
            }
    __syncthreads();

    if (ks > 1)
    {
        // Split K: publish this split's partial, the last arrival sums all splits in order.
        const int item_group = it.w >> 12;
        float* part = p.fc1_part + ((size_t) item_group * ks * UPR + (size_t) j * ks + ksi) * (16 * 256);
        for (int o = threadIdx.x; o < rows * 256; o += THREADS) part[o] = red[o];
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int* ctr = p.fc1_ctr + item_group * UPR + j;
            last = atomicAdd(ctr, 1) == ks - 1;
            if (last) *ctr = 0;
        }
        __syncthreads();
        if (!last) return;
        __threadfence();
        const float* base = p.fc1_part + ((size_t) item_group * ks * UPR + (size_t) j * ks) * (16 * 256);
        for (int o = threadIdx.x; o < rows * 256; o += THREADS)
        {
            float s = 0.f;
            for (int q = 0; q < ks; ++q) s += __ldcg(base + (size_t) q * 16 * 256 + o);
            red[o] = s;
        }
        __syncthreads();
    }
    for (int r = warp; r < rows; r += WARPS)
    {
        if constexpr (MB == 1)
            fc1_epilogue_row(p, e, j, red + r * 256, red + r * 256 + 128, p.h + (size_t) (first + r) * I, lane);
        else if (EXLR8_FP8H && p.fp8)
            fc1_epilogue_row(p, e, j, redh + r * 256, redh + r * 256 + 128, p.h + (size_t) (first + r) * I, lane,
                             p.h8 + (size_t) (first + r) * I,
                             (float*) (p.h8 + (size_t) p.M * p.R * I) + (size_t) (first + r) * UPR);
        else
            fc1_epilogue_row(p, e, j, redh + r * 256, redh + r * 256 + 128, p.h + (size_t) (first + r) * I, lane);
    }
}

template <int K, int MBMAX>
__device__ __forceinline__ void fc1_prefill(const Params& p, int4 it, int j, int mb, char* smem)
{
    if (mb == 2) fc1_item<K, 2, MBMAX>(p, it, j, smem);
    else if (MBMAX == 4 || mb == 4) fc1_item<K, 4, MBMAX>(p, it, j, smem);
    else fc1_item<K, MBMAX, MBMAX>(p, it, j, smem);
}

// Grid: one block per (item, owned unit).
template <int MBMAX>
__global__ void __launch_bounds__(THREADS, MBMAX == 1 ? FC1_DEC_BLOCKS : MBMAX == 4 ? 2 : 1) fc1_kernel(Params p)
{
    extern __shared__ __align__(16) char smem[];
    const int item = blockIdx.x / UPR, j = blockIdx.x % UPR;
    if (item >= p.n_items[MBMAX == 1 ? 0 : 1]) return;
    const int4 it = (MBMAX == 1 ? p.fc1_items : p.fc1p_items)[item];
    const int mb = (it.w >> 8) & 15;
    const int K = p.bits[it.x * 3];
    if constexpr (MBMAX == 1)
    {
        if (K == 2) fc1_item<2, 1, 1>(p, it, j, smem);
        else if (K == 3) fc1_item<3, 1, 1>(p, it, j, smem);
        else fc1_item<4, 1, 1>(p, it, j, smem);
    }
    else
    {
        if (K == 2) fc1_prefill<2, MBMAX>(p, it, j, mb, smem);
        else if (K == 3) fc1_prefill<3, MBMAX>(p, it, j, mb, smem);
        else fc1_prefill<4, MBMAX>(p, it, j, mb, smem);
    }
}

// Dynamic shared memory per launch.
constexpr int cmax(int a, int b) { return a > b ? a : b; }
constexpr int fc1_smem(int mbmax)
{
    return cmax(cmax(2 * mbmax * 16 * ASTRIDE * 2, mbmax * 16 * 256 * (mbmax == 1 ? 4 : 2)),
                mbmax > 1 && EXLR8_FP8A && EXLR8_PIPE ? (mbmax >= 8 ? Pipe<8>::BYTES : Pipe<4>::BYTES) + mbmax * 16 * 4 : 0);
}
constexpr int fc2_smem_core(int mbmax)
{
    return cmax(mbmax == 1 ? 16 * HSTRIDE * 2 : 2 * mbmax * 16 * ASTRIDE * 2,
                mbmax * 16 * 128 * (mbmax == 1 ? 1 : EXLR8_FC2NTW) * 2);
}
constexpr int FC2_EP_ROWS = EXLR8_FC2MB * 16;  // prefill FC2 items' per-row routes and scales follow the core
constexpr int fc2_smem(int mbmax) { return fc2_smem_core(mbmax) + (mbmax > 1 ? FC2_EP_ROWS * 8 : 0); }

// ------------------------------------------------------------------------------------------------
// FC2: down projection of one expert and row block, NTW * 128 output columns, K = I (one stage per owned unit).
// Warp w owns column tile w of each 128-column chunk t. Within a unit, tile (b, r, c) is at word (64 b + 8 r + c) TW.

template <int K, int MB, int NTW>
__device__ void fc2_item(const Params& p, int4 it, int ot, char* smem)
{
    constexpr int TW = 8 * K;
    constexpr int OC = 128 * NTW;              // output columns of the block
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int e = it.x, first = it.y, rows = it.z;
    half* As = (half*) smem;
    half* red = (half*) smem;                  // [MB*16][OC] GEMM output (fp16), over the A tile
    const TileLane<K> tl(lane);
    const bool loads = lane < TW;
    // The epilogue's per-row route and scale, fetched now so their loads hide under the GEMM (the GEMM's barriers
    // order these writes before the epilogue's reads).
    int* ep_route = (int*) (smem + fc2_smem_core(EXLR8_FC2MB));
    float* ep_scale = (float*) (ep_route + FC2_EP_ROWS);
    if (MB > 1)
        for (int r = threadIdx.x; r < rows; r += THREADS)
        {
            const int route = p.order[first + r];
            ep_route[r] = route;
            ep_scale[r] = route_weight(p, route);
        }

    FragC acc[MB][NTW][2];
    #pragma unroll
    for (int b = 0; b < MB; ++b)
        #pragma unroll
        for (int t = 0; t < NTW; ++t)
            #pragma unroll
            for (int c = 0; c < 2; ++c)
                acc[b][t][c] = FragC{{0.f, 0.f, 0.f, 0.f}};

    auto load_stage = [&](int j, uint32_t (&wd)[NTW][KSTAGE / 16])
    {
        const uint32_t* dt = unit_words(p, e, 2, j);
        #pragma unroll
        for (int t = 0; t < NTW; ++t)
            #pragma unroll
            for (int kk = 0; kk < KSTAGE / 16; ++kk)
            {
                const int b = ot * NTW + t;
                wd[t][kk] = loads ? __ldcs(dt + (size_t) (64 * b + 8 * kk + warp) * TW + lane) : 0u;
            }
    };
    uint32_t wd[NTW][KSTAGE / 16];
    load_stage(0, wd);
    if (MB == 1 && loads && !(lane & 7))
        for (int j = 1; j < UPR; ++j)
        {
            const uint32_t* dt = unit_words(p, e, 2, j);
            #pragma unroll
            for (int t = 0; t < NTW; ++t)
                #pragma unroll
                for (int kk = 0; kk < KSTAGE / 16; ++kk)
                    prefetch_l2(dt + (size_t) (64 * (ot * NTW + t) + 8 * kk + warp) * TW + lane);
        }
    // Decode items read all of h (16 rows x I) at once, so its load latency is paid once rather than per stage.
    constexpr bool HALL = MB == 1;
    if (HALL)
    {
        for (int r = warp; r < 16; r += WARPS)
        {
            uint2 v[UPR];
            #pragma unroll
            for (int s = 0; s < UPR; ++s)
                v[s] = r < rows ? __ldcg((const uint2*) (p.h + (size_t) (first + r) * I + s * KSTAGE + lane * 4))
                                : make_uint2(0u, 0u);
            #pragma unroll
            for (int s = 0; s < UPR; ++s) *((uint2*) (As + r * HSTRIDE + s * KSTAGE + lane * 4)) = v[s];
        }
        __syncthreads();
    }
    // Prefill items: stage j's h rows go into buffer j & 1 by cp.async while stage j - 1 multiplies. Under EXLR8_FP8H
    // they are E4M3 rows with a scale per 128-channel chunk; per row and stage, hratio moves the accumulators from
    // this stage's scale to the next (1 after the last).
    constexpr bool H8C = EXLR8_FP8H && MB > 1;             // compiled in
    const bool H8 = H8C && p.fp8;
    constexpr int HBUF = MB * 16 * ASTRIDE;                    // halves; an E4M3 buffer uses part of one
    const float* hsc = (const float*) (p.h8 + (size_t) p.M * p.R * I);
    __shared__ float hratio[H8C ? MB * 16 : 1][UPR];
    if (H8)
    {
        for (int i = threadIdx.x; i < MB * 16 * UPR; i += THREADS)
        {
            const int r = i / UPR, st = i % UPR;
            const float* rs = hsc + (size_t) (first + r) * UPR;
            hratio[r][st] = r < rows ? rs[st] / (st + 1 < UPR ? rs[st + 1] : 1.f) : 1.f;
        }
        __syncthreads();
    }
    auto stage_h = [&](int j)
    {
        half* dst = As + (j & 1) * HBUF;
        for (int r = warp; r < MB * 16; r += WARPS)
        {
            const int rr = r < rows ? r : 0;
            const size_t row = (size_t) (first + rr) * I + j * KSTAGE;
            if (H8)
                cp_async4((uint8_t*) dst + r * ASTRIDE8 + lane * 4, p.h8 + row + lane * 4, r < rows ? 4 : 0);
            else
                cp_async8(dst + r * ASTRIDE + lane * 4, p.h + row + lane * 4, r < rows ? 8 : 0);
        }
        cp_async_commit();
    };
    if (!HALL) stage_h(0);
    for (int j = 0; j < UPR; ++j)
    {
        const int k0 = j * KSTAGE;
        uint32_t nd[NTW][KSTAGE / 16];
        if (j + 1 < UPR) load_stage(j + 1, nd);
        const half* Ac = As + (j & 1) * HBUF;
        if (!HALL)
        {
            cp_async_wait_all();
            __syncthreads();                   // stage j landed for every warp; stage j - 1's buffer is free
            if (j + 1 < UPR) stage_h(j + 1);
        }
        if (H8C && H8)
        {
            #pragma unroll
            for (int kk = 0; kk < KSTAGE / 32; ++kk)
            {
                FragA8 a[MB];
                #pragma unroll
                for (int mbk = 0; mbk < MB; ++mbk) a[mbk] = load_a8((const uint8_t*) Ac, mbk, kk, lane);
                #pragma unroll
                for (int t = 0; t < NTW; ++t)
                {
                    uint32_t d0, d1, e0, e1;
                    decode_tile<K>(wd[t][2 * kk], tl, d0, d1);
                    decode_tile<K>(wd[t][2 * kk + 1], tl, e0, e1);
                    const FragB8 b0{{d0, e0}};
                    const FragB8 b1{{d1, e1}};
                    #pragma unroll
                    for (int mbk = 0; mbk < MB; ++mbk)
                    {
                        mma8(a[mbk], b0, acc[mbk][t][0]);
                        mma8(a[mbk], b1, acc[mbk][t][1]);
                    }
                }
            }
            #pragma unroll
            for (int mbk = 0; mbk < MB; ++mbk)
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                {
                    const float ratio = hratio[mbk * 16 + (lane >> 2) + hh * 8][j];
                    #pragma unroll
                    for (int t = 0; t < NTW; ++t)
                        #pragma unroll
                        for (int f = 0; f < 2; ++f)
                        {
                            acc[mbk][t][f].v[2 * hh] *= ratio;
                            acc[mbk][t][f].v[2 * hh + 1] *= ratio;
                        }
                }
        }
        else
        #pragma unroll
        for (int kk = 0; kk < KSTAGE / 16; ++kk)
        {
            FragA a[MB];
            #pragma unroll
            for (int mbk = 0; mbk < MB; ++mbk)
                a[mbk] = HALL ? load_a(As + k0, mbk, kk, lane, HSTRIDE) : load_a(Ac, mbk, kk, lane);
            #pragma unroll
            for (int t = 0; t < NTW; ++t)
            {
                FragB d0, d1;
                if (MB > 1 && EXLR8_FC2X >= 2)
                {
                    h2u u;
                    u.u = wd[t][kk];
                    d0 = d1 = FragB{{u.h, u.h}};
                }
                else decode_tile<K>(wd[t][kk], tl, d0, d1);
                #pragma unroll
                for (int mbk = 0; mbk < MB; ++mbk)
                {
                    if (MB > 1 && EXLR8_FC2X >= 3)
                    {
                        acc[mbk][t][0].v[0] += __low2float(d0.v[0]) + __low2float(a[mbk].v[0]);
                        continue;
                    }
                    mma(a[mbk], d0, acc[mbk][t][0]);
                    mma(a[mbk], d1, acc[mbk][t][1]);
                }
            }
        }
        if (j + 1 < UPR)                       // nd is unset after the last stage
            #pragma unroll
            for (int t = 0; t < NTW; ++t)
                #pragma unroll
                for (int kk = 0; kk < KSTAGE / 16; ++kk) wd[t][kk] = nd[t][kk];
    }
    __syncthreads();
    #pragma unroll
    for (int mbk = 0; mbk < MB; ++mbk)
        #pragma unroll
        for (int t = 0; t < NTW; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int r = mbk * 16 + (lane >> 2);
                const int c = t * 128 + warp * 16 + f * 8 + 2 * (lane & 3);
                *((half2*) (red + r * OC + c)) = __floats2half2_rn(acc[mbk][t][f].v[0], acc[mbk][t][f].v[1]);
                *((half2*) (red + (r + 8) * OC + c)) = __floats2half2_rn(acc[mbk][t][f].v[2], acc[mbk][t][f].v[3]);
            }
    __syncthreads();

    if (MB > 1 && EXLR8_FC2X >= 1)
    {
        if (__heq(red[threadIdx.x], __float2half(12345.f))) p.y[0] = red[threadIdx.x];   // keeps the GEMM alive
        return;
    }
    // Per row and 128-column chunk: down block scale * routing weight -> y[route]. The output Hadamard and signs
    // are the same for every expert of the layer, so they move to the per-token combine.
    const float* db = p.blk + (e * 3 + 2) * HB;
    uint32_t neg = 0;                          // bit 4 t + k: output sign of column k of this lane's chunk t
    #pragma unroll
    for (int t = 0; t < NTW; ++t)
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            neg |= (uint32_t) (__half2float(p.signs[H + ot * OC + t * 128 + lane * 4 + k]) < 0.f) << (4 * t + k);
    for (int r = warp; r < rows; r += WARPS)
    {
        const int route = MB > 1 ? ep_route[r] : p.order[first + r];
        const float w = MB > 1 ? ep_scale[r] : route_weight(p, route);
        #pragma unroll
        for (int t = 0; t < NTW; ++t)
        {
            const int col = ot * OC + t * 128 + lane * 4;
            const half2* src = (const half2*) (red + r * OC + t * 128 + lane * 4);
            const float s = db[ot * NTW + t] * w;
            float q[4] = {__low2float(src[0]) * s, __high2float(src[0]) * s, __low2float(src[1]) * s,
                          __high2float(src[1]) * s};
            if (p.rows_out)
            {
                // The output rotation per route instead of per token: linear, so the caller's sum is the same.
                had128(q[0], q[1], q[2], q[3], lane);
                #pragma unroll
                for (int k = 0; k < 4; ++k) q[k] = neg >> (4 * t + k) & 1 ? -q[k] : q[k];
            }
            if (EXLR8_Y8)
            {
                float amax = fmaxf(fmaxf(fabsf(q[0]), fabsf(q[1])), fmaxf(fabsf(q[2]), fabsf(q[3])));
                #pragma unroll
                for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                const float inv = amax > 0.f ? 127.f / amax : 0.f;
                uint32_t packed = 0;
                #pragma unroll
                for (int k = 0; k < 4; ++k) packed |= ((uint32_t) (uint8_t) (int8_t) __float2int_rn(q[k] * inv)) << (8 * k);
                uint8_t* y8 = (uint8_t*) p.y;
                *((uint32_t*) (y8 + (size_t) route * H + col)) = packed;
                if (lane == 0) ((float*) (y8 + (size_t) p.M * p.R * H))[route * HB + col / 128] = amax / 127.f;
                continue;
            }
            half* yr = p.y + (size_t) route * H + col;
            *((half2*) yr) = __floats2half2_rn(q[0], q[1]);
            *((half2*) (yr + 2)) = __floats2half2_rn(q[2], q[3]);
        }
    }
    if (p.rows_out) return;
    __threadfence();
    __syncthreads();
    // Combine: each route of this block arrives at its token's 128-column chunks; the last arrival at a chunk sums
    // the token's routes in route order.
    __shared__ int done[MAXMB * 16 * NTW];
    for (int i = threadIdx.x; i < rows * NTW; i += THREADS)
    {
        const int r = i / NTW, t = i % NTW;
        const int token = p.order[first + r] / p.R;
        int* ctr = p.y_ctr + token * HB + ot * NTW + t;
        const int prev = atomicAdd(ctr, 1);
        done[i] = prev == p.R - 1 ? token : -1;
        if (prev == p.R - 1) *ctr = 0;
    }
    __syncthreads();
    __threadfence();
    for (int i = warp; i < rows * NTW; i += WARPS)
    {
        const int token = done[i];
        if (token < 0) continue;
        const int col = ot * OC + (i % NTW) * 128 + lane * 4;
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k = 0; k < p.R; ++k)
        {
            if (EXLR8_Y8)
            {
                const uint8_t* y8 = (const uint8_t*) p.y;
                const uint32_t q = __ldcg((const uint32_t*) (y8 + (size_t) (token * p.R + k) * H + col));
                const float sc = __ldcg((const float*) (y8 + (size_t) p.M * p.R * H) + (token * p.R + k) * HB + col / 128);
                #pragma unroll
                for (int j = 0; j < 4; ++j) s[j] += sc * (float) (int8_t) (q >> (8 * j));
                continue;
            }
            const uint2 v = __ldcg((const uint2*) (p.y + (size_t) (token * p.R + k) * H + col));
            const half2* h2 = (const half2*) &v;
            s[0] += __low2float(h2[0]);
            s[1] += __high2float(h2[0]);
            s[2] += __low2float(h2[1]);
            s[3] += __high2float(h2[1]);
        }
        had128(s[0], s[1], s[2], s[3], lane);
        const half* sgn = p.signs + H + col;
        #pragma unroll
        for (int q = 0; q < 4; ++q) s[q] = __half2float(sgn[q]) < 0.f ? -s[q] : s[q];
        if (p.out_bf16)
        {
            __nv_bfloat162* o = (__nv_bfloat162*) ((__nv_bfloat16*) p.out + (size_t) token * H + col);
            o[0] = __floats2bfloat162_rn(s[0], s[1]);
            o[1] = __floats2bfloat162_rn(s[2], s[3]);
            continue;
        }
        float* o = (float*) p.out + (size_t) token * H + col;
        #pragma unroll
        for (int q = 0; q < 4; ++q) o[q] = s[q];
    }
}

template <int K, int NTW>
__device__ __forceinline__ void fc2_prefill(const Params& p, int4 it, int ot, int mb, char* smem)
{
    if (mb == 2) fc2_item<K, 2, NTW>(p, it, ot, smem);
    else if (EXLR8_FC2MB == 4 || mb == 4) fc2_item<K, 4, NTW>(p, it, ot, smem);
    else fc2_item<K, EXLR8_FC2MB, NTW>(p, it, ot, smem);
}

template <int MBMAX>
__host__ __device__ constexpr int fc2_ntw() { return MBMAX == 1 ? 1 : EXLR8_FC2NTW; }

template <int MBMAX>
__global__ void __launch_bounds__(THREADS, MBMAX > 1 ? 2 : 1) fc2_kernel(Params p)
{
    extern __shared__ __align__(16) char smem[];
    constexpr int NTW = fc2_ntw<MBMAX>();
    constexpr int OTILES = H / (128 * NTW);
    const int item = blockIdx.x / OTILES, ot = blockIdx.x % OTILES;
    if (item >= p.n_items[MBMAX == 1 ? 2 : 3]) return;
    const int4 it = (MBMAX == 1 ? p.fc2_items : p.fc2p_items)[item];
    const int mb = it.w;
    const int K = p.bits[it.x * 3 + 2];
    if constexpr (MBMAX == 1)
    {
        if (K == 2) fc2_item<2, 1, NTW>(p, it, ot, smem);
        else if (K == 3) fc2_item<3, 1, NTW>(p, it, ot, smem);
        else fc2_item<4, 1, NTW>(p, it, ot, smem);
    }
    else
    {
        if (K == 2) fc2_prefill<2, NTW>(p, it, ot, mb, smem);
        else if (K == 3) fc2_prefill<3, NTW>(p, it, ot, mb, smem);
        else fc2_prefill<4, NTW>(p, it, ot, mb, smem);
    }
}


}  // namespace exlr8

// ------------------------------------------------------------------------------------------------
// Host side (torch extension).

#ifdef EXLR8_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace exlr8 {

// Worst-case list sizes for M tokens. Decode row blocks: at most one per touched expert. Prefill row blocks: at
// most one per 17 routes plus one per touched expert, and none up to 16 tokens (an expert takes at most one route
// per token).
// Scratch for decode items covers every call up to mcap tokens; a call of M tokens launches dec_blocks_max(M).
static int dec_blocks_cap(int M) { return std::min(NE, M * RMAX); }
static int dec_blocks_max(int M) { return M >= EXLR8_FP8_MIN_M ? 0 : dec_blocks_cap(M); }
static int item_mb(int M) { return M >= EXLR8_MB4_MIN_M && M >= EXLR8_FP8_MIN_M ? 4 : MAXMB; }
static int pre_blocks_max(int M)
{
    if (M <= 16) return 0;
    return (M >= EXLR8_FP8_MIN_M ? NE : std::min(NE, M * RMAX / 17)) + (M * RMAX) / (item_mb(M) * 16) + 1;
}

// Bytes of the per-route rows for up to mcap tokens: int8 rows and their scales, or fp16 rows.
static int64_t y_bytes(int64_t mcap)
{
    return EXLR8_Y8 ? mcap * RMAX * (H + 4 * HB) : mcap * RMAX * H * 2;
}

// Bytes of the E4M3 h rows and their scales for up to mcap tokens (EXLR8_FP8H), else 0.
static int64_t h8_bytes(int64_t mcap) { return EXLR8_FP8H ? mcap * RMAX * (I + 4 * UPR) : 0; }

struct Layout { int64_t n_int, n_float; };
static Layout layout(int64_t mcap, int64_t ks_max)
{
    const int dcap = dec_blocks_cap((int) mcap), pcap = pre_blocks_max((int) mcap);
    const int64_t n_int = (int64_t) dcap * UPR + mcap * HB + NE + 4 + mcap * RMAX + 4
                        + 4 * (dcap * ks_max + pcap + dcap + 2 * pcap);
    return {n_int, (int64_t) dcap * UPR * ks_max * 16 * 256};
}

static Params make_params(torch::Tensor x, torch::Tensor ids, torch::Tensor rw, bool shared, double routed_scale,
                          double swiglu_limit, torch::Tensor slab, torch::Tensor off, torch::Tensor bits,
                          torch::Tensor signs, torch::Tensor blk, torch::Tensor ch, torch::Tensor scratch_i,
                          torch::Tensor scratch_f, torch::Tensor xr, torch::Tensor xr8, torch::Tensor xr8s,
                          torch::Tensor h, torch::Tensor h8, torch::Tensor y, torch::Tensor out, int64_t ks_max,
                          int64_t mcap, bool pre_rot, bool rows_out)
{
    // pre_rot: x is unused (the rotated rows are already in xr8 / xr8s), so M comes from ids.
    TORCH_CHECK(pre_rot || (x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == H), "x must be bf16 [M, H]");
    TORCH_CHECK(ids.scalar_type() == at::kLong && ids.size(1) == TOPK, "ids must be int64 [M, top-k]");
    TORCH_CHECK(rw.scalar_type() == at::kFloat && rw.size(1) == TOPK, "rw must be float32 [M, top-k]");
    TORCH_CHECK(ks_max >= 1 && ks_max <= 15, "ks_max in 1..15");
    const int M = (int) ids.size(0);
    TORCH_CHECK(M >= 1 && M <= mcap, "M must be in 1..mcap");
    TORCH_CHECK(!rows_out || EXLR8_Y8, "rows_out needs the int8 rows (EXLR8_Y8)");
    TORCH_CHECK(!pre_rot || (EXLR8_FP8A && M >= EXLR8_FP8_MIN_M), "pre_rot needs the FP8 path");
    TORCH_CHECK(off.numel() == NE * 3 * UPR && bits.numel() == NE * 3 && blk.numel() == NE * 3 * HB
                && ch.numel() == (int64_t) NE * 3 * I && signs.numel() == 2 * H, "weight tables do not match the build");
    const Layout lay = layout(mcap, ks_max);
    TORCH_CHECK(scratch_i.numel() >= lay.n_int && scratch_f.numel() >= lay.n_float, "scratch too small");
    TORCH_CHECK(xr.numel() >= (int64_t) mcap * H && h.numel() >= (int64_t) mcap * RMAX * I
                && y.nbytes() >= (int64_t) y_bytes(mcap), "activation scratch too small");

    Params p = {};
    p.x = pre_rot ? nullptr : (const __nv_bfloat16*) x.data_ptr();
    p.ids = ids.data_ptr<int64_t>();
    p.rw = rw.data_ptr<float>();
    p.M = M;
    p.R = shared ? TOPK + 1 : TOPK;
    p.routed_scale = (float) routed_scale;
    p.swiglu_limit = (float) swiglu_limit;
    p.slab = slab.data_ptr<uint8_t>();
    p.off = off.data_ptr<int64_t>();
    p.bits = bits.data_ptr<int>();
    p.signs = (const half*) signs.data_ptr();
    p.blk = blk.data_ptr<float>();
    p.ch = (const half*) ch.data_ptr();

    // The layout depends only on mcap and ks_max, never on M: the counters it holds persist across calls (each
    // kernel resets the ones it uses).
    const int dcap = dec_blocks_cap((int) mcap), pcap = pre_blocks_max((int) mcap);
    int* si = scratch_i.data_ptr<int>();
    p.fc1_ctr = si;                    si += dcap * UPR;
    p.y_ctr = si;                      si += (int) mcap * HB;
    p.count = si;                      si += NE;
    p.n_items = si;                    si += 4;
    p.order = si;                      si += (int) mcap * RMAX;
    si = (int*) (((uintptr_t) si + 15) & ~(uintptr_t) 15);
    p.fc1_items = (int4*) si;          si += 4 * dcap * (int) ks_max;
    p.fc1p_items = (int4*) si;         si += 4 * pcap;
    p.fc2_items = (int4*) si;          si += 4 * dcap;
    p.fc2p_items = (int4*) si;         si += 4 * 2 * pcap;
    p.fc1_part = scratch_f.data_ptr<float>();
    TORCH_CHECK(xr.scalar_type() == at::kHalf && h.scalar_type() == at::kHalf, "xr and h must be fp16");
    p.xr = (half*) xr.data_ptr();
    TORCH_CHECK(xr8.scalar_type() == at::kByte && xr8.numel() >= (int64_t) M * H, "xr8 must be uint8 [M * H]");
    TORCH_CHECK(xr8s.scalar_type() == at::kFloat && xr8s.numel() >= M, "xr8s must be float32 [M]");
    p.xr8 = xr8.data_ptr<uint8_t>();
    p.xr8s = xr8s.data_ptr<float>();
    p.pre_rot = pre_rot;
    p.rows_out = rows_out;
    if (pre_rot) p.xr = nullptr;
    p.h = (half*) h.data_ptr();
    TORCH_CHECK(h8.nbytes() >= (size_t) h8_bytes(mcap), "h8 scratch too small");
    p.h8 = (uint8_t*) h8.data_ptr();
    p.y = (half*) y.data_ptr();
    TORCH_CHECK(out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16, "out must be fp32 or bf16");
    p.out = out.data_ptr();
    p.out_bf16 = out.scalar_type() == at::kBFloat16;
    p.ks_max = (int) ks_max;
    p.fp8 = M >= EXLR8_FP8_MIN_M;
    p.item_mb = item_mb(M);
    return p;
}

void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor rw, bool shared, double routed_scale, double swiglu_limit,
         torch::Tensor slab, torch::Tensor off, torch::Tensor bits, torch::Tensor signs, torch::Tensor blk,
         torch::Tensor ch, torch::Tensor scratch_i, torch::Tensor scratch_f, torch::Tensor xr, torch::Tensor xr8,
         torch::Tensor xr8s, torch::Tensor h, torch::Tensor h8, torch::Tensor y, torch::Tensor out, int64_t ks_max,
         int64_t mcap, bool pre_rot, bool rows_out)
{
    Params p = make_params(x, ids, rw, shared, routed_scale, swiglu_limit, slab, off, bits, signs, blk, ch,
                           scratch_i, scratch_f, xr, xr8, xr8s, h, h8, y, out, ks_max, mcap, pre_rot, rows_out);
    static int slots = 0;
    if (!slots)
    {
        cudaFuncSetAttribute(fc1_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, fc1_smem(1));
        cudaFuncSetAttribute(fc1_kernel<MAXMB>, cudaFuncAttributeMaxDynamicSharedMemorySize, fc1_smem(MAXMB));
        cudaFuncSetAttribute(fc1_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, fc1_smem(4));
        cudaFuncSetAttribute(fc2_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, fc2_smem(1));
        cudaFuncSetAttribute(fc2_kernel<EXLR8_FC2MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, fc2_smem(EXLR8_FC2MB));
        int per_sm = 1, sms = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fc1_kernel<1>, THREADS, fc1_smem(1));
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.device().index());
        slots = std::max(1, per_sm) * sms;
    }
    p.slots = slots;
    const int M = p.M;
    const int dmax = dec_blocks_max(M), pmax = pre_blocks_max(M);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (pre_rot) {}
    else if (EXLR8_FP8A && pmax && p.fp8) rot8_kernel<<<M, THREADS, 0, stream>>>(p);
    else rot_kernel<<<(M * HB + WARPS - 1) / WARPS, THREADS, 0, stream>>>(p);
    plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(p);
    if (dmax) fc1_kernel<1><<<dmax * (int) ks_max * UPR, THREADS, fc1_smem(1), stream>>>(p);
    if (pmax && p.item_mb == 4) fc1_kernel<4><<<pmax * UPR, THREADS, fc1_smem(4), stream>>>(p);
    else if (pmax) fc1_kernel<MAXMB><<<pmax * UPR, THREADS, fc1_smem(MAXMB), stream>>>(p);
    if (dmax) fc2_kernel<1><<<dmax * (H / (128 * fc2_ntw<1>())), THREADS, fc2_smem(1), stream>>>(p);
    if (pmax) fc2_kernel<EXLR8_FC2MB><<<2 * pmax * (H / (128 * fc2_ntw<EXLR8_FC2MB>())), THREADS, fc2_smem(EXLR8_FC2MB), stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x_rot of a shard as E4M3 with its per-token scales, for a caller that gathers it (the SP path).
void rot8(torch::Tensor x, torch::Tensor signs, torch::Tensor xr8, torch::Tensor xr8s)
{
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == H && x.is_contiguous(), "x must be bf16 [n, H]");
    const int n = (int) x.size(0);
    TORCH_CHECK(xr8.scalar_type() == at::kByte && xr8.numel() >= (int64_t) n * H, "xr8 must be uint8 [n * H]");
    TORCH_CHECK(xr8s.scalar_type() == at::kFloat && xr8s.numel() >= n, "xr8s must be float32 [n]");
    TORCH_CHECK(signs.numel() == 2 * H, "signs must be [2, H]");
    if (n == 0) return;
    Params p = {};
    p.x = (const __nv_bfloat16*) x.data_ptr();
    p.M = n;
    p.signs = (const half*) signs.data_ptr();
    p.xr8 = xr8.data_ptr<uint8_t>();
    p.xr8s = xr8s.data_ptr<float>();
    rot8_kernel<<<n, THREADS, 0, at::cuda::getCurrentCUDAStream().stream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Scratch sizes for up to mcap tokens: int32 elements, float elements, and bytes of y.
std::vector<int64_t> scratch_sizes(int64_t mcap, int64_t ks_max)
{
    const Layout lay = layout(mcap, ks_max);
    return {lay.n_int, lay.n_float, y_bytes(mcap), h8_bytes(mcap)};
}

}  // namespace exlr8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe", &exlr8::moe, "exlr8 v2 routed + shared expert layer, prefill and decode");
    m.def("rot8", &exlr8::rot8, "x_rot as E4M3 with per-token scales");
    m.def("scratch_sizes", &exlr8::scratch_sizes, "int32 and float scratch elements and y bytes for up to mcap tokens");
    m.attr("H") = exlr8::H;
    m.attr("E") = exlr8::E;
    m.attr("UPR") = exlr8::UPR;
    m.attr("TOPK") = exlr8::TOPK;
}
#endif
