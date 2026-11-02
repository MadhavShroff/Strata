// src/kernels/nvfp4_decode_bench.cpp - the decode path's grouped GPU experts (native_expert_grouped) timed on random
// blobs of a format pair, as weight bytes per second against the card's DRAM bandwidth, plus a hash of the output so
// a kernel change can be checked bit for bit.
//
//     nvfp4_decode_bench [gu_type d_type] [groups] [tokens per group] [iters]
//
// Defaults: 40 40 (NVFP4), 64 groups (distinct ~2.7 MB blobs: far past L2), 2 tokens each, 50 iterations.  UD-IQ4_XS's
// experts: 21 20 (IQ3_S gate/up, IQ4_NL down).  Random bytes: the values are meaningless, the timing is not (the
// kernels have no data-dependent branches but the NVFP4 scale decode), and the hash is stable for a given build.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
}  // namespace

int main(int argc, char** argv) {
    {   // the UE4M3 decode built from bits (iq_kernels.cu) against the ldexpf form it replaced, all 256 bytes
        int bad = 0;
        for (int x = 0; x < 256; ++x) {
            const int e = (x >> 3) & 0xF, m = x & 7;
            const float old = (x == 0 || x == 0x7F) ? 0.0f
                              : (e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7)) * 0.5f;
            uint32_t bits = ((uint32_t) (e + 119) << 23) | ((uint32_t) m << 20);
            float nw;
            std::memcpy(&nw, &bits, 4);
            if (x == 0 || x == 0x7F) nw = 0.0f;
            else if (e == 0) nw = (float) m * 0.0009765625f;
            bad += std::memcmp(&old, &nw, 4) != 0;
        }
        std::printf("ue4m3 decode: %d of 256 bytes differ from the ldexpf form\n", bad);
        if (bad) return 1;
    }
    const int gu = argc > 2 ? std::atoi(argv[1]) : 40, dt = argc > 2 ? std::atoi(argv[2]) : 40;
    const int G = argc > 3 ? std::atoi(argv[3]) : 64, TPG = argc > 4 ? std::atoi(argv[4]) : 2;
    const int IT = argc > 5 ? std::atoi(argv[5]) : 50;
    // argv[6] (or NVFP4_BENCH_T): the window's token rows, 1..16 (the engine's kVerifyMaxT windows; the entries'
    // tok ids are taken modulo it).  12 is the raised window; 8 the current one.
    int T = 8;
    if (const char* e = std::getenv("NVFP4_BENCH_T")) T = std::atoi(e);
    if (argc > 6) T = std::atoi(argv[6]);
    if (T < 1 || T > 16) { std::fprintf(stderr, "window tokens 1..16\n"); return 2; }
    constexpr int64_t N = 2560, FF = 640;
    if (!k::native_expert_supported(gu, dt, N, FF)) { std::fprintf(stderr, "types %d/%d not supported\n", gu, dt); return 2; }
    const k::NativeExpertLayout L = k::native_expert_layout(gu, dt, N, FF);
    const int E = G * TPG;                             // entries; T = the window's tokens (activations)
    std::mt19937 rng(20261012);
    std::vector<uint8_t> blobs((size_t) G * L.bytes);
    for (auto& b : blobs) b = (uint8_t) (rng() & 0xFF);
    if (gu == 40 || dt == 40) {                        // NVFP4: UE4M3 scales in the checkpoint's range, sane tails
        for (int g = 0; g < G; ++g) {
            uint8_t* b = blobs.data() + (size_t) g * L.bytes;
            auto fix = [&](size_t off, size_t bytes) {
                for (size_t o = off; o + 36 <= off + bytes; o += 36)
                    for (int s = 0; s < 4; ++s) b[o + s] = (uint8_t) (0x20 + rng() % 0x40);
            };
            if (gu == 40) fix(0, L.down_off);
            if (dt == 40) fix(L.down_off, (size_t) N * L.d_row);
            if (L.tail_off) {
                const float t[4] = {8e-5f, 8e-5f, 1e-4f, 0.f};
                std::memcpy(b + L.tail_off, t, sizeof t);
            }
        }
    } else {                                           // the other formats: fp16 block scales kept finite (|d| < 2)
        for (size_t i = 1; i < blobs.size(); i += 2) blobs[i] &= 0x3B;
    }
    // NVFP4_BENCH_FILE=<experts.bin or its head>: the blobs are read from it (the first G whole blobs, cycled when it
    // holds fewer), real GPTQ / ModelOpt weights instead of random ones
    if (const char* fp = std::getenv("NVFP4_BENCH_FILE")) {
        FILE* f = std::fopen(fp, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", fp); return 2; }
        int have = 0;
        while (have < G && std::fread(blobs.data() + (size_t) have * L.bytes, 1, L.bytes, f) == L.bytes) ++have;
        std::fclose(f);
        if (have == 0) { std::fprintf(stderr, "%s holds no whole blob\n", fp); return 2; }
        for (int g = have; g < G; ++g)
            std::memcpy(blobs.data() + (size_t) g * L.bytes, blobs.data() + (size_t) (g % have) * L.bytes, L.bytes);
        std::printf("blobs: %d real from %s (cycled to %d)\n", have, fp, G);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) T * N);
    for (float& v : x) v = nd(rng);
    std::vector<unsigned long long> ptr(G);
    std::vector<int32_t> start(G + 1), dst(E), tok(E);
    uint8_t* db;
    ck(cudaMalloc(&db, blobs.size()), "blobs");
    ck(cudaMemcpy(db, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "blobs");
    for (int g = 0; g < G; ++g) {
        ptr[g] = (unsigned long long) (db + (size_t) g * L.bytes);
        start[g] = g * TPG;
        for (int j = 0; j < TPG; ++j) { dst[g * TPG + j] = (g * TPG + j) % T + T * ((g * TPG + j) / T); tok[g * TPG + j] = (g + j) % T; }
    }
    start[G] = E;
    for (int e = 0; e < E; ++e) dst[e] = e;            // one output row per entry (no summation races in the hash)
    float *dx, *dout; void *dxq, *scr; unsigned long long* dptr; int32_t *dstart, *dng, *ddst, *dtok;
    ck(cudaMalloc(&dx, x.size() * 4), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dxq, (size_t) T * N / 32 * 36), "xq");
    ck(cudaMalloc(&dout, (size_t) E * N * 4), "out");
    ck(cudaMalloc(&scr, k::native_expert_scratch_bytes(E, FF)), "scratch");
    ck(cudaMalloc(&dptr, 8 * G), "p");
    ck(cudaMemcpy(dptr, ptr.data(), 8 * G, cudaMemcpyHostToDevice), "p");
    ck(cudaMalloc(&dstart, 4 * (G + 1)), "s");
    ck(cudaMemcpy(dstart, start.data(), 4 * (G + 1), cudaMemcpyHostToDevice), "s");
    ck(cudaMalloc(&dng, 4), "n");
    ck(cudaMemcpy(dng, &G, 4, cudaMemcpyHostToDevice), "n");
    ck(cudaMalloc(&ddst, 4 * E), "d");
    ck(cudaMemcpy(ddst, dst.data(), 4 * E, cudaMemcpyHostToDevice), "d");
    ck(cudaMalloc(&dtok, 4 * E), "t");
    ck(cudaMemcpy(dtok, tok.data(), 4 * E, cudaMemcpyHostToDevice), "t");
    k::quantize_q8_1_rows(dx, T, N, dxq, nullptr);
    // NVFP4_BENCH_PHASE=1: gate/up + SwiGLU + quantize only, 2: down only (native_expert_set_mode), 0/unset: all
    int ph = 0;
    if (const char* p = std::getenv("NVFP4_BENCH_PHASE")) ph = std::atoi(p);
    if (ph) k::native_expert_set_mode(-1, ph);
    auto run = [&] { k::native_expert_grouped(L, dptr, dstart, dng, ddst, dtok, G, E, dxq, scr, dout, nullptr); };
    // the output hash of one arm (STRATA_EXPERT_LOADS off = the direct 4-byte loads, on = the shared-staged path)
    auto hash_of = [&](bool loads) {
        k::iq_set_expert_loads(loads);
        ck(cudaMemset(dout, 0xFF, (size_t) E * N * 4), "clr");
        run();
        ck(cudaDeviceSynchronize(), "run");
        std::vector<float> out((size_t) E * N);
        ck(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost), "out");
        uint64_t h = 1469598103934665603ull;           // FNV-1a over the output bytes
        for (const float& v : out) {
            const auto* p = reinterpret_cast<const uint8_t*>(&v);
            for (int i = 0; i < 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
        }
        return h;
    };
    const uint64_t h_direct = hash_of(false);
    const bool ab = gu == 40 || dt == 40;
    const uint64_t h_staged = ab ? hash_of(true) : h_direct;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    // the arms ALTERNATE, three rounds each keeping its minimum (a card under sustained load steps its clocks;
    // a fixed order would flatter whichever arm ran first)
    float best[2] = {1e30f, 1e30f};
    for (int round = 0; round < 3; ++round)
        for (int arm = 0; arm < (ab ? 2 : 1); ++arm) {
            k::iq_set_expert_loads(arm == 1);
            for (int i = 0; i < 5; ++i) run();         // warm
            for (int i = 0; i < IT; ++i) {
                cudaEventRecord(a);
                run();
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, a, b);
                if (ms < best[arm]) best[arm] = ms;
            }
        }
    k::iq_set_expert_loads(false);
    ck(cudaGetLastError(), "run");
    int dev = 0, bus = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, dev);   // kHz
    const double peak = 2.0 * clk * 1e3 * (bus / 8) / 1e9;            // GB/s, DDR
    // the weight bytes the phase actually streams: gate/up is down_off bytes per expert, down the rest to the
    // tail (the whole blob when there is no tail)
    const double gu_b = (double) G * (double) L.down_off;
    const double d_b = (double) G * (double) (L.tail_off ? L.tail_off - L.down_off : L.bytes - L.down_off);
    const double bytes = ph == 1 ? gu_b : ph == 2 ? d_b : gu_b + d_b;
    std::printf("types %d/%d  blob %zu B  T %d  groups %d x %d tokens  phase %d:  direct %.3f ms -> %.1f GB/s",
                gu, dt, L.bytes, T, G, TPG, ph, best[0], bytes / (best[0] * 1e6));
    if (ab)
        std::printf("  |  staged %.3f ms -> %.1f GB/s (%.0f%% of direct)  hashes %s", best[1],
                    bytes / (best[1] * 1e6), 100.0 * best[0] / best[1],
                    h_direct == h_staged ? "bitwise equal" : "DIFFER");
    std::printf("  (%.0f%% of the %.0f GB/s DDR peak)  out hash %016llx\n", 100.0 * bytes / (best[0] * 1e6) / peak,
                peak, (unsigned long long) h_direct);
    return h_direct == h_staged ? 0 : 1;
}
