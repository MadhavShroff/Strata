// aarch64 port: a shim for GCC's <cpuid.h>.  AArch64 has no CPUID leaf, so __cpuid_count
// reports zero for every feature bit.  The engine's cpu_avx512_ok() / cpu_avx2_ok() /
// cpu_avx1_ok() probes therefore answer false, and its dispatch falls through to the
// ggml-cpu path, which this build compiles with NEON.
#ifndef STRATA_COMPAT_CPUID_H
#define STRATA_COMPAT_CPUID_H

#define __cpuid_count(leaf, sub, a, b, c, d) \
    do { (void) (leaf); (void) (sub); (a) = 0u; (b) = 0u; (c) = 0u; (d) = 0u; } while (0)
#define __cpuid(leaf, a, b, c, d) __cpuid_count(leaf, 0, a, b, c, d)

static inline int __get_cpuid_max(unsigned int leaf, unsigned int* sig) {
    (void) leaf;
    if (sig != 0) *sig = 0u;
    return 0;
}
static inline int __get_cpuid(unsigned int leaf, unsigned int* a, unsigned int* b,
                              unsigned int* c, unsigned int* d) {
    (void) leaf;
    if (a) *a = 0u;
    if (b) *b = 0u;
    if (c) *c = 0u;
    if (d) *d = 0u;
    return 0;
}
#endif
