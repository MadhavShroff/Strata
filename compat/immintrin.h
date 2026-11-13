// aarch64 port: a shim for <immintrin.h>.  The engine's CPU kernels are written in x86
// SSE/AVX2/AVX-512 intrinsics; SIMDe maps them to NEON.  This directory goes on the include
// path ONLY for non-x86 hosts, so an x86 build still gets the compiler's own header.
#ifndef STRATA_COMPAT_IMMINTRIN_H
#define STRATA_COMPAT_IMMINTRIN_H
#include "strata_x86_compat.h"
#endif
