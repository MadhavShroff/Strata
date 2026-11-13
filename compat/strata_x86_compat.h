// aarch64 port: translate the engine's x86 SSE/AVX2/AVX-512 intrinsics to NEON via SIMDe.
// Included in place of <immintrin.h> on non-x86 hosts (see CMakeLists / immintrin.h shim).
#ifndef STRATA_X86_COMPAT_H
#define STRATA_X86_COMPAT_H
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__i386__)

#define SIMDE_ENABLE_NATIVE_ALIASES 1
#include <simde/x86/avx512.h>
#include <simde/x86/fma.h>
#include <simde/x86/f16c.h>

#include <cstdint>

// ---- intrinsics SIMDe 0.8.2 does not provide ----

// _mm_pause: on AArch64 the closest hint is YIELD (or ISB for a tighter spin).
#ifndef _mm_pause
static inline void strata_mm_pause(void) {
#if defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  __asm__ __volatile__("" ::: "memory");
#endif
}
#define _mm_pause() strata_mm_pause()
#endif

// _mm_sfence / _mm_lfence / _mm_mfence: store/load/full barriers.
#ifndef _mm_sfence
static inline void strata_mm_sfence(void) { __atomic_thread_fence(__ATOMIC_RELEASE); }
#define _mm_sfence() strata_mm_sfence()
#endif
#ifndef _mm_lfence
static inline void strata_mm_lfence(void) { __atomic_thread_fence(__ATOMIC_ACQUIRE); }
#define _mm_lfence() strata_mm_lfence()
#endif
#ifndef _mm_mfence
static inline void strata_mm_mfence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
#define _mm_mfence() strata_mm_mfence()
#endif

// AVX-512 horizontal reductions (missing from SIMDe 0.8.2): exact same association order
// as Intel's reference sequence (pairwise halving), so results match the x86 kernels bit for bit
// for the add reductions on IEEE doubles/floats where the halving order is what Intel documents.
static inline float strata_mm512_reduce_add_ps(simde__m512 v) {
  simde__m256 lo = simde_mm512_castps512_ps256(v);
  simde__m256 hi = simde_mm512_extractf32x8_ps(v, 1);
  simde__m256 s8 = simde_mm256_add_ps(lo, hi);
  simde__m128 s4 = simde_mm_add_ps(simde_mm256_castps256_ps128(s8), simde_mm256_extractf128_ps(s8, 1));
  s4 = simde_mm_add_ps(s4, simde_mm_movehl_ps(s4, s4));
  s4 = simde_mm_add_ss(s4, simde_mm_movehdup_ps(s4));
  return simde_mm_cvtss_f32(s4);
}
#ifndef _mm512_reduce_add_ps
#define _mm512_reduce_add_ps(v) strata_mm512_reduce_add_ps(v)
#endif

static inline float strata_mm512_reduce_max_ps(simde__m512 v) {
  simde__m256 lo = simde_mm512_castps512_ps256(v);
  simde__m256 hi = simde_mm512_extractf32x8_ps(v, 1);
  simde__m256 m8 = simde_mm256_max_ps(lo, hi);
  simde__m128 m4 = simde_mm_max_ps(simde_mm256_castps256_ps128(m8), simde_mm256_extractf128_ps(m8, 1));
  m4 = simde_mm_max_ps(m4, simde_mm_movehl_ps(m4, m4));
  m4 = simde_mm_max_ss(m4, simde_mm_movehdup_ps(m4));
  return simde_mm_cvtss_f32(m4);
}
#ifndef _mm512_reduce_max_ps
#define _mm512_reduce_max_ps(v) strata_mm512_reduce_max_ps(v)
#endif

static inline int strata_mm512_reduce_add_epi32(simde__m512i v) {
  simde__m256i lo = simde_mm512_castsi512_si256(v);
  simde__m256i hi = simde_mm512_extracti32x8_epi32(v, 1);
  simde__m256i s8 = simde_mm256_add_epi32(lo, hi);
  simde__m128i s4 = simde_mm_add_epi32(simde_mm256_castsi256_si128(s8), simde_mm256_extracti128_si256(s8, 1));
  s4 = simde_mm_add_epi32(s4, simde_mm_shuffle_epi32(s4, 0x4E));
  s4 = simde_mm_add_epi32(s4, simde_mm_shuffle_epi32(s4, 0xB1));
  return simde_mm_cvtsi128_si32(s4);
}
#ifndef _mm512_reduce_add_epi32
#define _mm512_reduce_add_epi32(v) strata_mm512_reduce_add_epi32(v)
#endif

// ---- mask types and conversions SIMDe 0.8.2 does not alias ----
// SIMDe's avx512/types.h aliases __mmask16, __mmask32 and __mmask64 under SIMDE_ENABLE_NATIVE_ALIASES
// but never __mmask8 (it only ever defines simde__mmask8), so that one alone is added here.
#if !defined(__mmask8)
typedef simde__mmask8 __mmask8;
#endif
// _cvtu64_mask64 / _cvtu32_mask32 are plain reinterpretations of the integer as a mask.
#ifndef _cvtu64_mask64
#define _cvtu64_mask64(x) ((__mmask64) (x))
#endif
#ifndef _cvtu32_mask32
#define _cvtu32_mask32(x) ((__mmask32) (x))
#endif
#ifndef _cvtmask64_u64
#define _cvtmask64_u64(x) ((uint64_t) (x))
#endif

// ---- widening converts and narrowing stores SIMDe 0.8.2 does not provide ----

// _mm512_cvtepu16_epi64: the low 8 u16 lanes of a 128-bit value, zero-extended into 8 qwords.
static inline simde__m512i strata_mm512_cvtepu16_epi64(simde__m128i a) {
  uint16_t s[8];
  simde_mm_storeu_si128((simde__m128i*) s, a);
  return simde_mm512_set_epi64((int64_t) s[7], (int64_t) s[6], (int64_t) s[5], (int64_t) s[4],
                               (int64_t) s[3], (int64_t) s[2], (int64_t) s[1], (int64_t) s[0]);
}
#ifndef _mm512_cvtepu16_epi64
#define _mm512_cvtepu16_epi64(a) strata_mm512_cvtepu16_epi64(a)
#endif

// _mm256_cvtepu16_epi64: the low 4 u16 lanes, zero-extended into 4 qwords.
static inline simde__m256i strata_mm256_cvtepu16_epi64(simde__m128i a) {
  uint16_t s[8];
  simde_mm_storeu_si128((simde__m128i*) s, a);
  return simde_mm256_set_epi64x((int64_t) s[3], (int64_t) s[2], (int64_t) s[1], (int64_t) s[0]);
}
#ifndef _mm256_cvtepu16_epi64
#define _mm256_cvtepu16_epi64(a) strata_mm256_cvtepu16_epi64(a)
#endif

// _mm512_cvtepu8_epi32: 16 u8 lanes of a 128-bit value, zero-extended into 16 dwords.
static inline simde__m512i strata_mm512_cvtepu8_epi32(simde__m128i a) {
  uint8_t s[16];
  simde_mm_storeu_si128((simde__m128i*) s, a);
  return simde_mm512_set_epi32((int32_t) s[15], (int32_t) s[14], (int32_t) s[13], (int32_t) s[12],
                               (int32_t) s[11], (int32_t) s[10], (int32_t) s[9],  (int32_t) s[8],
                               (int32_t) s[7],  (int32_t) s[6],  (int32_t) s[5],  (int32_t) s[4],
                               (int32_t) s[3],  (int32_t) s[2],  (int32_t) s[1],  (int32_t) s[0]);
}
#ifndef _mm512_cvtepu8_epi32
#define _mm512_cvtepu8_epi32(a) strata_mm512_cvtepu8_epi32(a)
#endif

// _mm512_cvtepu16_epi32: 16 u16 lanes of a 256-bit value, zero-extended into 16 dwords.
static inline simde__m512i strata_mm512_cvtepu16_epi32(simde__m256i a) {
  uint16_t s[16];
  simde_mm256_storeu_si256((simde__m256i*) s, a);
  return simde_mm512_set_epi32((int32_t) s[15], (int32_t) s[14], (int32_t) s[13], (int32_t) s[12],
                               (int32_t) s[11], (int32_t) s[10], (int32_t) s[9],  (int32_t) s[8],
                               (int32_t) s[7],  (int32_t) s[6],  (int32_t) s[5],  (int32_t) s[4],
                               (int32_t) s[3],  (int32_t) s[2],  (int32_t) s[1],  (int32_t) s[0]);
}
#ifndef _mm512_cvtepu16_epi32
#define _mm512_cvtepu16_epi32(a) strata_mm512_cvtepu16_epi32(a)
#endif

// _mm512_cvtepi32_epi8: TRUNCATING narrow (keeps each dword's low byte), which is what the
// activation quantizer relies on - not the saturating _mm512_cvtsepi32_epi8.
static inline simde__m128i strata_mm512_cvtepi32_epi8(simde__m512i a) {
  int32_t s[16];
  simde_mm512_storeu_si512((void*) s, a);
  int8_t o[16];
  for (int i = 0; i < 16; ++i) o[i] = (int8_t) (s[i] & 0xFF);
  return simde_mm_loadu_si128((const simde__m128i*) o);
}
#ifndef _mm512_cvtepi32_epi8
#define _mm512_cvtepi32_epi8(a) strata_mm512_cvtepi32_epi8(a)
#endif

// _mm512_cvtsepi32_epi8: the SATURATING narrow (signed clamp to int8).
static inline simde__m128i strata_mm512_cvtsepi32_epi8(simde__m512i a) {
  int32_t s[16];
  simde_mm512_storeu_si512((void*) s, a);
  int8_t o[16];
  for (int i = 0; i < 16; ++i) {
    int32_t v = s[i];
    o[i] = (int8_t) (v > 127 ? 127 : (v < -128 ? -128 : v));
  }
  return simde_mm_loadu_si128((const simde__m128i*) o);
}
#ifndef _mm512_cvtsepi32_epi8
#define _mm512_cvtsepi32_epi8(a) strata_mm512_cvtsepi32_epi8(a)
#endif

// ---- masked loads ----

// _mm_maskz_loadu_epi16: 8 u16 lanes; a zero mask bit zeroes its lane and suppresses its load.
static inline simde__m128i strata_mm_maskz_loadu_epi16(__mmask8 k, const void* p) {
  const uint16_t* src = (const uint16_t*) p;
  uint16_t o[8];
  for (int i = 0; i < 8; ++i) o[i] = ((k >> i) & 1u) ? src[i] : (uint16_t) 0;
  return simde_mm_loadu_si128((const simde__m128i*) o);
}
#ifndef _mm_maskz_loadu_epi16
#define _mm_maskz_loadu_epi16(k, p) strata_mm_maskz_loadu_epi16(k, p)
#endif

// _mm256_maskz_loadu_epi16: the same for 16 lanes.
static inline simde__m256i strata_mm256_maskz_loadu_epi16(__mmask16 k, const void* p) {
  const uint16_t* src = (const uint16_t*) p;
  uint16_t o[16];
  for (int i = 0; i < 16; ++i) o[i] = ((k >> i) & 1u) ? src[i] : (uint16_t) 0;
  return simde_mm256_loadu_si256((const simde__m256i*) o);
}
#ifndef _mm256_maskz_loadu_epi16
#define _mm256_maskz_loadu_epi16(k, p) strata_mm256_maskz_loadu_epi16(k, p)
#endif

// ---- gathers ----
// _mm512_i32gather_epi32(vindex, base, scale): 16 dword loads at base + index*scale.
// `scale` is a compile-time 1/2/4/8 on x86; here it is an ordinary multiply.
static inline simde__m512i strata_mm512_i32gather_epi32(simde__m512i vindex, const void* base, int scale) {
  int32_t idx[16];
  simde_mm512_storeu_si512((void*) idx, vindex);
  const char* b = (const char*) base;
  int32_t o[16];
  for (int i = 0; i < 16; ++i) {
    int32_t v;
    __builtin_memcpy(&v, b + (int64_t) idx[i] * (int64_t) scale, sizeof(v));
    o[i] = v;
  }
  return simde_mm512_loadu_si512((const void*) o);
}
#ifndef _mm512_i32gather_epi32
#define _mm512_i32gather_epi32(vindex, base, scale) strata_mm512_i32gather_epi32(vindex, base, scale)
#endif

static inline simde__m256i strata_mm256_i32gather_epi32(const void* base, simde__m256i vindex, int scale) {
  int32_t idx[8];
  simde_mm256_storeu_si256((simde__m256i*) idx, vindex);
  const char* b = (const char*) base;
  int32_t o[8];
  for (int i = 0; i < 8; ++i) {
    int32_t v;
    __builtin_memcpy(&v, b + (int64_t) idx[i] * (int64_t) scale, sizeof(v));
    o[i] = v;
  }
  return simde_mm256_loadu_si256((const simde__m256i*) o);
}
#ifndef _mm256_i32gather_epi32
#define _mm256_i32gather_epi32(base, vindex, scale) strata_mm256_i32gather_epi32(base, vindex, scale)
#endif

#endif  // !x86
#endif  // STRATA_X86_COMPAT_H
