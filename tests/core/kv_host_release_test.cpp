// The memory guard (measurement-34): a lazy host K/V copy's committed pages go back to the OS when the
// conversation that wrote them ends. This tests the release mechanism itself - the mmap(MAP_NORESERVE) region
// src/core/layer.cpp carves under STRATA_KV_HOST_LAZY=1 - without a GPU state: a QsaState that names only the
// region, exactly the fields qsa_state_init sets on that path.
//
// What it proves: qsa_state_host_release drops exactly the pages a conversation wrote (the next read sees
// zeros, as no reader names a cell before its sequence writes it); a state without a lazy copy - the pinned
// path - releases nothing and its bytes are untouched; a fresh region releases nothing; the running total
// counts every release. That the release happens only at a fresh start, never at a resume, is placement: its
// only callers are qsa_state_zero (a fresh sequence) and the slot admission in generate.cpp (an overwrite) -
// the resume paths (conversation_kv_restore, checkpoint_restore, copy_from_slot) never zero a live state.
#include "strata/core/layer.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "kv_host_release_test: FAILED: %s\n", what);
        g_failed = 1;
    }
}

constexpr uint64_t kPage = 4096;

// The region the way qsa_state_init maps it under STRATA_KV_HOST_LAZY=1 (layer.cpp): reserved, not committed.
uint8_t* map_region(uint64_t bytes) {
    void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    check(m != MAP_FAILED, "the test region maps");
    return (uint8_t*) m;
}

uint64_t resident_pages(const void* base, uint64_t bytes) {
    const size_t pages = (size_t) ((bytes + kPage - 1) / kPage);
    std::vector<unsigned char> vec(pages);
    if (mincore(const_cast<void*>(base), bytes, vec.data()) != 0) return UINT64_MAX;
    uint64_t n = 0;
    for (const unsigned char c : vec) n += (c & 1) != 0;
    return n;
}

}  // namespace

int main() {
    const uint64_t bytes = 64 * kPage;   // 64 pages: small, but the same page semantics as the GiB regions

    // 1. A conversation wrote every other page; a fresh start releases exactly what it wrote.
    {
        uint8_t* m = map_region(bytes);
        for (uint64_t off = 0; off < bytes; off += 2 * kPage) m[off] = 0x5a;
        strata::core::QsaState st{};
        st.host_lazy_base = m;
        st.host_lazy_bytes = bytes;
        const uint64_t before = resident_pages(m, bytes);
        check(before == 32, "half the region is committed after the writes");
        const uint64_t total0 = strata::core::kv_host_lazy_released_bytes();
        const uint64_t released = strata::core::qsa_state_host_release(st);
        check(released == before * kPage, "the release reports exactly the resident bytes");
        check(strata::core::kv_host_lazy_released_bytes() == total0 + released, "the running total counts it");
        check(resident_pages(m, bytes) == 0, "nothing stays committed after the release");
        check(m[0] == 0 && m[2 * kPage] == 0, "a read after the release sees zeros (a fresh sequence's unwritten cells)");
        // 2. Releasing again gives nothing back: the release is a fresh start's, not a repeated one.
        check(strata::core::qsa_state_host_release(st) == 0, "a second release releases nothing");
        munmap(m, bytes);
    }

    // 3. The pinned path (no lazy region named) releases nothing and reads nothing: host_lazy_bytes == 0 is the
    //    pinned or no-copy state, and the memory guard leaves it alone.
    {
        strata::core::QsaState st{};
        const uint64_t total0 = strata::core::kv_host_lazy_released_bytes();
        check(strata::core::qsa_state_host_release(st) == 0, "a state without a lazy copy releases nothing");
        check(strata::core::qsa_state_host_committed(st) == 0, "a state without a lazy copy holds nothing to report");
        check(strata::core::kv_host_lazy_released_bytes() == total0, "the total did not move");
    }

    // 4. A fresh region (a fresh start with no conversation behind it, the engine's own startup zeroing)
    //    releases nothing but is still a legal call.
    {
        uint8_t* m = map_region(bytes);
        strata::core::QsaState st{};
        st.host_lazy_base = m;
        st.host_lazy_bytes = bytes;
        check(strata::core::qsa_state_host_release(st) == 0, "a never-written region releases nothing");
        check(strata::core::qsa_state_host_committed(st) == 0, "a never-written region reports nothing held");
        munmap(m, bytes);
    }

    if (g_failed == 0) std::printf("kv_host_release_test: ok\n");
    return g_failed;
}
