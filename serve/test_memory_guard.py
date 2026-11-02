"""The memory guard (measurement-34): the server's admission rule, pure Python - no engine, no GPU.

A request enters a batch slot only while free host memory (MemAvailable, tightened by the cgroup) covers the
host K/V it can add - (prompt + max_new - the cells the slot already holds) x 12,672 bytes - above a floor
("memory_guard_floor_gib" / STRATA_MEMORY_GUARD_FLOOR_GIB, default 8 GiB, 0 disables). A request that does not
fit is held in the slot wait and admitted when memory frees; it is never rejected. These tests cover the rule
(admits under the floor, holds above it, releases when memory frees), the floor's sources, the free-memory
reader and the held-cells credit."""
import contextlib
import io
import os
import threading
import unittest
from unittest import mock

from serve.server import (KV_HOST_BYTES_PER_CELL, MemoryGuard, StrataEngine, available_host_bytes)

GIB = 1 << 30


class Estimate(unittest.TestCase):
    def test_each_cell_costs_the_constant(self):
        g = MemoryGuard(floor_gib=0)
        self.assertEqual(KV_HOST_BYTES_PER_CELL, 12672)          # 1,056 B/cell/layer x 12 QSA layers, KV int8
        self.assertEqual(g.estimate_bytes(1000, 500), 1500 * 12672)

    def test_the_cells_a_slot_holds_are_not_added_again(self):
        g = MemoryGuard(floor_gib=0)
        self.assertEqual(g.estimate_bytes(1000, 500, held_cells=400), 1100 * 12672)

    def test_more_held_cells_than_the_prompt_clamp_to_zero(self):
        g = MemoryGuard(floor_gib=0)
        self.assertEqual(g.estimate_bytes(100, 50, held_cells=500), 0)


class Admits(unittest.TestCase):
    def test_admits_when_the_estimate_leaves_the_floor_free(self):
        g = MemoryGuard(floor_gib=8)
        self.assertTrue(g.admits(20 * GIB, 10 * GIB))            # 20 - 10 = 10 GiB >= the 8 GiB floor

    def test_holds_when_the_estimate_eats_into_the_floor(self):
        g = MemoryGuard(floor_gib=8)
        self.assertFalse(g.admits(15 * GIB, 10 * GIB))           # 15 - 10 = 5 GiB < 8 GiB

    def test_the_boundary_itself_admits(self):
        g = MemoryGuard(floor_gib=8)
        self.assertTrue(g.admits(18 * GIB, 10 * GIB))            # exactly the floor: not below it

    def test_floor_zero_disables_the_guard(self):
        g = MemoryGuard(floor_gib=0)
        self.assertTrue(g.admits(1, 10 ** 12))                   # everything admits, as before the guard

    def test_unknown_free_memory_admits(self):
        g = MemoryGuard(floor_gib=8)
        self.assertTrue(g.admits(None, 10 ** 12))                # an unreadable metric never holds every request


class FloorSources(unittest.TestCase):
    def said_default(self, cfg, env=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_MEMORY_GUARD_FLOOR_GIB", None)   # restored when the patch context ends
            if env:
                os.environ.update(env)
            g = MemoryGuard.from_config(cfg)
        return g, buf.getvalue()

    def test_the_default_is_eight_gib(self):
        g, _ = self.said_default({})
        self.assertEqual(g.floor_bytes, 8 * GIB)

    def test_the_config_key_sets_it(self):
        g, _ = self.said_default({"memory_guard_floor_gib": 2})
        self.assertEqual(g.floor_bytes, 2 * GIB)

    def test_zero_disables(self):
        g, _ = self.said_default({"memory_guard_floor_gib": 0})
        self.assertEqual(g.floor_bytes, 0)

    def test_the_env_overrides_the_config(self):
        g, _ = self.said_default({"memory_guard_floor_gib": 2}, {"STRATA_MEMORY_GUARD_FLOOR_GIB": "4"})
        self.assertEqual(g.floor_bytes, 4 * GIB)

    def test_a_bad_value_is_said_and_ignored(self):
        g, said = self.said_default({"memory_guard_floor_gib": "plenty"})
        self.assertEqual(g.floor_bytes, 8 * GIB)
        self.assertIn("memory_guard_floor_gib", said)


class CheckTransitions(unittest.TestCase):
    """check() is what the slot wait calls every pass: it decides and says one line when a request starts
    waiting and one when it goes in - never one per pass."""

    def guard(self, available_values):
        """A guard with a fake clock (5 s a pass) and a scripted free-memory reading: one value per check,
        the last one repeating."""
        clock = [0.0]
        seq = list(available_values)
        it = iter(seq)

        def fake_available():
            try:
                return next(it)
            except StopIteration:
                return seq[-1]

        g = MemoryGuard(floor_gib=8, clock=lambda: clock[0], available=fake_available)

        def tick():
            clock[0] += 5.0
        return g, tick

    def lines(self, g, tick, calls):
        out = io.StringIO()
        results = []
        with contextlib.redirect_stdout(out):
            for prompt, max_new in calls:
                results.append(g.check(prompt, max_new))
                tick()
        return results, out.getvalue()

    def test_admits_under_the_floor_and_says_nothing(self):
        g, tick = self.guard([30 * GIB])
        results, said = self.lines(g, tick, [(1000, 1000)])
        self.assertEqual(results, [True])
        self.assertEqual(said, "")

    def test_holds_above_the_floor_then_releases_when_memory_frees(self):
        # 100k cells x 12,672 B = ~1.18 GiB per 100k: 1M cells is ~11.8 GiB of host K/V
        g, tick = self.guard([15 * GIB, 15 * GIB, 15 * GIB, 30 * GIB])
        results, said = self.lines(g, tick, [(1_000_000, 0)] * 3 + [(1_000_000, 0)])
        self.assertEqual(results[:3], [False, False, False])     # held while memory is short
        self.assertTrue(results[3])                              # and in as soon as it frees
        self.assertEqual(said.count("holding a request"), 1)     # one line for the hold
        self.assertEqual(said.count("goes in"), 1)               # one line for the release
        self.assertIn("after 15 s", said)                        # three passes of the fake clock
        self.assertIn("never rejected", said)

    def test_a_hold_then_a_smaller_request_admits_and_ends_the_hold(self):
        # the release line also comes when a SMALLER request fits while the big one still waits
        g, tick = self.guard([15 * GIB, 30 * GIB])
        results, said = self.lines(g, tick, [(1_000_000, 0), (10_000, 0)])
        self.assertEqual(results, [False, True])
        self.assertEqual(said.count("holding a request"), 1)
        self.assertEqual(said.count("goes in"), 1)


class AvailableHostBytes(unittest.TestCase):
    def read(self, files):
        def fake_open(path, *a, **k):
            path = str(path)
            if path not in files:
                raise OSError(path)
            return io.BytesIO(files[path])
        with mock.patch("builtins.open", side_effect=fake_open):
            return available_host_bytes()

    def test_memavailable_alone(self):
        n = self.read({"/proc/meminfo": b"MemTotal: 100 kB\nMemAvailable: 4096 kB\n"})
        self.assertEqual(n, 4096 * 1024)

    def test_a_tighter_cgroup_limit_wins(self):
        n = self.read({
            "/proc/meminfo": b"MemAvailable: 4194304 kB\n",
            "/proc/self/cgroup": b"0::/user.slice/user-1000.slice/strata.service\n",
            "/sys/fs/cgroup/user.slice/user-1000.slice/strata.service/memory.max": b"2147483648",
            "/sys/fs/cgroup/user.slice/user-1000.slice/strata.service/memory.current": b"1073741824",
            "/sys/fs/cgroup/user.slice/user-1000.slice/memory.max": b"max",
            "/sys/fs/cgroup/user.slice/user-1000.slice/memory.current": b"0",
        })
        self.assertEqual(n, 1 * GIB)                             # min(4 GiB MemAvailable, 2 - 1 GiB cgroup)

    def test_max_means_no_limit_from_that_level(self):
        n = self.read({
            "/proc/meminfo": b"MemAvailable: 4194304 kB\n",
            "/proc/self/cgroup": b"0::/\n",
            "/sys/fs/cgroup/memory.max": b"max",
        })
        self.assertEqual(n, 4 * GIB)

    def test_unreadable_memavailable_is_none(self):
        self.assertIsNone(self.read({}))

    def test_clean_inactive_file_cache_is_credited(self):
        # E1 2026-11-02: a group at 69 of 80 GiB, 12 GiB of it clean inactive cache, held a one-word request 20 min.
        n = self.read({
            "/proc/meminfo": b"MemAvailable: 104857600 kB\n",
            "/proc/self/cgroup": b"0::/strata\n",
            "/sys/fs/cgroup/strata/memory.max": str(80 * GIB).encode(),
            "/sys/fs/cgroup/strata/memory.current": str(69 * GIB).encode(),
            "/sys/fs/cgroup/strata/memory.stat": (
                f"anon 1\ninactive_file {12 * GIB}\nactive_file {GIB}\nfile_dirty {GIB}\nfile_writeback 0\n").encode(),
        })
        self.assertEqual(n, (80 - 69 + 12 - 1) * GIB)            # room + inactive file - dirty; active not credited

    def test_the_credit_never_exceeds_usage(self):
        n = self.read({
            "/proc/meminfo": b"MemAvailable: 104857600 kB\n",
            "/proc/self/cgroup": b"0::/strata\n",
            "/sys/fs/cgroup/strata/memory.max": str(10 * GIB).encode(),
            "/sys/fs/cgroup/strata/memory.current": str(2 * GIB).encode(),
            "/sys/fs/cgroup/strata/memory.stat": f"inactive_file {50 * GIB}\n".encode(),
        })
        self.assertEqual(n, 10 * GIB)                            # credit clamps to the 2 GiB charged

    def test_a_missing_memory_stat_gives_no_credit(self):
        n = self.read({
            "/proc/meminfo": b"MemAvailable: 104857600 kB\n",
            "/proc/self/cgroup": b"0::/strata\n",
            "/sys/fs/cgroup/strata/memory.max": str(10 * GIB).encode(),
            "/sys/fs/cgroup/strata/memory.current": str(6 * GIB).encode(),
        })
        self.assertEqual(n, 4 * GIB)

    def test_cgroup_v1_limit_wins_too(self):
        n = self.read({
            "/proc/meminfo": b"MemAvailable: 4194304 kB\n",
            "/proc/self/cgroup": b"11:memory:/strata\n",
            "/sys/fs/cgroup/memory/strata/memory.limit_in_bytes": b"1073741824",
            "/sys/fs/cgroup/memory/strata/memory.usage_in_bytes": b"737418240",
        })
        self.assertEqual(n, 1024 * 1024 * 1024 - 737418240)


class SlotAdmission(unittest.TestCase):
    """The engine-level rule the slot wait uses: a free slot is not taken while the guard holds it, and the
    cells the slot already holds for THIS conversation are credited (a slot that holds a foreign conversation
    or nothing is not)."""

    def engine(self, floor_gib, held):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = 2
        e.slot_order = [0, 1]
        e.slot_busy = [False, False]
        e.slot_held = [held, []]
        e.slot_used = [0.0, 0.0]
        e.slot_live = [None, None]
        e.slot_group = [0, 0]
        e.slot_cv = threading.Condition()
        e.mem_guard = MemoryGuard(floor_gib=floor_gib, available=lambda: 10 * GIB)
        return e

    def blocked(self, e, prompt, left):
        with e.slot_cv:
            return e._memory_blocks_slot(list(prompt), left)

    def test_a_request_the_free_memory_covers_takes_the_slot(self):
        e = self.engine(8, [])
        self.assertFalse(self.blocked(e, [1, 2, 3], 10))          # tiny: admits

    def test_the_same_request_with_a_full_slot_is_held(self):
        # 800k cells x 12,672 B ~ 9.4 GiB on 9 GiB free with an 8 GiB floor: held
        e = self.engine(8, [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.blocked(e, list(range(700_000)), 100_000))

    def test_the_cells_the_slot_holds_for_this_conversation_are_credited(self):
        # the slot holds the prompt's first 699,999 tokens: only the one left to re-read plus the 100k to
        # generate count, which fits (a held prefix is never the whole prompt - pick_slot's own rule)
        e = self.engine(8, list(range(699_999)))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.blocked(e, list(range(700_000)), 100_000))

    def test_a_foreign_conversations_cells_are_not_credited(self):
        e = self.engine(8, [9, 9, 9])                             # not a prefix of this prompt
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.blocked(e, list(range(700_000)), 100_000))

    def test_no_free_slot_is_not_a_memory_hold(self):
        e = self.engine(8, [])
        e.slot_busy = [True, True]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.blocked(e, list(range(700_000)), 100_000))   # the plain slot wait owns it

    def test_a_bare_engine_keeps_todays_behaviour(self):
        # a StrataEngine nothing configured (a test's) has no guard: however big the request, it goes in
        e = self.engine(8, [])
        e.mem_guard = None
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.blocked(e, list(range(2_000_000)), 100_000))


if __name__ == "__main__":
    unittest.main()
