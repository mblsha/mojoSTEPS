# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in synchronized stage timings; absent from ordinary forecast execution.

Build with -D STEPS_GPU_PROFILE=1. Synchronization changes scheduling, so use
these timings to locate work, and benchmark changes with profiling disabled.
"""
from std.sys import get_defined_int
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

comptime ENABLED = get_defined_int["STEPS_GPU_PROFILE", 0]() != 0

struct StageTimer(Movable):
    var previous: Int
    var seconds: List[Float64]

    @inline(.always)
    def __init__(out self):
        self.previous = 0
        self.seconds = List[Float64]()
        comptime if ENABLED:
            self.previous = perf_counter_ns()
            self.seconds = List[Float64](length=12, fill=0)

    @inline(.always)
    def mark(mut self, context: DeviceContext, stage: Int) raises:
        comptime if ENABLED:
            context.synchronize()
            var now = perf_counter_ns()
            self.seconds[stage] += Float64(now - self.previous) * 1e-9
            self.previous = now

    @inline(.always)
    def report(self):
        comptime if ENABLED:
            var names: List[String] = ["setup", "rng", "forward_fft", "filter", "inverse_fft", "normalize", "evolve_mask", "sort", "cdf", "mask_update", "advection", "unpack"]
            for i in range(len(names)):
                print("STEPS_GPU_STAGE", names[i], self.seconds[i])
