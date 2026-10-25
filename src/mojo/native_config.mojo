# SPDX-License-Identifier: BSD-3-Clause
# Legacy defaults and runtime physical options for the generic kernels.
comptime GRID_SIZE = 192
comptime KM_PER_PIXEL = 0.5
comptime TIMESTEP = 5.0
comptime PRECIP_THR = 10.0
comptime NORAIN_THR = -10.0
comptime MAX_MEMBERS = 96
comptime MAX_LEADS = 14
comptime MASK_RADIUS = 4
comptime VELOCITY_SCALE = 6.0
comptime MERGE_PASSES = 6

from std.math import isfinite

struct RuntimeOptions(Copyable):
    var kmperpixel: Float64
    var timestep: Float64
    var precip_thr: Float64
    var norain_thr: Float64

    def __init__(out self, kmperpixel: Float64 = 0.5, timestep: Float64 = 5.0, precip_thr: Float64 = 10.0, norain_thr: Float64 = -10.0):
        self.kmperpixel = kmperpixel
        self.timestep = timestep
        self.precip_thr = precip_thr
        self.norain_thr = norain_thr

    def valid(self) -> Bool:
        return isfinite(self.kmperpixel) and isfinite(self.timestep) and isfinite(self.precip_thr) and isfinite(self.norain_thr) and self.kmperpixel >= 0.01 and self.kmperpixel <= 20 and self.timestep >= 0.1 and self.timestep <= 60 and self.norain_thr >= -100 and self.norain_thr < self.precip_thr and self.precip_thr <= 100

    def mask_radius(self) -> Int:
        return max(1, Int((self.timestep / self.kmperpixel - 1.0) / 2.0))

    def velocity_scale(self) -> Float64:
        return 60.0 * self.kmperpixel / self.timestep


def merge_passes(pixels: Int) -> Int:
    var count = 0
    var width = 1024
    while width < pixels:
        width *= 4
        count += 2
    return count
