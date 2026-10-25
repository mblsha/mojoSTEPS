# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""MT19937 and legacy polar normals compatible with NumPy RandomState."""

from std.math import log, sqrt
from std.ffi import external_call


struct MT19937(Movable):
    var state: List[UInt32]
    var position: Int
    var cached: Float64
    var has_cached: Bool

    def __init__(out self, seed: UInt32):
        self.state = List[UInt32](length=624, fill=0)
        self.state[0] = seed
        for i in range(1, 624):
            var previous = self.state[i - 1]
            self.state[i] = UInt32(1812433253) * (previous ^ (previous >> 30)) + UInt32(i)
        self.position = 624
        self.cached = 0
        self.has_cached = False

    def next_u32(mut self) -> UInt32:
        if self.position >= 624:
            for i in range(624):
                var value = (self.state[i] & UInt32(0x80000000)) | (self.state[(i + 1) % 624] & UInt32(0x7fffffff))
                self.state[i] = self.state[(i + 397) % 624] ^ (value >> 1)
                if value & 1:
                    self.state[i] ^= UInt32(0x9908b0df)
            self.position = 0
        var value = self.state[self.position]
        self.position += 1
        value ^= value >> 11
        value ^= (value << 7) & UInt32(0x9d2c5680)
        value ^= (value << 15) & UInt32(0xefc60000)
        value ^= value >> 18
        return value

    def uniform(mut self) -> Float64:
        var a = self.next_u32() >> 5
        var b = self.next_u32() >> 6
        return (Float64(a) * 67108864.0 + Float64(b)) / 9007199254740992.0

    def below(mut self, limit: UInt32) -> UInt32:
        var mask = limit - 1
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        var value = self.next_u32() & mask
        while value >= limit:
            value = self.next_u32() & mask
        return value

    def normal(mut self) -> Float64:
        if self.has_cached:
            self.has_cached = False
            return self.cached
        var x = 0.0
        var y = 0.0
        var radius = 2.0
        while radius >= 1.0 or radius == 0.0:
            x = 2.0 * self.uniform() - 1.0
            y = 2.0 * self.uniform() - 1.0
            radius = x * x + y * y
        var scale = external_call["sqrt", Float64](-2.0 * external_call["log", Float64](radius) / radius)
        self.cached = scale * x
        self.has_cached = True
        return scale * y

    def laplace(mut self) -> Float64:
        var value = self.uniform()
        if value >= 0.5:
            return -external_call["log", Float64](2.0 - value - value) / sqrt(2.0)
        return external_call["log", Float64](value + value) / sqrt(2.0)
