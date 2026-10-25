# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""Native double-precision PocketFFT transforms; no Python dependency."""

from std.ffi import external_call
from std.memory import Pointer

from native_config import GRID_SIZE

comptime N = GRID_SIZE
comptime PIXELS = N * N
comptime FREQUENCIES = N * (N // 2 + 1)
comptime F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime Plan = Pointer[NoneType, MutUntrackedOrigin]


def data(mut values: List[Float64]) -> F64Ptr:
    return values.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


struct FFT[SIZE: Int = 192](Movable):
    comptime N = Self.SIZE
    comptime PIXELS = Self.SIZE * Self.SIZE
    comptime FREQUENCIES = Self.SIZE * (Self.SIZE // 2 + 1)
    var real: List[Float64]
    var complex: List[Float64]
    var forward_plan: Plan
    var inverse_plan: Plan

    def __init__(out self):
        self.real = List[Float64](length=Self.PIXELS, fill=0)
        self.complex = List[Float64](length=Self.FREQUENCIES * 2, fill=0)
        self.forward_plan = external_call["mojosteps_fft_plan", Plan](Int32(Self.N), data(self.real), data(self.complex), Int32(0))
        self.inverse_plan = external_call["mojosteps_fft_plan", Plan](Int32(Self.N), data(self.real), data(self.complex), Int32(1))

    def __deinit__(deinit self):
        external_call["mojosteps_fft_destroy", NoneType](self.forward_plan)
        external_call["mojosteps_fft_destroy", NoneType](self.inverse_plan)

    def forward(mut self, values: F64Ptr):
        for i in range(Self.PIXELS):
            self.real[i] = values[unsafe_offset=i]
        external_call["mojosteps_fft_execute", NoneType](self.forward_plan)

    def inverse(mut self):
        external_call["mojosteps_fft_execute", NoneType](self.inverse_plan)
        for i in range(Self.PIXELS):
            self.real[i] /= Float64(Self.PIXELS)
