# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
from native_fft import FFT, F64Ptr, PIXELS, data
from native_rng import MT19937
from native_fields import align, advect, dilated_mask


@export
def native_rng_probe(seed: UInt32, values: F64Ptr, count: Int) abi("C"):
    var rng = MT19937(seed)
    values[unsafe_offset=0] = Float64(rng.below(UInt32(1000000000)))
    for i in range(count):
        values[unsafe_offset=i + 1] = rng.normal()


@export
def native_fft_probe(values: F64Ptr, output: F64Ptr) abi("C"):
    var fft = FFT()
    fft.forward(values)
    fft.inverse()
    for i in range(PIXELS):
        output[unsafe_offset=i] = fft.real[i]


@export
def native_align_probe(values: F64Ptr, velocity: F64Ptr, output: F64Ptr, steps: Int, velocity32: Int32) abi("C"):
    align(values, velocity, steps, output, velocity32 != 0)


@export
def native_mask_probe(values: F64Ptr, output: F64Ptr) abi("C"):
    var distances = List[Int](length=PIXELS, fill=0)
    dilated_mask(values, output, distances.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())


@export
def native_advect_probe(values: F64Ptr, velocity: F64Ptr, output: F64Ptr, steps: Int) abi("C"):
    var displacement = List[Float64](length=2 * PIXELS, fill=0)
    for step in range(steps):
        advect(values, velocity, data(displacement), step == 0, output + step * PIXELS)
