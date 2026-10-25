# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""Native MAX GPU mixed-radix 192x192 FFT primitives and focused probes.

Each complex value uses two float32 components per real/imaginary part. This
keeps approximately 48 significant bits without requiring native GPU float64.
The CPU forecast implementation is unchanged.
"""

from std.math import cos, sin, fma
from std.memory import unsafe_stack_allocation
from std.runtime import initialize_runtime
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from native_fft import F64Ptr, N, PIXELS

comptime D2 = SIMD[DType.float32, 2]
comptime C4 = SIMD[DType.float32, 4]


@inline(.always)
def dd_add(a: D2, b: D2) -> D2:
    var s = a[0] + b[0]
    var v = s - a[0]
    var error = (a[0] - (s - v)) + (b[0] - v)
    error += a[1] + b[1]
    var hi = s + error
    return D2(hi, error - (hi - s))


@inline(.always)
def dd_mul(a: D2, b: D2) -> D2:
    var product = a[0] * b[0]
    var error = fma(a[0], b[0], -product)
    error += a[0] * b[1] + a[1] * b[0]
    var hi = product + error
    return D2(hi, error - (hi - product))


@inline(.always)
def complex_add(a: C4, b: C4) -> C4:
    var r = dd_add(D2(a[0], a[1]), D2(b[0], b[1]))
    var i = dd_add(D2(a[2], a[3]), D2(b[2], b[3]))
    return C4(r[0], r[1], i[0], i[1])


@inline(.always)
def complex_mul(a: C4, b: C4) -> C4:
    var ar = D2(a[0], a[1])
    var ai = D2(a[2], a[3])
    var br = D2(b[0], b[1])
    var bi = D2(b[2], b[3])
    var r = dd_add(dd_mul(ar, br), -dd_mul(ai, bi))
    var i = dd_add(dd_mul(ar, bi), dd_mul(ai, br))
    return C4(r[0], r[1], i[0], i[1])


@inline(.always)
def load_complex(p: MutPointer[Float32, MutAnyOrigin], i: Int) -> C4:
    return p.load[width=4](4 * i)


def fft_axis(
    source: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    twiddles: MutPointer[Float32, MutAnyOrigin],
):
    var t = thread_idx.x
    var line = block_idx.x % N
    var batch = block_idx.x // N
    var group = t // 64
    var k = t % 64
    var reverse = 0
    var bits = k
    comptime for _ in range(6):
        reverse = (reverse << 1) | (bits & 1)
        bits >>= 1
    var shared = unsafe_stack_allocation[192, C4, address_space=.SHARED]()
    shared[t] = load_complex(source, batch * PIXELS + line * N + 3 * reverse + group)
    barrier()
    comptime for stage in range(6):
        comptime half = 1 << stage
        var base = group * 64 + (k // (2 * half)) * (2 * half)
        var j = k % half
        var a = shared[base + j]
        var b = complex_mul(shared[base + j + half], load_complex(twiddles, j * (192 // (2 * half))))
        var value = complex_add(a, b if k % (2 * half) < half else -b)
        barrier()
        shared[t] = value
        barrier()
    var a = shared[t % 64]
    var b = complex_mul(shared[64 + t % 64], load_complex(twiddles, t))
    var c = complex_mul(shared[128 + t % 64], load_complex(twiddles, (2 * t) % 192))
    var result = complex_add(complex_add(a, b), c)
    var index = 4 * (batch * PIXELS + t * N + line)
    comptime for component in range(4):
        output[index + component] = result[component]


def inspect_arithmetic(source: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin], twiddles: MutPointer[Float32, MutAnyOrigin], mode: Int32):
    var t = thread_idx.x
    var index = block_idx.x * N + t
    if mode == 8:
        comptime for component in range(4):
            output[4 * index + component] = source[4 * index + component]
        return
    var a = load_complex(source, index)
    var value = a
    if mode == 4:
        value = complex_add(a, C4(0))
    elif mode == 5:
        value = complex_mul(a, load_complex(twiddles, 0))
    elif mode == 6:
        var shared = unsafe_stack_allocation[192, C4, address_space=.SHARED]()
        shared[t] = a
        barrier()
        value = shared[t]
    elif mode == 7:
        value = C4(3, 0, 4, 0)
    comptime for i in range(4):
        output[4 * index + i] = value[i]


struct GpuFFT(Movable):
    var context: DeviceContext
    var first: DeviceBuffer[DType.float32]
    var second: DeviceBuffer[DType.float32]
    var twiddles: DeviceBuffer[DType.float32]
    var batches: Int

    def __init__(out self, batches: Int, inverse: Bool = False) raises:
        self.context = DeviceContext()
        self.batches = batches
        self.first = self.context.enqueue_create_buffer[DType.float32](4 * PIXELS * batches)
        self.second = self.context.enqueue_create_buffer[DType.float32](4 * PIXELS * batches)
        self.twiddles = self.context.enqueue_create_buffer[DType.float32](4 * N)
        with self.twiddles.map_to_host() as table:
            for k in range(N):
                var angle = -2.0 * 3.14159265358979323846 * Float64(k) / N
                if inverse:
                    angle = -angle
                var r = cos(angle)
                var i = sin(angle)
                var rh = Float32(r)
                var ih = Float32(i)
                table[4 * k] = rh
                table[4 * k + 1] = Float32(r - Float64(rh))
                table[4 * k + 2] = ih
                table[4 * k + 3] = Float32(i - Float64(ih))

    def upload(mut self, source: F64Ptr) raises:
        with self.first.map_to_host() as target:
            for i in range(self.batches * PIXELS):
                var r = source[unsafe_offset=2 * i]
                var im = source[unsafe_offset=2 * i + 1]
                var rh = Float32(r)
                var ih = Float32(im)
                target[4 * i] = rh
                target[4 * i + 1] = Float32(r - Float64(rh))
                target[4 * i + 2] = ih
                target[4 * i + 3] = Float32(im - Float64(ih))

    def execute(mut self) raises:
        self.context.enqueue_function[fft_axis](self.first, self.second, self.twiddles, grid_dim=self.batches * N, block_dim=192)
        self.context.enqueue_function[fft_axis](self.second, self.first, self.twiddles, grid_dim=self.batches * N, block_dim=192)

    def download(mut self, output: F64Ptr, inverse: Bool = False) raises:
        self.context.synchronize()
        var scale = Float64(PIXELS) if inverse else 1.0
        with self.first.map_to_host() as result:
            for i in range(self.batches * PIXELS):
                output[unsafe_offset=2 * i] = (Float64(result[4 * i]) + Float64(result[4 * i + 1])) / scale
                output[unsafe_offset=2 * i + 1] = (Float64(result[4 * i + 2]) + Float64(result[4 * i + 3])) / scale


@export
def steps_gpu_fft_probe(source: F64Ptr, output: F64Ptr, batches: Int, inverse: Int32) abi("C") -> Int32:
    try:
        initialize_runtime()
        var fft = GpuFFT(batches, inverse != 0)
        fft.upload(source)
        fft.execute()
        fft.download(output, inverse != 0)
        _ = fft
        return 0
    except error:
        print(error)
        return 1


@export
def steps_gpu_fft_stage_probe(source: F64Ptr, output: F64Ptr, stage: Int32) abi("C") -> Int32:
    try:
        initialize_runtime()
        var fft = GpuFFT(1)
        fft.upload(source)
        if stage == 1 or stage >= 3:
            if stage == 1:
                fft.context.enqueue_function[fft_axis](fft.first, fft.second, fft.twiddles, grid_dim=N, block_dim=192)
            else:
                fft.context.enqueue_function[inspect_arithmetic](fft.first, fft.second, fft.twiddles, stage, grid_dim=N, block_dim=192)
            fft.context.synchronize()
            with fft.second.map_to_host() as result:
                for i in range(PIXELS):
                    output[unsafe_offset=2 * i] = Float64(result[4 * i]) + Float64(result[4 * i + 1])
                    output[unsafe_offset=2 * i + 1] = Float64(result[4 * i + 2]) + Float64(result[4 * i + 3])
        else:
            if stage == 2:
                fft.execute()
            fft.download(output)
        _ = fft
        return 0
    except error:
        print(error)
        return 1
