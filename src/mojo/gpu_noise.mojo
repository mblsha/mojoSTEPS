# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""Batched native GPU FFTs and cascade normalization with cached host output."""

from native_config import RuntimeOptions, merge_passes
from std.runtime import initialize_runtime
from std.time import perf_counter_ns
from std.math import cos, sin, sqrt
from max.algorithm import parallelize
from max.gpu import global_idx, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from native_fft import FFT, F64Ptr, N, PIXELS, FREQUENCIES, data
from native_fields import normalize
from native_core import Model, decompose, LEVELS
from std.ffi import external_call
from std.memory import unsafe_stack_allocation, Pointer
from gpu_fft import D2, C4, dd_add, dd_mul, load_complex, fft_axis

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]


def release_buffer(var buffer: DeviceBuffer[DType.float32]):
    pass


@inline(.always)
def dd_div(a: D2, b: D2) -> D2:
    var first = a[0] / b[0]
    var remainder = dd_add(a, -dd_mul(b, D2(first, 0)))
    var second = (remainder[0] + remainder[1]) / b[0]
    return dd_add(D2(first, 0), D2(second, 0))


def normalize_cascades[SIZE: Int = 192](source: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin]):
    comptime PIXELS = SIZE * SIZE
    comptime INV = Float64(1) / Float64(PIXELS)
    comptime INVERSE = D2(Float32(INV), Float32(INV - Float64(Float32(INV))))
    var field = block_idx.x
    var thread = thread_idx.x
    var sum = D2(0)
    var squares = D2(0)
    for pixel in range(thread, PIXELS, 256):
        var value = source.load[width=2](4 * (field * PIXELS + pixel))
        sum = dd_add(sum, value)
        squares = dd_add(squares, dd_mul(value, value))
    var shared = unsafe_stack_allocation[256, C4, address_space=.SHARED]()
    shared[thread] = C4(sum[0], sum[1], squares[0], squares[1])
    barrier()
    comptime for stage in range(8):
        comptime offset = 128 >> stage
        if thread < offset:
            var a = shared[thread]
            var b = shared[thread + offset]
            var s = dd_add(D2(a[0], a[1]), D2(b[0], b[1]))
            var sq = dd_add(D2(a[2], a[3]), D2(b[2], b[3]))
            shared[thread] = C4(s[0], s[1], sq[0], sq[1])
        barrier()
    var totals = shared[0]
    var mean = dd_mul(D2(totals[0], totals[1]), INVERSE)
    var variance = dd_add(dd_mul(D2(totals[2], totals[3]), INVERSE), -dd_mul(mean, mean))
    var root = sqrt(variance[0])
    var residual = dd_add(variance, -dd_mul(D2(root, 0), D2(root, 0)))
    var deviation = dd_add(D2(root, 0), D2((residual[0] + residual[1]) / (2 * root), 0))
    for pixel in range(thread, PIXELS, 256):
        var i = field * PIXELS + pixel
        var value = source.load[width=2](4 * i)
        var normalized = dd_div(dd_add(value, -mean), deviation)
        output[2 * i] = normalized[0]
        output[2 * i + 1] = normalized[1]


def apply_filters[SIZE: Int = 192](
    spectrum: MutPointer[Float32, MutAnyOrigin],
    filters: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], total: Int32,
):
    comptime PIXELS = SIZE * SIZE
    var i = global_idx.x
    if i >= Int(total):
        return
    var member = i // (LEVELS * PIXELS)
    var level = (i // PIXELS) % LEVELS
    var pixel = i % PIXELS
    var value = load_complex(spectrum, member * PIXELS + pixel)
    var weight = filters.load[width=2](2 * (level * PIXELS + pixel))
    var r = dd_mul(D2(value[0], value[1]), weight)
    var im = dd_mul(D2(value[2], value[3]), weight)
    output[4 * i] = r[0]
    output[4 * i + 1] = r[1]
    output[4 * i + 2] = im[0]
    output[4 * i + 3] = im[1]


struct GpuNoise[SIZE: Int = 192](Movable):
    comptime N = Self.SIZE
    comptime PIXELS = Self.SIZE * Self.SIZE
    comptime FREQUENCIES = Self.SIZE * (Self.SIZE // 2 + 1)
    var context: DeviceContext
    var input: DeviceBuffer[DType.float32]
    var noise_row: DeviceBuffer[DType.float32]
    var spectrum: DeviceBuffer[DType.float32]
    var frequency: DeviceBuffer[DType.float32]
    var cascade_row: DeviceBuffer[DType.float32]
    var cascades: DeviceBuffer[DType.float32]
    var normalized: DeviceBuffer[DType.float32]
    var filters: DeviceBuffer[DType.float32]
    var forward_twiddles: DeviceBuffer[DType.float32]
    var inverse_twiddles: DeviceBuffer[DType.float32]
    var host_input: HostBuffer[DType.float32]
    var host_output: F32Ptr
    var members: Int
    var workers: Int

    def __init__(out self, noise_filter: F64Ptr, weights: F64Ptr, members: Int, workers: Int, initialize_filters: Bool = True) raises:
        self.members = members
        self.workers = workers
        self.context = DeviceContext()
        self.input = self.context.enqueue_create_buffer[DType.float32](4 * members * Self.PIXELS)
        self.noise_row = self.context.enqueue_create_buffer[DType.float32](4 * members * Self.PIXELS)
        self.spectrum = self.context.enqueue_create_buffer[DType.float32](4 * members * Self.PIXELS)
        self.frequency = self.context.enqueue_create_buffer[DType.float32](4 * members * LEVELS * Self.PIXELS)
        self.cascade_row = self.context.enqueue_create_buffer[DType.float32](4 * members * LEVELS * Self.PIXELS)
        self.cascades = self.context.enqueue_create_buffer[DType.float32](4 * members * LEVELS * Self.PIXELS)
        self.filters = self.context.enqueue_create_buffer[DType.float32](2 * LEVELS * Self.PIXELS)
        self.forward_twiddles = self.context.enqueue_create_buffer[DType.float32](4 * Self.N)
        self.inverse_twiddles = self.context.enqueue_create_buffer[DType.float32](4 * Self.N)
        self.host_input = self.context.enqueue_create_host_buffer[DType.float32](4 * members * Self.PIXELS)
        var cached = F32Ptr.unsafe_dangling()
        var bytes = 2 * members * LEVELS * Self.PIXELS * 4
        var status = external_call["posix_memalign", Int32](Pointer(to=cached), Int(16384), bytes)
        if status != 0:
            raise Error("cached GPU readback allocation failed")
        self.host_output = cached
        self.normalized = self.context.wrap_host_memory[DType.float32](cached, 2 * members * LEVELS * Self.PIXELS)
        var host_filter = self.context.enqueue_create_host_buffer[DType.float32](2 * LEVELS * Self.PIXELS)
        if initialize_filters:
            for level in range(LEVELS):
                for y in range(Self.N):
                    for x in range(Self.N):
                        var sy = y
                        var sx = x
                        if x > Self.N // 2:
                            sy = (Self.N - y) % Self.N
                            sx = Self.N - x
                        var source = sy * (Self.N // 2 + 1) + sx
                        var value = noise_filter[unsafe_offset=source] * weights[unsafe_offset=level * Self.FREQUENCIES + source]
                        var high = Float32(value)
                        var index = 2 * (level * Self.PIXELS + y * Self.N + x)
                        host_filter[index] = high
                        host_filter[index + 1] = Float32(value - Float64(high))
        self.filters.enqueue_copy_from(host_filter)
        var forward = self.context.enqueue_create_host_buffer[DType.float32](4 * Self.N)
        var inverse = self.context.enqueue_create_host_buffer[DType.float32](4 * Self.N)
        for i in range(Self.N):
            var angle = -2.0 * 3.14159265358979323846 * Float64(i) / Float64(Self.N)
            var r = cos(angle)
            var im = sin(angle)
            var rh = Float32(r)
            var ih = Float32(im)
            forward[4 * i] = rh
            forward[4 * i + 1] = Float32(r - Float64(rh))
            forward[4 * i + 2] = ih
            forward[4 * i + 3] = Float32(im - Float64(ih))
            inverse[4 * i] = rh
            inverse[4 * i + 1] = Float32(r - Float64(rh))
            inverse[4 * i + 2] = -ih
            inverse[4 * i + 3] = -Float32(im - Float64(ih))
        self.forward_twiddles.enqueue_copy_from(forward)
        self.inverse_twiddles.enqueue_copy_from(inverse)
        self.context.synchronize()

    def __deinit__(deinit self):
        try:
            release_buffer(self.normalized^)
            self.context.synchronize()
            external_call["free", NoneType](self.host_output)
        except error:
            print("GPU buffer cleanup:", error)

    def generate(mut self, white: F64Ptr, output: F64Ptr, times: F64Ptr) raises:
        var start = perf_counter_ns()
        var input = self.host_input.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        def pack(member: Int) {imm}:
            for pixel in range(Self.PIXELS):
                var i = member * Self.PIXELS + pixel
                var value = white[unsafe_offset=i]
                var high = Float32(value)
                input[unsafe_offset=4 * i] = high
                input[unsafe_offset=4 * i + 1] = Float32(value - Float64(high))
                input[unsafe_offset=4 * i + 2] = 0
                input[unsafe_offset=4 * i + 3] = 0
        parallelize(pack, self.members, min(self.members, self.workers))
        self.input.enqueue_copy_from(self.host_input)
        var packed = perf_counter_ns()
        self.context.enqueue_function[fft_axis[Self.SIZE]](self.input, self.noise_row, self.forward_twiddles, grid_dim=self.members * Self.N, block_dim=Self.N)
        self.context.enqueue_function[fft_axis[Self.SIZE]](self.noise_row, self.spectrum, self.forward_twiddles, grid_dim=self.members * Self.N, block_dim=Self.N)
        var total = self.members * LEVELS * Self.PIXELS
        self.context.enqueue_function[apply_filters[Self.SIZE]](self.spectrum, self.filters, self.frequency, Int32(total), grid_dim=(total + 255) // 256, block_dim=256)
        self.context.enqueue_function[fft_axis[Self.SIZE]](self.frequency, self.cascade_row, self.inverse_twiddles, grid_dim=self.members * LEVELS * Self.N, block_dim=Self.N)
        self.context.enqueue_function[fft_axis[Self.SIZE]](self.cascade_row, self.cascades, self.inverse_twiddles, grid_dim=self.members * LEVELS * Self.N, block_dim=Self.N)
        self.context.enqueue_function[normalize_cascades[Self.SIZE]](self.cascades, self.normalized, grid_dim=self.members * LEVELS, block_dim=256)
        self.context.synchronize()
        var executed = perf_counter_ns()
        var results = self.host_output
        def unpack(member: Int) {imm}:
            for level in range(LEVELS):
                for pixel in range(Self.PIXELS):
                    var i = (member * LEVELS + level) * Self.PIXELS + pixel
                    output[unsafe_offset=i] = Float64(results[unsafe_offset=2 * i]) + Float64(results[unsafe_offset=2 * i + 1])
        parallelize(unpack, self.members, min(self.members, self.workers))
        var end = perf_counter_ns()
        times[unsafe_offset=0] = Float64(packed - start) * 1e-9
        times[unsafe_offset=1] = Float64(executed - packed) * 1e-9
        times[unsafe_offset=2] = Float64(end - executed) * 1e-9


@export
def steps_gpu_noise_probe(input: F64Ptr, velocity: F64Ptr, white: F64Ptr, output: F64Ptr, members: Int, workers: Int, runs: Int, times: F64Ptr) abi("C") -> Int32:
    try:
        initialize_runtime()
        var model = Model[192](input, velocity, True)
        var gpu = GpuNoise[192](data(model.noise_filter), data(model.weights), members, workers)
        print("GPU noise backend:", gpu.context.name())
        for run in range(runs):
            gpu.generate(white, output, times + run * 3)
        _ = gpu
        _ = model
        return 0
    except error:
        print(error)
        return 1


@export
def steps_cpu_noise_probe(input: F64Ptr, velocity: F64Ptr, white: F64Ptr, output: F64Ptr, members: Int, workers: Int, runs: Int, times: F64Ptr) abi("C") -> Int32:
    initialize_runtime()
    var model = Model[192](input, velocity, True)
    var filter = data(model.noise_filter)
    var weights = data(model.weights)
    def generate(member: Int) {imm}:
        var fft = FFT[192]()
        var scratch = List[Float64](length=PIXELS, fill=0)
        var means = List[Float64](length=LEVELS, fill=0)
        var deviations = List[Float64](length=LEVELS, fill=0)
        fft.forward(white + member * PIXELS)
        for i in range(FREQUENCIES):
            fft.complex[2 * i] *= filter[unsafe_offset=i]
            fft.complex[2 * i + 1] *= filter[unsafe_offset=i]
        fft.inverse()
        _ = normalize[192](data(fft.real), data(scratch))
        decompose[192](fft, data(fft.real), weights, output + member * LEVELS * PIXELS, data(means), data(deviations))
        _ = scratch
        _ = means
        _ = deviations
    for run in range(runs):
        var start = perf_counter_ns()
        parallelize(generate, members, min(members, workers))
        times[unsafe_offset=run] = Float64(perf_counter_ns() - start) * 1e-9
    _ = model
    return 0
