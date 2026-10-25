# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""GPU alignment, spectral model fitting, AR coefficients and initial states.

Metal generates the Gaussian/taper tables and fits the input history. The
host supplies seven fixed scalar band constants and marshals input arrays.
"""
from native_config import RuntimeOptions, merge_passes
from native_config import MERGE_PASSES
from std.math import pow, log, exp, sqrt, cos, isfinite
from std.memory import unsafe_stack_allocation, bitcast
from max.gpu import global_idx, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer
from native_fft import N, PIXELS, F64Ptr
from native_core import LEVELS
from gpu_fft import D2, C4, dd_add, dd_mul, fft_axis, load_complex
from gpu_noise import GpuNoise, dd_div
from gpu_rng import dd_sqrt, dd_log
from gpu_fields import (get, put, less, lesser, sample, field_minimum,
    local_sort, merge_sort, PARAM_TARGET, PARAM_DIRECTION, PARAM_VELOCITY, FPtr)


@inline(.always)
def dd_exp(x: D2) -> D2:
    comptime ln2 = 0.693147180559945309417232121458176568
    var k = Int(x[0] * 1.4426950408889634)
    if k < -126:
        return D2(0)
    var r = dd_add(x, -dd_mul(D2(Float32(k), 0), D2(Float32(ln2), Float32(ln2-Float64(Float32(ln2))))))
    var term = D2(1, 0)
    var total = term
    comptime for i in range(1, 19):
        comptime c = Float64(1) / Float64(i)
        term = dd_mul(dd_mul(term, r), D2(Float32(c), Float32(c-Float64(Float32(c)))))
        total = dd_add(total, term)
    var scale = bitcast[DType.float32](UInt32(k+127) << 23)
    return dd_mul(total, D2(scale, 0))


@inline(.always)
def dd_cos(x: D2) -> D2:
    var negative_square = -dd_mul(x, x)
    var term = D2(1, 0)
    var total = term
    comptime for i in range(1, 19):
        comptime c = Float64(1) / Float64((2*i-1)*(2*i))
        term = dd_mul(dd_mul(term, negative_square), D2(Float32(c), Float32(c-Float64(Float32(c)))))
        total = dd_add(total, term)
    return total


def fixed_tables[SIZE: Int = 192](constants: FPtr, weights: FPtr, taper: FPtr):
    comptime N = SIZE
    comptime PIXELS = SIZE * SIZE
    var p = global_idx.x
    if p >= PIXELS:
        return
    var x = p % N
    var y = p // N
    var xx = min(x, N-x)
    var yy = min(y, N-y)
    var radius = dd_sqrt(D2(Float32(xx*xx+yy*yy), 0))
    var logarithm = D2(0)
    if radius[0] > 0:
        logarithm = dd_mul(dd_log(radius), get(constants, 6))
    var values = SIMD[DType.float32, 16](0)
    var total = D2(0)
    for level in range(LEVELS):
        var d = dd_add(logarithm, -get(constants, level))
        var w = dd_exp(dd_mul(D2(-2, 0), dd_mul(d, d)))
        values[2*level] = w[0]
        values[2*level+1] = w[1]
        total = dd_add(total, w)
    for level in range(LEVELS):
        var w = dd_div(D2(values[2*level], values[2*level+1]), total)
        if radius[0] == 0:
            w = D2(Float32(Int(level == 0)), 0)
        put(weights, level*PIXELS+p, w)
    var tx = dd_add(dd_div(D2(Float32(x), 0), D2(Float32(N), 0)), D2(-.5, 0))
    var ty = dd_add(dd_div(D2(Float32(y), 0), D2(Float32(N), 0)), D2(-.5, 0))
    radius = dd_sqrt(dd_add(dd_mul(tx, tx), dd_mul(ty, ty)))
    var value = D2(1, 0)
    if radius[0] >= .5:
        value = D2(0)
    elif radius[0] > .4:
        # Use the same binary64 0.1 and pi constants as the reference.
        comptime tenth = .1
        comptime pi = 3.14159265358979323846
        var angle = dd_mul(dd_add(dd_div(radius, D2(Float32(tenth), Float32(tenth-Float64(Float32(tenth))))), D2(-4, 0)), D2(Float32(pi), Float32(pi-Float64(Float32(pi)))))
        value = dd_mul(D2(.5, 0), dd_add(D2(1, 0), dd_cos(angle)))
    put(taper, p, value)


def align_history[SIZE: Int = 192](source: FPtr, aligned: FPtr, params: FPtr, v32: Int32):
    comptime N = SIZE
    comptime PIXELS = SIZE * SIZE
    comptime PARAM_DIRECTION = 60 + 2 * PIXELS
    comptime PARAM_VELOCITY = 60 + 6 * PIXELS
    var i = global_idx.x
    if i >= 3 * PIXELS:
        return
    var frame = i // PIXELS
    var p = i % PIXELS
    var velocity = source + 6 * PIXELS
    var x = D2(Float32(p % N), 0)
    var y = D2(Float32(p // N), 0)
    var vx = get(velocity, p)
    var vy = get(velocity, PIXELS + p)
    var dx = D2(0)
    var dy = D2(0)
    var low = get(params, 30 + frame)
    for step in range(2 - frame):
        var hx = dd_add(x, dd_add(dx, -dd_mul(vx, D2(.5, 0))))
        var hy = dd_add(y, dd_add(dy, -dd_mul(vy, D2(.5, 0))))
        vx = sample[SIZE](velocity, 0, hx, hy, True)
        vy = sample[SIZE](velocity, PIXELS, hx, hy, True)
        if v32 != 0:
            vx = D2(vx[0] + vx[1], 0)
            vy = D2(vy[0] + vy[1], 0)
        dx = dd_add(dx, -vx)
        dy = dd_add(dy, -vy)
        vx = sample[SIZE](velocity, 0, dd_add(x, dx), dd_add(y, dy), True)
        vy = sample[SIZE](velocity, PIXELS, dd_add(x, dx), dd_add(y, dy), True)
        if v32 != 0:
            vx = D2(vx[0] + vx[1], 0)
            vy = D2(vy[0] + vy[1], 0)
    var xx = dd_add(x, dx)
    var yy = dd_add(y, dy)
    var value = low
    if not (less(xx, D2(0)) or less(yy, D2(0)) or less(D2(Float32(N - 1), 0), xx) or less(D2(Float32(N - 1), 0), yy)):
        value = sample[SIZE](source, frame * PIXELS, xx, yy, False, v32 != 0)
    put(aligned, i, value)
    if frame == 0:
        var original_x = get(velocity, p)
        var original_y = get(velocity, PIXELS + p)
        put(params + PARAM_VELOCITY, p, original_x)
        put(params + PARAM_VELOCITY, PIXELS + p, original_y)
        var length = dd_sqrt(dd_add(dd_mul(original_x, original_x), dd_mul(original_y, original_y)))
        var ux = D2(0)
        var uy = D2(0)
        if v32 != 0:
            var length32 = sqrt(original_x[0] * original_x[0] + original_y[0] * original_y[0])
            if length32 > 1e-12:
                ux = D2(original_x[0] / length32, 0)
                uy = D2(original_y[0] / length32, 0)
        elif length[0] > 1e-12:
            ux = dd_div(original_x, length)
            uy = dd_div(original_y, length)
        put(params + PARAM_DIRECTION, p, ux)
        put(params + PARAM_DIRECTION, PIXELS + p, uy)


def input_lows[SIZE: Int = 192](source: FPtr, params: FPtr):
    comptime PIXELS = SIZE * SIZE
    var frame = block_idx.x
    var t = thread_idx.x
    var low = D2(3.4e38, 0)
    for i in range(t, PIXELS, 256):
        low = lesser(low, get(source, frame * PIXELS + i))
    var shared = unsafe_stack_allocation[256, D2, address_space=.SHARED]()
    shared[t] = low
    barrier()
    comptime for stage in range(8):
        comptime h = 128 >> stage
        if t < h:
            shared[t] = lesser(shared[t], shared[t+h])
        barrier()
    if t == 0:
        put(params, 30 + frame, shared[0])


def remove_discontinuity[SIZE: Int = 192](source: FPtr, lows: FPtr, output: FPtr):
    comptime PIXELS = SIZE * SIZE
    var t = thread_idx.x
    var low = lesser(lesser(get(lows, 0), get(lows, 1)), get(lows, 2))
    var first = D2(3.4e38, 0)
    for i in range(t, 3 * PIXELS, 256):
        var value = get(source, i)
        if less(low, value):
            first = lesser(first, value)
    var shared = unsafe_stack_allocation[256, D2, address_space=.SHARED]()
    shared[t] = first
    barrier()
    comptime for stage in range(8):
        comptime h = 128 >> stage
        if t < h:
            shared[t] = lesser(shared[t], shared[t+h])
        barrier()
    var discontinuity = dd_add(shared[0], -low)
    for i in range(t, 3 * PIXELS, 256):
        var value = get(source, i)
        if less(low, value):
            value = dd_add(value, -discontinuity)
        put(output, i, value)


def prepare_fft[SIZE: Int = 192](source: FPtr, output: FPtr, lows: FPtr, taper: FPtr, use_taper: Int32):
    comptime PIXELS = SIZE * SIZE
    var i = global_idx.x
    if i >= 3 * PIXELS:
        return
    var value = get(source, i)
    if use_taper != 0:
        value = dd_mul(dd_add(value, -get(lows, i // PIXELS)), get(taper, i % PIXELS))
    output[4*i] = value[0]
    output[4*i+1] = value[1]
    output[4*i+2] = 0
    output[4*i+3] = 0


def model_filters[SIZE: Int = 192](spectrum: FPtr, weights: FPtr, filters: FPtr):
    comptime PIXELS = SIZE * SIZE
    var i = global_idx.x
    if i >= PIXELS:
        return
    var re = D2(0)
    var im = D2(0)
    for f in range(3):
        var c = load_complex(spectrum, f * PIXELS + i)
        re = dd_add(re, D2(c[0], c[1]))
        im = dd_add(im, D2(c[2], c[3]))
    re = dd_div(re, D2(3, 0))
    im = dd_div(im, D2(3, 0))
    var magnitude = dd_sqrt(dd_add(dd_mul(re, re), dd_mul(im, im)))
    for level in range(LEVELS):
        put(filters, level * PIXELS + i, dd_mul(magnitude, get(weights, level * PIXELS + i)))


def model_bands[SIZE: Int = 192](spectrum: FPtr, weights: FPtr, output: FPtr):
    comptime PIXELS = SIZE * SIZE
    var i = global_idx.x
    if i >= 3 * LEVELS * PIXELS:
        return
    var p = i % PIXELS
    var level = (i // PIXELS) % LEVELS
    var frame = i // (LEVELS * PIXELS)
    var c = load_complex(spectrum, frame * PIXELS + p)
    var w = get(weights, level * PIXELS + p)
    var re = dd_mul(D2(c[0], c[1]), w)
    var im = dd_mul(D2(c[2], c[3]), w)
    output[4*i] = re[0]
    output[4*i+1] = re[1]
    output[4*i+2] = im[0]
    output[4*i+3] = im[1]


def normalize_model[SIZE: Int = 192](source: FPtr, output: FPtr, params: FPtr):
    comptime PIXELS = SIZE * SIZE
    comptime inv = Float64(1) / Float64(PIXELS)
    comptime inverse = D2(Float32(inv), Float32(inv - Float64(Float32(inv))))
    var f = block_idx.x
    var t = thread_idx.x
    var sum = D2(0)
    var squares = D2(0)
    for i in range(t, PIXELS, 256):
        var v = source.load[width=2](4 * (f * PIXELS + i))
        sum = dd_add(sum, v)
        squares = dd_add(squares, dd_mul(v, v))
    var shared = unsafe_stack_allocation[256, C4, address_space=.SHARED]()
    shared[t] = C4(sum[0], sum[1], squares[0], squares[1])
    barrier()
    comptime for stage in range(8):
        comptime h = 128 >> stage
        if t < h:
            var a = shared[t]
            var b = shared[t+h]
            var s = dd_add(D2(a[0], a[1]), D2(b[0], b[1]))
            var q = dd_add(D2(a[2], a[3]), D2(b[2], b[3]))
            shared[t] = C4(s[0], s[1], q[0], q[1])
        barrier()
    var total = shared[0]
    var mean = dd_mul(D2(total[0], total[1]), inverse)
    var variance = dd_add(dd_mul(D2(total[2], total[3]), inverse), -dd_mul(mean, mean))
    var deviation = dd_sqrt(variance)
    if t == 0 and f >= 2 * LEVELS:
        put(params, f % LEVELS, dd_mul(mean, inverse))
        put(params, 6 + f % LEVELS, dd_mul(deviation, inverse))
    for i in range(t, PIXELS, 256):
        put(output, f * PIXELS + i, dd_div(dd_add(source.load[width=2](4 * (f * PIXELS + i)), -mean), deviation))


def fit_ar[SIZE: Int = 192](cascades: FPtr, params: FPtr):
    comptime PIXELS = SIZE * SIZE
    var level = block_idx.x
    var t = thread_idx.x
    # Normalized cascades have zero mean. Their residual paired-float means
    # contribute below 1e-24 to these centered cross products.
    var a1 = D2(0)
    var a2 = D2(0)
    var aa = D2(0)
    var b1 = D2(0)
    var b2 = D2(0)
    for i in range(t, PIXELS, 256):
        var a = get(cascades, (2 * LEVELS + level) * PIXELS + i)
        var lag1 = get(cascades, (LEVELS + level) * PIXELS + i)
        var lag2 = get(cascades, level * PIXELS + i)
        a1 = dd_add(a1, dd_mul(a, lag1))
        a2 = dd_add(a2, dd_mul(a, lag2))
        aa = dd_add(aa, dd_mul(a, a))
        b1 = dd_add(b1, dd_mul(lag1, lag1))
        b2 = dd_add(b2, dd_mul(lag2, lag2))
    var shared = unsafe_stack_allocation[1280, D2, address_space=.SHARED]()
    shared[t] = a1
    shared[256+t] = a2
    shared[512+t] = aa
    shared[768+t] = b1
    shared[1024+t] = b2
    barrier()
    comptime for stage in range(8):
        comptime h = 128 >> stage
        if t < h:
            for j in range(5):
                shared[j*256+t] = dd_add(shared[j*256+t], shared[j*256+t+h])
        barrier()
    if t == 0:
        var root = dd_sqrt(shared[512])
        var g1 = dd_div(dd_div(shared[0], root), dd_sqrt(shared[768]))
        var g2 = dd_div(dd_div(shared[256], root), dd_sqrt(shared[1024]))
        g1 = lesser(D2(1, 0), g1)
        if less(g1, D2(-1, 0)):
            g1 = D2(-1, 0)
        g2 = lesser(D2(1, 0), g2)
        if less(g2, D2(-1, 0)):
            g2 = D2(-1, 0)
        var bound1 = dd_add(dd_mul(D2(2, 0), dd_mul(g1, g2)), D2(-1, 0))
        if less(g2, bound1):
            g2 = bound1
        var square = dd_mul(g1, g1)
        var complement = dd_add(D2(1, 0), -square)
        var bound = D2(0)
        if square[0] > 1e-20:
            bound = dd_div(dd_add(dd_add(dd_mul(D2(3, 0), square), D2(-2, 0)), dd_mul(D2(2, 0), dd_mul(complement, dd_sqrt(complement)))), square)
        if less(g2, bound):
            g2 = bound
        var p1 = dd_div(dd_add(g2, -square), complement)
        var p0 = dd_add(g1, -dd_mul(g1, p1))
        var residual = dd_add(dd_add(D2(1, 0), -dd_mul(g1, p0)), -dd_mul(g2, p1))
        if less(residual, D2(0)):
            residual = D2(0)
        put(params, 12 + 3 * level, p0)
        put(params, 12 + 3 * level + 1, p1)
        put(params, 12 + 3 * level + 2, dd_sqrt(residual))


def finish_model[SIZE: Int = 192](cascades: FPtr, sorted_target: FPtr, initial: FPtr, params: FPtr, source: FPtr):
    comptime PIXELS = SIZE * SIZE
    comptime PARAM_TARGET = 60
    var i = global_idx.x
    if i < 2 * LEVELS * PIXELS:
        put(initial, i, get(cascades, LEVELS * PIXELS + i))
    if i < PIXELS:
        put(initial, 2 * LEVELS * PIXELS + i, get(source, 2 * PIXELS + i))
        put(params + PARAM_TARGET, i, get(sorted_target, i))


def copy_latest[SIZE: Int = 192](source: FPtr, latest: FPtr):
    comptime PIXELS = SIZE * SIZE
    var i = global_idx.x
    if i < PIXELS:
        put(latest, i, get(source, 2 * PIXELS + i))


def validate_model(params: FPtr, status: MutPointer[Int32, MutAnyOrigin]):
    var valid = True
    for i in range(30):
        var value = get(params, i)
        valid = valid and isfinite(value[0]) and isfinite(value[1])
        if i >= 6 and i < 12:
            valid = valid and value[0] > 0
    status[0] = Int32(valid)


def append_pair(mut values: List[Float32], v: Float64):
    var hi = Float32(v)
    values.append(hi)
    values.append(Float32(v - Float64(hi)))


struct GpuModel[SIZE: Int = 192](Movable):
    comptime N = Self.SIZE
    comptime PIXELS = Self.SIZE * Self.SIZE
    comptime MERGE_PASSES = merge_passes(Self.PIXELS)
    var constants: DeviceBuffer[DType.float32]
    var valid: Bool
    var latest: DeviceBuffer[DType.float32]
    var source: DeviceBuffer[DType.float32]
    var weights: DeviceBuffer[DType.float32]
    var taper: DeviceBuffer[DType.float32]
    var aligned: DeviceBuffer[DType.float32]
    var prepared: DeviceBuffer[DType.float32]
    var lows: DeviceBuffer[DType.float32]
    var fft_input: DeviceBuffer[DType.float32]
    var row: DeviceBuffer[DType.float32]
    var spectrum: DeviceBuffer[DType.float32]
    var bands: DeviceBuffer[DType.float32]
    var band_row: DeviceBuffer[DType.float32]
    var band_result: DeviceBuffer[DType.float32]
    var normalized: DeviceBuffer[DType.float32]
    var keys0: DeviceBuffer[DType.float32]
    var keys1: DeviceBuffer[DType.float32]
    var indices0: DeviceBuffer[DType.int32]
    var indices1: DeviceBuffer[DType.int32]

    def __init__(out self, mut gpu: GpuNoise[Self.SIZE], input: F64Ptr, velocity: F64Ptr, initial: DeviceBuffer[DType.float32], params: DeviceBuffer[DType.float32], v32: Int32) raises:
        var host_source = List[Float32]()
        for i in range(3 * Self.PIXELS):
            append_pair(host_source, input[unsafe_offset=i])
        for i in range(2 * Self.PIXELS):
            append_pair(host_source, velocity[unsafe_offset=i])
        var host_constants = List[Float32]()
        var q = pow(Float64(Self.N) / 2, 1.0 / LEVELS)
        for level in range(LEVELS):
            append_pair(host_constants, log((pow(q, Float64(level)) + pow(q, Float64(level+1))) / 2) / log(q))
        append_pair(host_constants, 1.0 / log(q))
        self.valid = False
        self.latest = gpu.context.enqueue_create_buffer[DType.float32](2 * Self.PIXELS)
        self.source = gpu.context.enqueue_create_buffer[DType.float32](10 * Self.PIXELS)
        self.weights = gpu.context.enqueue_create_buffer[DType.float32](2 * LEVELS * Self.PIXELS)
        self.taper = gpu.context.enqueue_create_buffer[DType.float32](2 * Self.PIXELS)
        self.source.enqueue_copy_from(host_source)
        self.constants = gpu.context.enqueue_create_buffer[DType.float32](14)
        self.constants.enqueue_copy_from(host_constants)
        gpu.context.enqueue_function[fixed_tables[Self.SIZE]](self.constants, self.weights, self.taper, grid_dim=Self.PIXELS//256, block_dim=256)
        self.aligned = gpu.context.enqueue_create_buffer[DType.float32](6 * Self.PIXELS)
        self.prepared = gpu.context.enqueue_create_buffer[DType.float32](6 * Self.PIXELS)
        self.lows = gpu.context.enqueue_create_buffer[DType.float32](6)
        self.fft_input = gpu.context.enqueue_create_buffer[DType.float32](12 * Self.PIXELS)
        self.row = gpu.context.enqueue_create_buffer[DType.float32](12 * Self.PIXELS)
        self.spectrum = gpu.context.enqueue_create_buffer[DType.float32](12 * Self.PIXELS)
        self.bands = gpu.context.enqueue_create_buffer[DType.float32](12 * LEVELS * Self.PIXELS)
        self.band_row = gpu.context.enqueue_create_buffer[DType.float32](12 * LEVELS * Self.PIXELS)
        self.band_result = gpu.context.enqueue_create_buffer[DType.float32](12 * LEVELS * Self.PIXELS)
        self.normalized = gpu.context.enqueue_create_buffer[DType.float32](6 * LEVELS * Self.PIXELS)
        self.keys0 = gpu.context.enqueue_create_buffer[DType.float32](2 * Self.PIXELS)
        self.keys1 = gpu.context.enqueue_create_buffer[DType.float32](2 * Self.PIXELS)
        self.indices0 = gpu.context.enqueue_create_buffer[DType.int32](Self.PIXELS)
        self.indices1 = gpu.context.enqueue_create_buffer[DType.int32](Self.PIXELS)
        gpu.context.enqueue_function[input_lows[Self.SIZE]](self.source, params, grid_dim=3, block_dim=256)
        gpu.context.enqueue_function[align_history[Self.SIZE]](self.source, self.aligned, params, v32, grid_dim=3*Self.PIXELS//256, block_dim=256)
        gpu.context.enqueue_function[field_minimum[Self.SIZE]](self.aligned, self.lows, grid_dim=3, block_dim=256)
        gpu.context.enqueue_function[remove_discontinuity[Self.SIZE]](self.aligned, self.lows, self.prepared, grid_dim=1, block_dim=256)
        gpu.context.enqueue_function[field_minimum[Self.SIZE]](self.prepared, self.lows, grid_dim=3, block_dim=256)
        gpu.context.enqueue_function[prepare_fft[Self.SIZE]](self.prepared, self.fft_input, self.lows, self.taper, Int32(1), grid_dim=3*Self.PIXELS//256, block_dim=256)
        gpu.context.enqueue_function[fft_axis[Self.SIZE]](self.fft_input, self.row, gpu.forward_twiddles, grid_dim=3*Self.N, block_dim=Self.N)
        gpu.context.enqueue_function[fft_axis[Self.SIZE]](self.row, self.spectrum, gpu.forward_twiddles, grid_dim=3*Self.N, block_dim=Self.N)
        gpu.context.enqueue_function[model_filters[Self.SIZE]](self.spectrum, self.weights, gpu.filters, grid_dim=Self.PIXELS//256, block_dim=256)
        gpu.context.enqueue_function[prepare_fft[Self.SIZE]](self.aligned, self.fft_input, self.lows, self.taper, Int32(0), grid_dim=3*Self.PIXELS//256, block_dim=256)
        gpu.context.enqueue_function[fft_axis[Self.SIZE]](self.fft_input, self.row, gpu.forward_twiddles, grid_dim=3*Self.N, block_dim=Self.N)
        gpu.context.enqueue_function[fft_axis[Self.SIZE]](self.row, self.spectrum, gpu.forward_twiddles, grid_dim=3*Self.N, block_dim=Self.N)
        gpu.context.enqueue_function[model_bands[Self.SIZE]](self.spectrum, self.weights, self.bands, grid_dim=3*LEVELS*Self.PIXELS//256, block_dim=256)
        gpu.context.enqueue_function[fft_axis[Self.SIZE]](self.bands, self.band_row, gpu.inverse_twiddles, grid_dim=3*LEVELS*Self.N, block_dim=Self.N)
        gpu.context.enqueue_function[fft_axis[Self.SIZE]](self.band_row, self.band_result, gpu.inverse_twiddles, grid_dim=3*LEVELS*Self.N, block_dim=Self.N)
        gpu.context.enqueue_function[normalize_model[Self.SIZE]](self.band_result, self.normalized, params, grid_dim=3*LEVELS, block_dim=256)
        gpu.context.enqueue_function[fit_ar[Self.SIZE]](self.normalized, params, grid_dim=LEVELS, block_dim=256)
        # Use a separate view of the latest source frame without any readback.
        gpu.context.enqueue_function[copy_latest[Self.SIZE]](self.source, self.latest, grid_dim=Self.PIXELS//256, block_dim=256)
        gpu.context.enqueue_function[local_sort[Self.SIZE]](self.latest, self.keys0, self.indices0, grid_dim=Self.PIXELS//1024, block_dim=256)
        comptime for stage in range(Self.MERGE_PASSES):
            if stage % 2 == 0:
                gpu.context.enqueue_function[merge_sort[Self.SIZE]](self.keys0, self.indices0, self.keys1, self.indices1, Int32(1024 << stage), Int32(Self.PIXELS), grid_dim=Self.PIXELS//256, block_dim=256)
            else:
                gpu.context.enqueue_function[merge_sort[Self.SIZE]](self.keys1, self.indices1, self.keys0, self.indices0, Int32(1024 << stage), Int32(Self.PIXELS), grid_dim=Self.PIXELS//256, block_dim=256)
        gpu.context.enqueue_function[finish_model[Self.SIZE]](self.normalized, self.keys0, initial, params, self.source, grid_dim=2*LEVELS*Self.PIXELS//256, block_dim=256)
        var status = gpu.context.enqueue_create_buffer[DType.int32](1)
        gpu.context.enqueue_function[validate_model](params, status, grid_dim=1, block_dim=1)
        with status.map_to_host() as host_status:
            self.valid = host_status[0] != 0
        gpu.context.synchronize()
        _ = host_source
        _ = host_constants
