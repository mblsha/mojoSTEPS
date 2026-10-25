# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""Native GPU forecast stages, with staged variants for matched benchmarks.

The all-GPU entrypoint fits the model, creates reference-compatible random
streams and evolves the ensemble on Metal. Host work validates/marshals input,
launches kernels and converts the final paired-float output to caller float64.
"""
from native_config import RuntimeOptions, merge_passes
from native_config import TIMESTEP, PRECIP_THR, MAX_MEMBERS, MAX_LEADS, MERGE_PASSES
from std.ffi import external_call
from std.runtime import initialize_runtime
from std.math import isfinite, pow
from std.memory import Pointer
from max.algorithm import parallelize
from max.gpu.host import DeviceBuffer, DeviceContext
from native_fft import N, PIXELS, F64Ptr, data
from native_core import Model, LEVELS, forecast_native
from native_rng import MT19937
from gpu_rng import generate_normals, initialize_random
from gpu_model import GpuModel
from gpu_core import steps_forecast_native, steps_forecast_gpu, steps_gpu_model_probe
from gpu_noise import GpuNoise, F32Ptr, normalize_cascades, apply_filters, release_buffer
from gpu_fft import fft_axis
from gpu_profile import StageTimer

from gpu_fields import (
    PARAM_TARGET, PARAM_DIRECTION, PARAM_VELOCITY, PARAM_EPS,
    initialize_members, evolve, field_minimum, apply_mask, count_dry,
    local_sort, merge_sort, merge_sorted_tiles, cdf_match, prepare_cdf, apply_cdf, mask_rows, mask_columns, perturb, advection,
)


struct CachedResult(Movable):
    var context: DeviceContext
    var buffer: DeviceBuffer[DType.float32]
    var pointer: F32Ptr

    def __init__(out self, context: DeviceContext, count: Int) raises:
        self.context = context
        var pointer = F32Ptr.unsafe_dangling()
        if external_call["posix_memalign", Int32](Pointer(to=pointer), Int(16384), count * 4) != 0:
            raise Error("cached result allocation failed")
        self.pointer = pointer
        try:
            self.buffer = self.context.wrap_host_memory[DType.float32](pointer, count)
        except error:
            external_call["free", NoneType](pointer)
            raise error^

    def __deinit__(deinit self):
        try:
            self.context.synchronize()
            release_buffer(self.buffer^)
            external_call["free", NoneType](self.pointer)
        except error:
            print("GPU result cleanup:", error)


def pack(mut host: List[Float32], source: F64Ptr, count: Int):
    for i in range(count):
        var hi = Float32(source[unsafe_offset=i])
        host.append(hi)
        host.append(Float32(source[unsafe_offset=i] - Float64(hi)))


@export
def steps_resident_model_probe(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, velocity_float32: Int32) abi("C"):
    steps_gpu_model_probe(input, velocity, output, velocity_float32)


def forecast_resident[SIZE: Int, GPU_RNG: Bool, GPU_MODEL: Bool = False, GPU_SEED: Bool = False](input: F64Ptr, velocity: F64Ptr, output: F64Ptr, members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32, options: RuntimeOptions = RuntimeOptions()) -> Int32:
    comptime N = SIZE
    comptime PIXELS = SIZE * SIZE
    comptime PARAM_EPS = 60 + 10 * PIXELS
    comptime MERGE_PASSES = merge_passes(PIXELS)
    if members < 1 or members > 4096 or leads < 1 or leads > 288 or workers < 1 or velocity_float32 < 0 or velocity_float32 > 1:
        return 1
    var echoes = 0
    for i in range(3 * PIXELS):
        if not isfinite(input[unsafe_offset=i]):
            return 2
        echoes += Int(input[unsafe_offset=i] >= options.precip_thr)
    for i in range(2 * PIXELS):
        if not isfinite(velocity[unsafe_offset=i]):
            return 2
    if echoes < 16:
        return forecast_native[SIZE](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    try:
        initialize_runtime()
        var timer = StageTimer()
        var model = Optional[Model[SIZE]]()
        var noise_filter = input
        var weights = input
        if not GPU_MODEL:
            model = Optional[Model[SIZE]](Model[SIZE](input, velocity, velocity_float32 != 0))
            noise_filter = data(model.value().noise_filter)
            weights = data(model.value().weights)
            for i in range(LEVELS):
                if not isfinite(model.value().deviations[i]) or model.value().deviations[i] <= 0:
                    return 2
            for i in range(3 * LEVELS):
                if not isfinite(model.value().coefficients[i]):
                    return 2
        var gpu = GpuNoise[SIZE](noise_filter, weights, members, workers, not GPU_MODEL)
        var rngs = List[MT19937]()
        var params = List[Float32]()
        if not GPU_MODEL:
            pack(params, data(model.value().means), LEVELS)
            pack(params, data(model.value().deviations), LEVELS)
            pack(params, data(model.value().coefficients), 3 * LEVELS)
            pack(params, data(model.value().target), PIXELS)
            pack(params, data(model.value().direction), 2 * PIXELS)
            pack(params, velocity, 2 * PIXELS)
        var rng_state = List[UInt32]()
        var rng_position = List[Int32]()
        if GPU_MODEL and not GPU_SEED:
            params = List[Float32](length=PARAM_EPS, fill=0)
        if not GPU_SEED:
            var next_seed = seed
            for m in range(members):
                var rng = MT19937(next_seed)
                next_seed = rng.below(UInt32(1000000000))
                var motion = MT19937(next_seed)
                next_seed = motion.below(UInt32(1000000000))
                var eps = List[Float64]()
                eps.append(motion.laplace())
                eps.append(motion.laplace())
                pack(params, data(eps), 2)
                _ = eps
                for i in range(624):
                    rng_state.append(rng.state[i])
                rng_position.append(Int32(rng.position))
                rngs.append(rng^)
        var initial = List[Float32]()
        if not GPU_MODEL:
            pack(initial, data(model.value().initial), 2 * LEVELS * PIXELS)
            pack(initial, input + 2 * PIXELS, PIXELS)
        var initial_gpu = gpu.context.enqueue_create_buffer[DType.float32](2 * (2 * LEVELS + 1) * PIXELS)
        if not GPU_MODEL:
            initial_gpu.enqueue_copy_from(initial)
        var params_gpu = gpu.context.enqueue_create_buffer[DType.float32](PARAM_EPS + 4 * members)
        if not GPU_MODEL or not GPU_SEED:
            params_gpu.enqueue_copy_from(params)
        var gpu_model = Optional[GpuModel[SIZE]]()
        if GPU_MODEL:
            gpu_model = Optional[GpuModel[SIZE]](GpuModel[SIZE](gpu, input, velocity, initial_gpu, params_gpu, velocity_float32))
            if not gpu_model.value().valid:
                return 2
        var rng_state_gpu = gpu.context.enqueue_create_buffer[DType.uint32](members * 624)
        var motion_state_gpu = gpu.context.enqueue_create_buffer[DType.uint32](624)
        var rng_position_gpu = gpu.context.enqueue_create_buffer[DType.int32](members)
        if GPU_SEED:
            gpu.context.enqueue_function[initialize_random[SIZE]](rng_state_gpu, motion_state_gpu, rng_position_gpu, params_gpu, seed, Int32(members), grid_dim=1, block_dim=1)
        else:
            rng_state_gpu.enqueue_copy_from(rng_state)
            rng_position_gpu.enqueue_copy_from(rng_position)
        var state = gpu.context.enqueue_create_buffer[DType.float32](2 * members * 2 * LEVELS * PIXELS)
        var field = gpu.context.enqueue_create_buffer[DType.float32](2 * members * PIXELS)
        var matched = gpu.context.enqueue_create_buffer[DType.float32](2 * members * PIXELS)
        var mask = gpu.context.enqueue_create_buffer[DType.float32](2 * members * PIXELS)
        var distance = gpu.context.enqueue_create_buffer[DType.int32](members * PIXELS)
        var displacement = gpu.context.enqueue_create_buffer[DType.float32](4 * members * PIXELS)
        var perturbed = gpu.context.enqueue_create_buffer[DType.float32](4 * members * PIXELS)
        var lows = gpu.context.enqueue_create_buffer[DType.float32](2 * members)
        var dry = gpu.context.enqueue_create_buffer[DType.int32](members)
        var cutoffs = gpu.context.enqueue_create_buffer[DType.float32](2 * members)
        var keys0 = gpu.context.enqueue_create_buffer[DType.float32](2 * members * PIXELS)
        var keys1 = gpu.context.enqueue_create_buffer[DType.float32](2 * members * PIXELS)
        var indices0 = gpu.context.enqueue_create_buffer[DType.int32](members * PIXELS)
        var indices1 = gpu.context.enqueue_create_buffer[DType.int32](members * PIXELS)
        var readback = CachedResult(gpu.context, 2 * members * leads * PIXELS)
        var cached = readback.pointer
        gpu.context.enqueue_function[initialize_members[SIZE]](initial_gpu, state, field, displacement, Int32(members), grid_dim=(members * 2 * LEVELS * PIXELS + 255) // 256, block_dim=256)
        gpu.context.enqueue_function[mask_rows[SIZE]](field, distance, Float32(options.precip_thr), Float32(options.precip_thr - Float64(Float32(options.precip_thr))), grid_dim=members * N // 64, block_dim=64)
        gpu.context.enqueue_function[mask_columns[SIZE]](distance, mask, Int32(options.mask_radius()), grid_dim=members * N // 64, block_dim=64)
        var rp = rngs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var host_input = gpu.host_input.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var total = members * PIXELS
        timer.mark(gpu.context, 0)
        for lead in range(leads):
            if GPU_RNG:
                gpu.context.enqueue_function[generate_normals[SIZE]](rng_state_gpu, rng_position_gpu, gpu.input, grid_dim=members, block_dim=256)
            else:
                # Synchronize before overwriting the previous upload's host storage.
                gpu.context.synchronize()
                def white(m: Int) {imm}:
                    for p in range(PIXELS):
                        var value = rp[unsafe_offset=m].normal()
                        var high = Float32(value)
                        var i = m * PIXELS + p
                        host_input[unsafe_offset=4 * i] = high
                        host_input[unsafe_offset=4 * i + 1] = Float32(value - Float64(high))
                        host_input[unsafe_offset=4 * i + 2] = 0
                        host_input[unsafe_offset=4 * i + 3] = 0
                parallelize(white, members, min(workers, members))
                gpu.input.enqueue_copy_from(gpu.host_input)
            timer.mark(gpu.context, 1)
            gpu.context.enqueue_function[fft_axis[SIZE]](gpu.input, gpu.noise_row, gpu.forward_twiddles, grid_dim=members * N, block_dim=N)
            gpu.context.enqueue_function[fft_axis[SIZE]](gpu.noise_row, gpu.spectrum, gpu.forward_twiddles, grid_dim=members * N, block_dim=N)
            timer.mark(gpu.context, 2)
            gpu.context.enqueue_function[apply_filters[SIZE]](gpu.spectrum, gpu.filters, gpu.frequency, Int32(total * LEVELS), grid_dim=(total * LEVELS + 255) // 256, block_dim=256)
            timer.mark(gpu.context, 3)
            gpu.context.enqueue_function[fft_axis[SIZE]](gpu.frequency, gpu.cascade_row, gpu.inverse_twiddles, grid_dim=members * LEVELS * N, block_dim=N)
            gpu.context.enqueue_function[fft_axis[SIZE]](gpu.cascade_row, gpu.cascades, gpu.inverse_twiddles, grid_dim=members * LEVELS * N, block_dim=N)
            timer.mark(gpu.context, 4)
            gpu.context.enqueue_function[normalize_cascades[SIZE]](gpu.cascades, gpu.normalized, grid_dim=members * LEVELS, block_dim=256)
            timer.mark(gpu.context, 5)
            gpu.context.enqueue_function[evolve[SIZE]](state, gpu.normalized, params_gpu, field, Int32(members), grid_dim=(total + 255) // 256, block_dim=256)
            gpu.context.enqueue_function[field_minimum[SIZE]](field, lows, grid_dim=members, block_dim=256)
            gpu.context.enqueue_function[apply_mask[SIZE]](field, mask, lows, Int32(members), grid_dim=(total + 255) // 256, block_dim=256)
            gpu.context.enqueue_function[count_dry[SIZE]](field, lows, dry, grid_dim=members, block_dim=256)
            timer.mark(gpu.context, 6)
            gpu.context.enqueue_function[local_sort[SIZE]](field, keys0, indices0, grid_dim=total // 1024, block_dim=256)
            # An even pass count leaves the result in keys0.
            comptime for stage in range(MERGE_PASSES):
                if stage % 2 == 0:
                    gpu.context.enqueue_function[merge_sorted_tiles[SIZE]](keys0, indices0, keys1, indices1, Int32(1024 << stage), Int32(total), grid_dim=total // 1024, block_dim=256)
                else:
                    gpu.context.enqueue_function[merge_sorted_tiles[SIZE]](keys1, indices1, keys0, indices0, Int32(1024 << stage), Int32(total), grid_dim=total // 1024, block_dim=256)
            timer.mark(gpu.context, 7)
            gpu.context.enqueue_function[prepare_cdf[SIZE]](dry, params_gpu, cutoffs, Int32(members), grid_dim=(members + 127) // 128, block_dim=128)
            gpu.context.enqueue_function[apply_cdf[SIZE]](keys0, indices0, lows, params_gpu, cutoffs, matched, Int32(members), grid_dim=(total + 255) // 256, block_dim=256)
            timer.mark(gpu.context, 8)
            gpu.context.enqueue_function[mask_rows[SIZE]](matched, distance, Float32(options.precip_thr), Float32(options.precip_thr - Float64(Float32(options.precip_thr))), grid_dim=members * N // 64, block_dim=64)
            gpu.context.enqueue_function[mask_columns[SIZE]](distance, mask, Int32(options.mask_radius()), grid_dim=members * N // 64, block_dim=64)
            timer.mark(gpu.context, 9)
            var minutes = Float64(lead + 1) * options.timestep
            var along = 10.88 * pow(minutes, 0.23) - 7.68
            var across = 5.76 * pow(minutes, 0.31) - 2.72
            var ah = Float32(along)
            var ch = Float32(across)
            gpu.context.enqueue_function[perturb[SIZE]](params_gpu, perturbed, ah, Float32(along - Float64(ah)), ch, Float32(across - Float64(ch)), Int32(members), Float32(options.velocity_scale()), Float32(options.velocity_scale() - Float64(Float32(options.velocity_scale()))), grid_dim=(total + 255) // 256, block_dim=256)
            gpu.context.enqueue_function[advection[SIZE]](matched, perturbed, displacement, readback.buffer, Int32(lead), Int32(leads), Int32(members), grid_dim=(total + 255) // 256, block_dim=256)
            timer.mark(gpu.context, 10)
        gpu.context.synchronize()
        def unpack(m: Int) {imm}:
            for p in range(leads * PIXELS):
                var i = m * leads * PIXELS + p
                output[unsafe_offset=i] = Float64(cached[unsafe_offset=2 * i]) + Float64(cached[unsafe_offset=2 * i + 1])
        parallelize(unpack, members, min(members, workers))
        timer.mark(gpu.context, 11)
        timer.report()
        _ = readback
        _ = gpu_model
        _ = gpu
        _ = model
        _ = rng_state
        _ = rng_position
        _ = rngs
        _ = params
        _ = initial
        return 0
    except error:
        print("GPU-resident forecast failed:", error)
        return 3


@export
def steps_forecast_gpu_resident(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32) abi("C") -> Int32:
    return forecast_resident[192, False](input, velocity, output, members, leads, seed, workers, velocity_float32)


@export
def steps_forecast_gpu_rng(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32) abi("C") -> Int32:
    return forecast_resident[192, True](input, velocity, output, members, leads, seed, workers, velocity_float32)


@export
def steps_forecast_gpu_all(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32) abi("C") -> Int32:
    return forecast_resident[192, True, True, True](input, velocity, output, members, leads, seed, workers, velocity_float32)


@export
def steps_all_model_probe(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, velocity_float32: Int32) abi("C"):
    try:
        initialize_runtime()
        var gpu = GpuNoise[192](input, input, 1, 1, False)
        var initial = gpu.context.enqueue_create_buffer[DType.float32](2 * (2 * LEVELS + 1) * PIXELS)
        var params = gpu.context.enqueue_create_buffer[DType.float32](PARAM_EPS)
        var model = GpuModel[192](gpu, input, velocity, initial, params, velocity_float32)
        with params.map_to_host() as host:
            for i in range(30):
                output[unsafe_offset=i] = Float64(host[2*i]) + Float64(host[2*i+1])
        with gpu.filters.map_to_host() as host:
            for y in range(N):
                for x in range(N//2+1):
                    var value = 0.0
                    for level in range(LEVELS):
                        var i = 2 * (level * PIXELS + y*N+x)
                        value += Float64(host[i]) + Float64(host[i+1])
                    output[unsafe_offset=30+y*(N//2+1)+x] = value
        _ = model
        _ = gpu
    except error:
        print("GPU model probe failed:", error)


@export
def steps_forecast_gpu_model(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32) abi("C") -> Int32:
    return forecast_resident[192, True, True, False](input, velocity, output, members, leads, seed, workers, velocity_float32)


@export
def steps_gpu_seed_probe(seed: UInt32, members: Int, output: F64Ptr) abi("C"):
    try:
        initialize_runtime()
        var ctx = DeviceContext()
        var states = ctx.enqueue_create_buffer[DType.uint32](members * 624)
        var motion = ctx.enqueue_create_buffer[DType.uint32](624)
        var positions = ctx.enqueue_create_buffer[DType.int32](members)
        var params = ctx.enqueue_create_buffer[DType.float32](PARAM_EPS + 4 * members)
        ctx.enqueue_function[initialize_random[192]](states, motion, positions, params, seed, Int32(members), grid_dim=1, block_dim=1)
        with positions.map_to_host() as host:
            for m in range(members):
                output[unsafe_offset=m*627] = Float64(host[m])
        with params.map_to_host() as host:
            for m in range(members):
                for k in range(2):
                    var i = PARAM_EPS + 4*m+2*k
                    output[unsafe_offset=m*627+1+k] = Float64(host[i]) + Float64(host[i+1])
        with states.map_to_host() as host:
            for m in range(members):
                for k in range(624):
                    output[unsafe_offset=m*627+3+k] = Float64(host[m*624+k])
    except error:
        print("seed probe:", error)


@export
def steps_gpu_initial_probe(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, v32: Int32, backend: Int32) abi("C"):
    from native_fields import align
    if backend == 0:
        var model = Model[192](input, velocity, v32 != 0)
        for i in range(2 * LEVELS * PIXELS):
            output[unsafe_offset=i] = model.initial[i]
        for i in range(2 * PIXELS):
            output[unsafe_offset=2 * LEVELS * PIXELS+i] = model.direction[i]
        align[192](input, velocity, 2, output + (2*LEVELS+2)*PIXELS, v32 != 0)
        align[192](input + PIXELS, velocity, 1, output + (2*LEVELS+3)*PIXELS, v32 != 0)
        for i in range(PIXELS):
            output[unsafe_offset=(2*LEVELS+4)*PIXELS+i] = input[unsafe_offset=2*PIXELS+i]
        _ = model
    else:
        try:
            initialize_runtime()
            var gpu = GpuNoise[192](input, input, 1, 1, False)
            var initial = gpu.context.enqueue_create_buffer[DType.float32](2 * (2 * LEVELS + 1) * PIXELS)
            var params = gpu.context.enqueue_create_buffer[DType.float32](PARAM_EPS)
            var model = GpuModel[192](gpu, input, velocity, initial, params, v32)
            with initial.map_to_host() as host:
                for i in range(2 * LEVELS * PIXELS):
                    output[unsafe_offset=i] = Float64(host[2*i]) + Float64(host[2*i+1])
            with params.map_to_host() as host:
                for i in range(2 * PIXELS):
                    var j = PARAM_DIRECTION + 2*i
                    output[unsafe_offset=2*LEVELS*PIXELS+i] = Float64(host[j]) + Float64(host[j+1])
            with model.aligned.map_to_host() as host:
                for i in range(3*PIXELS):
                    output[unsafe_offset=(2*LEVELS+2)*PIXELS+i] = Float64(host[2*i]) + Float64(host[2*i+1])
            _ = model
            _ = gpu
        except error:
            print("initial probe:", error)


@export
def steps_gpu_fields_probe(input: F64Ptr, target: F64Ptr, output: F64Ptr) abi("C") -> Int32:
    try:
        initialize_runtime()
        var ctx = DeviceContext()
        var host_field = List[Float32]()
        var host_params = List[Float32](length=PARAM_TARGET, fill=0)
        pack(host_field, input, PIXELS)
        pack(host_params, target, PIXELS)
        var field = ctx.enqueue_create_buffer[DType.float32](2 * PIXELS)
        var params = ctx.enqueue_create_buffer[DType.float32](len(host_params))
        field.enqueue_copy_from(host_field)
        params.enqueue_copy_from(host_params)
        var lows = ctx.enqueue_create_buffer[DType.float32](2)
        var dry = ctx.enqueue_create_buffer[DType.int32](1)
        var k0 = ctx.enqueue_create_buffer[DType.float32](2*PIXELS)
        var k1 = ctx.enqueue_create_buffer[DType.float32](2*PIXELS)
        var i0 = ctx.enqueue_create_buffer[DType.int32](PIXELS)
        var i1 = ctx.enqueue_create_buffer[DType.int32](PIXELS)
        var matched = ctx.enqueue_create_buffer[DType.float32](2*PIXELS)
        var distance = ctx.enqueue_create_buffer[DType.int32](PIXELS)
        var mask = ctx.enqueue_create_buffer[DType.float32](2*PIXELS)
        ctx.enqueue_function[field_minimum[192]](field, lows, grid_dim=1, block_dim=256)
        ctx.enqueue_function[count_dry[192]](field, lows, dry, grid_dim=1, block_dim=256)
        ctx.enqueue_function[local_sort[192]](field, k0, i0, grid_dim=PIXELS//1024, block_dim=256)
        comptime for stage in range(MERGE_PASSES):
            if stage % 2 == 0:
                ctx.enqueue_function[merge_sorted_tiles[192]](k0, i0, k1, i1, Int32(1024 << stage), Int32(PIXELS), grid_dim=PIXELS//1024, block_dim=256)
            else:
                ctx.enqueue_function[merge_sorted_tiles[192]](k1, i1, k0, i0, Int32(1024 << stage), Int32(PIXELS), grid_dim=PIXELS//1024, block_dim=256)
        var cutoffs = ctx.enqueue_create_buffer[DType.float32](2)
        ctx.enqueue_function[prepare_cdf[192]](dry, params, cutoffs, Int32(1), grid_dim=1, block_dim=128)
        ctx.enqueue_function[apply_cdf[192]](k0, i0, lows, params, cutoffs, matched, Int32(1), grid_dim=PIXELS//256, block_dim=256)
        ctx.enqueue_function[mask_rows[192]](field, distance, Float32(10), Float32(0), grid_dim=N//64, block_dim=64)
        ctx.enqueue_function[mask_columns[192]](distance, mask, Int32(4), grid_dim=N//64, block_dim=64)
        with matched.map_to_host() as host:
            for i in range(PIXELS):
                output[unsafe_offset=i] = Float64(host[2*i]) + Float64(host[2*i+1])
        with mask.map_to_host() as host:
            for i in range(PIXELS):
                output[unsafe_offset=PIXELS+i] = Float64(host[2*i]) + Float64(host[2*i+1])
        _ = host_field
        _ = host_params
        return 0
    except error:
        print("field probe:", error)
        return 3


@export
def steps_forecast_gpu_v2(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, size: Int, members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32, kmperpixel: Float64, timestep: Float64, precip_thr: Float64, norain_thr: Float64) abi("C") -> Int32:
    var options = RuntimeOptions(kmperpixel, timestep, precip_thr, norain_thr)
    if not options.valid():
        return 1
    if size == 64:
        return forecast_resident[64, True, True, True](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    if size == 128:
        return forecast_resident[128, True, True, True](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    if size == 192:
        return forecast_resident[192, True, True, True](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    # Large-grid CDF ranks need the reference-compatible float64 CPU fit.
    if size == 256:
        return forecast_resident[256, True, False, True](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    if size == 512:
        return forecast_resident[512, True, False, True](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    return 1


def sort_probe[SIZE: Int](input: F64Ptr, output: F64Ptr, members: Int) raises:
    comptime PIXELS = SIZE * SIZE
    comptime PASSES = merge_passes(PIXELS)
    var context = DeviceContext()
    var host = List[Float32]()
    pack(host, input, members * PIXELS)
    var field = context.enqueue_create_buffer[DType.float32](2 * members * PIXELS)
    field.enqueue_copy_from(host)
    var k0 = context.enqueue_create_buffer[DType.float32](2 * members * PIXELS)
    var k1 = context.enqueue_create_buffer[DType.float32](2 * members * PIXELS)
    var i0 = context.enqueue_create_buffer[DType.int32](members * PIXELS)
    var i1 = context.enqueue_create_buffer[DType.int32](members * PIXELS)
    context.enqueue_function[local_sort[SIZE]](field, k0, i0, grid_dim=members * PIXELS // 1024, block_dim=256)
    comptime for stage in range(PASSES):
        if stage % 2 == 0:
            context.enqueue_function[merge_sorted_tiles[SIZE]](k0, i0, k1, i1, Int32(1024 << stage), Int32(members * PIXELS), grid_dim=members * PIXELS // 1024, block_dim=256)
        else:
            context.enqueue_function[merge_sorted_tiles[SIZE]](k1, i1, k0, i0, Int32(1024 << stage), Int32(members * PIXELS), grid_dim=members * PIXELS // 1024, block_dim=256)
    with k0.map_to_host() as keys:
        for i in range(members * PIXELS):
            output[unsafe_offset=i] = Float64(keys[2 * i]) + Float64(keys[2 * i + 1])
    with i0.map_to_host() as indices:
        for i in range(members * PIXELS):
            output[unsafe_offset=members * PIXELS + i] = Float64(indices[i])
    _ = host


@export
def steps_gpu_sort_probe(input: F64Ptr, output: F64Ptr, size: Int, members: Int) abi("C") -> Int32:
    try:
        initialize_runtime()
        if members < 1 or members > 4096:
            return 1
        if size == 64:
            sort_probe[64](input, output, members)
        elif size == 128:
            sort_probe[128](input, output, members)
        elif size == 192:
            sort_probe[192](input, output, members)
        elif size == 256:
            sort_probe[256](input, output, members)
        elif size == 512:
            sort_probe[512](input, output, members)
        else:
            return 1
        return 0
    except error:
        print("sort probe:", error)
        return 3
