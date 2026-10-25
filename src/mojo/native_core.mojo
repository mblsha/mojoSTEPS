# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""Standalone generic STEPS forecast core. No Python imports.

The numerical method follows pySTEPS 1.21.2's spatial STEPS defaults used by
this project. PocketFFT is a native math dependency; MAX supplies CPU parallelism.
"""

from std.ffi import external_call
from std.math import sqrt, exp, log, cos, pow, isfinite
from std.runtime import initialize_runtime
from max.algorithm import parallelize
from native_fft import FFT, F64Ptr, N, PIXELS, FREQUENCIES, data
from native_rng import MT19937
from native_config import RuntimeOptions
from native_fields import (
    IndexPtr, sum_values, normalize, statistics, minimum, align, advect,
    dilated_mask, sort_indices, probability_match,
)

comptime LEVELS = 6
comptime PI = 3.141592653589793238462643383279502884


def decompose[SIZE: Int = 192](
    mut fft: FFT[SIZE], source: F64Ptr, weights: F64Ptr, cascades: F64Ptr,
    means: F64Ptr, deviations: F64Ptr,
):
    comptime PIXELS = SIZE * SIZE
    comptime FREQUENCIES = SIZE * (SIZE // 2 + 1)
    fft.forward(source)
    var spectrum = List[Float64](length=2 * FREQUENCIES, fill=0)
    var scratch = List[Float64](length=PIXELS, fill=0)
    for i in range(2 * FREQUENCIES):
        spectrum[i] = fft.complex[i]
    for level in range(LEVELS):
        for i in range(FREQUENCIES):
            var weight = weights[unsafe_offset=level * FREQUENCIES + i]
            fft.complex[2 * i] = spectrum[2 * i] * weight
            fft.complex[2 * i + 1] = spectrum[2 * i + 1] * weight
        fft.inverse()
        var mean, deviation = normalize[SIZE](data(fft.real), data(scratch))
        means[unsafe_offset=level] = mean
        deviations[unsafe_offset=level] = deviation
        for i in range(PIXELS):
            cascades[unsafe_offset=level * PIXELS + i] = fft.real[i]


def correlation[SIZE: Int = 192](a: F64Ptr, b: F64Ptr) -> Float64:
    comptime PIXELS = SIZE * SIZE
    var ma = sum_values(a, PIXELS) / Float64(PIXELS)
    var mb = sum_values(b, PIXELS) / Float64(PIXELS)
    # Pairwise products reduce correlation round-off amplified by CDF ranks.
    comptime if SIZE >= 256:
        var products = List[Float64](length=3 * PIXELS, fill=0)
        for i in range(PIXELS):
            var da = a[unsafe_offset=i] - ma
            var db = b[unsafe_offset=i] - mb
            products[i] = da * db
            products[PIXELS + i] = da * da
            products[2 * PIXELS + i] = db * db
        var ab = sum_values(data(products), PIXELS)
        var aa = sum_values(data(products) + PIXELS, PIXELS)
        var bb = sum_values(data(products) + 2 * PIXELS, PIXELS)
        return max(-1.0, min(1.0, ab / sqrt(aa) / sqrt(bb)))
    else:
        var ab = 0.0
        var aa = 0.0
        var bb = 0.0
        for i in range(PIXELS):
            var da = a[unsafe_offset=i] - ma
            var db = b[unsafe_offset=i] - mb
            ab += da * db
            aa += da * da
            bb += db * db
        return max(-1.0, min(1.0, ab / sqrt(aa) / sqrt(bb)))



struct Model[SIZE: Int = 192](Movable):
    comptime N = Self.SIZE
    comptime PIXELS = Self.SIZE * Self.SIZE
    comptime FREQUENCIES = Self.SIZE * (Self.SIZE // 2 + 1)
    var weights: List[Float64]
    var noise_filter: List[Float64]
    var initial: List[Float64]
    var means: List[Float64]
    var deviations: List[Float64]
    var coefficients: List[Float64]
    var target: List[Float64]
    var direction: List[Float64]

    def __init__(out self, input: F64Ptr, velocity: F64Ptr, velocity32: Bool):
        self.weights = List[Float64](length=LEVELS * Self.FREQUENCIES, fill=0)
        self.noise_filter = List[Float64](length=Self.FREQUENCIES, fill=0)
        self.initial = List[Float64](length=2 * LEVELS * Self.PIXELS, fill=0)
        self.means = List[Float64](length=LEVELS, fill=0)
        self.deviations = List[Float64](length=LEVELS, fill=0)
        self.coefficients = List[Float64](length=LEVELS * 3, fill=0)
        self.target = List[Float64](length=Self.PIXELS, fill=0)
        self.direction = List[Float64](length=2 * Self.PIXELS, fill=0)
        var q = pow(Float64(Self.N) / 2.0, 1.0 / LEVELS)
        var centers = List[Float64](length=LEVELS, fill=0)
        for level in range(LEVELS):
            var radius = (pow(q, Float64(level)) + pow(q, Float64(level + 1))) / 2.0
            centers[level] = log(radius) / log(q)
        for y in range(Self.N):
            var fy = min(y, Self.N - y)
            for x in range(Self.N // 2 + 1):
                var i = y * (Self.N // 2 + 1) + x
                var radius = sqrt(Float64(x * x + fy * fy))
                var logarithm = 0.0
                if radius > 0:
                    logarithm = log(radius) / log(q)
                var total = 0.0
                for level in range(LEVELS):
                    var difference = logarithm - centers[level]
                    var weight = exp(-(difference * difference) / 0.5)
                    self.weights[level * Self.FREQUENCIES + i] = weight
                    total += weight
                for level in range(LEVELS):
                    self.weights[level * Self.FREQUENCIES + i] /= total
        for level in range(LEVELS):
            self.weights[level * Self.FREQUENCIES] = Float64(level == 0)

        var aligned = List[Float64](length=3 * Self.PIXELS, fill=0)
        align[Self.SIZE](input, velocity, 2, data(aligned), velocity32)
        align[Self.SIZE](input + Self.PIXELS, velocity, 1, data(aligned) + Self.PIXELS, velocity32)
        for i in range(Self.PIXELS):
            aligned[2 * Self.PIXELS + i] = input[unsafe_offset=2 * Self.PIXELS + i]
            var vx = velocity[unsafe_offset=i]
            var vy = velocity[unsafe_offset=Self.PIXELS + i]
            var length = sqrt(vx * vx + vy * vy)
            if velocity32:
                var x32 = Float32(vx)
                var y32 = Float32(vy)
                var length32 = sqrt(x32 * x32 + y32 * y32)
                length = Float64(length32)
            if length > 1e-12:
                self.direction[i] = vx / length
                self.direction[Self.PIXELS + i] = vy / length
                if velocity32:
                    self.direction[i] = Float64(Float32(vx) / Float32(length))
                    self.direction[Self.PIXELS + i] = Float64(Float32(vy) / Float32(length))

        var low = minimum[Self.SIZE](data(aligned))
        for i in range(3 * Self.PIXELS):
            low = min(low, aligned[i])
        var first_wet = Float64(1e300)
        for i in range(3 * Self.PIXELS):
            if aligned[i] > low:
                first_wet = min(first_wet, aligned[i])
        var discontinuity = first_wet - low
        var fft = FFT[Self.SIZE]()
        var temporary = List[Float64](length=Self.PIXELS, fill=0)
        var sum_spectrum = List[Float64](length=2 * Self.FREQUENCIES, fill=0)
        for frame in range(3):
            var frame_min = Float64(1e300)
            for i in range(Self.PIXELS):
                var value = aligned[frame * Self.PIXELS + i]
                if value > low:
                    value -= discontinuity
                temporary[i] = value
                frame_min = min(frame_min, value)
            for i in range(Self.PIXELS):
                var xx = Float64(i % Self.N) / Float64(Self.N) - 0.5
                var yy = Float64(i // Self.N) / Float64(Self.N) - 0.5
                var radius = sqrt(xx * xx + yy * yy)
                var taper = 1.0
                if radius >= 0.5:
                    taper = 0.0
                elif radius > 0.4:
                    taper = 0.5 * (1.0 + cos(PI * (radius / 0.1 - 4.0)))
                temporary[i] = (temporary[i] - frame_min) * taper
            fft.forward(data(temporary))
            for i in range(2 * Self.FREQUENCIES):
                sum_spectrum[i] += fft.complex[i]
        for i in range(Self.FREQUENCIES):
            var re = sum_spectrum[2 * i] / 3.0
            var im = sum_spectrum[2 * i + 1] / 3.0
            self.noise_filter[i] = sqrt(re * re + im * im)

        var cascades = List[Float64](length=3 * LEVELS * Self.PIXELS, fill=0)
        for frame in range(3):
            decompose[Self.SIZE](fft, data(aligned) + frame * Self.PIXELS, data(self.weights), data(cascades) + frame * LEVELS * Self.PIXELS, data(self.means), data(self.deviations))
        for level in range(LEVELS):
            var latest = data(cascades) + 2 * LEVELS * Self.PIXELS + level * Self.PIXELS
            var lag1 = data(cascades) + LEVELS * Self.PIXELS + level * Self.PIXELS
            var lag2 = data(cascades) + level * Self.PIXELS
            var g1 = correlation[Self.SIZE](latest, lag1)
            var g2 = correlation[Self.SIZE](latest, lag2)
            g2 = max(g2, 2.0 * g1 * g2 - 1.0)
            var square = g1 * g1
            var bound = 0.0
            if square > 1e-20:
                bound = (3.0 * square - 2.0 + 2.0 * pow(1.0 - square, 1.5)) / square
            g2 = max(g2, bound)
            var p1 = (g2 - square) / (1.0 - square)
            var p0 = g1 - g1 * p1
            var innovation = 1.0 - g1 * p0
            innovation -= g2 * p1
            self.coefficients[3 * level] = p0
            self.coefficients[3 * level + 1] = p1
            self.coefficients[3 * level + 2] = sqrt(max(0.0, innovation))
            for i in range(Self.PIXELS):
                self.initial[level * Self.PIXELS + i] = lag1[unsafe_offset=i]
                self.initial[(LEVELS + level) * Self.PIXELS + i] = latest[unsafe_offset=i]
        var order = List[Int](length=Self.PIXELS, fill=0)
        for i in range(Self.PIXELS):
            order[i] = i
        sort_indices(input + 2 * Self.PIXELS, order.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 0, Self.PIXELS - 1)
        for i in range(Self.PIXELS):
            self.target[i] = input[unsafe_offset=2 * Self.PIXELS + order[i]]
        _ = cascades


def simulate_member[SIZE: Int = 192](
    weights: F64Ptr, noise_filter: F64Ptr, initial: F64Ptr,
    means: F64Ptr, deviations: F64Ptr, coefficients: F64Ptr,
    target: F64Ptr, direction: F64Ptr, velocity: F64Ptr,
    latest: F64Ptr, seed: UInt32, eps_par: Float64, eps_perp: Float64,
    leads: Int, output: F64Ptr, options: RuntimeOptions,
):
    comptime PIXELS = SIZE * SIZE
    comptime FREQUENCIES = SIZE * (SIZE // 2 + 1)
    var rng = MT19937(seed)
    _ = rng.below(UInt32(1000000000))
    var fft = FFT[SIZE]()
    var state = List[Float64](length=2 * LEVELS * PIXELS, fill=0)
    for i in range(2 * LEVELS * PIXELS):
        state[i] = initial[unsafe_offset=i]
    var noise = List[Float64](length=PIXELS, fill=0)
    var noise_cascades = List[Float64](length=LEVELS * PIXELS, fill=0)
    var noise_means = List[Float64](length=LEVELS, fill=0)
    var noise_deviations = List[Float64](length=LEVELS, fill=0)
    var scratch = List[Float64](length=PIXELS, fill=0)
    var field = List[Float64](length=PIXELS, fill=0)
    var matched = List[Float64](length=PIXELS, fill=0)
    var mask = List[Float64](length=PIXELS, fill=0)
    var displacement = List[Float64](length=2 * PIXELS, fill=0)
    var perturbed_velocity = List[Float64](length=2 * PIXELS, fill=0)
    var indices = List[Int](length=PIXELS, fill=0)
    var ip = indices.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    dilated_mask[SIZE](latest, data(mask), ip, options.precip_thr, options.mask_radius())
    for lead in range(leads):
        for i in range(PIXELS):
            noise[i] = rng.normal()
        fft.forward(data(noise))
        for i in range(FREQUENCIES):
            fft.complex[2 * i] *= noise_filter[unsafe_offset=i]
            fft.complex[2 * i + 1] *= noise_filter[unsafe_offset=i]
        fft.inverse()
        _ = normalize[SIZE](data(fft.real), data(scratch))
        decompose[SIZE](fft, data(fft.real), weights, data(noise_cascades), data(noise_means), data(noise_deviations))
        for level in range(LEVELS):
            var p0 = coefficients[unsafe_offset=level * 3]
            var p1 = coefficients[unsafe_offset=level * 3 + 1]
            var pe = coefficients[unsafe_offset=level * 3 + 2]
            for i in range(PIXELS):
                var old_index = level * PIXELS + i
                var recent_index = (LEVELS + level) * PIXELS + i
                var recent = state[recent_index]
                var value = p0 * recent
                value += p1 * state[old_index]
                value += pe * noise_cascades[old_index]
                state[old_index] = recent
                state[recent_index] = value
        for i in range(PIXELS):
            var value = 0.0
            for level in range(LEVELS):
                value += state[(LEVELS + level) * PIXELS + i] * deviations[unsafe_offset=level] + means[unsafe_offset=level]
            field[i] = value
        var low = minimum[SIZE](data(field))
        for i in range(PIXELS):
            field[i] = low + (field[i] - low) * mask[i]
            if not field[i] > low:
                field[i] = low
        probability_match[SIZE](data(field), target, ip, data(matched))
        dilated_mask[SIZE](data(matched), data(mask), ip, options.precip_thr, options.mask_radius())
        var minutes = Float64(lead + 1) * options.timestep
        var along = (10.88 * pow(minutes, 0.23) - 7.68) * eps_par
        var across = (5.76 * pow(minutes, 0.31) - 2.72) * eps_perp
        for i in range(PIXELS):
            var vx = direction[unsafe_offset=i]
            var vy = direction[unsafe_offset=PIXELS + i]
            perturbed_velocity[i] = velocity[unsafe_offset=i] + (along * vx - across * vy) / options.velocity_scale()
            perturbed_velocity[PIXELS + i] = velocity[unsafe_offset=PIXELS + i] + (along * vy + across * vx) / options.velocity_scale()
        advect[SIZE](data(matched), data(perturbed_velocity), data(displacement), lead == 0, output + lead * PIXELS)
    # Untracked C-ABI pointers must not outlive their owning Lists.
    _ = indices


def forecast_native[SIZE: Int = 192](
    input: F64Ptr, velocity: F64Ptr, output: F64Ptr,
    members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32, options: RuntimeOptions = RuntimeOptions(),
) -> Int32:
    comptime PIXELS = SIZE * SIZE
    if members < 1 or members > 4096 or leads < 1 or leads > 288 or workers < 1 or velocity_float32 < 0 or velocity_float32 > 1:
        return 1
    for i in range(3 * PIXELS):
        if not isfinite(input[unsafe_offset=i]):
            return 2
    for i in range(2 * PIXELS):
        if not isfinite(velocity[unsafe_offset=i]):
            return 2
    var echoes = 0
    for i in range(3 * PIXELS):
        echoes += Int(input[unsafe_offset=i] >= options.precip_thr)
    if echoes < 16:
        for member in range(members):
            for lead in range(leads):
                for i in range(PIXELS):
                    output[unsafe_offset=(member * leads + lead) * PIXELS + i] = input[unsafe_offset=2 * PIXELS + i]
        return 0
    initialize_runtime()
    var model = Model[SIZE](input, velocity, velocity_float32 != 0)
    for i in range(LEVELS):
        if not isfinite(model.deviations[i]) or model.deviations[i] <= 0:
            return 2
    for i in range(3 * LEVELS):
        if not isfinite(model.coefficients[i]):
            return 2
    var seeds = List[UInt32](length=members, fill=0)
    var par = List[Float64](length=members, fill=0)
    var perp = List[Float64](length=members, fill=0)
    var next_seed = seed
    for member in range(members):
        seeds[member] = next_seed
        var precip_rng = MT19937(next_seed)
        next_seed = precip_rng.below(UInt32(1000000000))
        var motion_rng = MT19937(next_seed)
        next_seed = motion_rng.below(UInt32(1000000000))
        par[member] = motion_rng.laplace()
        perp[member] = motion_rng.laplace()
    var seed_pointer = seeds.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var par_pointer = data(par)
    var perp_pointer = data(perp)
    var weights = data(model.weights)
    var noise_filter = data(model.noise_filter)
    var initial = data(model.initial)
    var means = data(model.means)
    var deviations = data(model.deviations)
    var coefficients = data(model.coefficients)
    var target = data(model.target)
    var direction = data(model.direction)

    def member_forecast(member: Int) {imm}:
        simulate_member[SIZE](weights, noise_filter, initial, means, deviations, coefficients, target, direction, velocity, input + 2 * PIXELS, seed_pointer[unsafe_offset=member], par_pointer[unsafe_offset=member], perp_pointer[unsafe_offset=member], leads, output + member * leads * PIXELS, options)

    parallelize(member_forecast, members, min(workers, members))
    # Mojo drops values at their last use; raw-pointer captures do not extend
    # ownership. Retain the model and RNG/perturbation arrays through the join.
    _ = model
    _ = seeds
    _ = par
    _ = perp
    return 0


@export
def steps_native_model_probe(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, velocity_float32: Int32) abi("C"):
    var model = Model[192](input, velocity, velocity_float32 != 0)
    var position = 0
    for i in range(LEVELS):
        output[unsafe_offset=position] = model.means[i]
        position += 1
    for i in range(LEVELS):
        output[unsafe_offset=position] = model.deviations[i]
        position += 1
    for i in range(3 * LEVELS):
        output[unsafe_offset=position] = model.coefficients[i]
        position += 1
    for i in range(FREQUENCIES):
        output[unsafe_offset=position] = model.noise_filter[i]
        position += 1


@export
def steps_forecast_native(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32) abi("C") -> Int32:
    if members > 96 or leads > 14:
        return 1
    return forecast_native[192](input, velocity, output, members, leads, seed, workers, velocity_float32)


@export
def steps_forecast_cpu_v2(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, size: Int, members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32, kmperpixel: Float64, timestep: Float64, precip_thr: Float64, norain_thr: Float64) abi("C") -> Int32:
    var options = RuntimeOptions(kmperpixel, timestep, precip_thr, norain_thr)
    if not options.valid():
        return 1
    if size == 64:
        return forecast_native[64](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    if size == 128:
        return forecast_native[128](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    if size == 192:
        return forecast_native[192](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    if size == 256:
        return forecast_native[256](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    if size == 512:
        return forecast_native[512](input, velocity, output, members, leads, seed, workers, velocity_float32, options)
    return 1
