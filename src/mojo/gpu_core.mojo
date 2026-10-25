# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""Standalone native CPU/GPU STEPS: GPU FFT/cascades, native CPU remainder.

No Python executes inside this forecast. The existing CPU entrypoint remains
available and unchanged. GPU failures return an error, never a CPU fallback.
"""

from native_config import TIMESTEP, PRECIP_THR, MAX_MEMBERS, MAX_LEADS, MERGE_PASSES
from std.ffi import external_call
from std.runtime import initialize_runtime
from std.math import pow, isfinite
from max.algorithm import parallelize
from native_fft import F64Ptr, PIXELS, data
from native_core import Model, LEVELS, steps_forecast_native, steps_native_model_probe
from native_rng import MT19937
from native_fields import IndexPtr, minimum, probability_match, dilated_mask, advect
from gpu_noise import GpuNoise


@export
def steps_gpu_model_probe(input: F64Ptr, velocity: F64Ptr, output: F64Ptr, velocity_float32: Int32) abi("C"):
    steps_native_model_probe(input, velocity, output, velocity_float32)


def update_member(
    state: F64Ptr, innovations: F64Ptr, means: F64Ptr, deviations: F64Ptr,
    coefficients: F64Ptr, target: F64Ptr, direction: F64Ptr, velocity: F64Ptr,
    mask: F64Ptr, indices: IndexPtr, displacement: F64Ptr,
    eps_par: Float64, eps_perp: Float64, lead: Int, output: F64Ptr,
):
    for level in range(LEVELS):
        var p0 = coefficients[unsafe_offset=3 * level]
        var p1 = coefficients[unsafe_offset=3 * level + 1]
        var pe = coefficients[unsafe_offset=3 * level + 2]
        for i in range(PIXELS):
            var old = level * PIXELS + i
            var recent = (LEVELS + level) * PIXELS + i
            var previous = state[unsafe_offset=recent]
            var value = p0 * previous
            value += p1 * state[unsafe_offset=old]
            value += pe * innovations[unsafe_offset=old]
            state[unsafe_offset=old] = previous
            state[unsafe_offset=recent] = value
    var field = List[Float64](length=PIXELS, fill=0)
    var matched = List[Float64](length=PIXELS, fill=0)
    var perturbed = List[Float64](length=2 * PIXELS, fill=0)
    for i in range(PIXELS):
        var value = 0.0
        for level in range(LEVELS):
            value += state[unsafe_offset=(LEVELS + level) * PIXELS + i] * deviations[unsafe_offset=level] + means[unsafe_offset=level]
        field[i] = value
    var low = minimum(data(field))
    for i in range(PIXELS):
        field[i] = low + (field[i] - low) * mask[unsafe_offset=i]
        if not field[i] > low:
            field[i] = low
    probability_match(data(field), target, indices, data(matched))
    dilated_mask(data(matched), mask, indices)
    var minutes = Float64(lead + 1) * Float64(TIMESTEP)
    var along = (10.88 * pow(minutes, 0.23) - 7.68) * eps_par
    var across = (5.76 * pow(minutes, 0.31) - 2.72) * eps_perp
    for i in range(PIXELS):
        var vx = direction[unsafe_offset=i]
        var vy = direction[unsafe_offset=PIXELS + i]
        perturbed[i] = velocity[unsafe_offset=i] + (along * vx - across * vy) / 6.0
        perturbed[PIXELS + i] = velocity[unsafe_offset=PIXELS + i] + (along * vy + across * vx) / 6.0
    advect(data(matched), data(perturbed), displacement, lead == 0, output)
    _ = field
    _ = matched
    _ = perturbed


@export
def steps_forecast_gpu(
    input: F64Ptr, velocity: F64Ptr, output: F64Ptr,
    members: Int, leads: Int, seed: UInt32, workers: Int, velocity_float32: Int32,
) abi("C") -> Int32:
    if members < 1 or members > MAX_MEMBERS or leads < 1 or leads > MAX_LEADS or workers < 1 or velocity_float32 < 0 or velocity_float32 > 1:
        return 1
    var echoes = 0
    for i in range(3 * PIXELS):
        if not isfinite(input[unsafe_offset=i]):
            return 2
        echoes += Int(input[unsafe_offset=i] >= PRECIP_THR)
    for i in range(2 * PIXELS):
        if not isfinite(velocity[unsafe_offset=i]):
            return 2
    if echoes < 16:
        return steps_forecast_native(input, velocity, output, members, leads, seed, workers, velocity_float32)
    try:
        initialize_runtime()
        var model = Model(input, velocity, velocity_float32 != 0)
        for i in range(LEVELS):
            if not isfinite(model.deviations[i]) or model.deviations[i] <= 0:
                return 2
        for i in range(3 * LEVELS):
            if not isfinite(model.coefficients[i]):
                return 2
        var gpu = GpuNoise(data(model.noise_filter), data(model.weights), members, workers)
        var rngs = List[MT19937]()
        var par = List[Float64](length=members, fill=0)
        var perp = List[Float64](length=members, fill=0)
        var next_seed = seed
        for member in range(members):
            var precip_rng = MT19937(next_seed)
            next_seed = precip_rng.below(UInt32(1000000000))
            var motion_rng = MT19937(next_seed)
            next_seed = motion_rng.below(UInt32(1000000000))
            par[member] = motion_rng.laplace()
            perp[member] = motion_rng.laplace()
            rngs.append(precip_rng^)
        var state = List[Float64](length=members * 2 * LEVELS * PIXELS, fill=0)
        var masks = List[Float64](length=members * PIXELS, fill=0)
        var indices = List[Int](length=members * PIXELS, fill=0)
        var displacements = List[Float64](length=members * 2 * PIXELS, fill=0)
        var white = List[Float64](length=members * PIXELS, fill=0)
        var cascades = List[Float64](length=members * LEVELS * PIXELS, fill=0)
        var phase_times = List[Float64](length=3, fill=0)
        var rp = rngs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var sp = data(state)
        var mp = data(masks)
        var ip = indices.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var dp = data(displacements)
        var wp = data(white)
        var cp = data(cascades)
        var initial = data(model.initial)
        var latest = input + 2 * PIXELS
        var means = data(model.means)
        var deviations = data(model.deviations)
        var coefficients = data(model.coefficients)
        var target = data(model.target)
        var direction = data(model.direction)
        var along = data(par)
        var across = data(perp)
        def initialize(member: Int) {imm}:
            for i in range(2 * LEVELS * PIXELS):
                sp[unsafe_offset=member * 2 * LEVELS * PIXELS + i] = initial[unsafe_offset=i]
            dilated_mask(latest, mp + member * PIXELS, ip + member * PIXELS)
        parallelize(initialize, members, min(workers, members))
        for lead in range(leads):
            def generate_white(member: Int) {imm}:
                for pixel in range(PIXELS):
                    wp[unsafe_offset=member * PIXELS + pixel] = rp[unsafe_offset=member].normal()
            parallelize(generate_white, members, min(workers, members))
            gpu.generate(wp, cp, data(phase_times))
            def update(member: Int) {imm}:
                update_member(sp + member * 2 * LEVELS * PIXELS, cp + member * LEVELS * PIXELS, means, deviations, coefficients, target, direction, velocity, mp + member * PIXELS, ip + member * PIXELS, dp + member * 2 * PIXELS, along[unsafe_offset=member], across[unsafe_offset=member], lead, output + (member * leads + lead) * PIXELS)
            parallelize(update, members, min(workers, members))
        _ = gpu
        _ = model
        _ = rngs
        _ = par
        _ = perp
        _ = state
        _ = masks
        _ = indices
        _ = displacements
        _ = white
        _ = cascades
        _ = phase_times
        return 0
    except error:
        print("Native GPU forecast failed:", error)
        return 3
