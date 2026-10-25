# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""Spatial statistics, interpolation, CDF matching and masks in native Mojo."""

from std.math import floor, sqrt, nan
from native_fft import N, PIXELS, F64Ptr
from std.memory import Pointer

comptime IndexPtr = Pointer[Int, MutUntrackedOrigin]


def sum_values(values: F64Ptr, count: Int) -> Float64:
    if count < 8:
        var total = -0.0
        for i in range(count):
            total += values[unsafe_offset=i]
        return total
    if count > 128:
        var half = count // 2
        half -= half % 8
        return sum_values(values, half) + sum_values(values + half, count - half)
    var sums = SIMD[DType.float64, 8](0)
    for j in range(8):
        sums[j] = values[unsafe_offset=j]
    var i = 8
    while i <= count - 8:
        for j in range(8):
            sums[j] += values[unsafe_offset=i + j]
        i += 8
    var result = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]))
    while i < count:
        result += values[unsafe_offset=i]
        i += 1
    return result


def statistics(values: F64Ptr, scratch: F64Ptr) -> Tuple[Float64, Float64]:
    var mean = sum_values(values, PIXELS) / PIXELS
    for i in range(PIXELS):
        var difference = values[unsafe_offset=i] - mean
        scratch[unsafe_offset=i] = difference * difference
    return mean, sqrt(sum_values(scratch, PIXELS) / PIXELS)


def normalize(values: F64Ptr, scratch: F64Ptr) -> Tuple[Float64, Float64]:
    var mean, deviation = statistics(values, scratch)
    for i in range(PIXELS):
        values[unsafe_offset=i] = (values[unsafe_offset=i] - mean) / deviation
    return mean, deviation


def minimum(values: F64Ptr) -> Float64:
    var result = values[unsafe_offset=0]
    for i in range(1, PIXELS):
        result = min(result, values[unsafe_offset=i])
    return result


def bilinear(values: F64Ptr, x: Float64, y: Float64, nearest: Bool, outside: Float64) -> Float64:
    var xx = x
    var yy = y
    if nearest:
        xx = max(0.0, min(Float64(N - 1), xx))
        yy = max(0.0, min(Float64(N - 1), yy))
    elif xx < 0 or yy < 0 or xx > N - 1 or yy > N - 1:
        return outside
    var ix = Int(floor(xx))
    var iy = Int(floor(yy))
    var ix1 = min(ix + 1, N - 1)
    var iy1 = min(iy + 1, N - 1)
    var fx = xx - Float64(ix)
    var fy = yy - Float64(iy)
    var result = values[unsafe_offset=iy * N + ix] * (1.0 - fy) * (1.0 - fx)
    result += values[unsafe_offset=iy * N + ix1] * (1.0 - fy) * fx
    result += values[unsafe_offset=iy1 * N + ix] * fy * (1.0 - fx)
    result += values[unsafe_offset=iy1 * N + ix1] * fy * fx
    return result


def motion_sample(velocity: F64Ptr, x: Float64, y: Float64, velocity32: Bool) -> Float64:
    var value = bilinear(velocity, x, y, True, 0)
    # scipy.ndimage.map_coordinates preserves the source array's dtype.
    if velocity32:
        value = Float64(Float32(value))
    return value


def align(source: F64Ptr, velocity: F64Ptr, steps: Int, output: F64Ptr, velocity32: Bool = False):
    var low = minimum(source)
    for i in range(PIXELS):
        var x = Float64(i % N)
        var y = Float64(i // N)
        var dx = 0.0
        var dy = 0.0
        var vx = velocity[unsafe_offset=i]
        var vy = velocity[unsafe_offset=PIXELS + i]
        for step in range(steps):
            var hx = x + (dx - vx / 2.0)
            var hy = y + (dy - vy / 2.0)
            vx = motion_sample(velocity, hx, hy, velocity32)
            vy = motion_sample(velocity + PIXELS, hx, hy, velocity32)
            dx -= vx
            dy -= vy
            vx = motion_sample(velocity, x + dx, y + dy, velocity32)
            vy = motion_sample(velocity + PIXELS, x + dx, y + dy, velocity32)
        output[unsafe_offset=i] = bilinear(source, x + dx, y + dy, False, low)


def advect(source: F64Ptr, velocity: F64Ptr, displacement: F64Ptr, first: Bool, output: F64Ptr):
    var outside = nan[DType.float64]()
    for i in range(PIXELS):
        var x = Float64(i % N)
        var y = Float64(i // N)
        var dx = displacement[unsafe_offset=i]
        var dy = displacement[unsafe_offset=PIXELS + i]
        var vx = velocity[unsafe_offset=i]
        var vy = velocity[unsafe_offset=PIXELS + i]
        if not first:
            vx = bilinear(velocity, x + dx, y + dy, True, 0)
            vy = bilinear(velocity + PIXELS, x + dx, y + dy, True, 0)
        var hx = x + (dx - vx / 2.0)
        var hy = y + (dy - vy / 2.0)
        dx -= bilinear(velocity, hx, hy, True, 0)
        dy -= bilinear(velocity + PIXELS, hx, hy, True, 0)
        displacement[unsafe_offset=i] = dx
        displacement[unsafe_offset=PIXELS + i] = dy
        output[unsafe_offset=i] = bilinear(source, x + dx, y + dy, False, outside)


def dilated_mask(field: F64Ptr, output: F64Ptr, distance: IndexPtr):
    var has_wet = False
    for i in range(PIXELS):
        if field[unsafe_offset=i] >= 10.0:
            has_wet = True
            break
    if not has_wet:
        # pySTEPS normalizes this all-zero mask by its zero maximum.
        for i in range(PIXELS):
            output[unsafe_offset=i] = nan[DType.float64]()
        return
    for y in range(N):
        for x in range(N):
            var i = y * N + x
            var value = 100000
            if field[unsafe_offset=i] >= 10.0:
                value = 0
            if x > 0:
                value = min(value, distance[unsafe_offset=i - 1] + 1)
            if y > 0:
                value = min(value, distance[unsafe_offset=i - N] + 1)
            distance[unsafe_offset=i] = value
    for yy in range(N):
        var y = N - 1 - yy
        for xx in range(N):
            var x = N - 1 - xx
            var i = y * N + x
            var value = distance[unsafe_offset=i]
            if x + 1 < N:
                value = min(value, distance[unsafe_offset=i + 1] + 1)
            if y + 1 < N:
                value = min(value, distance[unsafe_offset=i + N] + 1)
            distance[unsafe_offset=i] = value
            output[unsafe_offset=i] = Float64(max(0, min(11, 15 - value))) / 11.0


def sort_indices(values: F64Ptr, order: IndexPtr, left: Int, right: Int):
    var lo = left
    var hi = right
    var pivot = values[unsafe_offset=order[unsafe_offset=(left + right) // 2]]
    while lo <= hi:
        while values[unsafe_offset=order[unsafe_offset=lo]] < pivot:
            lo += 1
        while values[unsafe_offset=order[unsafe_offset=hi]] > pivot:
            hi -= 1
        if lo <= hi:
            var temporary = order[unsafe_offset=lo]
            order[unsafe_offset=lo] = order[unsafe_offset=hi]
            order[unsafe_offset=hi] = temporary
            lo += 1
            hi -= 1
    if left < hi:
        sort_indices(values, order, left, hi)
    if lo < right:
        sort_indices(values, order, lo, right)


def probability_match(field: F64Ptr, target_sorted: F64Ptr, order: IndexPtr, output: F64Ptr):
    var low = minimum(field)
    var target_min = target_sorted[unsafe_offset=0]
    var wet = 0
    var target_wet = 0
    for i in range(PIXELS):
        order[unsafe_offset=i] = i
        wet += Int(field[unsafe_offset=i] > low)
        target_wet += Int(target_sorted[unsafe_offset=i] > target_min)
    var cutoff = target_min
    if target_wet > wet:
        var rank = (1.0 - Float64(wet) / PIXELS) * Float64(PIXELS - 1)
        var r = Int(floor(rank))
        var fraction = rank - Float64(r)
        cutoff = target_sorted[unsafe_offset=r] * (1.0 - fraction) + target_sorted[unsafe_offset=min(r + 1, PIXELS - 1)] * fraction
    sort_indices(field, order, 0, PIXELS - 1)
    for rank in range(PIXELS):
        var i = order[unsafe_offset=rank]
        var value = target_sorted[unsafe_offset=rank]
        if value < cutoff or field[unsafe_offset=i] == low:
            value = target_min
        output[unsafe_offset=i] = value
