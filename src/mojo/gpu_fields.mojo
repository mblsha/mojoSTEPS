# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""Paired-float GPU member evolution, sorting, masks and interpolation."""
from std.math import floor, nan
from std.memory import unsafe_stack_allocation
from max.gpu import global_idx, block_idx, thread_idx
from max.gpu.sync import barrier
from native_fft import N, PIXELS
from native_core import LEVELS
from gpu_fft import D2, dd_add, dd_mul
from gpu_noise import dd_div

comptime PARAM_TARGET = 60
comptime PARAM_DIRECTION = PARAM_TARGET + 2 * PIXELS
comptime PARAM_VELOCITY = PARAM_DIRECTION + 4 * PIXELS
comptime PARAM_EPS = PARAM_VELOCITY + 4 * PIXELS
comptime FPtr = MutPointer[Float32, MutAnyOrigin]
comptime IPtr = MutPointer[Int32, MutAnyOrigin]

@inline(.always)
def get(p: FPtr, i: Int) -> D2:
    return p.load[width=2](2 * i)

@inline(.always)
def put(p: FPtr, i: Int, v: D2):
    p[2 * i] = v[0]
    p[2 * i + 1] = v[1]

@inline(.always)
def less(a: D2, b: D2) -> Bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

@inline(.always)
def equal(a: D2, b: D2) -> Bool:
    return a[0] == b[0] and a[1] == b[1]

@inline(.always)
def lesser(a: D2, b: D2) -> D2:
    if less(a, b):
        return a
    return b

@inline(.always)
def ordered(a: D2, ai: Int32, b: D2, bi: Int32) -> Bool:
    return less(a, b) or (equal(a, b) and ai < bi)

@inline(.always)
def dd_floor(a: D2) -> Int:
    var base = Int(floor(a[0]))
    if less(a, D2(Float32(base), 0)):
        base -= 1
    elif not less(a, D2(Float32(base + 1), 0)):
        base += 1
    return base

@inline(.always)
def sample(p: FPtr, offset: Int, x0: D2, y0: D2, nearest: Bool, affine: Bool = False) -> D2:
    var x = x0
    var y = y0
    if nearest:
        if less(x, D2(0)):
            x = D2(0)
        if less(D2(N - 1, 0), x):
            x = D2(N - 1, 0)
        if less(y, D2(0)):
            y = D2(0)
        if less(D2(N - 1, 0), y):
            y = D2(N - 1, 0)
    elif less(x, D2(0)) or less(y, D2(0)) or less(D2(N - 1, 0), x) or less(D2(N - 1, 0), y):
        return D2(nan[DType.float32](), 0)
    var ix = dd_floor(x)
    var iy = dd_floor(y)
    var ix1 = min(ix + 1, N - 1)
    var iy1 = min(iy + 1, N - 1)
    var fx = dd_add(x, -D2(Float32(ix), 0))
    var fy = dd_add(y, -D2(Float32(iy), 0))
    var rx = dd_add(D2(1, 0), -fx)
    var ry = dd_add(D2(1, 0), -fy)
    if affine:
        var a = get(p, offset + iy * N + ix)
        var b = get(p, offset + iy * N + ix1)
        var c = get(p, offset + iy1 * N + ix)
        var d = get(p, offset + iy1 * N + ix1)
        var top = dd_add(a, dd_mul(dd_add(b, -a), fx))
        var bottom = dd_add(c, dd_mul(dd_add(d, -c), fx))
        return dd_add(top, dd_mul(dd_add(bottom, -top), fy))
    var v = dd_mul(dd_mul(get(p, offset + iy * N + ix), ry), rx)
    v = dd_add(v, dd_mul(dd_mul(get(p, offset + iy * N + ix1), ry), fx))
    v = dd_add(v, dd_mul(dd_mul(get(p, offset + iy1 * N + ix), fy), rx))
    v = dd_add(v, dd_mul(dd_mul(get(p, offset + iy1 * N + ix1), fy), fx))
    return v


def initialize_members(initial: FPtr, state: FPtr, field: FPtr, displacement: FPtr, count: Int32):
    var i = global_idx.x
    if i < Int(count) * 2 * LEVELS * PIXELS:
        put(state, i, get(initial, i % (2 * LEVELS * PIXELS)))
    if i < Int(count) * PIXELS:
        put(field, i, get(initial, 2 * LEVELS * PIXELS + i % PIXELS))
        put(displacement, 2 * i, D2(0))
        put(displacement, 2 * i + 1, D2(0))


def evolve(state: FPtr, innovation: FPtr, params: FPtr, field: FPtr, count: Int32):
    var i = global_idx.x
    if i >= Int(count) * PIXELS:
        return
    var member = i // PIXELS
    var pixel = i % PIXELS
    var result = D2(0)
    for level in range(LEVELS):
        var old = (member * 2 * LEVELS + level) * PIXELS + pixel
        var recent = old + LEVELS * PIXELS
        var previous = get(state, recent)
        var value = dd_mul(get(params, 12 + 3 * level), previous)
        value = dd_add(value, dd_mul(get(params, 12 + 3 * level + 1), get(state, old)))
        value = dd_add(value, dd_mul(get(params, 12 + 3 * level + 2), get(innovation, (member * LEVELS + level) * PIXELS + pixel)))
        put(state, old, previous)
        put(state, recent, value)
        result = dd_add(result, dd_add(dd_mul(value, get(params, 6 + level)), get(params, level)))
    put(field, i, result)


def field_minimum(field: FPtr, lows: FPtr):
    var member = block_idx.x
    var t = thread_idx.x
    var low = D2(3.4e38, 0)
    for p in range(t, PIXELS, 256):
        low = lesser(low, get(field, member * PIXELS + p))
    var shared = unsafe_stack_allocation[256, D2, address_space=.SHARED]()
    shared[t] = low
    barrier()
    comptime for stage in range(8):
        comptime half = 128 >> stage
        if t < half:
            shared[t] = lesser(shared[t], shared[t + half])
        barrier()
    if t == 0:
        put(lows, member, shared[0])


def apply_mask(field: FPtr, mask: FPtr, lows: FPtr, count: Int32):
    var i = global_idx.x
    if i >= Int(count) * PIXELS:
        return
    var low = get(lows, i // PIXELS)
    var value = dd_add(low, dd_mul(dd_add(get(field, i), -low), get(mask, i)))
    if not less(low, value):
        value = low
    put(field, i, value)


def count_dry(field: FPtr, lows: FPtr, dry: IPtr):
    var m = block_idx.x
    var t = thread_idx.x
    var count = Int32(0)
    var low = get(lows, m)
    for p in range(t, PIXELS, 256):
        count += Int32(equal(get(field, m * PIXELS + p), low))
    var s = unsafe_stack_allocation[256, Int32, address_space=.SHARED]()
    s[t] = count
    barrier()
    comptime for stage in range(8):
        comptime half = 128 >> stage
        if t < half:
            s[t] += s[t + half]
        barrier()
    if t == 0:
        dry[m] = s[0]


def local_sort(field: FPtr, keys: FPtr, indices: IPtr):
    var t = thread_idx.x
    var group = block_idx.x
    var base = group * 1024
    var sk = unsafe_stack_allocation[1024, D2, address_space=.SHARED]()
    var si = unsafe_stack_allocation[1024, Int32, address_space=.SHARED]()
    for j in range(t, 1024, 256):
        sk[j] = get(field, base + j)
        si[j] = Int32((base + j) % PIXELS)
    barrier()
    comptime for power in range(1, 11):
        comptime size = 1 << power
        comptime for stage in range(power):
            comptime stride = 1 << (power - 1 - stage)
            for j in range(t, 1024, 256):
                var other = j ^ stride
                if other > j:
                    var a = sk[j]
                    var b = sk[other]
                    var ai = si[j]
                    var bi = si[other]
                    var ascending = (j & size) == 0
                    if ordered(b, bi, a, ai) == ascending:
                        sk[j] = b
                        sk[other] = a
                        si[j] = bi
                        si[other] = ai
            barrier()
    for j in range(t, 1024, 256):
        put(keys, base + j, sk[j])
        indices[base + j] = si[j]


def merge_sort(source: FPtr, source_i: IPtr, target: FPtr, target_i: IPtr, width: Int32, total: Int32):
    var i = global_idx.x
    if i >= Int(total):
        return
    var member = i // PIXELS
    var pixel = i % PIXELS
    var w = Int(width)
    var start = (pixel // (2 * w)) * 2 * w
    var middle = min(start + w, PIXELS)
    var end = min(start + 2 * w, PIXELS)
    var left = start
    var right = middle
    var own_start = middle
    if pixel < middle:
        left = middle
        right = end
        own_start = start
    var key = get(source, i)
    var index = source_i[i]
    var begin = left
    while left < right:
        var mid = (left + right) // 2
        var other = member * PIXELS + mid
        if ordered(get(source, other), source_i[other], key, index):
            left = mid + 1
        else:
            right = mid
    var rank = start + (pixel - own_start) + (left - begin)
    var dest = member * PIXELS + rank
    put(target, dest, key)
    target_i[dest] = index


def cdf_match(keys: FPtr, indices: IPtr, lows: FPtr, dry: IPtr, params: FPtr, matched: FPtr, count: Int32):
    var i = global_idx.x
    if i >= Int(count) * PIXELS:
        return
    var m = i // PIXELS
    var rank = i % PIXELS
    var target = params + PARAM_TARGET
    var low = get(target, 0)
    var cutoff = low
    var wet = PIXELS - Int(dry[m])
    # A target rank calculation rather than a second count reduction. The
    # sorted target's first value above its minimum determines target wetness.
    var a = 0
    var b = PIXELS
    while a < b:
        var mid = (a + b) // 2
        if not less(low, get(target, mid)):
            a = mid + 1
        else:
            b = mid
    if PIXELS - a > wet:
        var r = dd_mul(dd_div(D2(Float32(dry[m]), 0), D2(PIXELS, 0)), D2(PIXELS - 1, 0))
        var index = dd_floor(r)
        var fraction = dd_add(r, -D2(Float32(index), 0))
        cutoff = dd_add(dd_mul(get(target, index), dd_add(D2(1, 0), -fraction)), dd_mul(get(target, min(index + 1, PIXELS - 1)), fraction))
    var value = get(target, rank)
    if less(value, cutoff) or equal(get(keys, i), get(lows, m)):
        value = low
    put(matched, m * PIXELS + Int(indices[i]), value)


def mask_rows(field: FPtr, distance: IPtr):
    var row = global_idx.x
    var base = row * N
    var dist = Int32(100000)
    for x in range(N):
        if not less(get(field, base + x), D2(10, 0)):
            dist = 0
        else:
            dist += 1
        distance[base + x] = dist
    dist = 100000
    for xx in range(N):
        var x = N - 1 - xx
        dist = min(distance[base + x], dist + 1)
        distance[base + x] = dist


def mask_columns(distance: IPtr, mask: FPtr):
    var column = global_idx.x
    var m = column // N
    var x = column % N
    var dist = Int32(100000)
    for y in range(N):
        var i = m * PIXELS + y * N + x
        dist = min(distance[i], dist + 1)
        distance[i] = dist
    dist = 100000
    var has_wet = False
    for yy in range(N):
        var y = N - 1 - yy
        var i = m * PIXELS + y * N + x
        dist = min(distance[i], dist + 1)
        # Empty fields have no finite distance anywhere, including this column.
        var value = dd_div(D2(Float32(max(0, min(11, 15 - Int(dist)))), 0), D2(11, 0))
        if dist >= 100000:
            value = D2(nan[DType.float32](), 0)
        put(mask, i, value)


def perturb(params: FPtr, velocity: FPtr, along_h: Float32, along_l: Float32, across_h: Float32, across_l: Float32, count: Int32):
    var i = global_idx.x
    if i >= Int(count) * PIXELS:
        return
    var m = i // PIXELS
    var p = i % PIXELS
    var vx = get(params + PARAM_DIRECTION, p)
    var vy = get(params + PARAM_DIRECTION, PIXELS + p)
    var along = dd_mul(D2(along_h, along_l), get(params + PARAM_EPS, 2 * m))
    var across = dd_mul(D2(across_h, across_l), get(params + PARAM_EPS, 2 * m + 1))
    put(velocity, (m * 2) * PIXELS + p, dd_add(get(params + PARAM_VELOCITY, p), dd_div(dd_add(dd_mul(along, vx), -dd_mul(across, vy)), D2(6, 0))))
    put(velocity, (m * 2 + 1) * PIXELS + p, dd_add(get(params + PARAM_VELOCITY, PIXELS + p), dd_div(dd_add(dd_mul(along, vy), dd_mul(across, vx)), D2(6, 0))))


def advection(field: FPtr, velocity: FPtr, displacement: FPtr, output: FPtr, lead: Int32, leads: Int32, count: Int32):
    var i = global_idx.x
    if i >= Int(count) * PIXELS:
        return
    var m = i // PIXELS
    var p = i % PIXELS
    var base = m * 2 * PIXELS
    var x = D2(Float32(p % N), 0)
    var y = D2(Float32(p // N), 0)
    var dx = get(displacement, base + p)
    var dy = get(displacement, base + PIXELS + p)
    var vx = get(velocity, base + p)
    var vy = get(velocity, base + PIXELS + p)
    if lead != 0:
        vx = sample(velocity, base, dd_add(x, dx), dd_add(y, dy), True)
        vy = sample(velocity, base + PIXELS, dd_add(x, dx), dd_add(y, dy), True)
    var hx = dd_add(x, dd_add(dx, -dd_mul(vx, D2(0.5, 0))))
    var hy = dd_add(y, dd_add(dy, -dd_mul(vy, D2(0.5, 0))))
    dx = dd_add(dx, -sample(velocity, base, hx, hy, True))
    dy = dd_add(dy, -sample(velocity, base + PIXELS, hx, hy, True))
    put(displacement, base + p, dx)
    put(displacement, base + PIXELS + p, dy)
    put(output, (m * Int(leads) + Int(lead)) * PIXELS + p, sample(field, m * PIXELS, dd_add(x, dx), dd_add(y, dy), False))
