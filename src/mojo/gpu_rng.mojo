# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
# Adapted algorithm notices: see NOTICE and third_party/.
"""NumPy legacy MT19937/polar normals on Metal, preserving rejection order."""
from native_config import RuntimeOptions, merge_passes
from std.math import sqrt
from std.memory import bitcast
from std.memory import unsafe_stack_allocation
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from gpu_fft import D2, dd_add, dd_mul
from gpu_noise import dd_div
from native_fft import PIXELS


@inline(.always)
def dd_sqrt(a: D2) -> D2:
    if a[0] == 0:
        return D2(0)
    var root = sqrt(a[0])
    var residual = dd_add(a, -dd_mul(D2(root, 0), D2(root, 0)))
    return dd_add(D2(root, 0), D2((residual[0] + residual[1]) / (2 * root), 0))


@inline(.always)
def dd_log(a: D2) -> D2:
    var bits = bitcast[DType.uint32](a[0])
    var exponent = Int((bits >> 23) & 255) - 127
    var scale = bitcast[DType.float32](UInt32(127 - exponent) << 23)
    var m = dd_mul(a, D2(scale, 0))
    if m[0] > 1.4142135623730951:
        m = dd_mul(m, D2(0.5, 0))
        exponent += 1
    var y = dd_div(dd_add(m, D2(-1, 0)), dd_add(m, D2(1, 0)))
    var square = dd_mul(y, y)
    var term = y
    var result = y
    comptime for k in range(1, 10):
        term = dd_mul(term, square)
        comptime c = Float64(1) / Float64(2 * k + 1)
        result = dd_add(result, dd_mul(term, D2(Float32(c), Float32(c - Float64(Float32(c))))))
    comptime ln2 = 0.693147180559945309417232121458176568
    return dd_add(dd_mul(result, D2(2, 0)), dd_mul(D2(Float32(exponent), 0), D2(Float32(ln2), Float32(ln2 - Float64(Float32(ln2))))))


@inline(.always)
def temper(v: UInt32) -> UInt32:
    var value = v
    value ^= value >> 11
    value ^= (value << 7) & UInt32(0x9d2c5680)
    value ^= (value << 15) & UInt32(0xefc60000)
    return value ^ (value >> 18)


@inline(.always)
def int_pair(a: UInt32) -> D2:
    var hi = Float32(a)
    return D2(hi, Float32(Int(a) - Int(hi)))


@inline(.always)
def uniform(a: UInt32, b: UInt32) -> D2:
    return dd_mul(dd_add(dd_mul(int_pair(a >> 5), D2(67108864, 0)), int_pair(b >> 6)), D2(1.1102230246251565e-16, 0))


def generate_normals[SIZE: Int = 192](states: MutPointer[UInt32, MutAnyOrigin], positions: MutPointer[Int32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin]):
    comptime PIXELS = SIZE * SIZE
    var m = block_idx.x
    var t = thread_idx.x
    var old = unsafe_stack_allocation[624, UInt32, address_space=.SHARED]()
    var next = unsafe_stack_allocation[624, UInt32, address_space=.SHARED]()
    var prefix = unsafe_stack_allocation[256, Int32, address_space=.SHARED]()
    var used = unsafe_stack_allocation[1, Int32, address_space=.SHARED]()
    for i in range(t, 624, 256):
        old[i] = states[m * 624 + i]
    var position = Int(positions[m])
    var produced = 0
    barrier()
    while produced < PIXELS // 2:
        # The in-place reference twist has three dependency segments and a
        # final wraparound update. Barriers preserve that exact integer stream.
        for i in range(t, 227, 256):
            var y = (old[i] & UInt32(0x80000000)) | (old[i + 1] & UInt32(0x7fffffff))
            next[i] = old[i + 397] ^ (y >> 1) ^ (UInt32(0x9908b0df) if y & 1 else UInt32(0))
        barrier()
        if t < 227:
            var i = t + 227
            var y = (old[i] & UInt32(0x80000000)) | (old[i + 1] & UInt32(0x7fffffff))
            next[i] = next[i - 227] ^ (y >> 1) ^ (UInt32(0x9908b0df) if y & 1 else UInt32(0))
        barrier()
        if t < 169:
            var i = t + 454
            var y = (old[i] & UInt32(0x80000000)) | (old[i + 1] & UInt32(0x7fffffff))
            next[i] = next[i - 227] ^ (y >> 1) ^ (UInt32(0x9908b0df) if y & 1 else UInt32(0))
        barrier()
        if t == 0:
            var y = (old[623] & UInt32(0x80000000)) | (next[0] & UInt32(0x7fffffff))
            next[623] = next[396] ^ (y >> 1) ^ (UInt32(0x9908b0df) if y & 1 else UInt32(0))
            used[0] = 624
        barrier()
        var x = D2(0)
        var y = D2(0)
        var radius = D2(2, 0)
        var valid = False
        if t < 156:
            var words = SIMD[DType.uint32, 4](0)
            for j in range(4):
                var i = position + 4 * t + j
                var v = UInt32(0)
                if i < 624:
                    v = old[i]
                else:
                    v = next[i - 624]
                words[j] = temper(v)
            x = dd_add(dd_mul(uniform(words[0], words[1]), D2(2, 0)), D2(-1, 0))
            y = dd_add(dd_mul(uniform(words[2], words[3]), D2(2, 0)), D2(-1, 0))
            radius = dd_add(dd_mul(x, x), dd_mul(y, y))
            valid = radius[0] > 0 and (radius[0] < 1 or (radius[0] == 1 and radius[1] < 0))
        prefix[t] = Int32(valid)
        barrier()
        comptime for stage in range(8):
            comptime offset = 1 << stage
            var v = prefix[t]
            if t >= offset:
                v += prefix[t - offset]
            barrier()
            prefix[t] = v
            barrier()
        var rank = Int(prefix[t]) - 1
        if valid and produced + rank < PIXELS // 2:
            var scale = dd_sqrt(dd_div(dd_mul(D2(-2, 0), dd_log(radius)), radius))
            var first = dd_mul(scale, y)
            var second = dd_mul(scale, x)
            var i = m * PIXELS + 2 * (produced + rank)
            output[4 * i] = first[0]
            output[4 * i + 1] = first[1]
            output[4 * i + 2] = 0
            output[4 * i + 3] = 0
            output[4 * i + 4] = second[0]
            output[4 * i + 5] = second[1]
            output[4 * i + 6] = 0
            output[4 * i + 7] = 0
            if produced + rank + 1 == PIXELS // 2:
                used[0] = Int32((t + 1) * 4)
        barrier()
        produced += Int(prefix[255])
        if produced >= PIXELS // 2:
            var consumed = position + Int(used[0])
            if consumed <= 624:
                for i in range(t, 624, 256):
                    states[m * 624 + i] = old[i]
            else:
                for i in range(t, 624, 256):
                    states[m * 624 + i] = next[i]
                consumed -= 624
            if t == 0:
                positions[m] = Int32(consumed)
        else:
            for i in range(t, 624, 256):
                old[i] = next[i]
        barrier()


@inline(.always)
def seed_state(state: MutPointer[UInt32, MutUntrackedOrigin], seed: UInt32):
    state[0] = seed
    for i in range(1, 624):
        var previous = state[i-1]
        state[i] = UInt32(1812433253) * (previous ^ (previous >> 30)) + UInt32(i)


def next_word(state: MutPointer[UInt32, MutUntrackedOrigin], mut position: Int) -> UInt32:
    if position >= 624:
        for i in range(624):
            var y = (state[i] & UInt32(0x80000000)) | (state[(i+1) % 624] & UInt32(0x7fffffff))
            state[i] = state[(i+397) % 624] ^ (y >> 1) ^ (UInt32(0x9908b0df) if y & 1 else UInt32(0))
        position = 0
    var value = temper(state[position])
    position += 1
    return value


def below_billion(state: MutPointer[UInt32, MutUntrackedOrigin], mut position: Int) -> UInt32:
    var value = next_word(state, position) & UInt32(0x3fffffff)
    while value >= UInt32(1000000000):
        value = next_word(state, position) & UInt32(0x3fffffff)
    return value


def initialize_random[SIZE: Int = 192](states: MutPointer[UInt32, MutAnyOrigin], motion_storage: MutPointer[UInt32, MutAnyOrigin], positions: MutPointer[Int32, MutAnyOrigin], params: MutPointer[Float32, MutAnyOrigin], seed: UInt32, members: Int32):
    comptime PIXELS = SIZE * SIZE
    # Member seeds form a serial reference-defined chain. Only this small
    # initialization is serial; subsequent normal generation is cooperative.
    var motion = motion_storage.unsafe_origin_cast[MutUntrackedOrigin]()
    var next_seed = seed
    for m in range(Int(members)):
        var precip = (states + m * 624).unsafe_origin_cast[MutUntrackedOrigin]()
        seed_state(precip, next_seed)
        var position = 624
        next_seed = below_billion(precip, position)
        positions[m] = Int32(position)
        seed_state(motion, next_seed)
        position = 624
        next_seed = below_billion(motion, position)
        for k in range(2):
            var a = next_word(motion, position)
            var b = next_word(motion, position)
            var u = uniform(a, b)
            var value = D2(0)
            if u[0] > .5 or (u[0] == .5 and u[1] >= 0):
                value = -dd_log(dd_add(D2(2, 0), -dd_mul(D2(2, 0), u)))
            else:
                value = dd_log(dd_mul(D2(2, 0), u))
            comptime root2 = 1.414213562373095048801688724209698
            value = dd_div(value, D2(Float32(root2), Float32(root2-Float64(Float32(root2)))))
            var i = 60 + 10 * PIXELS + 4 * m + 2 * k
            params[i] = value[0]
            params[i+1] = value[1]
