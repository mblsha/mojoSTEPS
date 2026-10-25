// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Michael Pishchagin and mojoSTEPS contributors.
#define POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_CACHE_SIZE 16
#include "../third_party/pocketfft/pocketfft_hdronly.h"
#include <complex>
#include <cstdint>
#include <cstdlib>

namespace {
struct Plan {
    int n;
    double *real;
    std::complex<double> *frequency;
    bool inverse;
};
}
extern "C" void *mojosteps_fft_plan(std::int32_t n, double *real,
                                     double *frequency, std::int32_t inverse) noexcept {
    try { return new Plan{n, real, reinterpret_cast<std::complex<double> *>(frequency), inverse != 0}; }
    catch (...) { std::abort(); }
}
extern "C" void mojosteps_fft_destroy(void *plan) noexcept {
    delete static_cast<Plan *>(plan);
}
extern "C" void mojosteps_fft_execute(void *opaque) noexcept {
    auto &p = *static_cast<Plan *>(opaque);
    const pocketfft::shape_t shape{std::size_t(p.n), std::size_t(p.n)};
    const pocketfft::shape_t axes{0, 1};
    const pocketfft::stride_t real_stride{p.n * std::ptrdiff_t(sizeof(double)), sizeof(double)};
    const pocketfft::stride_t frequency_stride{(p.n / 2 + 1) * std::ptrdiff_t(sizeof(std::complex<double>)), sizeof(std::complex<double>)};
    try {
        // Unscaled transforms preserve the existing Mojo inverse / N^2 convention.
        if (p.inverse)
            pocketfft::c2r(shape, frequency_stride, real_stride, axes, false,
                          p.frequency, p.real, 1.0, 1);
        else
            pocketfft::r2c(shape, real_stride, frequency_stride, axes, true,
                          p.real, p.frequency, 1.0, 1);
    } catch (...) { std::abort(); }
}
