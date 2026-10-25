// SPDX-License-Identifier: BSD-3-Clause
// Shared harness: one r2c + c2r + normalization, with planning outside the timer.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
extern "C" void *mojosteps_fft_plan(std::int32_t,double*,double*,std::int32_t) noexcept;
extern "C" void mojosteps_fft_execute(void*) noexcept;
extern "C" void mojosteps_fft_destroy(void*) noexcept;
extern "C" double bench_fft_pairs(const double *input, int count, double *error) {
    constexpr int n=192,p=n*n;
    auto *real=static_cast<double*>(std::malloc(p*sizeof(double)));
    auto *frequency=static_cast<double*>(std::malloc(2*n*(n/2+1)*sizeof(double)));
    if (!real || !frequency) std::abort();
    void *forward=mojosteps_fft_plan(n,real,frequency,0);
    void *inverse=mojosteps_fft_plan(n,real,frequency,1);
    std::memcpy(real,input,p*sizeof(double));
    auto start=std::chrono::steady_clock::now();
    for (int rep=0; rep<count; ++rep) {
        mojosteps_fft_execute(forward); mojosteps_fft_execute(inverse);
        for (int i=0;i<p;++i) real[i]/=p;
    }
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    *error=0;
    for (int i=0;i<p;++i) *error=std::max(*error,std::abs(real[i]-input[i]));
    mojosteps_fft_destroy(forward); mojosteps_fft_destroy(inverse);
    std::free(real);std::free(frequency);
    return seconds;
}
