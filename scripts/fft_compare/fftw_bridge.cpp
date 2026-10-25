// SPDX-License-Identifier: BSD-3-Clause
// Optional local benchmark adapter. FFTW itself is an external GPL dependency.
#include <fftw3.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
namespace {
std::mutex planner_mutex;
unsigned flags=FFTW_ESTIMATE;
double plan_seconds=0;
std::uint64_t plan_calls=0;
std::string descriptions[2];
}
extern "C" int bench_fftw_configure(unsigned mode, const char *wisdom) {
    std::lock_guard<std::mutex> lock(planner_mutex);
    flags=mode | (wisdom ? FFTW_WISDOM_ONLY : 0);
    fftw_forget_wisdom();
    return wisdom ? fftw_import_wisdom_from_filename(wisdom) : 1;
}
extern "C" int bench_fftw_export(const char *path) {
    std::lock_guard<std::mutex> lock(planner_mutex);
    return fftw_export_wisdom_to_filename(path);
}
extern "C" const char *bench_fftw_version() { return fftw_version; }
extern "C" const char *bench_fftw_plan_description(int inverse) { return descriptions[inverse].c_str(); }
extern "C" double bench_fftw_plan_seconds() { return plan_seconds; }
extern "C" std::uint64_t bench_fftw_plan_calls() { return plan_calls; }
extern "C" void *mojosteps_fft_plan(std::int32_t n, double *real, double *frequency, std::int32_t inverse) noexcept {
    std::lock_guard<std::mutex> lock(planner_mutex);
    auto start=std::chrono::steady_clock::now();
    fftw_plan p=inverse ? fftw_plan_dft_c2r_2d(n,n,reinterpret_cast<fftw_complex *>(frequency),real,flags)
                        : fftw_plan_dft_r2c_2d(n,n,real,reinterpret_cast<fftw_complex *>(frequency),flags);
    plan_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    ++plan_calls;
    if (!p) { std::fputs("FFTW plan creation failed\n",stderr); std::abort(); }
    if (descriptions[inverse].empty()) {
        char *text=fftw_sprint_plan(p); descriptions[inverse]=text; fftw_free(text);
    }
    return p;
}
extern "C" void mojosteps_fft_execute(void *p) noexcept { fftw_execute(static_cast<fftw_plan>(p)); }
extern "C" void mojosteps_fft_destroy(void *p) noexcept {
    std::lock_guard<std::mutex> lock(planner_mutex);
    fftw_destroy_plan(static_cast<fftw_plan>(p));
}
