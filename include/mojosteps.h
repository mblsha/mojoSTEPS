/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef MOJOSTEPS_H
#define MOJOSTEPS_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Fixed grid: history [3,192,192], velocity [2,192,192],
 * output [members,leads,192,192], contiguous float64, disjoint buffers.
 * velocity_float32=1 requires velocity pre-rounded to float32, stored as double.
 * members 1..96; leads 1..14; workers >=1. Serialize calls across all instances.
 * Status: 0 success, 1 invalid options, 2 nonfinite/degenerate model, 3 GPU failure.
 * Caller owns and retains all buffers until the synchronous call returns.
 */
typedef int32_t (*mojosteps_forecast_fn)(const double*, const double*, double*,
    intptr_t members, intptr_t leads, uint32_t seed, intptr_t workers, int32_t velocity_float32);
int32_t steps_forecast_native(const double*, const double*, double*, intptr_t, intptr_t, uint32_t, intptr_t, int32_t);
int32_t steps_forecast_gpu_all(const double*, const double*, double*, intptr_t, intptr_t, uint32_t, intptr_t, int32_t);
#ifdef __cplusplus
}
#endif
#endif
