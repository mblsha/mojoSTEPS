/* SPDX-License-Identifier: BSD-3-Clause
 * No Python interpreter needed: load a built library and forecast raw demo input.
 * cc -O2 examples/forecast.c -o .build/forecast-c [-ldl on Linux]
 * .build/forecast-c LIBRARY cpu|gpu INPUT.f64 OUTPUT.f64 MEMBERS LEADS WORKERS
 * Input is five 192x192 little-endian float64 planes, float32-rounded velocity.
 */
#include "../include/mojosteps.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 8) { fprintf(stderr,"usage: %s LIBRARY cpu|gpu INPUT OUTPUT MEMBERS LEADS WORKERS\n",argv[0]); return 1; }
    int members=atoi(argv[5]), leads=atoi(argv[6]), workers=atoi(argv[7]);
    if (members<1 || members>96 || leads<1 || leads>14 || workers<1) return 1;
    size_t pixels=192*192, outputs=(size_t)members*leads*pixels;
    double *input=malloc(5*pixels*sizeof(double)), *output=malloc(outputs*sizeof(double));
    if (!input || !output) return 1;
    FILE *f=fopen(argv[3],"rb");
    if (!f || fread(input,sizeof(double),5*pixels,f)!=5*pixels) return 1;
    fclose(f);
    void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if (!library) { fprintf(stderr,"%s\n",dlerror()); return 1; }
    const char *name = argv[2][0]=='g' ? "steps_forecast_gpu_all" : "steps_forecast_native";
    mojosteps_forecast_fn forecast=(mojosteps_forecast_fn)dlsym(library,name);
    if (!forecast) { fprintf(stderr,"%s\n",dlerror()); return 1; }
    int status=forecast(input,input+3*pixels,output,members,leads,20260817,workers,1);
    if (status) { fprintf(stderr,"forecast status %d\n",status); return status; }
    f=fopen(argv[4],"wb");
    if (!f || fwrite(output,sizeof(double),outputs,f)!=outputs) return 1;
    if (fclose(f)) return 1;
    dlclose(library); free(input); free(output);
    return 0;
}
