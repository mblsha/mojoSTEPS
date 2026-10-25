// SPDX-License-Identifier: BSD-3-Clause
// Process-local Mach/Metal memory observer; independent of forecast code.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
static NSArray<id<MTLDevice>> *devices;
static std::atomic<bool> running{false};
static std::thread sampler;
static std::uint64_t peaks[8]{};
static std::uint64_t samples=0;
extern "C" int mem_init(int metal) {
    @autoreleasepool {
        if (metal) devices=MTLCopyAllDevices();
        return metal ? int(devices.count) : 0;
    }
}
extern "C" const char *mem_device_name() {
    return devices.count ? [devices[0].name UTF8String] : "none";
}
extern "C" int mem_unified() { return devices.count ? devices[0].hasUnifiedMemory : 0; }
extern "C" int mem_snapshot(std::uint64_t *out) {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    kern_return_t status=task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count);
    if (status!=KERN_SUCCESS) return int(status);
    out[0]=info.resident_size;
    out[1]=info.resident_size_peak;
    out[2]=info.phys_footprint;
    out[3]=std::uint64_t(info.ledger_phys_footprint_peak);
    out[4]=0;
    @autoreleasepool { for (id<MTLDevice> device in devices) out[4]+=device.currentAllocatedSize; }
    out[5]=info.compressed;
    out[6]=std::uint64_t(std::max<std::int64_t>(0,info.ledger_tag_graphics_footprint));
    out[7]=std::uint64_t(std::max<std::int64_t>(0,info.ledger_tag_graphics_nofootprint));
    return 0;
}
static void sample() {
    std::uint64_t current[8]{};
    if (!mem_snapshot(current)) {
        for (int i=0;i<8;++i) peaks[i]=std::max(peaks[i],current[i]);
        ++samples;
    }
}
extern "C" void mem_start() {
    std::fill(std::begin(peaks),std::end(peaks),0);
    samples=0;sample();running=true;
    sampler=std::thread([] {
        while (running.load()) {
            sample();
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    });
}
extern "C" std::uint64_t mem_stop(std::uint64_t *out) {
    running=false;if (sampler.joinable()) sampler.join();sample();
    std::copy(std::begin(peaks),std::end(peaks),out);
    return samples;
}
extern "C" int mem_control(std::uint64_t *out) {
    if (!devices.count) return 0;
    @autoreleasepool {
        id<MTLDevice> device=devices[0];
        out[0]=device.currentAllocatedSize;
        id<MTLBuffer> buffer=[device newBufferWithLength:64*1024*1024 options:MTLResourceStorageModeShared];
        if (!buffer) return 0;
        std::memset(buffer.contents,1,buffer.length);
        out[1]=device.currentAllocatedSize;
        buffer=nil;
    }
    out[2]=devices[0].currentAllocatedSize;
    return 1;
}
