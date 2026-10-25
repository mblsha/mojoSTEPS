// SPDX-License-Identifier: BSD-3-Clause
// Inventory public Metal counter sets; no claim about Instruments-only counters.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main() { @autoreleasepool {
  NSMutableArray *devices = [NSMutableArray array];
  for (id<MTLDevice> d in MTLCopyAllDevices()) {
    NSMutableArray *sets = [NSMutableArray array];
    for (id<MTLCounterSet> s in d.counterSets) {
      NSMutableArray *names = [NSMutableArray array];
      for (id<MTLCounter> c in s.counters) [names addObject:c.name];
      [sets addObject:@{@"name":s.name,@"counters":names}];
    }
    [devices addObject:@{@"device":d.name,@"counter_sets":sets}];
  }
  NSData *json = [NSJSONSerialization dataWithJSONObject:devices options:NSJSONWritingPrettyPrinted error:nil];
  puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
} }
