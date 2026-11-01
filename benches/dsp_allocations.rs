//! Separate instrumented process: counters never contaminate timing samples.
mod common;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn main() {
    let Some(options) = common::options() else {
        return;
    };
    let prepared = common::prepare(&options);
    for _ in 0..options.warmup {
        prepared.run();
    }
    for _ in 0..options.repetitions {
        let region = Region::new(GLOBAL);
        prepared.run();
        let stats = region.change();
        println!(
            "{{\"kind\":\"allocation\",\"case\":{},\"allocations\":{},\"deallocations\":{},\"reallocations\":{},\"bytes_allocated\":{},\"bytes_deallocated\":{},\"bytes_reallocated\":{}}}",
            prepared.case.json(),
            stats.allocations,
            stats.deallocations,
            stats.reallocations,
            stats.bytes_allocated,
            stats.bytes_deallocated,
            stats.bytes_reallocated
        );
    }
}
