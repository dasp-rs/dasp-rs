//! Uninstrumented release latency measurements; allocation probes are separate.
mod common;
use std::time::Instant;
fn main() {
    let Some(options) = common::options() else {
        return;
    };
    let prepared = common::prepare(&options);
    for _ in 0..options.warmup {
        prepared.run();
    }
    let mut elapsed = Vec::with_capacity(options.repetitions);
    for _ in 0..options.repetitions {
        let start = Instant::now();
        prepared.run();
        elapsed.push(start.elapsed().as_secs_f64());
    }
    let values = elapsed
        .iter()
        .map(|v| format!("{v:.12}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"kind\":\"timing\",\"case\":{},\"warmup\":{},\"seconds\":[{}]}}",
        prepared.case.json(),
        options.warmup,
        values
    );
}
