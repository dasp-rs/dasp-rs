//! Array and signal utilities: framing, padding, peak-picking, sync, and interval matching.
//!
//! Run with: `cargo run --example array_utilities`

use dasp_rs::{generate::tone, util, util::SyncAggregate};
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(1.0).compute();

    // ── Frame ─────────────────────────────────────────────────────────────────
    // Slice a 1-D signal into overlapping frames of shape (frame_length, n_frames).
    let frames = util::frame(&y, 2048).hop_length(512).compute();
    println!("Frame:         {:?} (frame_length × n_frames)", frames.shape());

    // ── Pad center ────────────────────────────────────────────────────────────
    // Zero-pad symmetrically so the original data is centred in a buffer of `size`.
    let padded = util::pad_center(&y[..100], 256);
    println!("Pad center:    100 samples → {} (centered)", padded.len());

    // ── Fix length ────────────────────────────────────────────────────────────
    // Truncate or zero-pad to exactly `size` samples.
    let fixed_short = util::fix_length(&y, 22050 / 2);
    let fixed_long  = util::fix_length(&y, y.len() + 1000);
    println!("Fix (shorter): {} → {}", y.len(), fixed_short.len());
    println!("Fix (longer):  {} → {}", y.len(), fixed_long.len());

    // ── Local maxima / minima ─────────────────────────────────────────────────
    // Returns a bool mask, true at strict local peaks / troughs.
    let ramp: Vec<f32> = (0..20).map(|i| (i as f32 * 0.5).sin()).collect();
    let maxima = util::localmax(&ramp);
    let minima = util::localmin(&ramp);
    println!("Local max:     {} peaks in ramp", maxima.iter().filter(|&&b| b).count());
    println!("Local min:     {} troughs in ramp", minima.iter().filter(|&&b| b).count());

    // ── Peak pick ─────────────────────────────────────────────────────────────
    // Fine-grained selector: must be a local max within pre/post_max window AND
    // exceed mean+delta within pre/post_avg window AND respect a minimum spacing.
    let onset_env: Vec<f32> = (0..200).map(|i| (i as f32 * 0.15).sin().abs()).collect();
    let peaks = util::peak_pick(&onset_env)
        .pre_max(3)
        .post_max(3)
        .pre_avg(3)
        .post_avg(5)
        .delta(0.07)
        .wait(10)
        .compute();
    println!("Peak pick:     {} peaks found", peaks.len());

    // ── Sync ─────────────────────────────────────────────────────────────────
    // Aggregate a feature matrix to event-aligned frames (e.g., beat or segment boundaries).
    // Returns shape (n_features, n_events+1) depending on padding.
    let data: Array2<f32> = Array2::from_shape_fn((12, 100), |(i, j)| (i + j) as f32);
    let event_frames = vec![15_usize, 35, 60, 85];
    let synced_mean = util::sync(&data, &event_frames)
        .aggregate(SyncAggregate::Mean)
        .compute();
    let synced_max = util::sync(&data, &event_frames)
        .aggregate(SyncAggregate::Max)
        .compute();
    println!("Sync (mean):   {:?}", synced_mean.shape());
    println!("Sync (max):    {:?}", synced_max.shape());

    // ── Match intervals ───────────────────────────────────────────────────────
    // For each interval in `from`, returns the index of the best-matching interval
    // in `to` (maximises overlap). Useful for aligning annotations to segments.
    let from: &[(f32, f32)] = &[(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)];
    let to:   &[(f32, f32)] = &[(0.0, 0.4), (0.4, 0.9), (0.9, 1.5)];
    let matched = util::match_intervals(from, to);
    println!("Match intervals: {:?}", matched);

    // ── Expand to ─────────────────────────────────────────────────────────────
    // Broadcast a 1-D array into a 2-D matrix.
    //   axis=0 → column vector (n, 1) — use for per-frequency weights
    //   axis=1 → row vector    (1, n) — use for per-frame envelopes
    let weights = Array1::from_vec(vec![0.5_f32, 1.0, 1.5, 2.0]);
    let col = util::expand_to(&weights, 0); // (4, 1)
    let row = util::expand_to(&weights, 1); // (1, 4)
    println!("Expand axis=0: {:?}", col.shape());
    println!("Expand axis=1: {:?}", row.shape());

    Ok(())
}
