//! Audio effects and signal manipulation: trim, time-stretch, pitch-shift, remix.
//!
//! Run with: `cargo run --example effects`

use dasp_rs::{generate::tone, proc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(2.0).compute();

    // ── Trim silence ──────────────────────────────────────────────────────────
    // Removes leading and trailing silence below top_db below the peak level.
    let (trimmed, (start, end)) = proc::trim(&y).top_db(60.0).compute();
    println!("Trim:          [{start}..{end}] → {} samples (was {})", trimmed.len(), y.len());

    // ── Split on silence ──────────────────────────────────────────────────────
    // Returns non-silent intervals as (start, end) sample pairs.
    let intervals = proc::split(&y).top_db(60.0).compute();
    println!("Split:         {} interval(s)", intervals.len());

    // ── Time stretching ───────────────────────────────────────────────────────
    // rate < 1.0 → slower, rate > 1.0 → faster. Preserves pitch.
    let y_slow = proc::time_stretch(&y, 0.75).compute()?;
    let y_fast = proc::time_stretch(&y, 1.5).compute()?;
    println!("Time stretch ×0.75: {} samples", y_slow.len());
    println!("Time stretch ×1.50: {} samples", y_fast.len());

    // ── Pitch shifting ────────────────────────────────────────────────────────
    // n_steps in semitones (positive = up, negative = down). Preserves duration.
    let y_up = proc::pitch_shift(&y, sr, 4.0).compute()?;
    let y_dn = proc::pitch_shift(&y, sr, -7.0).compute()?;
    println!("Pitch +4 semi:  {} samples", y_up.len());
    println!("Pitch -7 semi:  {} samples", y_dn.len());

    // ── Preemphasis / deemphasis ─────────────────────────────────────────────
    // Preemphasis boosts high frequencies (common before speech feature extraction).
    let y_pre = proc::preemphasis(&y).coef(0.97).compute();
    let y_de  = proc::deemphasis(&y_pre).coef(0.97).compute();
    println!("Preemphasis:   {} samples", y_pre.len());
    println!("Deemphasis:    {} samples (roundtrip)", y_de.len());

    // ── Remix ─────────────────────────────────────────────────────────────────
    // Concatenate arbitrary non-overlapping segments in any order.
    // intervals: (start_sample, end_sample)
    let segs: &[(usize, usize)] = &[(0, 8000), (16000, 22050)];
    let y_remix = proc::remix(&y, segs).align_zeros(true).compute();
    println!("Remix:         {} samples from {} segments", y_remix.len(), segs.len());

    Ok(())
}
