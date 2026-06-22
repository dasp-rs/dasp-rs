//! Pitch detection and tuning estimation: YIN, pYIN, piptrack, tuning.
//!
//! Run with: `cargo run --example pitch_detection`

use dasp_rs::{generate::tone, pitch};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let fmin = 50.0_f32;
    let fmax = 2000.0_f32;

    // A 440 Hz tone — pitch detectors should return ≈440 Hz.
    let y = tone(440.0, sr).duration(1.0).compute();

    // ── YIN ──────────────────────────────────────────────────────────────────
    // Classic YIN algorithm; returns one F0 estimate per frame.
    let f0_yin = pitch::yin(&y, fmin, fmax)
        .sample_rate(sr)
        .frame_length(2048)
        .hop_length(512)
        .compute()?;
    let mean_f0 = voiced_mean(&f0_yin);
    println!("YIN mean F0:       {mean_f0:.1} Hz over {} frames", f0_yin.len());

    // ── pYIN (probabilistic) ──────────────────────────────────────────────────
    // More robust than YIN; also returns per-frame voiced probability.
    let f0_pyin = pitch::pyin(&y, fmin, fmax)
        .sample_rate(sr)
        .frame_length(2048)
        .hop_length(512)
        .compute()?;
    let mean_pyin = voiced_mean(&f0_pyin);
    println!("pYIN mean F0:      {mean_pyin:.1} Hz over {} frames", f0_pyin.len());

    // ── Piptrack ─────────────────────────────────────────────────────────────
    // Spectral peak-picking pitch tracker. Returns (pitches, magnitudes).
    let (pitches, magnitudes) = pitch::piptrack(&y)
        .sample_rate(sr)
        .n_fft(2048)
        .hop_length(512)
        .compute()?;
    println!("Piptrack:          {:?} pitches, {:?} magnitudes", pitches.shape(), magnitudes.shape());

    // Find the loudest pitch in each frame.
    let n_frames = pitches.shape()[1];
    let mut dominant: Vec<f32> = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let col_mag = magnitudes.column(t);
        if let Some(idx) = col_mag.iter().copied().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i) {
            dominant.push(pitches[[idx, t]]);
        }
    }
    let mean_pip = voiced_mean(&dominant);
    println!("Piptrack dominant: {mean_pip:.1} Hz mean");

    // ── Tuning estimation ─────────────────────────────────────────────────────
    // Returns the deviation of the signal's pitch from equal temperament in cents.
    let deviation_cents = pitch::estimate_tuning(&y)
        .sample_rate(sr)
        .compute()?;
    println!("Tuning deviation:  {deviation_cents:+.1} cents from A440");

    // pitch_tuning converts a list of detected frequencies to a tuning offset.
    let detected = vec![438.0_f32, 441.0, 440.5];
    let tuning = pitch::pitch_tuning(&detected, None)?;
    println!("pitch_tuning:      {tuning:+.3} (fractional semitone offset)");

    Ok(())
}

fn voiced_mean(f0: &[f32]) -> f32 {
    let voiced: Vec<f32> = f0.iter().copied().filter(|&x| x > 0.0).collect();
    if voiced.is_empty() { 0.0 } else { voiced.iter().sum::<f32>() / voiced.len() as f32 }
}
