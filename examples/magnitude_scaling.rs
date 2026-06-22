//! Magnitude scaling and loudness: dB conversions, perceptual weighting, PCEN.
//!
//! Run with: `cargo run --example magnitude_scaling`

use dasp_rs::{feat, generate::tone, mag, proc, util};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(1.0).compute();

    // Compute a power spectrogram to work with.
    let spec = proc::stft(&y).n_fft(2048).hop_length(512).compute()?;
    let power = spec.mapv(|c| c.norm_sqr()); // |S|²
    let amplitude = spec.mapv(|c| c.norm()); // |S|

    // ── Amplitude → dB ───────────────────────────────────────────────────────
    // amplitude_to_db(A) ≈ 20 log₁₀(A / ref)
    let db_amp = mag::amplitude_to_db(&amplitude)
        .ref_val(1.0)
        .top_db(80.0)
        .compute()?;
    println!("Amplitude dB: {:?}, min={:.1}", db_amp.shape(), db_amp.iter().copied().fold(f32::INFINITY, f32::min));

    // ── Power → dB ───────────────────────────────────────────────────────────
    // power_to_db(P) ≈ 10 log₁₀(P / ref)
    let db_power = mag::power_to_db(&power)
        .ref_val(1.0)
        .top_db(80.0)
        .compute()?;
    println!("Power dB:     {:?}", db_power.shape());

    // ── dB → amplitude / power (inverse) ─────────────────────────────────────
    // ref_val: None → defaults to 1.0; returns Result so needs ?
    let amp_back = mag::db_to_amplitude(&db_amp, None)?;
    let pow_back = mag::db_to_power(&db_power, None)?;
    println!("dB→amp:       {:?}", amp_back.shape());
    println!("dB→power:     {:?}", pow_back.shape());

    // ── Perceptual weighting ─────────────────────────────────────────────────
    // a_weighting takes a frequency slice; use fft_frequencies() to build it.
    let freqs = util::fft_frequencies().sample_rate(sr).n_fft(2048).compute();
    let a_weights = mag::a_weighting(&freqs, None)?;
    println!("A-weights:    {} bins", a_weights.len());

    // perceptual_weighting multiplies a power spec by a frequency-dependent curve.
    // kind: None → "A"-weighting by default.
    let pw = mag::perceptual_weighting(&power, &freqs, None)?;
    println!("Perceptual W: {:?}", pw.shape());

    // ── PCEN (Per-Channel Energy Normalisation) ───────────────────────────────
    // Replaces log-mel compression; more robust to level variation.
    // Builder setters: .gain(), .bias(), .hop_length(), .sample_rate()
    let mel = feat::spectral(&y, sr).n_fft(2048).hop_length(512).melspectrogram()?;
    let pcen = mag::pcen(&mel)
        .gain(0.98)
        .bias(2.0)
        .hop_length(512)
        .sample_rate(sr)
        .compute()?;
    println!("PCEN:         {:?}", pcen.shape());

    Ok(())
}
