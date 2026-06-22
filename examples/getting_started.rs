//! Getting started with dasp-rs.
//!
//! Run with: `cargo run --example getting_started`
//!
//! This example walks through the core workflow: generate a signal, transform
//! it into the frequency domain, and extract a couple of common features.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;

    // ── 1. Generate a 440 Hz sine tone (2 seconds) ──────────────────────────
    let y = dasp_rs::generate::tone(440.0, sr)
        .duration(2.0)
        .compute();
    println!("Signal: {} samples at {} Hz ({:.1}s)", y.len(), sr, y.len() as f32 / sr as f32);

    // ── 2. Short-Time Fourier Transform ─────────────────────────────────────
    let spec = dasp_rs::proc::stft(&y)
        .n_fft(2048)
        .hop_length(512)
        .compute()?;
    println!("STFT:   {} bins × {} frames", spec.nrows(), spec.ncols());

    // ── 3. MFCCs ────────────────────────────────────────────────────────────
    let mfcc = dasp_rs::feat::spectral(&y, sr)
        .n_fft(2048)
        .hop_length(512)
        .mfcc()?;
    println!("MFCC:   {:?} (coefficients × frames)", mfcc.shape());

    // ── 4. Tempo estimate ───────────────────────────────────────────────────
    let bpm = dasp_rs::feat::tempo(&y, sr).compute()?;
    println!("Tempo:  {bpm:.1} BPM");

    // ── 5. Pitch (YIN) ──────────────────────────────────────────────────────
    let f0 = dasp_rs::pitch::yin(&y, 50.0, 2000.0)
        .sample_rate(sr)
        .compute()?;
    let voiced: Vec<f32> = f0.iter().copied().filter(|&x| x > 0.0).collect();
    if !voiced.is_empty() {
        let mean_f0 = voiced.iter().sum::<f32>() / voiced.len() as f32;
        println!("F0:     {mean_f0:.1} Hz (mean over voiced frames)");
    }

    Ok(())
}
