//! Non-negative Matrix Factorization (NMF) spectral decomposition.
//!
//! Run with: `cargo run --example decomposition`

use dasp_rs::{feat, generate::tone, proc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(2.0).compute();

    // Compute a magnitude spectrogram to factorize (NMF requires non-negative input).
    let spec = proc::stft(&y).n_fft(2048).hop_length(512).compute()?;
    let mag = spec.mapv(|c| c.norm());
    println!("Input magnitude: {:?}", mag.shape());

    // ── NMF decomposition ─────────────────────────────────────────────────────
    // Finds W (spectral templates) and H (temporal activations) such that W @ H ≈ mag.
    //   W shape: (n_bins, n_components)
    //   H shape: (n_components, n_frames)
    let n_components = 8_usize;
    let (w, h) = feat::decompose(&mag, n_components)
        .n_iter(200)
        .random_seed(42)
        .compute()?;

    println!("Templates W:   {:?} (freq bins × components)", w.shape());
    println!("Activations H: {:?} (components × frames)", h.shape());

    // ── Reconstruction ────────────────────────────────────────────────────────
    // W @ H approximates the original magnitude spectrogram.
    let reconstruction = w.dot(&h);
    let error = (&mag - &reconstruction).mapv(|x| x * x).sum().sqrt();
    println!("Frobenius reconstruction error: {error:.4}");

    // ── Fewer components / iterations ─────────────────────────────────────────
    // Fewer components → coarser, faster decomposition.
    let (w4, _h4) = feat::decompose(&mag, 4).n_iter(50).compute()?;
    println!("Coarse (4 comps, 50 iters): W={:?}", w4.shape());

    // ── Component inspection ──────────────────────────────────────────────────
    // Each column of W is a spectral template; the peak frequency of each is:
    let sr_f = sr as f32;
    let bin_hz = sr_f / (2048.0 * 2.0); // Hz per FFT bin
    for k in 0..n_components {
        let col = w.column(k);
        let peak_bin = col
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        println!("  Component {k}: peak at bin {peak_bin} ({:.1} Hz)", peak_bin as f32 * bin_hz);
    }

    Ok(())
}
