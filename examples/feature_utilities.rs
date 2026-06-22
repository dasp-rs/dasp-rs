//! Feature matrix utilities: normalization, CMVN, softmask, sparsify, stack memory,
//! temporal kurtosis, and zero-crossing rate.
//!
//! Run with: `cargo run --example feature_utilities`

use dasp_rs::{feat, feat::FeatureNorm, generate::tone, proc};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(2.0).compute();
    let n_fft = 2048_usize;
    let hop = 512_usize;

    // ── Normalize ────────────────────────────────────────────────────────────
    // Divides each column (frame) by its norm. Silent frames are left unchanged.
    // FeatureNorm: L1 | L2 (default) | LInf
    // axis: 0 = normalise each column (frame), 1 = normalise each row (feature).
    let chroma = feat::spectral(&y, sr).hop_length(hop).chroma_cqt()?;
    let normed_l2   = feat::normalize(&chroma).norm(FeatureNorm::L2).axis(0).compute();
    let normed_linf = feat::normalize(&chroma).norm(FeatureNorm::LInf).axis(0).compute();
    println!("Chroma:      {:?}", chroma.shape());
    println!("Normed L2:   {:?}", normed_l2.shape());
    println!("Normed LInf: {:?}", normed_linf.shape());

    // ── CMVN ─────────────────────────────────────────────────────────────────
    // Cepstral Mean–Variance Normalisation: z-score each feature dimension over time.
    // Result has zero mean and unit variance per row.
    let mfcc = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).mfcc()?;
    let mfcc_cmvn = feat::cmvn(&mfcc).compute()?;
    println!("MFCC:        {:?}", mfcc.shape());
    println!("CMVN:        {:?}", mfcc_cmvn.shape());

    // ── Softmask ─────────────────────────────────────────────────────────────
    // mask[i,j] = x[i,j]^p / (x[i,j]^p + x_ref[i,j]^p + ε)
    // power=2 → smooth mask; power=∞ → hard (binary) mask
    let spec = proc::stft(&y).n_fft(n_fft).hop_length(hop).compute()?;
    let mag  = spec.mapv(|c| c.norm());
    let ones = Array2::from_elem(mag.raw_dim(), 1.0_f32);
    let mask_soft = feat::softmask(&mag, &ones).power(2.0).compute();
    let mask_hard = feat::softmask(&mag, &ones).power(f32::INFINITY).compute();
    println!("Softmask(2): {:?}, mean={:.3}", mask_soft.shape(), mask_soft.mean().unwrap_or(0.0));
    println!("Softmask(∞): {:?}, unique values are 0.0 or 1.0", mask_hard.shape());

    // ── Sparsify rows ─────────────────────────────────────────────────────────
    // Zeros entries below the quantile-th percentile in each row.
    // Useful for cleaning noisy feature matrices before further processing.
    let sparse = feat::sparsify_rows(&chroma).quantile(0.1).compute();
    let nonzero = sparse.iter().filter(|&&x| x != 0.0).count();
    println!("Sparsify:    {:?}, {nonzero} nonzero entries", sparse.shape());

    // ── Stack memory ──────────────────────────────────────────────────────────
    // Augments each frame with n_steps-1 delayed copies for temporal context.
    // Output rows = n_features * n_steps; columns stay the same.
    let stacked = feat::stack_memory(&chroma, Some(3), Some(1));
    println!("Stack×3:     {:?} (3× rows, same cols)", stacked.shape());

    // ── Temporal kurtosis ─────────────────────────────────────────────────────
    // Per-frame excess kurtosis: positive → heavy tails (impulsive content).
    let kurtosis = feat::temporal_kurtosis(&y)
        .frame_length(n_fft)
        .hop_length(hop)
        .compute()?;
    println!("Kurtosis:    {} frames, mean={:.4}", kurtosis.len(), kurtosis.mean().unwrap_or(0.0));

    // ── Zero-crossing rate ───────────────────────────────────────────────────
    // Fraction of sign changes per frame. High for noisy or fricative signals.
    let zcr = feat::zero_crossing_rate(&y)
        .frame_length(n_fft)
        .hop_length(hop)
        .compute();
    println!("ZCR:         {} frames, mean={:.4}", zcr.len(), zcr.mean().unwrap_or(0.0));

    Ok(())
}
