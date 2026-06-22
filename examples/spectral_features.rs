//! Spectral and MIR features via the unified `spectral` builder.
//!
//! Run with: `cargo run --example spectral_features`
//!
//! `spectral(&y, sr)` is the main entry point. Configure shared parameters
//! with setters (`.n_fft()`, `.hop_length()`, …) then call one terminal method
//! per feature type. All terminal methods consume the builder.

use dasp_rs::{feat, generate::tone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(2.0).compute();
    let n_fft = 2048_usize;
    let hop   = 512_usize;

    // ── Mel spectrogram ──────────────────────────────────────────────────────
    let mel = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).n_mels(128).melspectrogram()?;
    println!("Mel spectrogram:      {:?}", mel.shape());

    // ── MFCCs (20 coefficients by default) ───────────────────────────────────
    let mfcc = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).mfcc()?;
    println!("MFCC:                 {:?}", mfcc.shape());

    // ── Chroma features ───────────────────────────────────────────────────────
    let chroma_cqt  = feat::spectral(&y, sr).hop_length(hop).chroma_cqt()?;
    let chroma_cens = feat::spectral(&y, sr).hop_length(hop).chroma_cens()?;
    println!("Chroma CQT:           {:?}", chroma_cqt.shape());
    println!("Chroma CENS:          {:?}", chroma_cens.shape());

    // Standalone builders for STFT-chroma and VQT-chroma
    let chroma_stft = feat::chroma_stft(&y, sr).n_fft(n_fft).hop_length(hop).compute()?;
    let chroma_vqt  = feat::chroma_vqt(&y, sr).hop_length(hop).compute()?;
    println!("Chroma STFT:          {:?}", chroma_stft.shape());
    println!("Chroma VQT:           {:?}", chroma_vqt.shape());

    // ── Frame-level energy ────────────────────────────────────────────────────
    let rms = feat::spectral(&y, sr).hop_length(hop).rms()?;
    println!("RMS energy:           {} frames", rms.len());

    // ── Spectral shape descriptors (return Array1 — one value per frame) ─────
    let centroid  = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).spectral_centroid()?;
    let bandwidth = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).spectral_bandwidth()?;
    let flatness  = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).spectral_flatness()?;
    let rolloff   = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).spectral_rolloff()?;
    let flux      = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).spectral_flux()?;
    println!("Spectral centroid:    {} frames", centroid.len());
    println!("Spectral bandwidth:   {} frames", bandwidth.len());
    println!("Spectral flatness:    {} frames", flatness.len());
    println!("Spectral rolloff:     {} frames", rolloff.len());
    println!("Spectral flux:        {} frames", flux.len());

    // ── Spectral contrast ─────────────────────────────────────────────────────
    let contrast = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).spectral_contrast()?;
    println!("Spectral contrast:    {:?}", contrast.shape());

    // ── Tonal centroid (Tonnetz) ──────────────────────────────────────────────
    let tonnetz = feat::spectral(&y, sr).hop_length(hop).tonnetz()?;
    println!("Tonnetz:              {:?}", tonnetz.shape());

    // ── HPSS — harmonic/percussive separation ─────────────────────────────────
    let (h_spec, p_spec) = feat::spectral(&y, sr).n_fft(n_fft).hop_length(hop).hpss()?;
    println!("HPSS harmonic:        {:?}", h_spec.shape());
    println!("HPSS percussive:      {:?}", p_spec.shape());

    // ── CMVN: cepstral mean-variance normalisation ────────────────────────────
    let normalised = feat::cmvn(&mfcc).compute()?;
    println!("CMVN MFCC:            {:?}", normalised.shape());

    Ok(())
}
