//! Harmonic/percussive source separation and soft masking.
//!
//! Run with: `cargo run --example harmonic_percussive`

use dasp_rs::{feat, generate::tone, proc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(2.0).compute();

    // ── HPSS via spectral builder ─────────────────────────────────────────────
    // hpss() returns (harmonic_power_spec, percussive_power_spec).
    let (h_spec, p_spec) = feat::spectral(&y, sr)
        .n_fft(2048)
        .hop_length(512)
        .hpss()?;
    println!("HPSS harmonic:    {:?}", h_spec.shape());
    println!("HPSS percussive:  {:?}", p_spec.shape());

    // ── Convenience builders: harmonic() / percussive() ──────────────────────
    // These wrap hpss() to return just one component directly.
    let h_only = feat::harmonic(&y, sr)
        .n_fft(2048)
        .hop_length(512)
        .compute()?;
    let p_only = feat::percussive(&y, sr)
        .n_fft(2048)
        .hop_length(512)
        .compute()?;
    println!("harmonic():       {:?}", h_only.shape());
    println!("percussive():     {:?}", p_only.shape());

    // ── Soft mask ─────────────────────────────────────────────────────────────
    // softmask(x, x_ref) computes x^p / (x^p + x_ref^p + ε).
    // Commonly used to apply HPSS separation in the magnitude domain.
    let spec = proc::stft(&y).n_fft(2048).hop_length(512).compute()?;
    let mag = spec.mapv(|c| c.norm());

    // power=2.0 gives a smooth mask; power=∞ gives a binary (hard) mask
    let h_mask = feat::softmask(&h_spec, &p_spec).power(2.0).compute();
    let p_mask = feat::softmask(&p_spec, &h_spec).power(2.0).compute();
    println!("Harmonic mask:    {:?}, sum={:.1}", h_mask.shape(), h_mask.sum());
    println!("Percussive mask:  {:?}, sum={:.1}", p_mask.shape(), p_mask.sum());

    // Apply mask to the magnitude spectrogram
    let h_mag = &mag * &h_mask;
    let p_mag = &mag * &p_mask;
    println!("Masked harmonic mag:    {:?}", h_mag.shape());
    println!("Masked percussive mag:  {:?}", p_mag.shape());

    Ok(())
}
