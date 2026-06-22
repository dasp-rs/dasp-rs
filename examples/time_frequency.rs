//! Time–frequency transforms: STFT, CQT, VQT, magnitude/phase decomposition.
//!
//! Run with: `cargo run --example time_frequency`

use dasp_rs::{generate::tone, proc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(1.0).compute();

    // ── STFT / ISTFT ─────────────────────────────────────────────────────────
    let spec = proc::stft(&y)
        .n_fft(2048)
        .hop_length(512)
        .compute()?;
    println!("STFT:          {} bins × {} frames", spec.nrows(), spec.ncols());

    // istft returns Vec<f32> directly (no Result)
    let y_rec = proc::istft(&spec).hop_length(512).compute();
    println!("ISTFT:         {} samples reconstructed", y_rec.len());

    // ── Magnitude and phase ───────────────────────────────────────────────────
    // magphase splits a complex STFT into magnitude and unit-phase components.
    let (magnitude, phase) = proc::magphase(&spec, None);
    println!("Magnitude:     {:?}", magnitude.shape());
    println!("Phase:         {:?}", phase.shape());

    // ── Phasor: rebuild complex spec from angles + magnitude ──────────────────
    let angles = phase.mapv(|c| c.arg()); // convert unit-phase to radians
    let rebuilt = proc::phasor(&angles, Some(&magnitude));
    println!("Phasor rebuilt: {:?}", rebuilt.shape());

    // ── Reassigned spectrogram ────────────────────────────────────────────────
    // Returns a 2-D time-reassigned energy map (sharper spectral ridges).
    let reassigned = proc::reassigned_spectrogram(&y, sr).n_fft(512).compute()?;
    println!("Reassigned:    {:?}", reassigned.shape());

    // ── CQT / ICQT ───────────────────────────────────────────────────────────
    let c = proc::cqt(&y, sr)
        .hop_length(512)
        .n_bins(84)   // 7 octaves × 12 bins/octave
        .compute()?;
    println!("CQT:           {} bins × {} frames", c.nrows(), c.ncols());

    let y_cqt_rec = proc::icqt(&c).compute()?;
    println!("ICQT:          {} samples reconstructed", y_cqt_rec.len());

    // ── VQT (Variable-Q) ─────────────────────────────────────────────────────
    let v = proc::vqt(&y, sr).n_bins(84).hop_length(512).compute()?;
    println!("VQT:           {} bins × {} frames", v.nrows(), v.ncols());

    // ── Pseudo-CQT (magnitude only, much faster) ─────────────────────────────
    let pc = proc::pseudo_cqt(&y, sr).compute()?;
    println!("Pseudo-CQT:    {:?}", pc.shape());

    // ── Hybrid CQT ───────────────────────────────────────────────────────────
    let hc = proc::hybrid_cqt(&y, sr).compute()?;
    println!("Hybrid CQT:    {:?}", hc.shape());

    Ok(())
}
