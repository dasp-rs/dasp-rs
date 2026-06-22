//! Rhythm analysis: tempo, beat tracking, tempograms, onset detection.
//!
//! Run with: `cargo run --example rhythm_analysis`

use dasp_rs::{feat, generate::tone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    // Use a tone — tempo estimation works better with actual music,
    // but this exercises the API paths correctly.
    let y = tone(440.0, sr).duration(4.0).compute();

    // ── Tempo estimation ──────────────────────────────────────────────────────
    let bpm = feat::tempo(&y, sr)
        .hop_length(512)
        .compute()?;
    println!("Estimated tempo: {bpm:.1} BPM");

    // ── Autocorrelation tempogram ─────────────────────────────────────────────
    // tempogram() returns a 2-D autocorrelation lag × time matrix.
    let tg = feat::tempogram(
        Some(&y),
        Some(sr),
        None,        // pass None to compute onset envelope internally
        Some(512),   // hop_length
        Some(384),   // win_length (controls lag resolution)
    )?;
    println!("Tempogram:       {:?} (lag_bins × frames)", tg.shape());

    // ── Fourier tempogram ─────────────────────────────────────────────────────
    // Captures non-integer and fractional tempo multiples.
    let ftg = feat::fourier_tempogram(&y, sr)
        .hop_length(512)
        .win_length(384)
        .compute()?;
    println!("Fourier tempogram: {:?} (freq_bins × frames)", ftg.shape());

    // ── Multi-band onset strength ─────────────────────────────────────────────
    // Separate onset curves for N frequency sub-bands (useful for rhythm MIR).
    let odf = feat::onset_strength_multi(&y, sr)
        .n_bands(6)
        .hop_length(512)
        .compute()?;
    println!("Onset strength multi: {:?} (bands × frames)", odf.shape());

    // ── Beat tracking with click track ────────────────────────────────────────
    // beat_track returns (tempo_bpm, beat_frame_indices)
    let (beat_bpm, beat_frames) = feat::beat_track(&y, sr).compute()?;
    println!("Beat BPM:        {beat_bpm:.1}");
    println!("Beat frames:     {:?}", &beat_frames[..beat_frames.len().min(8)]);

    // Render beats as a click track
    let click = dasp_rs::generate::clicks()
        .frames(&beat_frames)
        .sample_rate(sr)
        .hop_length(512)
        .compute()?;
    println!("Beat click track: {} samples", click.len());

    Ok(())
}
