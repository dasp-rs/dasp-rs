//! Synthetic signal generation: tones, chirps, and click tracks.
//!
//! Run with: `cargo run --example signal_generation`

use dasp_rs::generate::{chirp, clicks, tone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 44100_u32;

    // ── Sine tone ────────────────────────────────────────────────────────────
    // tone(frequency_hz, sample_rate) → ToneBuilder
    let y_tone = tone(440.0, sr)
        .duration(1.0)   // seconds (default 1.0)
        .phase(0.0)      // initial phase in radians (default 0.0)
        .compute();
    println!("Tone 440 Hz: {} samples, peak {:.4}", y_tone.len(), peak(&y_tone));

    // A 1 kHz tone, half a second
    let y_1k = tone(1000.0, sr).duration(0.5).compute();
    println!("Tone 1 kHz:  {} samples", y_1k.len());

    // ── Chirp (linear frequency sweep) ──────────────────────────────────────
    // chirp(fmin_hz, fmax_hz, sample_rate) → ChirpBuilder
    let y_chirp = chirp(200.0, 4000.0, sr)
        .duration(2.0)
        .compute();
    println!("Chirp 200→4000 Hz: {} samples, peak {:.4}", y_chirp.len(), peak(&y_chirp));

    // ── Click track ──────────────────────────────────────────────────────────
    // clicks() → ClicksBuilder — specify either seconds or frame indices
    let beat_times = [0.0_f32, 0.5, 1.0, 1.5, 2.0];
    let y_clicks = clicks()
        .times(&beat_times)
        .sample_rate(sr)
        .compute()?;
    println!("Click track: {} samples for {} beats", y_clicks.len(), beat_times.len());

    // Clicks from frame indices (e.g. beat frames from beat tracking)
    let beat_frames = [0_usize, 22, 44, 66, 88];
    let y_click_frames = clicks()
        .frames(&beat_frames)
        .sample_rate(sr)
        .hop_length(512)
        .compute()?;
    println!("Click frames: {} samples", y_click_frames.len());

    Ok(())
}

fn peak(x: &[f32]) -> f32 {
    x.iter().copied().fold(0.0_f32, f32::max)
}
