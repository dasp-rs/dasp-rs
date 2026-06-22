//! Audio input/output: write a WAV file, read it back, and stream it.
//!
//! Run with: `cargo run --example audio_io`

use dasp_rs::{generate::tone, io, types::AudioData, util};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 44100_u32;
    let path = "example_440hz.wav";

    // ── Generate and export ──────────────────────────────────────────────────
    let samples = tone(440.0, sr).duration(1.0).compute();
    let audio = AudioData::new(samples, sr, 1)?;
    io::export(path, &audio)?;
    println!("Exported '{}' ({} bytes)", path, std::fs::metadata(path)?.len());

    // ── Decoder builder (recommended for loading) ────────────────────────────
    // Decoder lets you set sample rate, mono-conversion, and time windows.
    let loaded = io::Decoder::new(path)
        .sample_rate(22050)   // resample on load
        .mono()               // mix to mono
        .load()?;
    println!(
        "Loaded:  {} samples at {} Hz ({} channel)",
        loaded.samples.len(), loaded.sample_rate, loaded.channels
    );

    // ── Duration helpers ─────────────────────────────────────────────────────
    let duration = util::get_duration(&loaded);
    println!("Duration: {duration:.3}s");

    // ── Low-level io::load (all options explicit) ────────────────────────────
    // load(path, target_sr, mono, offset_secs, duration_secs)
    let segment = io::load(path, Some(44100), Some(true), Some(0.0), Some(0.5))?;
    println!("Segment: {} samples (first 0.5s)", segment.samples.len());

    // ── Block streaming for large files ──────────────────────────────────────
    // stream(path, block_length, frame_length, hop_length) → Vec<Vec<f32>>
    let blocks = io::stream(path, 1024, 2048, Some(512))?;
    let total_samples: usize = blocks.iter().map(|b| b.len()).sum();
    println!("Streamed {} block(s), {} total samples", blocks.len(), total_samples);

    std::fs::remove_file(path)?;
    Ok(())
}
