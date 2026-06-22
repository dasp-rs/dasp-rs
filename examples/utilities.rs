//! Time, frequency, and notation utility functions.
//!
//! Run with: `cargo run --example utilities`

use dasp_rs::util;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let hop = 512_usize;

    // ── Frame / time / sample conversions ────────────────────────────────────
    let frames = vec![0_usize, 10, 20, 40, 80];

    let times = util::frames_to_time(&frames)
        .sample_rate(sr)
        .hop_length(hop)
        .compute();
    println!("frames → time:    {:?}", times);

    let samples = util::frames_to_samples(&frames, Some(hop));
    println!("frames → samples: {:?}", samples);

    let back_frames = util::time_to_frames(&times)
        .sample_rate(sr)
        .hop_length(hop)
        .compute();
    println!("time → frames:    {:?}", back_frames);

    let back_samples = util::time_to_samples(&times, Some(sr));
    println!("time → samples:   {:?}", back_samples);

    let back_from_samples = util::samples_to_time(&samples, Some(sr));
    println!("samples → time:   {:?}", back_from_samples);

    // ── Frequency unit conversions ────────────────────────────────────────────
    let hz: Vec<f32> = vec![261.63, 440.0, 880.0]; // C4, A4, A5

    let mels = util::hz_to_mel(&hz, None);       // None → HTK scale
    let hz_back = util::mel_to_hz(&mels, None);
    println!("\nhz → mel:  {:?}", mels);
    println!("mel → hz:  {:?}", hz_back);

    let midis = util::hz_to_midi(&hz);
    let hz_from_midi = util::midi_to_hz(&midis);
    println!("hz → midi: {:?}", midis);
    println!("midi → hz: {:?}", hz_from_midi);

    let octs = util::hz_to_octs(&hz, None);
    let hz_from_octs = util::octs_to_hz(&octs, None);
    println!("hz → octs: {:?}", octs);
    println!("octs → hz: {:?}", hz_from_octs);

    // ── Note / MIDI name conversions ──────────────────────────────────────────
    let notes = util::hz_to_note(&hz);
    println!("\nhz → note:   {:?}", notes);

    let notes_str: Vec<&str> = vec!["C4", "A4", "A5"];
    let hz_from_notes = util::note_to_hz(&notes_str);
    println!("note → hz:   {:?}", hz_from_notes);

    let midi_f: Vec<f32> = vec![60.0, 69.0, 81.0];
    let note_names = util::midi_to_note(&midi_f, Some(true));
    println!("midi → note: {:?}", note_names);

    let note_midis = util::note_to_midi(&notes_str, None);
    println!("note → midi: {:?}", note_midis);

    // ── Frequency bin arrays ──────────────────────────────────────────────────
    // fft_frequencies: linearly spaced 0 → Nyquist
    let fft_freqs = util::fft_frequencies().sample_rate(sr).n_fft(2048).compute();
    println!("\nFFT freqs:  {} bins, first={:.1} Hz, last={:.1} Hz",
        fft_freqs.len(), fft_freqs[0], fft_freqs.last().copied().unwrap_or(0.0));

    // mel_frequencies: Mel-scaled filter bank centre frequencies
    let mel_freqs = util::mel_frequencies()
        .n_mels(128)
        .fmin(0.0)
        .fmax(sr as f32 / 2.0)
        .compute();
    println!("Mel freqs:  {} bins, first={:.1} Hz, last={:.1} Hz",
        mel_freqs.len(), mel_freqs[0], mel_freqs.last().copied().unwrap_or(0.0));

    // cqt_frequencies: logarithmically spaced starting from fmin (C1 by default)
    let cqt_freqs = util::cqt_frequencies(84, None); // 7 octaves × 12 bins
    println!("CQT freqs:  {} bins, first={:.2} Hz, last={:.1} Hz",
        cqt_freqs.len(), cqt_freqs[0], cqt_freqs.last().copied().unwrap_or(0.0));

    // tempo_frequencies: BPM axis for tempogram analysis
    let tempo_freqs = util::tempo_frequencies(384).sample_rate(sr).hop_length(hop).compute();
    println!("Tempo freqs: {} bins, BPM range [{:.1}, {:.1}]",
        tempo_freqs.len(), tempo_freqs.iter().copied().fold(f32::INFINITY, f32::min),
        tempo_freqs.iter().copied().fold(f32::NEG_INFINITY, f32::max));

    // ── Tuning helpers ────────────────────────────────────────────────────────
    // a4_to_tuning: semitone offset from A4=440 Hz for a given A4 frequency
    let tuning = util::a4_to_tuning(442.0);
    println!("\nA4=442 Hz → tuning offset: {tuning:.4} semitones");

    // tuning_to_a4: recover A4 frequency from a semitone offset
    let a4 = util::tuning_to_a4(tuning);
    println!("tuning {tuning:.4} → A4={a4:.2} Hz");

    Ok(())
}
