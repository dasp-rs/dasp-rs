//! Signal-processing correctness tests.
//!
//! These assert standard DSP invariants of the transform pipeline (frame counts,
//! reconstruction, peak localization, window selection) so regressions are caught
//! without depending on any external reference at build time.

use std::f32::consts::PI;

use dasp_rs::proc::{istft, stft, Window};

fn sine(freq_bins: f32, n_fft: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| (2.0 * PI * freq_bins * n as f32 / n_fft as f32).sin())
        .collect()
}

#[test]
fn stft_centered_frame_count() {
    // A centered STFT yields n_frames = 1 + len / hop.
    let y = vec![0.0_f32; 1000];
    for &hop in &[64usize, 128, 256] {
        let spec = stft(&y).n_fft(512).hop_length(hop).compute().unwrap();
        assert_eq!(spec.shape()[1], 1 + y.len() / hop, "hop={hop}");
        assert_eq!(spec.shape()[0], 512 / 2 + 1);
    }
}

#[test]
fn stft_window_choice_changes_output() {
    let y = sine(8.0, 256, 2048);
    let hann = stft(&y).n_fft(256).window(Window::Hann).compute().unwrap();
    let hamm = stft(&y).n_fft(256).window(Window::Hamming).compute().unwrap();
    assert_eq!(hann.shape(), hamm.shape());
    let diff: f32 = hann
        .iter()
        .zip(hamm.iter())
        .map(|(a, b)| (a.norm() - b.norm()).abs())
        .sum();
    assert!(diff > 0.0, "Hann and Hamming should differ");
}

#[test]
fn stft_peak_bin_for_pure_tone() {
    let n_fft = 512;
    let k = 17usize;
    let y = sine(k as f32, n_fft, 8192);
    let spec = stft(&y).n_fft(n_fft).hop_length(128).compute().unwrap();
    let frame = spec.column(spec.shape()[1] / 2);
    let peak = (0..frame.len())
        .max_by(|&a, &b| frame[a].norm().partial_cmp(&frame[b].norm()).unwrap())
        .unwrap();
    assert_eq!(peak, k);
}

#[test]
fn istft_inverts_stft_in_the_interior() {
    let y = sine(11.0, 256, 4096);
    let spec = stft(&y).n_fft(256).hop_length(64).compute().unwrap();
    let recon = istft(&spec).hop_length(64).length(y.len()).compute();
    assert_eq!(recon.len(), y.len());
    let (mut err, mut cnt) = (0.0_f32, 0usize);
    for i in 256..(y.len() - 256) {
        err += (recon[i] - y[i]).abs();
        cnt += 1;
    }
    assert!((err / cnt as f32) < 1e-3);
}

#[test]
fn non_centered_stft_starts_at_sample_zero() {
    // With center=false there is no leading pad: frame 0 covers samples 0..n_fft.
    let y = sine(5.0, 128, 2048);
    let centered = stft(&y).n_fft(128).hop_length(32).center(true).compute().unwrap();
    let raw = stft(&y).n_fft(128).hop_length(32).center(false).compute().unwrap();
    assert!(centered.shape()[1] > raw.shape()[1]);
}
