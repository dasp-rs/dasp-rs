//! Offline integration workflows on three attributed four-second recordings.
//! Licenses, source/clip hashes, conversion details, and limitations are retained
//! in `fixtures/audio/manifest.json` and `fixtures/audio/ATTRIBUTION.md`.
//! Independent feature summaries use librosa/SciPy, not Rust-generated goldens.
#[path = "common/reference.rs"]
mod reference;
use dasp_rs::{feat, io, pitch, proc, types::AudioData, util};
use ndarray::{Array2, Axis};
use reference::{check_real, load};
use std::{io::Write, path::PathBuf};

const CLIPS: [&str; 3] = ["speech", "trumpet", "pitched_percussion"];
const FFT: usize = 1024;
const HOP: usize = 256;
const SELECTED_FRAMES: [usize; 5] = [0, 31, 89, 160, 250];
const SELECTED_SAMPLES: [usize; 7] = [0, 1, 17, 511, 5000, 31999, 63999];
fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/audio")
        .join(format!("{name}.flac"))
}
fn decode(name: &str) -> AudioData {
    let audio = io::Decoder::new(path(name)).load().unwrap();
    assert_eq!(audio.sample_rate, 16000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), 64000);
    assert!(audio.samples.iter().all(|v| v.is_finite()));
    audio
}
fn row(values: Vec<f32>) -> Array2<f32> {
    Array2::from_shape_vec((1, values.len()), values).unwrap()
}
fn means(matrix: &Array2<f32>) -> Array2<f32> {
    row(matrix
        .axis_iter(Axis(0))
        .map(|r| (r.iter().copied().map(f64::from).sum::<f64>() / r.len() as f64) as f32)
        .collect())
}

#[test]
fn real_recordings_decode_reconstruct_and_match_independent_feature_summaries() {
    let references = load(include_str!("fixtures/audio/corpus_reference.csv"));
    for name in CLIPS {
        let audio = decode(name);
        let y = &audio.samples;
        check_real(
            &format!("{name}_decoded_samples"),
            &row(SELECTED_SAMPLES.iter().map(|&i| y[i]).collect()),
            &references,
            0.0,
            0.0,
        );
        let summary = vec![
            y.iter().copied().fold(f32::INFINITY, f32::min),
            y.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            (y.iter().copied().map(f64::from).sum::<f64>() / y.len() as f64) as f32,
            (y.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / y.len() as f64).sqrt() as f32,
            (y.iter().map(|&x| f64::from(x).abs()).sum::<f64>() / y.len() as f64) as f32,
        ];
        check_real(
            &format!("{name}_waveform_summary"),
            &row(summary),
            &references,
            2e-8,
            3e-7,
        );
        let spectrum = proc::stft(y).n_fft(FFT).hop_length(HOP).compute().unwrap();
        assert_eq!(spectrum.dim(), (FFT / 2 + 1, 251));
        let reconstructed = proc::istft(&spectrum)
            .n_fft(FFT)
            .hop_length(HOP)
            .length(y.len())
            .compute()
            .unwrap();
        let max_error = y
            .iter()
            .zip(&reconstructed)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let rmse = (y
            .iter()
            .zip(&reconstructed)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();
        eprintln!(
            "REAL_AUDIO {name}: reconstruction_max={max_error:.6e} reconstruction_rmse={rmse:.6e}"
        );
        assert!(max_error < 2e-6 && rmse < 2e-7, "{name} reconstruction");
        let power = spectrum.mapv(|z| z.norm_sqr());
        let mel = feat::melspectrogram(
            &audio,
            Some(&power),
            Some(FFT),
            Some(HOP),
            Some(40),
            Some(0.0),
            Some(8000.0),
        )
        .unwrap();
        let mfcc = feat::mfcc(&audio, Some(&mel), Some(13), Some(2), Some("ortho")).unwrap();
        assert!(mel.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(mfcc.iter().all(|v| v.is_finite()));
        // Slaney area-normalized mel powers; natural-log mel DCT-II MFCCs.
        // Tolerances cover f32 FFT/filter/DCT accumulation versus f64 references.
        // Selected MFCCs allow 5e-4 natural-log coefficient units: log power
        // amplifies FFT roundoff in quiet spectral bands; temporal means use
        // the tighter 1e-4 budget below.
        check_real(
            &format!("{name}_mel_selected"),
            &mel.select(Axis(1), &SELECTED_FRAMES),
            &references,
            2e-6,
            1e-4,
        );
        check_real(
            &format!("{name}_mfcc_selected"),
            &mfcc.select(Axis(1), &SELECTED_FRAMES),
            &references,
            5e-4,
            3e-5,
        );
        check_real(
            &format!("{name}_mel_mean"),
            &means(&mel),
            &references,
            2e-6,
            1e-4,
        );
        check_real(
            &format!("{name}_mfcc_mean"),
            &means(&mfcc),
            &references,
            1e-4,
            3e-5,
        );
        check_hpss(name, &audio, &power);
    }
}

fn check_hpss(name: &str, audio: &AudioData, power: &Array2<f32>) {
    let (harmonic, percussive) =
        feat::hpss(audio, Some(power), Some(FFT), Some(HOP), Some(31), Some(31)).unwrap();
    for ((&h, &p), &original) in harmonic.iter().zip(&percussive).zip(power) {
        assert!(h.is_finite() && p.is_finite() && h >= 0.0 && p >= 0.0);
        assert!(
            (f64::from(h) + f64::from(p) - f64::from(original)).abs()
                <= 1e-6 * f64::from(original) + 1e-12,
            "{name}: HPSS power lost"
        );
    }
}

#[test]
fn recordings_resample_with_correct_duration_and_frame_timing() {
    let references = load(include_str!("fixtures/audio/corpus_reference.csv"));
    for name in CLIPS {
        let audio = decode(name);
        let resampled = io::Decoder::new(path(name))
            .sample_rate(12000)
            .mono()
            .load()
            .unwrap();
        assert_eq!(resampled.samples.len(), 48000);
        assert_eq!(resampled.sample_rate, 12000);
        assert_eq!(resampled.channels, 1);
        assert!(resampled.samples.iter().all(|v| v.is_finite()));
        let direct = proc::resample(&audio.samples, 16000, 12000).unwrap();
        assert_eq!(
            direct, resampled.samples,
            "decoder and direct resampling disagree"
        );
        let interior = &resampled.samples[256..resampled.samples.len() - 256];
        let rms = (interior.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
            / interior.len() as f64)
            .sqrt() as f32;
        // SciPy Kaiser and rubato filters differ. Compare interior signal RMS
        // within 1%, not sample parity or an unsupported identical-FIR claim.
        check_real(
            &format!("{name}_resampled_rms"),
            &row(vec![rms]),
            &references,
            1e-5,
            0.01,
        );
        let spectrum = proc::stft(&resampled.samples)
            .n_fft(FFT)
            .hop_length(HOP)
            .compute()
            .unwrap();
        assert_eq!(spectrum.ncols(), 1 + resampled.samples.len() / HOP);
        let frames = [0, 1, spectrum.ncols() - 1];
        let times = util::frames_to_time(&frames)
            .sample_rate(12000)
            .hop_length(HOP)
            .compute();
        for (&frame, &time) in frames.iter().zip(&times) {
            assert!((f64::from(time) - frame as f64 * HOP as f64 / 12000.0).abs() < 5e-7);
            assert!((0.0..=4.0).contains(&time));
        }
    }
}

#[test]
fn recordings_support_pitch_voicing_and_bounded_rhythm_workflows() {
    for name in CLIPS {
        let audio = decode(name);
        let y = &audio.samples;
        let yin = pitch::yin(y, 65.0, 1200.0)
            .sample_rate(audio.sample_rate)
            .frame_length(FFT)
            .hop_length(HOP)
            .compute()
            .unwrap();
        let pyin = pitch::pyin(y, 65.0, 1200.0)
            .sample_rate(audio.sample_rate)
            .frame_length(FFT)
            .hop_length(HOP)
            .compute_with_voicing()
            .unwrap();
        let frames = 1 + (y.len() - FFT) / HOP;
        assert_eq!(yin.len(), frames);
        assert_eq!(pyin.f0.len(), frames);
        assert_eq!(pyin.voiced_flag.len(), frames);
        assert_eq!(pyin.voiced_prob.len(), frames);
        for &pitch in yin.iter().chain(&pyin.f0) {
            assert!(pitch.is_finite() && (pitch == 0.0 || (65.0..=1200.0).contains(&pitch)));
        }
        for ((&f0, &voiced), &probability) in
            pyin.f0.iter().zip(&pyin.voiced_flag).zip(&pyin.voiced_prob)
        {
            assert!(probability.is_finite() && (0.0..=1.0).contains(&probability));
            assert_eq!(voiced, f0 > 0.0);
        }
        let onsets = feat::onset_detect(y, audio.sample_rate)
            .n_fft(FFT)
            .hop_length(HOP)
            .compute()
            .unwrap();
        assert!(onsets.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(onsets.iter().all(|&frame| frame < 1 + y.len() / HOP));
        let tempo = feat::tempo(y, audio.sample_rate)
            .win_length(FFT)
            .hop_length(HOP)
            .compute()
            .unwrap();
        assert!(tempo.is_finite() && (tempo == 0.0 || (29.0..=301.0).contains(&tempo)));
        eprintln!(
            "REAL_AUDIO {name}: voiced_frames={} total_frames={frames} onset_count={} tempo={tempo:.3}; no annotation-based accuracy claim",
            pyin.voiced_flag.iter().filter(|&&v| v).count(),
            onsets.len()
        );
    }
}

#[test]
fn truncated_real_flac_header_returns_an_error() {
    let bytes = std::fs::read(path("speech")).unwrap();
    let mut file = tempfile::Builder::new().suffix(".flac").tempfile().unwrap();
    file.write_all(&bytes[..12]).unwrap();
    file.flush().unwrap();
    assert!(io::Decoder::new(file.path()).load().is_err());
}
