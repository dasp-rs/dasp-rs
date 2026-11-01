//! Regression coverage for the correctness audit: timing, channel isolation,
//! transform dimensions, and builder parameter propagation.
use dasp_rs::{
    feat::{griffinlim, mfcc, spectral},
    io::{Decoder, export},
    proc::{Window, istft, resample, stft},
    types::AudioData,
};
use ndarray::Array2;
use num_complex::Complex;
use tempfile::NamedTempFile;

fn tone(len: usize, sr: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / sr as f32).sin() * 0.5)
        .collect()
}

#[test]
fn stereo_crop_preserves_frame_timing() {
    let audio = AudioData::new((0..16).map(|i| i as f32 / 20.0).collect(), 4, 2).unwrap();
    let file = NamedTempFile::new().unwrap();
    export(file.path(), &audio).unwrap();
    let crop = Decoder::new(file.path())
        .offset(0.5)
        .duration(0.5)
        .load()
        .unwrap();
    assert_eq!(crop.samples, [0.2, 0.25, 0.3, 0.35]);
    assert_eq!(crop.duration(), 0.5);
    let mono = Decoder::new(file.path())
        .offset(0.5)
        .duration(0.5)
        .mono()
        .load()
        .unwrap();
    assert_eq!(mono.samples.len(), 2);
    assert!((mono.samples[0] - 0.225).abs() < 1e-6);
}

#[test]
fn resampling_preserves_duration_for_short_and_long_clips() {
    for (input_sr, output_sr) in [(44_100, 22_050), (22_050, 44_100), (44_100, 48_000)] {
        for len in [1, 4, 512, 1025, 44_100] {
            let output = resample(&tone(len, input_sr), input_sr, output_sr).unwrap();
            let expected =
                (len as f64 * f64::from(output_sr) / f64::from(input_sr)).ceil() as usize;
            assert_eq!(output.len(), expected, "len={len}, {input_sr}->{output_sr}");
            assert!(output.iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn griffinlim_accepts_a_valid_stft_magnitude() {
    let signal = tone(1024, 8000);
    let spec = stft(&signal).n_fft(256).hop_length(64).compute().unwrap();
    let magnitude = spec.mapv(Complex::norm);
    let output = griffinlim(&magnitude)
        .n_iter(8)
        .hop_length(64)
        .compute()
        .unwrap();
    assert_eq!(output.len(), signal.len());
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn istft_rejects_invalid_shapes_without_panicking() {
    for shape in [(0, 4), (1, 4), (129, 0)] {
        let spec = Array2::<Complex<f32>>::zeros(shape);
        let result = std::panic::catch_unwind(|| istft(&spec).compute());
        assert!(
            result.expect("ISTFT must not panic").is_err(),
            "accepted {shape:?}"
        );
    }
}

#[test]
fn mfcc_builder_uses_requested_analysis_parameters() {
    let y = tone(4096, 8000);
    let audio = AudioData::new(y.clone(), 8000, 1).unwrap();
    for (n_fft, hop) in [(256, 64), (512, 128)] {
        let mel = spectral(&y, 8000)
            .n_fft(n_fft)
            .hop_length(hop)
            .n_mels(40)
            .fmax(3000.0)
            .melspectrogram()
            .unwrap();
        let expected = mfcc(&audio, Some(&mel), None, None, None).unwrap();
        let actual = spectral(&y, 8000)
            .n_fft(n_fft)
            .hop_length(hop)
            .n_mels(40)
            .fmax(3000.0)
            .mfcc()
            .unwrap();
        assert_eq!(actual.dim(), expected.dim());
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }
    }
}

#[test]
fn decoder_resampling_keeps_channels_independent() {
    let left = tone(4096, 44_100);
    let samples = left.iter().flat_map(|&v| [v, 0.0]).collect();
    let audio = AudioData::new(samples, 44_100, 2).unwrap();
    let file = NamedTempFile::new().unwrap();
    export(file.path(), &audio).unwrap();
    let output = Decoder::new(file.path())
        .sample_rate(22_050)
        .load()
        .unwrap();
    assert_eq!(output.channels, 2);
    assert!(
        output
            .samples
            .iter()
            .skip(1)
            .step_by(2)
            .all(|v| v.abs() < 1e-7),
        "silent channel contains crosstalk"
    );
    let expected = resample(&left, 44_100, 22_050).unwrap();
    assert_eq!(output.samples.len(), expected.len() * 2);
    for (actual, expected) in output.samples.iter().step_by(2).zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn resampling_preserves_impulse_timing_and_flushes_the_tail() {
    for position in [0, 1024, 2046] {
        let mut signal = vec![0.0; 2048];
        signal[position] = 1.0;
        let output = resample(&signal, 48_000, 24_000).unwrap();
        let (peak, value) = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .unwrap();
        assert_eq!(peak, position / 2);
        assert!(value.abs() > 0.4, "impulse at {position} lost: {value}");
    }
}

#[test]
fn resampling_preserves_passband_and_rejects_aliasing() {
    for frequency in [1000.0, 18_000.0] {
        let signal: Vec<f32> = (0..4800)
            .map(|i| (std::f32::consts::TAU * frequency * i as f32 / 48_000.0).sin())
            .collect();
        let output = resample(&signal, 48_000, 24_000).unwrap();
        let error = (128..output.len() - 128)
            .map(|i| {
                let expected = if frequency < 12_000.0 {
                    (std::f32::consts::TAU * frequency * i as f32 / 24_000.0).sin()
                } else {
                    0.0
                };
                (output[i] - expected).powi(2)
            })
            .sum::<f32>()
            / (output.len() - 256) as f32;
        assert!(
            error.sqrt() < 0.01,
            "frequency={frequency}, RMSE={}",
            error.sqrt()
        );
    }
}

#[test]
fn resampling_rejects_zero_sample_rates() {
    for signal in [&[][..], &[1.0][..]] {
        for rates in [(0, 0), (0, 44_100), (44_100, 0)] {
            assert!(resample(signal, rates.0, rates.1).is_err());
        }
    }
}

#[test]
fn istft_validates_parameters_and_coefficients() {
    let mut spec = Array2::<Complex<f32>>::zeros((129, 4));
    assert!(istft(&spec).hop_length(0).compute().is_err());
    assert!(istft(&spec).win_length(0).compute().is_err());
    assert!(istft(&spec).win_length(257).compute().is_err());
    assert!(istft(&spec).n_fft(512).compute().is_err());
    assert!(istft(&spec).hop_length(usize::MAX).compute().is_err());
    assert!(istft(&spec).length(usize::MAX).compute().is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        spec[[0, 0]] = Complex::new(value, 0.0);
        assert!(istft(&spec).compute().is_err());
    }
}

#[test]
fn istft_round_trips_even_and_explicit_odd_ffts_including_boundaries() {
    let signal = tone(1003, 8000);
    for n_fft in [255, 256] {
        for window in [Window::Hann, Window::Hamming] {
            let spec = stft(&signal)
                .n_fft(n_fft)
                .hop_length(64)
                .window(window)
                .compute()
                .unwrap();
            let output = istft(&spec)
                .n_fft(n_fft)
                .hop_length(64)
                .window(window)
                .length(signal.len())
                .compute()
                .unwrap();
            let max_error = signal
                .iter()
                .zip(output)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max_error < 1e-5, "n_fft={n_fft}, max error={max_error}");
        }
    }
    // With an odd FFT, centering removes n_fft - 1 samples, not n_fft.
    let signal = tone(1025, 8000);
    let spec = stft(&signal).n_fft(255).hop_length(64).compute().unwrap();
    assert_eq!(
        istft(&spec)
            .n_fft(255)
            .hop_length(64)
            .compute()
            .unwrap()
            .len(),
        signal.len()
    );
}

#[test]
fn griffinlim_reduces_spectral_error() {
    let signal = tone(2048, 8000);
    let magnitude = stft(&signal)
        .n_fft(256)
        .hop_length(64)
        .compute()
        .unwrap()
        .mapv(Complex::norm);
    let error = |iterations| {
        let reconstructed = griffinlim(&magnitude)
            .hop_length(64)
            .n_iter(iterations)
            .compute()
            .unwrap();
        let spec = stft(&reconstructed)
            .n_fft(256)
            .hop_length(64)
            .compute()
            .unwrap();
        assert_eq!(spec.dim(), magnitude.dim());
        let squared_error: f32 = spec
            .iter()
            .zip(&magnitude)
            .map(|(a, b)| (a.norm() - b).powi(2))
            .sum();
        squared_error / magnitude.iter().map(|v| v * v).sum::<f32>()
    };
    let initial = error(1);
    let final_error = error(32);
    assert!(
        final_error < initial,
        "initial={initial}, final={final_error}"
    );
    assert!(final_error < 0.1, "relative squared error={final_error}");
}

#[test]
fn griffinlim_handles_silence_single_frames_and_invalid_magnitudes() {
    for frames in [1, 2, 17] {
        let magnitude = Array2::<f32>::zeros((129, frames));
        let output = griffinlim(&magnitude)
            .hop_length(64)
            .n_iter(3)
            .compute()
            .unwrap();
        assert_eq!(output, vec![0.0; 64 * (frames - 1)]);
    }
    for shape in [(0, 4), (1, 4), (129, 0)] {
        assert!(griffinlim(&Array2::zeros(shape)).compute().is_err());
    }
    for value in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(
            griffinlim(&Array2::from_elem((129, 4), value))
                .compute()
                .is_err()
        );
    }
    let magnitude = Array2::zeros((129, 4));
    assert!(griffinlim(&magnitude).hop_length(0).compute().is_err());
    assert!(
        griffinlim(&magnitude)
            .hop_length(usize::MAX)
            .compute()
            .is_err()
    );
}

#[test]
fn decoder_validates_ranges_and_crops_multichannel_frames() {
    let signal = AudioData::new((0..24).map(|i| i as f32 / 32.0).collect(), 4, 3).unwrap();
    let file = NamedTempFile::new().unwrap();
    export(file.path(), &signal).unwrap();
    let crop = Decoder::new(file.path())
        .offset(0.25)
        .duration(0.5)
        .load()
        .unwrap();
    assert_eq!(crop.samples, signal.samples[3..9]);
    assert_eq!(crop.duration(), 0.5);
    for value in [f32::NAN, f32::INFINITY, -1.0] {
        assert!(Decoder::new(file.path()).offset(value).load().is_err());
        assert!(Decoder::new(file.path()).duration(value).load().is_err());
    }
    assert!(Decoder::new(file.path()).offset(99.0).load().is_err());
    assert!(Decoder::new(file.path()).duration(0.01).load().is_err());
    let tail = Decoder::new(file.path())
        .offset(1.5)
        .duration(99.0)
        .load()
        .unwrap();
    assert_eq!(tail.samples, signal.samples[18..]);
}

#[test]
fn mfcc_and_mel_builders_agree_with_nonzero_lower_frequency_bound() {
    let y = tone(4096, 8000);
    let audio = AudioData::new(y.clone(), 8000, 1).unwrap();
    let config = spectral(&y, 8000)
        .n_fft(512)
        .hop_length(128)
        .n_mels(40)
        .fmin(300.0)
        .fmax(3000.0);
    let mel = config.clone().melspectrogram().unwrap();
    let reference_mel = dasp_rs::feat::melspectrogram(
        &audio,
        None,
        Some(512),
        Some(128),
        Some(40),
        Some(300.0),
        Some(3000.0),
    )
    .unwrap();
    assert_eq!(mel, reference_mel);
    let expected = mfcc(&audio, Some(&mel), None, None, None).unwrap();
    let actual = config.mfcc().unwrap();
    for (a, b) in actual.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5, "mel/MFCC settings differ: {a} vs {b}");
    }
}

#[test]
fn resampling_preserves_tone_timing_for_rational_rate_changes() {
    for (input_sr, output_sr) in [
        (44_100, 48_000),
        (48_000, 44_100),
        (8000, 48_000),
        (48_000, 8000),
    ] {
        let len = input_sr as usize / 4;
        let signal = tone(len, input_sr);
        let output = resample(&signal, input_sr, output_sr).unwrap();
        let reference = tone(output.len(), output_sr);
        let margin = 512;
        let error = output[margin..output.len() - margin]
            .iter()
            .zip(&reference[margin..reference.len() - margin])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / (output.len() - 2 * margin) as f32;
        assert!(
            error.sqrt() < 1e-3,
            "{input_sr}->{output_sr}, RMSE={}",
            error.sqrt()
        );
    }
}
