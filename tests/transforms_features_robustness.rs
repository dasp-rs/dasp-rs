//! Adversarial numerical invariants for transforms and spectral features.
use dasp_rs::{
    feat::{
        compute_delta, f0_harmonics, interp_harmonics, phase_vocoder, salience, spectral_bandwidth,
        spectral_centroid,
    },
    proc::{magphase, reassigned_spectrogram, stft, vqt},
    types::AudioData,
};
use ndarray::{Array2, array};
use num_complex::Complex;

fn assert_close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
}

#[test]
fn delta_differentiates_along_the_selected_axis() {
    let input = Array2::from_shape_fn((5, 7), |(r, t)| 10.0 * r as f32 + 2.0 * t as f32);
    let time = compute_delta(&input).width(3).compute().unwrap();
    for r in 0..5 {
        for t in 1..6 {
            assert_close(time[[r, t]], 2.0);
        }
    }
    let coefficients = compute_delta(&input).width(3).axis(0).compute().unwrap();
    for r in 1..4 {
        for t in 0..7 {
            assert_close(coefficients[[r, t]], 10.0);
        }
    }
}

#[test]
fn delta_rejects_unsupported_axis() {
    assert!(
        compute_delta(&Array2::ones((5, 7)))
            .width(3)
            .axis(2)
            .compute()
            .is_err()
    );
}

#[test]
fn harmonic_interpolation_accepts_peaked_spectra_and_falling_pitch() {
    let spectrum = [0.0, 2.0, 1.0, 0.0];
    let frequencies = [0.0, 100.0, 200.0, 300.0];
    let interpolated = interp_harmonics(&spectrum, &frequencies, &[1.0]).unwrap();
    assert_eq!(interpolated.row(0).to_vec(), spectrum);
    let harmonic = f0_harmonics(&spectrum, &[200.0, 100.0], &frequencies, &[1.0]).unwrap();
    assert_eq!(harmonic.row(0).to_vec(), [1.0, 2.0]);
}

#[test]
fn harmonic_interpolation_handles_singletons_and_outside_the_grid() {
    assert_eq!(
        interp_harmonics(&[3.0], &[100.0], &[1.0, 2.0]).unwrap(),
        array![[3.0], [0.0]]
    );
    let harmonic =
        f0_harmonics(&[1.0, 3.0], &[50.0, 150.0, 250.0], &[100.0, 200.0], &[1.0]).unwrap();
    assert_eq!(harmonic.row(0).to_vec(), [0.0, 2.0, 0.0]);
}

#[test]
fn salience_default_weights_cover_all_harmonics() {
    let harmonics = vec![1.0; 20];
    let actual = salience(&array![[1.0], [2.0]], &[0.0, 100.0], &harmonics, None).unwrap();
    assert_eq!(actual, array![[20.0], [40.0]]);
}

#[test]
fn harmonic_interpolation_rejects_nonfinite_multipliers() {
    assert!(interp_harmonics(&[1.0, 2.0], &[0.0, 100.0], &[f32::NAN]).is_err());
}

#[test]
fn phase_vocoder_unit_rate_preserves_complex_spectra_including_final_frame() {
    for (n_fft, frames) in [(8, 6), (1024, 128)] {
        let input = Array2::from_shape_fn((n_fft / 2 + 1, frames), |(f, t)| {
            Complex::from_polar(1.0 + t as f32, 0.7 + 0.3 * f as f32 + 0.8 * t as f32)
        });
        let actual = phase_vocoder(&input, 1.0, Some(n_fft / 4), Some(n_fft)).unwrap();
        assert_eq!(actual.dim(), input.dim());
        for (a, b) in actual.iter().zip(input.iter()) {
            assert!((*a - *b).norm() < 2e-4, "{a} != {b}");
        }
    }
}

#[test]
fn phase_vocoder_interpolates_magnitude_and_preserves_tone_frequency() {
    let omega = std::f32::consts::TAU / 8.0;
    let input = Array2::from_shape_fn((5, 6), |(f, t)| {
        if f == 1 {
            Complex::from_polar(1.0 + t as f32, 0.4 + omega * 2.0 * t as f32)
        } else {
            Complex::new(0.0, 0.0)
        }
    });
    let actual = phase_vocoder(&input, 0.5, Some(2), Some(8)).unwrap();
    assert_eq!(actual.ncols(), 12);
    for t in 0..10 {
        let expected = Complex::from_polar(1.0 + 0.5 * t as f32, 0.4 + omega * 2.0 * t as f32);
        assert!((actual[[1, t]] - expected).norm() < 2e-4);
    }
}

#[test]
fn phase_vocoder_rejects_nonfinite_and_overflowing_rates() {
    let input = Array2::from_elem((5, 2), Complex::new(1.0, 0.0));
    for rate in [f32::NAN, f32::INFINITY, f32::MIN_POSITIVE] {
        assert!(phase_vocoder(&input, rate, Some(2), Some(8)).is_err());
    }
}

#[test]
fn magphase_has_a_unit_phase_for_silence_and_reconstructs_inputs() {
    let input = array![[Complex::new(0.0, 0.0), Complex::new(3.0, 4.0)]];
    let (mag, phase) = magphase(&input, None);
    for ((&m, &p), &original) in mag.iter().zip(phase.iter()).zip(input.iter()) {
        assert_close(p.norm(), 1.0);
        assert!((m * p - original).norm() < 1e-6);
    }
}

#[test]
fn bandwidth_uses_the_given_spectrum_and_frame_configuration() {
    let signal = AudioData::new(vec![0.0; 4096], 8000, 1).unwrap();
    let mut spectrum = Array2::zeros((65, 3));
    for t in 0..3 {
        spectrum[[8, t]] = 1.0;
        spectrum[[24, t]] = 1.0;
    }
    let actual = spectral_bandwidth(&signal, Some(&spectrum), Some(128), Some(32), None).unwrap();
    assert_eq!(actual.len(), 3);
    for &value in &actual {
        assert_close(value, 500.0);
    }
    let actual = spectral_bandwidth(&signal, None, Some(128), Some(32), None).unwrap();
    assert_eq!(actual.len(), 129);
}

#[test]
fn centroid_rejects_incompatible_supplied_spectrum() {
    let signal = AudioData::new(vec![0.0; 128], 8000, 1).unwrap();
    assert!(spectral_centroid(&signal, Some(&Array2::ones((4, 3))), Some(128), Some(32)).is_err());
}

#[test]
fn vqt_accepts_ordinary_audio_without_kernel_index_panics() {
    let signal: Vec<_> = (0..1024).map(|i| (i as f32 * 0.2).sin()).collect();
    let result = vqt(&signal, 8000)
        .hop_length(64)
        .fmin(100.0)
        .n_bins(12)
        .compute()
        .unwrap();
    assert_eq!(result.dim(), (12, 17));
    assert!(result.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
    assert!(result.iter().any(|c| c.norm() > 0.0));
}

#[test]
fn reassignment_uses_consistent_centered_frames_and_preserves_magnitude_mass() {
    let signal: Vec<_> = (0..256).map(|i| (i as f32 * 0.2).sin()).collect();
    let expected = stft(&signal).n_fft(64).compute().unwrap();
    let actual = reassigned_spectrogram(&signal, 8000)
        .n_fft(64)
        .compute()
        .unwrap();
    assert_eq!(actual.dim(), expected.dim());
    let magnitude_sum: f32 = expected.iter().map(|c| c.norm()).sum();
    assert!((actual.sum() - magnitude_sum).abs() < 0.002);
}

#[test]
fn stft_rejects_invalid_parameters_without_silently_clamping() {
    let y = [1.0; 16];
    assert!(stft(&y).n_fft(8).hop_length(0).compute().is_err());
    assert!(stft(&y).n_fft(8).win_length(0).compute().is_err());
    assert!(stft(&y).n_fft(8).win_length(9).compute().is_err());
    assert!(stft(&[]).n_fft(8).compute().is_err());
    assert!(stft(&[f32::NAN]).n_fft(8).compute().is_err());
    assert!(stft(&y).n_fft(usize::MAX).compute().is_err());
}

#[test]
fn constant_q_transforms_reject_invalid_frequency_and_geometry() {
    use dasp_rs::proc::{cqt, hybrid_cqt, icqt, pseudo_cqt};
    let y = [1.0; 16];
    let spectrum = Array2::from_elem((2, 3), Complex::new(0.0, 0.0));
    for fmin in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MIN_POSITIVE] {
        assert!(cqt(&y, 8000).hop_length(4).fmin(fmin).compute().is_err());
        assert!(
            hybrid_cqt(&y, 8000)
                .hop_length(4)
                .fmin(fmin)
                .compute()
                .is_err()
        );
        assert!(
            pseudo_cqt(&y, 8000)
                .hop_length(4)
                .fmin(fmin)
                .compute()
                .is_err()
        );
        assert!(
            icqt(&spectrum)
                .sample_rate(8000)
                .hop_length(4)
                .fmin(fmin)
                .compute()
                .is_err()
        );
    }
    assert!(cqt(&y, 0).hop_length(4).compute().is_err());
    assert!(cqt(&y, 8000).hop_length(0).compute().is_err());
    assert!(icqt(&spectrum).hop_length(usize::MAX).compute().is_err());
}

#[test]
fn griffinlim_cqt_silence_remains_finite_with_fixed_frame_geometry() {
    use dasp_rs::feat::griffinlim_cqt;
    let magnitude = Array2::zeros((3, 4));
    let y = griffinlim_cqt(&magnitude, 8000)
        .fmin(1000.0)
        .hop_length(4)
        .n_iter(3)
        .compute()
        .unwrap();
    assert_eq!(y.len(), 12);
    assert!(y.iter().all(|&v| v == 0.0));
}

#[test]
fn inverse_features_reject_nonfinite_parameters_and_invalid_dimensions() {
    use dasp_rs::feat::{mel_to_stft, mfcc_to_mel};
    let mel = Array2::ones((2, 3));
    for power in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        assert!(mel_to_stft(&mel).n_fft(16).power(power).compute().is_err());
    }
    assert!(
        mel_to_stft(&mel)
            .sample_rate(0)
            .n_fft(16)
            .compute()
            .is_err()
    );
    assert!(mel_to_stft(&mel).n_fft(usize::MAX).compute().is_err());
    assert!(mfcc_to_mel(&mel).n_mels(0).compute().is_err());
    assert!(mfcc_to_mel(&mel).n_mels(usize::MAX).compute().is_err());
    for value in [f32::NAN, f32::INFINITY] {
        let invalid = Array2::from_elem((2, 3), value);
        assert!(mel_to_stft(&invalid).n_fft(16).compute().is_err());
        assert!(mfcc_to_mel(&invalid).n_mels(2).compute().is_err());
        assert!(compute_delta(&invalid).width(3).compute().is_err());
    }
    assert!(
        mfcc_to_mel(&Array2::from_elem((2, 3), f32::MAX))
            .n_mels(2)
            .compute()
            .is_err()
    );
}

#[test]
fn tempo_builder_honors_window_length_and_rejects_zero_window() {
    use dasp_rs::feat::{tempo, tempogram};
    let y: Vec<_> = (0..512)
        .map(|i| ((i * 17 % 31) as f32 - 15.0) / 15.0)
        .collect();
    assert!(
        tempo(&y, 800)
            .hop_length(8)
            .win_length(0)
            .compute()
            .is_err()
    );
    // Independent reconstruction of the documented analysis pipeline using the
    // requested FFT size, followed by local-autocorrelation tempo selection.
    let spec = stft(&y).n_fft(32).hop_length(8).compute().unwrap();
    let envelope = spec.map_axis(ndarray::Axis(0), |col| col.iter().map(|z| z.norm()).sum());
    let gram = tempogram(None, Some(800), Some(&envelope), Some(8), None).unwrap();
    let valid: Vec<_> = (1..gram.nrows())
        .filter(|&lag| {
            let bpm = 60.0 * 800.0 / (8.0 * lag as f32);
            (30.0..=300.0).contains(&bpm)
        })
        .collect();
    let expected: f32 = gram
        .axis_iter(ndarray::Axis(1))
        .map(|col| {
            let lag = *valid
                .iter()
                .max_by(|&&a, &&b| col[a].total_cmp(&col[b]))
                .unwrap();
            60.0 * 800.0 / (8.0 * lag as f32)
        })
        .sum::<f32>()
        / gram.ncols() as f32;
    assert_close(
        tempo(&y, 800)
            .hop_length(8)
            .win_length(32)
            .compute()
            .unwrap(),
        expected,
    );
}

#[test]
fn phase_vocoder_supports_small_and_odd_fft_geometries() {
    for n_fft in [2, 3, 5, 7] {
        let spectrum = Array2::from_elem((n_fft / 2 + 1, 3), Complex::new(1.0, 0.5));
        let actual = phase_vocoder(&spectrum, 1.0, None, Some(n_fft)).unwrap();
        for (a, b) in actual.iter().zip(spectrum.iter()) {
            assert!((*a - *b).norm() < 1e-5);
        }
    }
    let spectrum = Array2::from_elem((5, 2), Complex::new(1.0, 0.0));
    assert!(phase_vocoder(&spectrum, 1.0, Some(1), Some(2)).is_err());
}
