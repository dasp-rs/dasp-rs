//! Analytic and boundary regressions for interleaved audio and time-domain DSP.
use dasp_rs::{ops, proc, types::AudioData};

fn audio(samples: Vec<f32>, channels: u16) -> AudioData {
    AudioData::new(samples, 4, channels).unwrap()
}

#[test]
fn time_edits_preserve_multichannel_frame_order() {
    for channels in [1, 2, 3, 6] {
        let signal = audio((0..4 * channels).map(f32::from).collect(), channels);
        let ch = usize::from(channels);
        assert_eq!(
            proc::time_crop(&signal, 0.25, 0.5).unwrap().samples,
            signal.samples[ch..3 * ch]
        );
        let expected: Vec<_> = signal
            .samples
            .chunks_exact(ch)
            .rev()
            .flatten()
            .copied()
            .collect();
        assert_eq!(proc::time_reversal(&signal).unwrap().samples, expected);
        let mut padded = vec![0.0; ch];
        padded.extend_from_slice(&signal.samples);
        padded.extend(vec![0.0; 2 * ch]);
        assert_eq!(
            proc::zero_padding(&signal, 0.25, 0.5).unwrap().samples,
            padded
        );
        let mut delayed = vec![0.0; ch];
        delayed.extend_from_slice(&signal.samples[..3 * ch]);
        assert_eq!(
            proc::delay(&signal, 0.25)
                .preserve_length(true)
                .compute()
                .unwrap()
                .samples,
            delayed
        );
    }
}

#[test]
fn preserved_delays_never_extend_the_input() {
    let signal = audio(vec![1.0, -1.0, 0.5, -0.5], 2);
    for seconds in [0.5, 1.0, 1000.0, f32::MAX] {
        let output = proc::delay(&signal, seconds)
            .preserve_length(true)
            .compute()
            .unwrap();
        assert_eq!(output.samples, [0.0; 4], "delay={seconds}");
    }
}

#[test]
fn time_edits_reject_nonfinite_and_unrepresentable_times() {
    let signal = audio(vec![1.0; 8], 2);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        assert!(proc::delay(&signal, value).compute().is_err());
        assert!(proc::time_crop(&signal, value, 0.25).is_err());
        assert!(proc::time_crop(&signal, 0.0, value).is_err());
        assert!(proc::zero_padding(&signal, value, 0.0).is_err());
        assert!(proc::zero_padding(&signal, 0.0, value).is_err());
    }
    assert!(proc::delay(&signal, f32::MAX).compute().is_err());
    assert!(proc::zero_padding(&signal, f32::MAX, 0.0).is_err());
    assert!(proc::zero_padding(&signal, 0.0, f32::MAX).is_err());
}

#[test]
fn fallible_frame_operations_reject_malformed_public_audio_data() {
    for (samples, sample_rate, channels) in
        [(vec![1.0], 4, 0), (vec![1.0], 0, 1), (vec![1.0], 4, 2)]
    {
        let signal = AudioData {
            samples,
            sample_rate,
            channels,
        };
        assert!(signal.split_channels().is_err());
        assert!(proc::time_reversal(&signal).is_err());
        assert!(proc::time_crop(&signal, 0.0, 0.25).is_err());
        assert!(proc::zero_padding(&signal, 0.0, 0.0).is_err());
        assert!(proc::delay(&signal, 0.0).compute().is_err());
        assert!(ops::mix_signals(&[signal]).is_err());
    }
}

#[test]
fn lpc_satisfies_independently_solved_toeplitz_equations_at_all_scales() {
    let y = [0.2, 0.8, -0.4, 0.1, 0.7, -0.3, 0.5, -0.1];
    for order in 1..=5 {
        let r: Vec<f64> = (0..=order)
            .map(|lag| {
                y[..y.len() - lag]
                    .iter()
                    .zip(&y[lag..])
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum()
            })
            .collect();
        let matrix = nalgebra::DMatrix::from_fn(order, order, |i, j| r[i.abs_diff(j)]);
        let rhs = nalgebra::DVector::from_iterator(order, r[1..].iter().map(|v| -v));
        let expected = matrix.lu().solve(&rhs).unwrap();
        for scale in [1e-25, 1.0, 1e25] {
            let signal = audio(y.iter().map(|v| v * scale).collect(), 1);
            let coefficients = proc::lpc(&signal, order).unwrap();
            assert_eq!(coefficients[0], 1.0);
            for (actual, expected) in coefficients[1..].iter().zip(expected.iter()) {
                assert!(
                    (f64::from(*actual) - expected).abs() < 1e-6,
                    "order={order}, scale={scale}: {coefficients:?}"
                );
            }
        }
    }
}

#[test]
fn lpc_rejects_silence_nonfinite_and_interleaved_inputs() {
    assert!(proc::lpc(&audio(vec![0.0; 8], 1), 2).is_err());
    assert!(proc::lpc(&audio(vec![1.0; 8], 2), 2).is_err());
    for value in [f32::NAN, f32::INFINITY] {
        assert!(proc::lpc(&audio(vec![value; 8], 1), 2).is_err());
    }
    assert_eq!(proc::lpc(&audio(vec![1.0], 1), 0).unwrap(), [1.0]);
}

#[test]
fn normalization_is_scale_invariant_including_subnormal_audio() {
    for level in [f32::from_bits(1), 1e-25, 1.0, 1e25, f32::MAX] {
        let signal = audio(vec![level, -level], 1);
        for mode in [proc::NormalizeMode::Peak, proc::NormalizeMode::Rms] {
            assert_eq!(
                proc::normalize(&signal, 0.5)
                    .mode(mode)
                    .compute()
                    .unwrap()
                    .samples,
                [0.5, -0.5]
            );
        }
    }
}

#[test]
fn amplitude_operations_reject_invalid_numbers_and_overflow() {
    let signal = audio(vec![0.0, 1.0], 1);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(proc::amplify(&signal, value).is_err());
        assert!(proc::attenuate(&signal, value).is_err());
        assert!(proc::normalize(&signal, value).compute().is_err());
        let invalid = audio(vec![value], 1);
        assert!(proc::amplify(&invalid, 1.0).is_err());
        assert!(proc::attenuate(&invalid, 0.5).is_err());
        assert!(proc::normalize(&invalid, 1.0).compute().is_err());
    }
    assert!(proc::amplify(&audio(vec![f32::MAX], 1), 2.0).is_err());
}

#[test]
fn averaging_large_finite_samples_does_not_overflow() {
    let signal = audio(vec![f32::MAX, f32::MAX], 2);
    assert_eq!(signal.to_mono().samples, [f32::MAX]);
    assert_eq!(
        ops::mix_signals(&[signal.clone(), signal.clone()])
            .unwrap()
            .samples,
        signal.samples
    );
    assert!(ops::mix_signals(&[audio(vec![f32::NAN], 1)]).is_err());
}

#[test]
fn mu_law_roundtrips_quiet_signals_and_small_positive_mu() {
    let y = vec![-1.0, -0.5, -1e-12, 0.0, 1e-12, 0.5, 1.0];
    for mu in [1e-12, 1.0, 255.0, f32::MAX] {
        let compressed = proc::mu_compress(&audio(y.clone(), 1))
            .mu(mu)
            .compute()
            .unwrap();
        let restored = proc::mu_expand(&audio(compressed, 1))
            .mu(mu)
            .compute()
            .unwrap();
        for (&expected, actual) in y.iter().zip(restored) {
            assert!(
                (actual - expected).abs() <= expected.abs() * 1e-5 + f32::MIN_POSITIVE,
                "mu={mu}: {expected} -> {actual}"
            );
        }
    }
}

#[test]
fn log_energy_handles_finite_values_whose_squares_overflow_f32() {
    let result = proc::log_energy(&audio(vec![1e25; 4], 1))
        .frame_length(4)
        .hop_length(1)
        .compute()
        .unwrap();
    let expected = (4.0_f64 * f64::from(1e25_f32).powi(2)).ln();
    assert!((f64::from(result[0]) - expected).abs() < 1e-5);
}

#[test]
fn time_stretch_has_exact_requested_length_for_non_aligned_clips() {
    let y: Vec<f32> = (0..1003).map(|i| (i as f32 * 0.13).sin()).collect();
    for rate in [0.7, 1.0, 1.3, 2.0] {
        let result = proc::time_stretch(&y, rate)
            .n_fft(256)
            .hop_length(64)
            .compute()
            .unwrap();
        assert_eq!(
            result.len(),
            (y.len() as f64 / f64::from(rate)).round() as usize
        );
        if rate.to_bits() == 1.0_f32.to_bits() {
            let max_error = y
                .iter()
                .zip(result)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max_error < 1e-4, "rate=1 error={max_error}");
        }
    }
}

#[test]
fn silence_detection_includes_partial_tail_frames_and_is_scale_invariant() {
    for scale in [1e-25, 1.0, 1e25] {
        let mut y = vec![0.0; 17];
        y[16] = scale;
        let intervals = proc::split(&y).frame_length(8).hop_length(4).compute();
        assert!(
            intervals
                .iter()
                .any(|&(start, end)| start <= 16 && end == 17),
            "scale={scale}, {intervals:?}"
        );
        let (trimmed, (_, end)) = proc::trim(&y).frame_length(8).hop_length(4).compute();
        assert_eq!(end, 17);
        assert_eq!(trimmed.last(), Some(&scale));
    }
    assert_eq!(
        proc::trim(&[0.0; 16])
            .frame_length(8)
            .hop_length(4)
            .compute()
            .0,
        Vec::<f32>::new()
    );
}

#[test]
fn quantized_mu_law_keeps_extreme_finite_values_finite() {
    let signal = audio(vec![f32::MAX], 1);
    let result = proc::mu_compress(&signal)
        .mu(f32::from_bits(1))
        .quantize(true)
        .compute()
        .unwrap();
    assert!(result[0].is_finite());
}

#[test]
fn autocorrelation_rejects_unrepresentable_results_and_nonfinite_samples() {
    for samples in [vec![1e20, 1e20, -1e20], vec![f32::NAN], vec![f32::INFINITY]] {
        assert!(proc::autocorrelate(&audio(samples, 1), None).is_err());
    }
    let result = proc::autocorrelate(&audio(vec![1.0, 2.0, -1.0], 1), None).unwrap();
    assert_eq!(result, [6.0, 0.0, -1.0]);
}

#[test]
fn panning_mixing_and_resampling_reject_invalid_signal_data() {
    let valid = audio(vec![1.0], 1);
    for value in [f32::NAN, f32::INFINITY] {
        assert!(proc::multi_channel_pan(&valid, value, 6).is_err());
        let invalid = audio(vec![value], 1);
        assert!(proc::stereo_mix(&invalid, &valid).is_err());
        assert!(proc::multi_channel_mix(&[&invalid, &valid], 2).is_err());
        assert!(proc::dry_wet_mix(&invalid, &valid, 0.5).is_err());
        assert!(proc::stereo_pan(&invalid, 0.0).is_err());
        assert!(proc::multi_channel_pan(&invalid, 0.0, 6).is_err());
        for sr in [4, 8] {
            assert!(proc::resample(&invalid.samples, 4, sr).is_err());
        }
    }
}
