//! Numerical contracts for framewise YIN and probabilistic pitch tracking.
use dasp_rs::pitch::{pyin, yin};

fn tone(frequency: f32, amplitude: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| amplitude * (std::f32::consts::TAU * frequency * i as f32 / 8000.0).sin())
        .collect()
}

fn assert_pitch(actual: &[f32], expected: f32, cents: f32) {
    assert_ne!(actual, []);
    for &f0 in actual {
        let error = (1200.0 * (f0 / expected).log2()).abs();
        assert!(
            f0 > 0.0 && error < cents,
            "expected {expected} Hz, got {f0} ({error} cents)"
        );
    }
}

#[test]
fn yin_tracks_fundamentals_and_is_gain_invariant() {
    for frequency in [100.0, 220.0, 440.0] {
        for amplitude in [0.5, 1e-10, 1e-30, 1e30] {
            let y = tone(frequency, amplitude, 2048);
            let f0 = yin(&y, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(512)
                .hop_length(128)
                .compute()
                .unwrap();
            assert_pitch(&f0, frequency, 8.0);
        }
    }
}

#[test]
fn pyin_tracks_fundamentals_and_is_gain_invariant() {
    for frequency in [100.0, 220.0, 440.0] {
        for amplitude in [0.5, 1e-10, 1e-30, 1e30] {
            let y = tone(frequency, amplitude, 2048);
            let f0 = pyin(&y, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(512)
                .hop_length(128)
                .compute()
                .unwrap();
            assert_pitch(&f0, frequency, 10.0);
        }
    }
}

#[test]
fn trackers_reject_frames_that_cannot_resolve_requested_periods() {
    let y = vec![0.1; 32];
    assert!(
        yin(&y, 50.0, 500.0)
            .sample_rate(8000)
            .frame_length(32)
            .compute()
            .is_err()
    );
    assert!(
        pyin(&y, 50.0, 500.0)
            .sample_rate(8000)
            .frame_length(32)
            .compute()
            .is_err()
    );
}

#[test]
fn trackers_reject_invalid_inputs_without_panicking() {
    let y = tone(220.0, 0.5, 512);
    for (fmin, fmax) in [
        (0.0, 600.0),
        (-1.0, 600.0),
        (f32::NAN, 600.0),
        (80.0, f32::INFINITY),
        (80.0, 5000.0),
        (300.0, 100.0),
        (f32::from_bits(1), 600.0),
    ] {
        assert!(
            yin(&y, fmin, fmax)
                .sample_rate(8000)
                .frame_length(512)
                .compute()
                .is_err()
        );
        assert!(
            pyin(&y, fmin, fmax)
                .sample_rate(8000)
                .frame_length(512)
                .compute()
                .is_err()
        );
    }
    for frame in [0, 1, 3, 4096] {
        assert!(
            yin(&y, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(frame)
                .compute()
                .is_err()
        );
        assert!(
            pyin(&y, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(frame)
                .compute()
                .is_err()
        );
    }
    assert!(yin(&y, 80.0, 600.0).sample_rate(0).compute().is_err());
    assert!(pyin(&y, 80.0, 600.0).sample_rate(0).compute().is_err());
    assert!(
        yin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(512)
            .hop_length(0)
            .compute()
            .is_err()
    );
    assert!(
        pyin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(512)
            .hop_length(0)
            .compute()
            .is_err()
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut signal = y.clone();
        signal[100] = bad;
        assert!(
            yin(&signal, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(512)
                .compute()
                .is_err()
        );
        assert!(
            pyin(&signal, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(512)
                .compute()
                .is_err()
        );
    }
}

#[test]
fn trackers_handle_range_endpoints_odd_frames_and_nonstandard_hops() {
    for frequency in [80.0, 600.0] {
        let y = tone(frequency, 0.5, 1513);
        let actual_yin = yin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(513)
            .hop_length(200)
            .compute()
            .unwrap();
        let actual_pyin = pyin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(513)
            .hop_length(200)
            .compute()
            .unwrap();
        assert_eq!(actual_yin.len(), 6);
        assert_eq!(actual_pyin.len(), 6);
        assert_pitch(&actual_yin, frequency, 10.0);
        assert_pitch(&actual_pyin, frequency, 10.0);
    }

    // A sub-bin pitch range still supports one pitch state, and even a hop
    // larger than the entire input produces its first complete frame.
    let y = tone(220.0, 0.5, 512);
    for hop in [128, usize::MAX] {
        let actual_yin = yin(&y, 219.9, 220.1)
            .sample_rate(8000)
            .frame_length(512)
            .hop_length(hop)
            .compute()
            .unwrap();
        let actual_pyin = pyin(&y, 219.9, 220.1)
            .sample_rate(8000)
            .frame_length(512)
            .hop_length(hop)
            .compute()
            .unwrap();
        assert_eq!(actual_yin.len(), 1);
        assert_eq!(actual_pyin.len(), 1);
        assert_pitch(&actual_yin, 220.0, 2.0);
        assert_pitch(&actual_pyin, 220.0, 2.0);
    }
}

#[test]
fn yin_threshold_controls_acceptance_of_a_noisy_tone() {
    let mut seed = 42_u32;
    let mut y = tone(220.0, 0.5, 2048);
    for sample in &mut y {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *sample += 0.5 * (f64::from(seed) / f64::from(u32::MAX) - 0.5) as f32;
    }
    let builder = yin(&y, 180.0, 260.0)
        .sample_rate(8000)
        .frame_length(512)
        .hop_length(128);
    let strict = builder.clone().trough_threshold(0.02).compute().unwrap();
    let tolerant = builder.trough_threshold(0.3).compute().unwrap();
    assert!(strict.iter().all(|&f0| f0 == 0.0));
    // With this noise level, individual troughs jitter. An independent
    // fixed-window calculation has a worst error of 42.192 cents on this input.
    assert_pitch(&tolerant, 220.0, 50.0);
}

#[test]
fn pyin_follows_pitch_changes_within_the_configured_transition_rate() {
    let mut y = tone(220.0, 0.5, 512 * 4);
    y.extend(tone(330.0, 0.5, 512 * 4));
    let builder = pyin(&y, 200.0, 350.0)
        .sample_rate(8000)
        .frame_length(512)
        .hop_length(512);
    let normal = builder.clone().compute_with_voicing().unwrap();
    assert_eq!(normal.f0.len(), 8);
    assert_pitch(&normal.f0[..4], 220.0, 10.0);
    assert_pitch(&normal.f0[4..], 330.0, 10.0);
    let slow = builder
        .max_transition_rate(0.01)
        .compute_with_voicing()
        .unwrap();
    // At this rate the permitted movement rounds down to zero pitch bins.
    // The tracker must not jump directly between these distinct pitches.
    let voiced: Vec<_> = slow.f0.iter().copied().filter(|&f0| f0 > 0.0).collect();
    assert_ne!(voiced, []);
    for pair in voiced.windows(2) {
        assert_eq!(pair[0], pair[1]);
    }
    assert_ne!(normal.f0, slow.f0);
    assert_eq!(normal.voiced_prob, slow.voiced_prob);
}

#[test]
fn trackers_follow_a_smooth_frequency_sweep() {
    let sr = 8000.0_f64;
    let slope = 50.0_f64; // Hz per second.
    let y: Vec<f32> = (0..8192)
        .map(|i| {
            let t = f64::from(i) / sr;
            (0.5 * (std::f64::consts::TAU * (200.0 * t + 0.5 * slope * t * t)).sin()) as f32
        })
        .collect();
    let actual_yin = yin(&y, 150.0, 350.0)
        .sample_rate(8000)
        .frame_length(512)
        .hop_length(128)
        .compute()
        .unwrap();
    let actual_pyin = pyin(&y, 150.0, 350.0)
        .sample_rate(8000)
        .frame_length(512)
        .hop_length(128)
        .compute()
        .unwrap();
    assert_eq!(actual_yin.len(), 61);
    assert_eq!(actual_pyin.len(), 61);
    for estimates in [&actual_yin, &actual_pyin] {
        for (frame, &f0) in estimates.iter().enumerate() {
            // A difference window compares pairs of samples separated by a
            // period. Its effective time is the window midpoint + half a period.
            let window_midpoint = (frame * 128) as f64 + 127.5;
            let midpoint_frequency = 200.0 + slope * window_midpoint / sr;
            let expected = midpoint_frequency + slope / (2.0 * midpoint_frequency);
            assert_pitch(&[f0], expected as f32, 15.0);
        }
    }
}

#[test]
fn pitch_estimates_scale_with_the_sample_rate() {
    for sr in [8000, 22_050, 44_100, 48_000, 96_000] {
        for frequency in [110.0, 440.0] {
            let y: Vec<f32> = (0..3072)
                .map(|i| {
                    (0.5 * (std::f64::consts::TAU * frequency * f64::from(i) / f64::from(sr)).sin())
                        as f32
                })
                .collect();
            let actual_yin = yin(&y, 100.0, 500.0)
                .sample_rate(sr)
                .frame_length(2048)
                .hop_length(512)
                .compute()
                .unwrap();
            let actual_pyin = pyin(&y, 100.0, 500.0)
                .sample_rate(sr)
                .frame_length(2048)
                .hop_length(512)
                .compute()
                .unwrap();
            assert_eq!(actual_yin.len(), 3);
            assert_eq!(actual_pyin.len(), 3);
            assert_pitch(&actual_yin, frequency as f32, 8.0);
            assert_pitch(&actual_pyin, frequency as f32, 10.0);
        }
    }
}

#[test]
fn pyin_exposes_sequence_voicing_and_preserves_frequency_only_api() {
    let mut y = vec![0.0; 512 * 3];
    y.extend(tone(220.0, 0.5, 512 * 5));
    y.extend(vec![0.0; 512 * 3]);
    let builder = pyin(&y, 80.0, 600.0)
        .sample_rate(8000)
        .frame_length(512)
        .hop_length(512);
    let result = builder.clone().compute_with_voicing().unwrap();
    assert_eq!(builder.compute().unwrap(), result.f0);
    assert_eq!(result.f0.len(), 11);
    for i in 0..11 {
        let voiced = (3..8).contains(&i);
        assert_eq!(result.voiced_flag[i], voiced, "frame {i}");
        assert!(result.voiced_prob[i].is_finite() && (0.0..=1.0).contains(&result.voiced_prob[i]));
        if voiced {
            assert_pitch(&result.f0[i..=i], 220.0, 10.0);
            assert!(result.voiced_prob[i] > 0.9);
        } else {
            assert_eq!(result.f0[i], 0.0);
            assert_eq!(result.voiced_prob[i], 0.0);
        }
    }
}

#[test]
fn silence_dc_and_incomplete_trailing_frames_have_defined_behavior() {
    for value in [0.0, 0.25] {
        let y = vec![value; 512 + 128 + 127];
        let result = pyin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(512)
            .hop_length(128)
            .compute_with_voicing()
            .unwrap();
        assert_eq!(result.f0, [0.0, 0.0]);
        assert_eq!(result.voiced_flag, [false, false]);
        assert_eq!(result.voiced_prob, [0.0, 0.0]);
        assert_eq!(
            yin(&y, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(512)
                .hop_length(128)
                .compute()
                .unwrap(),
            [0.0, 0.0]
        );
    }
}

#[test]
fn pitch_configuration_rejects_invalid_probabilities_and_thresholds() {
    let y = tone(220.0, 0.5, 512);
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            pyin(&y, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(512)
                .max_transition_rate(value)
                .compute()
                .is_err()
        );
    }
    for value in [0.0, 1.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            pyin(&y, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(512)
                .switch_probability(value)
                .compute()
                .is_err()
        );
        assert!(
            yin(&y, 80.0, 600.0)
                .sample_rate(8000)
                .frame_length(512)
                .trough_threshold(value)
                .compute()
                .is_err()
        );
    }
}

#[test]
fn trackers_follow_a_missing_fundamental() {
    let y: Vec<f32> = (0..2048)
        .map(|i| {
            let phase = std::f32::consts::TAU * 150.0 * i as f32 / 8000.0;
            0.5 * (2.0 * phase).sin() + 0.3 * (3.0 * phase).sin()
        })
        .collect();
    assert_pitch(
        &yin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(512)
            .compute()
            .unwrap(),
        150.0,
        10.0,
    );
    assert_pitch(
        &pyin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(512)
            .compute()
            .unwrap(),
        150.0,
        10.0,
    );
}

#[test]
fn pitch_matches_independent_librosa_reference_with_documented_tolerance() {
    let fixture = include_str!("fixtures/pitch_librosa_0_11.csv");
    for name in [
        "tone_100",
        "tone_220",
        "tone_440",
        "quiet_220",
        "missing_150",
        "silence",
    ] {
        let y = match name {
            "tone_100" => tone(100.0, 0.5, 2048),
            "tone_220" => tone(220.0, 0.5, 2048),
            "tone_440" => tone(440.0, 0.5, 2048),
            "quiet_220" => tone(220.0, 1e-10, 2048),
            "missing_150" => (0..2048)
                .map(|i| {
                    let phase = std::f32::consts::TAU * 150.0 * i as f32 / 8000.0;
                    0.5 * (2.0 * phase).sin() + 0.3 * (3.0 * phase).sin()
                })
                .collect(),
            _ => vec![0.0; 2048],
        };
        let actual_yin = yin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(512)
            .hop_length(128)
            .compute()
            .unwrap();
        let actual_pyin = pyin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(512)
            .hop_length(128)
            .compute_with_voicing()
            .unwrap();
        let mut frames = 0;
        for row in fixture.lines().filter(|row| row.starts_with(name)) {
            let columns: Vec<_> = row.split(',').collect();
            let index: usize = columns[1].parse().unwrap();
            let expected_yin: f32 = columns[2].parse().unwrap();
            let expected_pyin: f32 = columns[3].parse().unwrap();
            let voiced = columns[4] == "1";
            assert_eq!(
                actual_pyin.voiced_flag[index], voiced,
                "case {name}, frame {index}"
            );
            if voiced {
                assert_pitch(&actual_yin[index..=index], expected_yin, 15.0);
                assert_pitch(&actual_pyin.f0[index..=index], expected_pyin, 15.0);
            } else {
                assert_eq!(actual_pyin.f0[index], 0.0);
            }
            frames += 1;
        }
        assert_eq!(frames, actual_yin.len());
        assert_eq!(frames, actual_pyin.f0.len());
    }
}

#[test]
fn deterministic_white_noise_is_not_a_voiced_pitch_track() {
    for initial_seed in [0, 42, 123_456, u32::MAX] {
        let mut seed = initial_seed;
        let y: Vec<f32> = (0..4096)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (f64::from(seed) / f64::from(u32::MAX) - 0.5) as f32
            })
            .collect();
        let result = pyin(&y, 80.0, 600.0)
            .sample_rate(8000)
            .frame_length(512)
            .hop_length(128)
            .compute_with_voicing()
            .unwrap();
        assert!(result.voiced_prob.iter().all(|&p| p < 0.1));
        assert!(result.voiced_flag.iter().all(|&v| !v));
    }
}
