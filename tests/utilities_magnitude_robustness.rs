//! Adversarial numerical and boundary regressions for utilities and magnitude processing.

use dasp_rs::{generate, mag, util};
use ndarray::arr2;

#[test]
fn fft_bins_follow_dft_spacing_for_odd_and_even_sizes() {
    for n in [1, 2, 3, 5, 8, 17] {
        let bins = util::fft_frequencies()
            .sample_rate(12000)
            .n_fft(n)
            .compute();
        assert_eq!(bins.len(), n / 2 + 1);
        for (k, actual) in bins.iter().enumerate() {
            let expected = (k as f64 * 12000.0 / n as f64) as f32;
            assert!(
                (actual - expected).abs() < 0.001,
                "n={n}, k={k}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn note_names_round_trip_across_zero_and_octave_boundaries() {
    let midi: Vec<f32> = (-24..=127).map(|n| n as f32).collect();
    let names = util::midi_to_note(&midi, None);
    let borrowed: Vec<_> = names.iter().map(String::as_str).collect();
    assert_eq!(util::note_to_midi(&borrowed, None), midi);
}

#[test]
fn pythagorean_scale_contains_successive_fifths_without_duplicates() {
    assert_eq!(util::pythagorean_intervals(Some(3)), vec![1.0, 1.125, 1.5]);
    let scale = util::pythagorean_intervals(Some(12));
    assert!(scale.windows(2).all(|w| w[0] < w[1]));
    assert!(scale.iter().all(|&v| (1.0..2.0).contains(&v)));
}

#[test]
fn peak_windows_can_span_the_entire_signal_without_integer_overflow() {
    let peaks = util::peak_pick(&[0.0, 1.0, 0.0])
        .post_max(usize::MAX)
        .post_avg(usize::MAX)
        .pre_max(usize::MAX)
        .pre_avg(usize::MAX)
        .delta(0.1)
        .wait(0)
        .compute();
    assert_eq!(peaks, vec![1]);
}

#[test]
fn sync_mean_does_not_overflow_when_mean_is_representable() {
    let input = arr2(&[[f32::MAX, f32::MAX]]);
    assert_eq!(util::sync(&input, &[]).compute()[[0, 0]], f32::MAX);
}

#[test]
fn click_times_must_be_finite_and_nonnegative() {
    for time in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            generate::clicks().times(&[time]).compute().is_err(),
            "time={time}"
        );
    }
    assert!(
        generate::clicks()
            .times(&[0.0])
            .sample_rate(0)
            .compute()
            .is_err()
    );
    assert!(
        generate::clicks()
            .frames(&[0])
            .hop_length(0)
            .compute()
            .is_err()
    );
}

#[test]
fn click_frame_offset_overflow_returns_error() {
    assert!(
        generate::clicks()
            .frames(&[usize::MAX])
            .hop_length(2)
            .compute()
            .is_err()
    );
}

#[test]
fn default_click_track_duration_uses_selected_sample_rate() {
    let signal = generate::clicks().sample_rate(8000).compute().unwrap();
    assert_eq!(signal.len(), 8000);
    assert!(signal.iter().all(|&v| v == 0.0));
}

#[test]
fn tone_preserves_phase_at_high_audio_frequencies() {
    let actual = generate::tone(19000.0, 48000).duration(1.0).compute();
    for (n, &value) in actual.iter().enumerate() {
        let expected = (std::f64::consts::TAU * 19000.0 * n as f64 / 48000.0).cos() as f32;
        assert!(
            (value - expected).abs() < 1e-5,
            "sample {n}: {value} != {expected}"
        );
    }
}

#[test]
fn chirp_preserves_integrated_phase_at_high_audio_frequencies() {
    let actual = generate::chirp(12000.0, 19000.0, 48000)
        .duration(1.0)
        .compute();
    for (n, &value) in actual.iter().enumerate() {
        let t = n as f64 / 48000.0;
        let expected = (std::f64::consts::TAU * (12000.0 * t + 3500.0 * t * t)).cos() as f32;
        assert!(
            (value - expected).abs() < 1e-5,
            "sample {n}: {value} != {expected}"
        );
    }
}

#[test]
fn logarithmic_conversions_handle_extreme_finite_ratios() {
    let input = arr2(&[[1e30, 1e-30]]);
    let amplitude = mag::amplitude_to_db(&input)
        .ref_val(1e-30)
        .amin(1e-35)
        .top_db(2000.0)
        .compute()
        .unwrap();
    let power = mag::power_to_db(&input)
        .ref_val(1e-30)
        .amin(1e-35)
        .top_db(2000.0)
        .compute()
        .unwrap();
    assert!((amplitude[[0, 0]] - 1200.0).abs() < 0.001);
    assert_eq!(amplitude[[0, 1]], 0.0);
    assert!((power[[0, 0]] - 600.0).abs() < 0.001);
    assert_eq!(power[[0, 1]], 0.0);
}

#[test]
fn inverse_decibels_avoid_intermediate_overflow() {
    let amplitude = mag::db_to_amplitude(&arr2(&[[800.0]]), Some(1e-20)).unwrap();
    let power = mag::db_to_power(&arr2(&[[400.0]]), Some(1e-20)).unwrap();
    for value in [amplitude[[0, 0]], power[[0, 0]]] {
        assert!((value / 1e20 - 1.0).abs() < 1e-5);
    }
}

#[test]
fn magnitude_apis_reject_nonfinite_data_and_parameters() {
    let valid = arr2(&[[1.0, 2.0]]);
    for bad in [f32::NAN, f32::INFINITY] {
        let invalid = arr2(&[[bad]]);
        assert!(mag::amplitude_to_db(&invalid).compute().is_err());
        assert!(mag::power_to_db(&invalid).compute().is_err());
        assert!(mag::pcen(&invalid).compute().is_err());
        assert!(mag::amplitude_to_db(&valid).ref_val(bad).compute().is_err());
        assert!(mag::power_to_db(&valid).amin(bad).compute().is_err());
        assert!(mag::power_to_db(&valid).top_db(bad).compute().is_err());
        assert!(mag::db_to_amplitude(&valid, Some(bad)).is_err());
        assert!(mag::db_to_power(&valid, Some(bad)).is_err());
        assert!(mag::pcen(&valid).gain(bad).compute().is_err());
        assert!(mag::pcen(&valid).bias(bad).compute().is_err());
    }
}

#[test]
fn weighting_rejects_nonfinite_frequencies_and_thresholds() {
    for kind in ["A", "B", "C", "D"] {
        for bad in [f32::NAN, f32::INFINITY] {
            assert!(mag::frequency_weighting(&[bad], Some(kind)).is_err());
        }
    }
    assert!(mag::a_weighting(&[1000.0], Some(f32::NAN)).is_err());
}

#[test]
fn weighting_remains_finite_for_large_finite_frequencies() {
    for kind in ["A", "B", "C", "D"] {
        let weights = mag::frequency_weighting(&[1e10, f32::MAX], Some(kind)).unwrap();
        assert!(
            weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "{kind}: {weights:?}"
        );
    }
}

#[test]
fn pcen_validates_timebase() {
    let input = arr2(&[[1.0, 2.0]]);
    assert!(mag::pcen(&input).sample_rate(0).compute().is_err());
    assert!(mag::pcen(&input).hop_length(0).compute().is_err());
}

#[test]
fn pcen_short_time_constant_converges_to_instantaneous_smoothing() {
    let input = arr2(&[[1.0, 2.0, 4.0]]);
    let output = mag::pcen(&input)
        .sample_rate(1)
        .hop_length(1_000_000)
        .gain(1.0)
        .bias(0.0)
        .compute()
        .unwrap();
    for value in output {
        assert!((value - 1.0).abs() < 1e-5, "{value}");
    }
}

#[test]
fn pcen_preserves_small_positive_output_with_large_bias() {
    let output = mag::pcen(&arr2(&[[1.0]]))
        .gain(0.0)
        .bias(1e20)
        .compute()
        .unwrap();
    assert!((output[[0, 0]] / 5e-11 - 1.0).abs() < 1e-5);
}

#[test]
fn htk_mel_round_trip_preserves_small_positive_frequencies() {
    let input = [1e-6, 1e-4, 0.1, 1000.0];
    let mel = util::hz_to_mel(&input, Some(true));
    let back = util::mel_to_hz(&mel, Some(true));
    for (&actual, &expected) in back.iter().zip(&input) {
        assert!(
            (actual / expected - 1.0).abs() < 1e-5,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn prime_limit_intervals_include_octave_folded_prime_and_reciprocal() {
    let intervals = util::plimit_intervals(&[2, 3]);
    for expected in [1.0, 4.0 / 3.0, 1.5] {
        assert!(
            intervals.iter().any(|&v| (v - expected).abs() < 1e-6),
            "{intervals:?} missing {expected}"
        );
    }
    assert!(intervals.windows(2).all(|w| w[0] < w[1]));
    assert!(intervals.iter().all(|&v| (1.0..2.0).contains(&v)));
}

#[test]
fn pcen_instantaneous_smoothing_retains_quiet_sample_after_large_sample() {
    let input = arr2(&[[f32::MAX, 1.0]]);
    let output = mag::pcen(&input)
        .sample_rate(1)
        .hop_length(usize::MAX)
        .gain(1.0)
        .bias(0.0)
        .compute()
        .unwrap();
    assert!((output[[0, 1]] - 1.0).abs() < 1e-5);
}

#[test]
fn nonprime_zero_and_one_do_not_make_interval_expansion_loop_forever() {
    assert_eq!(
        util::plimit_intervals(&[0, 1, 2, 3]),
        util::plimit_intervals(&[2, 3])
    );
}

#[test]
fn large_fifth_offsets_do_not_overflow_before_octave_reduction() {
    assert_eq!(util::fifths_to_note("C", i32::MAX, None), "C#1252698794");
    assert_eq!(util::fifths_to_note("C", i32::MIN, None), "E-1252698794");
}

#[test]
#[should_panic(expected = "hop_length must be positive")]
fn samples_to_frames_rejects_zero_hop_explicitly() {
    util::samples_to_frames(&[0], Some(0));
}

#[test]
#[should_panic(expected = "hop_length must be positive")]
fn time_to_frames_rejects_zero_hop_explicitly() {
    util::time_to_frames(&[0.0]).hop_length(0).compute();
}

#[test]
#[should_panic(expected = "intervals must be non-empty when n_bins is positive")]
fn positive_interval_bin_count_requires_intervals() {
    util::interval_frequencies(1, 440.0, &[]);
}

#[test]
fn zero_interval_bin_count_accepts_empty_intervals() {
    assert_eq!(util::interval_frequencies(0, 440.0, &[]), [] as [f32; 0]);
}

#[test]
#[should_panic(expected = "frame sample index exceeds usize::MAX")]
fn frames_to_samples_checks_index_overflow() {
    util::frames_to_samples(&[usize::MAX], Some(2));
}

#[test]
#[should_panic(expected = "block frame index exceeds usize::MAX")]
fn blocks_to_frames_checks_index_overflow() {
    util::blocks_to_frames(&[usize::MAX], 2);
}

#[test]
#[should_panic(expected = "block sample index exceeds usize::MAX")]
fn blocks_to_samples_checks_frame_product_overflow() {
    util::blocks_to_samples(&[usize::MAX], 2, Some(1));
}

#[test]
#[should_panic(expected = "block sample index exceeds usize::MAX")]
fn blocks_to_samples_checks_hop_product_overflow() {
    util::blocks_to_samples(&[1], usize::MAX, Some(2));
}

#[test]
#[should_panic(expected = "column sample index exceeds usize::MAX")]
fn samples_like_checks_index_overflow() {
    util::samples_like(&arr2(&[[0.0, 0.0, 0.0]]), Some(usize::MAX));
}
