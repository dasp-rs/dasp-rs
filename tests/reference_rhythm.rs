//! Independent declared-convention references; see the pinned fixture generator.
//! This suite does not claim default librosa onset/beat/PLP parity.
//! Flux tolerances cover f32 FFT leakage through log differences; the second
//! FFT needs a larger absolute bound because it sums those envelope errors.
//! Tempograms use a pure relative bound to test quiet and loud energy alike.
#[path = "common/reference.rs"]
mod reference;
use dasp_rs::feat;
use ndarray::{Array1, Array2, Axis};
use reference::{check_real, load, signal};
const FIXTURE: &str = include_str!("fixtures/rhythm_reference.csv");
fn row(v: Array1<f32>) -> Array2<f32> {
    v.insert_axis(Axis(0))
}
#[test]
fn onset_flux_matches_independent_log_power_on_even_and_odd_grids() {
    let data = load(FIXTURE);
    for (name, n) in [
        ("even", 128),
        ("odd", 127),
        ("quiet", 128),
        ("huge", 128),
        ("silence", 128),
    ] {
        let y = signal(&data, &format!("{name}_input"));
        let actual = feat::onset_strength(&y, 4096)
            .n_fft(n)
            .hop_length(32)
            .compute()
            .unwrap();
        check_real(&format!("{name}_onset"), &row(actual), &data, 3e-5, 1e-6);
    }
}
#[test]
fn multiband_flux_preserves_the_same_frame_zero_and_timing() {
    let data = load(FIXTURE);
    for (name, n) in [
        ("even", 128),
        ("odd", 127),
        ("quiet", 128),
        ("huge", 128),
        ("silence", 128),
    ] {
        let y = signal(&data, &format!("{name}_input"));
        let actual = feat::onset_strength_multi(&y, 4096)
            .n_fft(n)
            .hop_length(32)
            .n_bands(3)
            .compute()
            .unwrap();
        check_real(&format!("{name}_multi"), &actual, &data, 1e-5, 1e-6);
    }
}
#[test]
fn tempograms_match_scipy_correlation_and_amplitude_squared() {
    let data = load(FIXTURE);
    for name in ["normal", "quiet", "loud"] {
        let env = Array1::from(signal(&data, &format!("env_{name}")));
        for win in [1, 9, 10] {
            let actual =
                feat::tempogram(None, Some(4096), Some(&env), Some(32), Some(win)).unwrap();
            check_real(&format!("tgram_{name}_{win}"), &actual, &data, 0.0, 3e-6);
        }
    }
}
#[test]
fn fourier_tempogram_matches_declared_magnitude_stft() {
    let data = load(FIXTURE);
    let y = signal(&data, "fourier_input");
    for win in [9, 10] {
        let actual = feat::fourier_tempogram(&y, 4096)
            .hop_length(128)
            .win_length(win)
            .compute()
            .unwrap();
        check_real(&format!("fourier_{win}"), &actual, &data, 5e-4, 3e-6);
    }
}
#[test]
fn isolated_pulses_have_analytical_onset_frames_and_times() {
    let data = load(FIXTURE);
    let y = signal(&data, "pulses_input");
    let actual = feat::onset_detect(&y, 4096)
        .n_fft(128)
        .hop_length(32)
        .wait(2)
        .compute()
        .unwrap();
    let expected: Vec<usize> = signal(&data, "pulse_onsets")
        .iter()
        .map(|&x| x as usize)
        .collect();
    assert_eq!(actual, expected);
    let times = feat::onset_detect(&y, 4096)
        .n_fft(128)
        .hop_length(32)
        .wait(2)
        .compute_times()
        .unwrap();
    check_real("pulse_times", &row(Array1::from(times)), &data, 0.0, 0.0);
}
#[test]
fn silence_has_no_tempo_onsets_beats_or_local_pulse() {
    let y = vec![0.0; 4096];
    assert_eq!(feat::tempo(&y, 4096).hop_length(64).compute().unwrap(), 0.0);
    assert_eq!(
        feat::onset_detect(&y, 4096)
            .hop_length(64)
            .delta(0.0)
            .compute()
            .unwrap(),
        Vec::<usize>::new()
    );
    let (bpm, beats) = feat::beat_track(&y, 4096).hop_length(64).compute().unwrap();
    assert_eq!(bpm, 0.0);
    assert_eq!(beats, Vec::<usize>::new());
    let (pulse, beats) = feat::plp(&y, 4096)
        .hop_length(64)
        .win_length(64)
        .compute()
        .unwrap();
    assert!(pulse.iter().all(|&x| x == 0.0));
    assert_eq!(beats, Vec::<usize>::new());
}
#[test]
fn regular_click_trains_report_grid_tempo_and_valid_beats() {
    let data = load(FIXTURE);
    for (name, bpm, period) in [("120", 120.0, 32usize), ("80", 80.0, 48)] {
        let y = signal(&data, &format!("clicks_{name}"));
        let estimated = feat::tempo(&y, 4096).hop_length(64).compute().unwrap();
        let (beat_bpm, beats) = feat::beat_track(&y, 4096).hop_length(64).compute().unwrap();
        eprintln!(
            "diagnostic_rhythm_clicks: target={bpm} estimated={estimated} beat_bpm={beat_bpm} beats={beats:?}"
        );
        assert!((estimated - bpm).abs() < 1.0, "known pulse train tempo");
        assert!((beat_bpm - bpm).abs() < 1.0);
        let expected: Vec<usize> = (0..12).map(|i| 1 + i * period).collect();
        assert_eq!(
            beats, expected,
            "centered impulse flux has a 15-frame lead; no unsupported terminal beat"
        );
        assert!(beats.windows(2).all(|w| w[1] > w[0]));
        assert!(beats.iter().all(|&b| b < 1 + y.len() / 64));
        assert!(beats.windows(2).filter(|w| w[1] - w[0] == period).count() >= 8);
    }
}
#[test]
fn invalid_geometry_and_nonfinite_envelopes_are_rejected() {
    let env = Array1::ones(8);
    let y = vec![0.0; 4096];
    assert!(feat::tempogram(None, Some(4096), Some(&env), Some(32), Some(0)).is_err());
    assert!(feat::tempogram(None, Some(4096), Some(&env), Some(0), Some(9)).is_err());
    assert!(
        feat::tempogram(
            None,
            Some(4096),
            Some(&Array1::from(vec![f32::NAN])),
            Some(32),
            Some(9)
        )
        .is_err()
    );
    assert!(feat::onset_detect(&y, 0).compute_times().is_err());
    assert!(
        feat::onset_detect(&y, 4096)
            .delta(f32::NAN)
            .compute()
            .is_err()
    );
    assert!(
        feat::beat_track(&y, 4096)
            .tightness(-1.0)
            .compute()
            .is_err()
    );
    assert!(
        feat::plp(&y, 4096)
            .min_tempo(300.0)
            .max_tempo(30.0)
            .compute()
            .is_err()
    );
}

#[test]
fn zero_threshold_does_not_detect_silent_frames() {
    assert_eq!(
        feat::onset_detect(&vec![0.0; 4096], 4096)
            .hop_length(64)
            .delta(0.0)
            .compute()
            .unwrap(),
        Vec::<usize>::new()
    );
}
#[test]
fn beat_tracking_does_not_invent_silent_beats() {
    let (tempo, beats) = feat::beat_track(&vec![0.0; 4096], 4096)
        .hop_length(64)
        .compute()
        .unwrap();
    assert!(beats.is_empty(), "silent beats: {beats:?}");
    assert_eq!(tempo, 0.0);
}

#[test]
fn local_pulse_matches_declared_autocorrelation_lobes() {
    let data = load(FIXTURE);
    let y = signal(&data, "fourier_input");
    for win in [32, 64] {
        let (pulse, beats) = feat::plp(&y, 4096)
            .hop_length(128)
            .win_length(win)
            .compute()
            .unwrap();
        assert!(beats.iter().all(|&i| i < pulse.len()));
        assert!(beats.windows(2).all(|w| w[0] < w[1]));
        check_real(&format!("plp_{win}"), &row(pulse), &data, 5e-7, 5e-7);
    }
}
#[test]
fn tempo_retains_pulse_period_across_large_and_tiny_gains() {
    let data = load(FIXTURE);
    let y = signal(&data, "clicks_120");
    for gain in [1e-20_f32, 1e20] {
        let scaled: Vec<f32> = y.iter().map(|&v| v * gain).collect();
        let bpm = feat::tempo(&scaled, 4096).hop_length(64).compute().unwrap();
        assert!((bpm - 120.0).abs() < 0.001, "gain={gain} BPM={bpm}");
    }
}
#[test]
fn unrepresentable_raw_autocorrelation_returns_an_error() {
    let env = Array1::from_elem(16, 1e30_f32);
    assert!(feat::tempogram(None, Some(4096), Some(&env), Some(32), Some(9)).is_err());
}

#[test]
fn appending_silence_does_not_extend_a_tied_beat_path() {
    let data = load(FIXTURE);
    let y = signal(&data, "clicks_120");
    let mut padded = y.clone();
    padded.extend(vec![0.0; 2048]);
    let (_, original) = feat::beat_track(&y, 4096).hop_length(64).compute().unwrap();
    let (_, extended) = feat::beat_track(&padded, 4096)
        .hop_length(64)
        .compute()
        .unwrap();
    assert_eq!(
        original, extended,
        "equal-score silent continuations must not extend the chosen path"
    );
}
