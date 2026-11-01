//! Independent IIR filterbank validation; see the pinned `SciPy` fixture generator.
#[path = "common/reference.rs"]
mod reference;
use dasp_rs::proc::iirt;
use reference::{check_real, load, signal};

#[test]
fn ordinary_iirt_is_a_twelve_band_semitone_transform() {
    let result = iirt(&vec![0.0; 4096], 44100).compute();
    assert!(result.is_ok(), "ordinary IIRT failed: {result:?}");
    assert_eq!(result.unwrap().dim(), (12, 5));
}

#[test]
fn iirt_zero_hop_returns_an_error_without_panicking() {
    let result = std::panic::catch_unwind(|| {
        iirt(&[0.0; 32], 8000)
            .win_length(16)
            .hop_length(0)
            .compute()
    });
    assert!(result.is_ok(), "zero hop panicked");
    assert!(result.unwrap().is_err());
}

#[test]
fn causal_semitone_rms_matches_independent_scipy_sosfilt() {
    let data = load(include_str!("fixtures/iir_reference.csv"));
    for name in ["silence", "tone", "chirp", "transient", "noise", "low_rate"] {
        let config = &data[&format!("{name}_iirt_config")];
        let y = signal(&data, &format!("{name}_input"));
        let result = iirt(&y, config[[0, 0]].re as u32)
            .win_length(config[[0, 1]].re as usize)
            .hop_length(config[[0, 2]].re as usize)
            .compute()
            .unwrap();
        // Continuous f64 SOS filtering followed by f32 output. A small absolute
        // floor protects near-zero startup tails; relative allowance covers
        // different mathematically equivalent SOS pole/zero pairings.
        check_real(&format!("{name}_iirt"), &result, &data, 1e-13, 2e-6);
    }
}

#[test]
fn long_tone_settles_at_the_expected_band_and_rms_gain() {
    let data = load(include_str!("fixtures/iir_reference.csv"));
    let y = signal(&data, "tone_input");
    let result = iirt(&y, 8000)
        .win_length(8000)
        .hop_length(8000)
        .compute()
        .unwrap();
    let last = result.ncols() - 1;
    let peak = result
        .column(last)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert_eq!(peak, 5);
    let steady = f64::from(result[[5, last]]);
    let ideal = 0.6_f64 / std::f64::consts::SQRT_2;
    let relative_gain_error = (steady / ideal - 1.0).abs();
    let max_adjacent = f64::from(result[[4, last]].max(result[[6, last]]));
    eprintln!(
        "REFERENCE iirt_long_tone: seconds=8 peak=5 relative_rms_gain_error={relative_gain_error:.6e} adjacent_ratio={:.6e} startup_rms={:.6e} settled_rms={steady:.6e}",
        max_adjacent / steady,
        result[[5, 0]]
    );
    // A one-second frame contains a nonintegral number of tone periods, leaving
    // a small finite-window RMS deviation from amplitude/sqrt(2).
    assert!(relative_gain_error < 0.003);
    assert!(max_adjacent < steady * 0.1);
    assert!(f64::from(result[[5, 0]]) < steady * 0.9);
}

#[test]
fn tiny_and_large_amplitudes_preserve_rms_without_underflow_or_overflow() {
    let data = load(include_str!("fixtures/iir_reference.csv"));
    let input = signal(&data, "tone_input");
    for scale in [1e-20_f32, 1e20_f32] {
        let y: Vec<_> = input.iter().map(|v| v * scale).collect();
        let actual = iirt(&y, 8000)
            .win_length(8000)
            .hop_length(8000)
            .compute()
            .unwrap();
        assert!(actual.iter().all(|v| v.is_finite()));
        let normalized = actual.mapv(|v| (f64::from(v) / f64::from(scale)) as f32);
        check_real("tone_iirt", &normalized, &data, 1e-12, 2e-6);
        eprintln!("REFERENCE iirt_amplitude: scale={scale:.6e} normalized_rms_matches_scipy=1");
    }
}

#[test]
fn iirt_rejects_invalid_geometry_and_input_and_supports_one_sample_windows() {
    let y = [0.0; 32];
    for sr in [0, 1, 64, 127] {
        assert!(iirt(&y, sr).win_length(16).compute().is_err());
    }
    assert!(iirt(&[], 8000).win_length(1).compute().is_err());
    assert!(iirt(&y, 8000).win_length(0).compute().is_err());
    assert!(iirt(&y, 8000).win_length(33).compute().is_err());
    assert!(iirt(&y, 8000).win_length(usize::MAX).compute().is_err());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = y;
        bad[5] = invalid;
        assert!(iirt(&bad, 8000).win_length(16).compute().is_err());
    }
    assert_eq!(
        iirt(&y, 128).win_length(1).compute().unwrap().dim(),
        (12, 32)
    );
    assert_eq!(
        iirt(&y, 8000)
            .win_length(16)
            .hop_length(usize::MAX)
            .compute()
            .unwrap()
            .dim(),
        (12, 1)
    );
}
