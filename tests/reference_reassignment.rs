//! Independent analytic-window/librosa magnitude-raster references.
//! See `fixtures/generate_reassignment_reference.py` for the explicit raster
//! convention and conversion from librosa's coordinate-returning API.
#[path = "common/reference.rs"]
mod reference;
use dasp_rs::proc::reassigned_spectrogram;
use reference::{check_real, load, signal};

#[test]
fn reassignment_matches_independent_periodic_window_raster() {
    let data = load(include_str!("fixtures/reassignment_reference.csv"));
    for (name, fft, scale) in [
        ("mix", 64, 1.0),
        ("odd", 63, 1.0),
        ("chirp", 64, 1.0),
        ("tone", 64, 1.0),
        ("half_bin", 64, 1.0),
        ("impulse", 64, 1.0),
        ("edges", 64, 1.0),
        ("half_time", 64, 1.0),
        ("silence", 64, 1.0),
        ("quiet", 64, 1e-30),
        ("loud", 64, 1e30),
        ("subnormal", 64, 1e-40),
        ("tiny_fft", 2, 1.0),
        ("dc", 64, 1.0),
        ("nyquist", 64, 1.0),
    ] {
        let y = signal(&data, &format!("input_{name}"));
        let actual = reassigned_spectrogram(&y, 1024)
            .n_fft(fft)
            .compute()
            .unwrap();
        // f64 FFTs/coordinate arithmetic followed by f32 raster storage. The
        // absolute tolerance scales with signal amplitude, including quiet data.
        check_real(
            &format!("raster_{name}"),
            &actual,
            &data,
            (2e-6_f64 * scale).max(f64::from(f32::from_bits(1))),
            3e-7,
        );
        let expected_mass: f64 = data[&format!("magnitude_{name}")]
            .iter()
            .map(|z| z.re)
            .sum();
        let actual_mass: f64 = actual.iter().copied().map(f64::from).sum();
        // Subnormal f32 raster storage rounds each output to the nearest
        // representable value; include at most half an ulp per cell.
        let bound = 1e-6 * expected_mass
            + 1e-12 * scale
            + 0.5 * f64::from(f32::from_bits(1)) * actual.len() as f64;
        assert!(
            (actual_mass - expected_mass).abs() <= bound,
            "{name}: input magnitude mass={expected_mass}, raster mass={actual_mass}"
        );
    }
}

#[test]
fn quiet_coefficients_are_not_discarded_by_an_absolute_threshold() {
    let data = load(include_str!("fixtures/reassignment_reference.csv"));
    let y = signal(&data, "input_quiet");
    let raster = reassigned_spectrogram(&y, 1024)
        .n_fft(64)
        .compute()
        .unwrap();
    assert!(
        raster.iter().any(|&v| v > 0.0),
        "nonzero quiet signal lost all magnitude"
    );
    check_real("raster_quiet", &raster, &data, 2e-36, 3e-7);
}

#[test]
fn impulse_localization_obeys_nearest_frame_and_half_grid_ties() {
    for (sample, expected_frame) in [(73, 5), (88, 6)] {
        let mut y = vec![0.0; 257];
        y[sample] = 0.8;
        let raster = reassigned_spectrogram(&y, 1024)
            .n_fft(64)
            .compute()
            .unwrap();
        let total: f64 = raster.iter().copied().map(f64::from).sum();
        let localized: f64 = raster
            .column(expected_frame)
            .iter()
            .copied()
            .map(f64::from)
            .sum();
        assert!(
            localized / total > 1.0 - 1e-6,
            "impulse sample{sample}: localized fraction {}",
            localized / total
        );
    }
}

#[test]
fn bin_centered_tone_localizes_without_upward_frequency_bias() {
    let data = load(include_str!("fixtures/reassignment_reference.csv"));
    let y = signal(&data, "input_tone");
    let raster = reassigned_spectrogram(&y, 1024)
        .n_fft(64)
        .compute()
        .unwrap();
    let mut total = 0.0_f64;
    let mut localized = 0.0_f64;
    for frame in 3..raster.ncols() - 3 {
        total += raster
            .column(frame)
            .iter()
            .copied()
            .map(f64::from)
            .sum::<f64>();
        localized += f64::from(raster[[8, frame]]);
    }
    assert!(
        localized / total > 0.999,
        "128 Hz tone localization fraction {}",
        localized / total
    );
}

#[test]
fn reassignment_rejects_unrepresentable_output_and_nonfinite_inputs() {
    let huge = vec![f32::MAX; 257];
    assert!(
        reassigned_spectrogram(&huge, 1024)
            .n_fft(64)
            .compute()
            .is_err()
    );
    for invalid in [f32::NAN, f32::INFINITY] {
        assert!(
            reassigned_spectrogram(&[invalid; 64], 1024)
                .n_fft(64)
                .compute()
                .is_err()
        );
    }
    assert!(
        reassigned_spectrogram(&[0.0; 64], 0)
            .n_fft(64)
            .compute()
            .is_err()
    );
    assert!(
        reassigned_spectrogram(&[0.0; 64], 1024)
            .n_fft(1)
            .compute()
            .is_err()
    );
}
