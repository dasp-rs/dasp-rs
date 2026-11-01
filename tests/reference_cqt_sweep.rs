//! Forward wavelet sweeps against independently computed librosa filters and
//! `SciPy` correlation. See `fixtures/generate_cqt_sweep_reference.py` for pinned
//! versions, source URLs, selected frames, and input quantization conventions.
//! Recursive librosa comparisons are measurements, never loose parity assertions.
#[path = "common/reference.rs"]
mod reference;
use dasp_rs::proc::{cqt, vqt};
use ndarray::{Array2, Axis};
use num_complex::Complex;
use reference::{References, check_complex, load, signal};

fn evaluate(data: &References, name: &str, transform: &str) -> Array2<Complex<f32>> {
    let config = &data[&format!("{name}_config")];
    let sr = config[[0, 0]].re as u32;
    let fmin = config[[0, 1]].re as f32;
    let bins = config[[0, 2]].re as usize;
    let hop = config[[0, 3]].re as usize;
    let y = signal(data, &format!("{name}_input"));
    let result = if transform == "cqt" {
        cqt(&y, sr)
            .hop_length(hop)
            .fmin(fmin)
            .n_bins(bins)
            .compute()
    } else {
        vqt(&y, sr)
            .hop_length(hop)
            .fmin(fmin)
            .n_bins(bins)
            .compute()
    }
    .unwrap_or_else(|error| panic!("{transform}_{name}: {error}"));
    assert_eq!(result.dim(), (bins, 1 + y.len() / hop));
    assert!(result.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
    result
}

#[test]
fn forward_geometry_signal_and_dynamic_range_sweep() {
    let data = load(include_str!("fixtures/cqt_sweep_reference.csv"));
    let mut count = 0;
    for key in data.keys().filter(|key| key.ends_with("_config")) {
        let name = key.strip_suffix("_config").unwrap();
        let amplitude = data[key][[0, 4]].re;
        let indices: Vec<_> = signal(&data, &format!("{name}_frames"))
            .into_iter()
            .map(|v| v as usize)
            .collect();
        for transform in ["cqt", "vqt"] {
            let actual = evaluate(&data, name, transform);
            let selected = actual.select(Axis(1), &indices);
            // A few f32 rounding units plus an amplitude-scaled cancellation
            // floor. Two subnormal quanta cover rounding at 1e-40 input levels.
            // Scaling the floor, instead of using a fixed 1e-6, prevents tiny
            // signals from passing merely because zero is below the tolerance.
            let atol = (amplitude * 1e-6).max(2.0 * f64::from(f32::from_bits(1)));
            check_complex(&format!("{transform}_{name}"), &selected, &data, atol, 2e-7);
            count += 1;
        }
    }
    assert_eq!(count, 28);
}

#[test]
fn calibrated_tones_preserve_gain_phase_and_frequency_localization() {
    let data = load(include_str!("fixtures/cqt_sweep_reference.csv"));
    for name in ["tone_16k", "tone_44k", "tone_48k", "tone_single_bin"] {
        let bin = data[&format!("{name}_config")][[0, 5]].re as usize;
        for transform in ["cqt", "vqt"] {
            let actual = evaluate(&data, name, transform);
            let middle = actual.ncols() / 2;
            let peak = actual
                .column(middle)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.norm_sqr().total_cmp(&b.1.norm_sqr()))
                .unwrap()
                .0;
            assert_eq!(peak, bin, "{transform}_{name} tone localization");
            let measured = actual[[bin, middle]];
            let measured = Complex::new(f64::from(measured.re), f64::from(measured.im));
            let ideal = data[&format!("{transform}_{name}_calibration")][[0, 0]];
            let response = measured / ideal;
            let gain_error = (response.norm() - 1.0).abs();
            let phase_error = response.arg().abs();
            eprintln!(
                "REFERENCE calibrated_{transform}_{name}: peak={peak} relative_gain_error={gain_error:.6e} phase_error_radians={phase_error:.6e}"
            );
            // A real cosine also has a negative-frequency image. Finite Hann
            // support leaves small image leakage; 5e-4 bounds that physical
            // deviation from the ideal analytic sinusoid (-66 dB).
            assert!(
                gain_error < 5e-4,
                "{transform}_{name} gain error {gain_error}"
            );
            assert!(
                phase_error < 5e-4,
                "{transform}_{name} phase error {phase_error}"
            );
        }
    }
}

#[test]
fn report_recursive_librosa_comparisons_separately_from_full_rate_parity() {
    let data = load(include_str!("fixtures/cqt_sweep_reference.csv"));
    for name in [
        "tone_16k",
        "tone_48k",
        "noise_16k",
        "transient_44k",
        "chirp_8k",
    ] {
        let indices: Vec<_> = signal(&data, &format!("{name}_frames"))
            .into_iter()
            .map(|v| v as usize)
            .collect();
        for transform in ["cqt", "vqt"] {
            let actual = evaluate(&data, name, transform).select(Axis(1), &indices);
            let expected = &data[&format!("recursive_librosa_{transform}_{name}")];
            let interior = signal(&data, &format!("{transform}_{name}_interior_mask"));
            // This independent mask uses the complete nonzero full-rate filter
            // support: short inputs can legitimately have no interior frames.
            for (region, is_interior) in [("interior", true), ("boundary", false)] {
                let columns: Vec<_> = interior
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| (**v > 0.5) == is_interior)
                    .map(|(i, _)| i)
                    .collect();
                if columns.is_empty() {
                    eprintln!(
                        "REFERENCE recursive_librosa_{transform}_{name}_{region}: selected_frames=0 (no complete-filter-support interior; comparison only)"
                    );
                    continue;
                }
                let mut squared_error = 0.0;
                let mut squared_energy = 0.0;
                let mut max_abs = 0.0_f64;
                for column in &columns {
                    for (a, b) in actual.column(*column).iter().zip(expected.column(*column)) {
                        let delta = (Complex::new(f64::from(a.re), f64::from(a.im)) - b).norm();
                        squared_error += delta * delta;
                        squared_energy += b.norm_sqr();
                        max_abs = max_abs.max(delta);
                    }
                }
                let relative_l2 = (squared_error / squared_energy).sqrt();
                eprintln!(
                    "REFERENCE recursive_librosa_{transform}_{name}_{region}: selected_frames={} relative_l2={relative_l2:.6e} max_abs={max_abs:.6e} (comparison only)",
                    columns.len()
                );
                assert!(relative_l2.is_finite());
            }
        }
    }
}

#[test]
fn tiny_signal_with_extremely_long_wavelet_retains_full_filter_scale() {
    // A full kernel here spans about 764 million taps (>12 GB as complex f64).
    // The centered one-sample analysis needs exactly one of them. This case is
    // deliberately run only after bounded-support construction was introduced.
    let fmin = 0.001_f32;
    let actual = cqt(&[1.0], 44_100).fmin(fmin).n_bins(1).compute().unwrap();
    assert_eq!(actual.dim(), (1, 1));
    let value = actual[[0, 0]];
    assert!(value.re.is_finite() && value.im.is_finite());
    // Analytic Hann center amplitude: full L1 norm N/2 and sqrt(L) scaling
    // approach 2/sqrt(L), with O(1/L) discretization error. This catches the
    // incorrect alternative of renormalizing the single retained tap to unity.
    let ratio = 2.0_f64.powf(2.0 / 12.0);
    let alpha = (ratio - 1.0) / (ratio + 1.0);
    let length = 44_100.0 / (alpha * f64::from(fmin));
    let expected = 2.0 / length.sqrt();
    let relative_error = (f64::from(value.re) / expected - 1.0).abs();
    eprintln!(
        "REFERENCE cqt_extreme_support: full_length={length:.6e} relative_center_scale_error={relative_error:.6e}"
    );
    assert!(relative_error < 1e-6);
    assert!(value.im.abs() < f32::EPSILON);
}
