//! Independent full-rate wavelet CQT/VQT reference benchmarks.
//!
//! References are librosa 0.11.0 wavelets correlated by `SciPy`, not Rust-produced
//! goldens. See `fixtures/generate_cqt_reference.py` for conventions and sources.
//! Production librosa.cqt/vqt recursively resample, so their remaining difference
//! is reported separately and is not hidden inside a loose parity tolerance.
#[path = "common/reference.rs"]
mod reference;
use dasp_rs::proc::{cqt, hybrid_cqt, icqt, pseudo_cqt, vqt};
use ndarray::{Array2, s};
use num_complex::Complex;
use reference::{check_complex, check_real, load, signal};

#[test]
fn cqt_vqt_match_independent_full_rate_wavelet_analysis() {
    let data = load(include_str!("fixtures/cqt_reference.csv"));
    for name in ["mix", "edge", "impulse", "tone"] {
        let y = signal(&data, &format!("input_{name}"));
        let hop = if name == "tone" { 256 } else { 32 };
        let c = cqt(&y, 4096)
            .hop_length(hop)
            .fmin(128.0)
            .n_bins(24)
            .compute()
            .unwrap();
        let v = vqt(&y, 4096)
            .hop_length(hop)
            .fmin(128.0)
            .n_bins(24)
            .compute()
            .unwrap();
        // Double accumulation followed by f32 output: absolute floor protects
        // cancellation bins; relative allowance is a few f32 rounding units.
        check_complex(&format!("cqt_{name}"), &c, &data, 1e-6, 2e-7);
        check_complex(&format!("vqt_{name}"), &v, &data, 1e-6, 2e-7);
    }
}

#[test]
fn calibrated_tone_localization_selectivity_and_recursive_librosa_gap() {
    let data = load(include_str!("fixtures/cqt_reference.csv"));
    let y = signal(&data, "input_tone");
    for (name, actual) in [
        (
            "cqt",
            cqt(&y, 4096)
                .hop_length(256)
                .fmin(128.0)
                .n_bins(24)
                .compute()
                .unwrap(),
        ),
        (
            "vqt",
            vqt(&y, 4096)
                .hop_length(256)
                .fmin(128.0)
                .n_bins(24)
                .compute()
                .unwrap(),
        ),
    ] {
        let reference = &data[&format!("librosa_{name}_tone")];
        let peak = |values: Vec<f64>| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        let amplitudes: Vec<_> = actual.column(8).mapv(|z| f64::from(z.norm())).to_vec();
        let ref_amplitudes = reference.column(8).mapv(Complex::norm).to_vec();
        assert_eq!(peak(amplitudes.clone()), 7);
        assert_eq!(peak(ref_amplitudes), 7);
        // CQT resolves semitone bands; VQT is intentionally broader at low f.
        let rejection_db = 20.0 * (amplitudes[10] / amplitudes[7]).log10();
        if name == "cqt" {
            assert!(rejection_db < -35.0, "{rejection_db}");
        }
        let mut error = 0.0;
        let mut energy = 0.0;
        let mut max_error = 0.0_f64;
        for (a, b) in actual
            .slice(s![.., 2..15])
            .iter()
            .zip(reference.slice(s![.., 2..15]))
        {
            let d = (Complex::new(f64::from(a.re), f64::from(a.im)) - b).norm();
            error += d * d;
            energy += b.norm_sqr();
            max_error = max_error.max(d);
        }
        let relative_l2 = (error / energy).sqrt();
        eprintln!(
            "REFERENCE recursive_librosa_{name}: interior_relative_l2={relative_l2:.6e} max_abs={max_error:.6e} tone_peak=7 third_semitone_db={rejection_db:.3}"
        );
        // This is a localization contract, not an elementwise parity claim.
        // Tight direct-reference comparisons above guard gain, phase, and timing.
    }
}

#[test]
fn inverse_matches_independent_svd_solution() {
    let data = load(include_str!("fixtures/cqt_reference.csv"));
    let c = data["inverse_input"].mapv(|z| Complex::new(z.re as f32, z.im as f32));
    let y = icqt(&c)
        .sample_rate(2048)
        .hop_length(1)
        .fmin(64.0)
        .compute()
        .unwrap();
    assert_eq!(y.len(), c.ncols());
    check_real(
        "inverse_solution",
        &Array2::from_shape_vec((1, y.len()), y).unwrap(),
        &data,
        1e-6,
        2e-6,
    );
}

#[test]
fn wavelets_reject_aliased_bands_and_accept_one_sample_with_zero_padding() {
    assert!(cqt(&[1.0], 4096).fmin(128.0).n_bins(24).compute().is_ok());
    assert!(vqt(&[1.0], 4096).fmin(128.0).n_bins(24).compute().is_ok());
    assert!(
        cqt(&[0.0; 32], 4096)
            .fmin(2000.0)
            .n_bins(1)
            .compute()
            .is_err()
    );
    assert!(
        vqt(&[0.0; 32], 4096)
            .fmin(128.0)
            .n_bins(usize::MAX)
            .compute()
            .is_err()
    );
}

#[test]
fn narrow_band_inverse_reports_conditioning_without_claiming_sample_parity() {
    let data = load(include_str!("fixtures/cqt_reference.csv"));
    let c = data["narrow_inverse_input"].mapv(|z| Complex::new(z.re as f32, z.im as f32));
    let y = icqt(&c)
        .sample_rate(2048)
        .hop_length(1)
        .fmin(256.0)
        .compute()
        .unwrap();
    let rebuilt = cqt(&y, 2048)
        .hop_length(1)
        .fmin(256.0)
        .n_bins(12)
        .compute()
        .unwrap();
    let rebuilt = rebuilt.slice(s![.., ..c.ncols()]).to_owned();
    // The SVD fixture has condition number 1.97e7: coefficient consistency is
    // meaningful, but a 32-step iterative inverse cannot promise sample parity.
    check_complex("narrow_inverse_input", &rebuilt, &data, 1e-6, 2e-6);
    let reference = signal(&data, "narrow_inverse_solution");
    let rmse = (y
        .iter()
        .zip(&reference)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / y.len() as f64)
        .sqrt();
    eprintln!(
        "REFERENCE narrow_inverse: waveform_rmse_vs_svd={rmse:.6e} condition=1.97215e7 (no waveform parity claim)"
    );
}

#[test]
fn inverse_reconstructs_an_in_band_calibrated_tone() {
    let data = load(include_str!("fixtures/cqt_reference.csv"));
    let y = signal(&data, "input_tone");
    // Dense time sampling avoids aliasing the filter envelopes. The input tone
    // is independently generated in Python and lies seven semitones above fmin.
    let c = cqt(&y, 4096)
        .hop_length(32)
        .fmin(128.0)
        .n_bins(24)
        .compute()
        .unwrap();
    let reconstructed = icqt(&c)
        .sample_rate(4096)
        .hop_length(32)
        .fmin(128.0)
        .compute()
        .unwrap();
    let mut error = 0.0;
    let mut energy = 0.0;
    for i in 512..y.len() - 512 {
        error += f64::from(y[i] - reconstructed[i]).powi(2);
        energy += f64::from(y[i]).powi(2);
    }
    let snr = 10.0 * (energy / error).log10();
    eprintln!("REFERENCE inverse_in_band_tone: interior_snr_db={snr:.3}");
    assert!(snr > 40.0, "In-band reconstruction SNR {snr:.3} dB");
}

#[test]
fn compatibility_wrappers_are_explicit_about_their_frequency_contract() {
    let y = [0.0_f32; 33];
    let full = cqt(&y, 16384).hop_length(8).compute().unwrap();
    let hybrid = hybrid_cqt(&y, 16384).hop_length(8).compute().unwrap();
    assert_eq!(full, hybrid);
    // Previously the out-of-Nyquist pseudo channels silently wrapped to DC.
    assert!(
        pseudo_cqt(&y, 4096)
            .fmin(128.0)
            .hop_length(8)
            .compute()
            .is_err()
    );
}

#[test]
fn singleton_vqt_window_matches_librosa() {
    let data = load(include_str!("fixtures/cqt_reference.csv"));
    let y = signal(&data, "input_singleton");
    let actual = vqt(&y, 32)
        .hop_length(1)
        .fmin(1.0)
        .n_bins(1)
        .compute()
        .unwrap();
    check_complex("vqt_singleton", &actual, &data, 1e-7, 2e-7);
}

#[test]
fn wavelet_overflow_returns_errors_instead_of_nonfinite_success() {
    let data = load(include_str!("fixtures/cqt_reference.csv"));
    let y: Vec<_> = signal(&data, "input_tone")
        .into_iter()
        .map(|v| v * f32::MAX)
        .collect();
    assert!(y.iter().all(|v| v.is_finite()));
    assert!(
        cqt(&y, 4096)
            .hop_length(256)
            .fmin(128.0)
            .n_bins(24)
            .compute()
            .is_err()
    );
    assert!(
        vqt(&y, 4096)
            .hop_length(256)
            .fmin(128.0)
            .n_bins(24)
            .compute()
            .is_err()
    );
    let c = Array2::from_elem((1, 1), Complex::new(f32::MAX, 0.0));
    assert!(
        icqt(&c)
            .sample_rate(4096)
            .hop_length(1)
            .fmin(128.0)
            .compute()
            .is_err()
    );
}
