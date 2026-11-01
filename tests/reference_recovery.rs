//! Phase recovery and NMF contracts: numerical accuracy, scale behavior and
//! reconstruction objectives. Phase is not uniquely determined by magnitude;
//! references explicitly match the deterministic initialization and iteration.
#[path = "common/reference.rs"]
mod reference;

use dasp_rs::{feat, proc};
use ndarray::Array2;

#[test]
fn griffin_lim_matches_independent_matched_iterations() {
    let data = reference::load(include_str!("fixtures/recovery_reference.csv"));
    for name in ["small", "ordinary", "short"] {
        let hop = data[&format!("{name}_metadata")][[0, 1]].re as usize;
        let magnitude = reference::real(&data, &format!("{name}_input"));
        for iterations in [1, 4, 12] {
            let actual = feat::griffinlim(&magnitude)
                .hop_length(hop)
                .n_iter(iterations)
                .compute()
                .unwrap();
            // Matching the broadband initial state makes phase comparisons
            // meaningful; allow accumulated f32 FFT/iteration roundoff.
            reference::check_real(
                &format!("{name}_iteration{iterations}"),
                &Array2::from_shape_vec((1, actual.len()), actual).unwrap(),
                &data,
                2e-6,
                2e-5,
            );
        }
    }
}

#[test]
fn nmf_recovers_independently_generated_dense_low_rank_input() {
    let data = reference::load(include_str!("fixtures/recovery_reference.csv"));
    let target = reference::real(&data, "nmf_dense_input");
    for seed in [0, 73] {
        let mut previous = f64::INFINITY;
        for iterations in [1, 5, 30, 200, 800] {
            let (w, h) = feat::decompose(&target, 3)
                .random_seed(seed)
                .n_iter(iterations)
                .compute()
                .unwrap();
            let reconstructed = w.mapv(f64::from).dot(&h.mapv(f64::from));
            let error = relative_error(
                reconstructed.iter().copied(),
                target.iter().map(|&v| f64::from(v)),
            );
            assert!(error <= previous + 2e-7);
            previous = error;
        }
        eprintln!("REFERENCE nmf_dense_seed{seed}: relative_reconstruction_error={previous:.6e}");
        assert!(previous < 0.01);
    }
}

fn relative_error(actual: impl Iterator<Item = f64>, expected: impl Iterator<Item = f64>) -> f64 {
    let (mut error, mut energy) = (0.0, 0.0);
    for (a, b) in actual.zip(expected) {
        error += (a - b).powi(2);
        energy += b * b;
    }
    (error / energy).sqrt()
}

#[test]
fn griffin_lim_retains_quiet_signal_phase_and_gain() {
    let y: Vec<f32> = (0..256)
        .map(|i| (i as f32 * 0.17).sin() * 0.6 + (i as f32 * 0.41).cos() * 0.2)
        .collect();
    let magnitude = proc::stft(&y)
        .n_fft(64)
        .hop_length(16)
        .compute()
        .unwrap()
        .mapv(num_complex::Complex::norm);
    let normal = feat::griffinlim(&magnitude)
        .n_iter(8)
        .hop_length(16)
        .compute()
        .unwrap();
    for scale in [1e-25_f32, 1e-12, 1e12, 1e25] {
        let scaled = feat::griffinlim(&magnitude.mapv(|v| v * scale))
            .n_iter(8)
            .hop_length(16)
            .compute()
            .unwrap();
        assert!(scaled.iter().all(|v| v.is_finite()));
        let error = relative_error(
            scaled.iter().map(|&v| f64::from(v) / f64::from(scale)),
            normal.iter().map(|&v| f64::from(v)),
        );
        eprintln!("REFERENCE recovery_gain_{scale:e}: relative_waveform_error={error:.6e}");
        assert!(
            error < 2e-4,
            "amplitude scale {scale}: relative error {error}"
        );
    }
}

#[test]
fn cqt_phase_recovery_preserves_gain_on_a_dense_small_system() {
    let y: Vec<f32> = (0..24)
        .map(|i| (i as f32 * 0.31).sin() * 0.6 + (i as f32 * 0.77).cos() * 0.2)
        .collect();
    let magnitude = proc::cqt(&y, 2048)
        .hop_length(1)
        .fmin(64.0)
        .n_bins(48)
        .compute()
        .unwrap()
        .mapv(num_complex::Complex::norm);
    let recover = |s: &Array2<f32>| {
        feat::griffinlim_cqt(s, 2048)
            .hop_length(1)
            .fmin(64.0)
            .n_iter(4)
            .compute()
            .unwrap()
    };
    let normal = recover(&magnitude);
    assert_eq!(normal.len(), y.len());
    for scale in [1e-20_f32, 1e20] {
        let scaled = recover(&magnitude.mapv(|v| v * scale));
        assert!(scaled.iter().all(|v| v.is_finite()));
        let error = relative_error(
            scaled.iter().map(|&v| f64::from(v) / f64::from(scale)),
            normal.iter().map(|&v| f64::from(v)),
        );
        eprintln!("REFERENCE recovery_cqt_gain_{scale:e}: relative_waveform_error={error:.6e}");
        assert!(error < 2e-4, "scaled dense CQT recovery error {error}");
    }
}

#[test]
fn cqt_phase_recovery_validates_bands_even_without_iterations_or_output() {
    for (frames, iterations) in [(1, 4), (2, 0)] {
        let magnitude = Array2::zeros((84, frames));
        assert!(
            feat::griffinlim_cqt(&magnitude, 2048)
                .fmin(128.0)
                .hop_length(8)
                .n_iter(iterations)
                .compute()
                .is_err()
        );
    }
}

#[test]
fn nmf_objective_decreases_and_rank_two_structure_is_recovered() {
    let w = Array2::from_shape_fn((8, 2), |(i, k)| {
        if i / 4 == k {
            0.3 + i as f32 * 0.1
        } else {
            0.02
        }
    });
    let h = Array2::from_shape_fn((2, 12), |(k, j)| {
        if j / 6 == k {
            0.4 + j as f32 * 0.03
        } else {
            0.01
        }
    });
    let matrix = w.dot(&h);
    for seed in [0, 17, u64::MAX] {
        for scale in [1e-25_f32, 1.0, 1e25] {
            let target = matrix.mapv(|v| v * scale);
            let mut previous = f64::INFINITY;
            for iterations in [1, 2, 5, 20, 100, 300] {
                let (w, h) = feat::decompose(&target, 2)
                    .random_seed(seed)
                    .n_iter(iterations)
                    .compute()
                    .unwrap();
                assert!(w.iter().chain(&h).all(|v| v.is_finite() && *v >= 0.0));
                // Evaluate factors in f64 so verification cannot introduce f32
                // overflow or underflow at the tested input scales.
                let rebuilt = w.mapv(f64::from).dot(&h.mapv(f64::from));
                let error = relative_error(
                    rebuilt.iter().copied(),
                    target.iter().map(|&v| f64::from(v)),
                );
                assert!(
                    error <= previous + 2e-7,
                    "seed={seed} scale={scale} iterations={iterations}: {error} > {previous}"
                );
                previous = error;
            }
            eprintln!(
                "REFERENCE nmf_rank2_{seed}_{scale:e}: relative_reconstruction_error={previous:.6e}"
            );
            assert!(previous < 0.01, "rank-two reconstruction error {previous}");
        }
    }
}
