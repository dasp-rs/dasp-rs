//! Inverse CQT solves real least squares for the direct wavelet operator, using
//! 32 CG steps from zero and output length frames*hop. Forward reanalysis has one
//! extra frame; residuals below use exactly the original coefficient frame count.
//! This is not recursive librosa iCQT parity or a universal waveform SNR claim.
//! Narrow bands, sparse hops, original-length ambiguity, and limited CG iterations
//! can all limit waveform recovery. All empirical quality output is diagnostic_.
#[path = "common/reference.rs"]
mod reference;

use dasp_rs::proc::{cqt, icqt};
use ndarray::Array2;
use num_complex::Complex;
use reference::{check_real, load};

#[derive(Clone, Copy, Debug)]
struct Geometry {
    sr: u32,
    hop: usize,
    fmin: f32,
    bins: usize,
}

impl Geometry {
    fn analyze(self, signal: &[f32]) -> Array2<Complex<f32>> {
        cqt(signal, self.sr)
            .hop_length(self.hop)
            .fmin(self.fmin)
            .n_bins(self.bins)
            .compute()
            .unwrap()
    }

    fn invert(self, coefficients: &Array2<Complex<f32>>) -> Vec<f32> {
        let output = icqt(coefficients)
            .sample_rate(self.sr)
            .hop_length(self.hop)
            .fmin(self.fmin)
            .compute()
            .unwrap();
        assert_eq!(output.len(), coefficients.ncols() * self.hop);
        assert!(output.iter().all(|value| value.is_finite()));
        output
    }
}

fn squared_norm(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|v| v * v).sum()
}

fn coefficient_norm(coefficients: &Array2<Complex<f32>>) -> f64 {
    coefficients
        .iter()
        .map(|z| f64::from(z.re).powi(2) + f64::from(z.im).powi(2))
        .sum()
}

fn objective(geometry: Geometry, output: &[f32], target: &Array2<Complex<f32>>) -> f64 {
    let rebuilt = geometry.analyze(output);
    assert_eq!(rebuilt.ncols(), target.ncols() + 1);
    target
        .indexed_iter()
        .map(|(index, &c)| {
            let value = rebuilt[index];
            (f64::from(value.re) - f64::from(c.re)).powi(2)
                + (f64::from(value.im) - f64::from(c.im)).powi(2)
        })
        .sum()
}

#[test]
fn varied_small_systems_match_independent_svd_minimizers() {
    let references = load(include_str!("fixtures/cqt_inverse_reference.csv"));
    for name in [
        "short_hop2",
        "inconsistent_hop3",
        "short_hop5",
        "partial_inconsistent",
    ] {
        let metadata = &references[&format!("{name}_metadata")];
        let geometry = Geometry {
            sr: metadata[[0, 0]].re as u32,
            hop: metadata[[0, 2]].re as usize,
            fmin: metadata[[0, 3]].re as f32,
            bins: metadata[[0, 4]].re as usize,
        };
        let input =
            references[&format!("{name}_input")].mapv(|z| Complex::new(z.re as f32, z.im as f32));
        let output = geometry.invert(&input);
        // These full-column-rank systems have independently measured condition
        // numbers 1.67, 3.74, 6.07 and 29.64. They fit within the 32-step budget.
        check_real(
            &format!("{name}_solution"),
            &Array2::from_shape_vec((1, output.len()), output.clone()).unwrap(),
            &references,
            1e-7,
            2e-7,
        );
        let actual = objective(geometry, &output, &input);
        let optimum = metadata[[0, 7]].re;
        let energy = metadata[[0, 8]].re;
        // Independent optimum, with allowance for public complex32 reanalysis.
        assert!(
            actual <= optimum + energy * 5e-9,
            "{name}: objective={actual} optimum={optimum}"
        );
        eprintln!(
            "diagnostic_svd_{name}: condition={:.6e} relative_residual={:.6e} optimal_relative_residual={:.6e} normalized_objective_gap={:.6e}",
            metadata[[0, 6]].re,
            (actual / energy).sqrt(),
            (optimum / energy).sqrt(),
            (actual - optimum) / energy
        );
    }
}

#[test]
fn finite_budget_case_reports_distance_to_the_svd_optimum() {
    let references = load(include_str!("fixtures/cqt_inverse_reference.csv"));
    let input = references["budget72_input"].mapv(|z| Complex::new(z.re as f32, z.im as f32));
    let metadata = &references["budget72_metadata"];
    let geometry = Geometry {
        sr: 2048,
        hop: 6,
        fmin: 64.0,
        bins: 48,
    };
    let output = geometry.invert(&input);
    let actual = objective(geometry, &output, &input);
    let energy = coefficient_norm(&input);
    let optimum = metadata[[0, 7]].re;
    assert!(
        actual <= energy * (1.0 + 2e-6),
        "CG must improve its zero initializer"
    );
    let reference = &references["budget72_solution"];
    let waveform_error = squared_norm(
        output
            .iter()
            .enumerate()
            .map(|(i, &y)| f64::from(y) - reference[[0, i]].re),
    );
    eprintln!(
        "diagnostic_budget72: condition={:.6e} relative_residual={:.6e} optimal_relative_residual={:.6e} normalized_objective_gap={:.6e} waveform_relative_l2_vs_svd={:.6e}",
        metadata[[0, 6]].re,
        (actual / energy).sqrt(),
        (optimum / energy).sqrt(),
        (actual - optimum) / energy,
        (waveform_error / metadata[[0, 9]].re).sqrt()
    );
}

fn signal_for(kind: &str, length: usize, geometry: Geometry) -> Vec<f32> {
    let in_band = f64::from(geometry.fmin) * 2.0_f64.powf(7.0 / 12.0);
    let out_band = f64::from(geometry.fmin) / 4.0;
    let mut state = 0x9e37_79b9_u32;
    (0..length)
        .map(|index| {
            let time = index as f64 / f64::from(geometry.sr);
            let inside = (2.0 * std::f64::consts::PI * in_band * time + 0.37).cos();
            let outside = (2.0 * std::f64::consts::PI * out_band * time - 0.12).cos();
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0;
            match kind {
                "in_band" => (0.6 * inside) as f32,
                "out_band" => (0.6 * outside) as f32,
                "mixture" => (0.4 * inside + 0.3 * outside + 0.1 * noise) as f32,
                "transient" => {
                    if index == 0 {
                        1.0
                    } else if index == length / 2 {
                        -0.7
                    } else {
                        0.0
                    }
                }
                "noise" => noise as f32,
                _ => unreachable!(),
            }
        })
        .collect()
}

#[test]
fn sweep_hop_band_length_and_signal_reports_objective_and_waveform_metrics() {
    // Lengths immediately below/on/above hop boundaries expose output-length
    // ambiguity. Band widths/hops vary independently enough to show loss of
    // observability; no universal waveform quality threshold is imposed.
    for (length, hop, fmin, bins) in [
        (127, 8, 128.0, 12),
        (128, 32, 128.0, 24),
        (129, 64, 128.0, 36),
        (255, 16, 64.0, 36),
        (256, 64, 256.0, 12),
        (257, 96, 128.0, 24),
    ] {
        let geometry = Geometry {
            sr: 2048,
            hop,
            fmin,
            bins,
        };
        for kind in ["in_band", "out_band", "mixture", "transient", "noise"] {
            let signal = signal_for(kind, length, geometry);
            let coefficients = geometry.analyze(&signal);
            let output = geometry.invert(&coefficients);
            let cost = objective(geometry, &output, &coefficients);
            let energy = coefficient_norm(&coefficients);
            // For this convex least-squares objective, even an incomplete solve
            // must not be worse than the zero starting point (within f32 error).
            assert!(
                cost <= energy * (1.0 + 2e-6),
                "{geometry:?} {kind}: zero-start objective increased"
            );
            let waveform_energy = squared_norm(signal.iter().map(|&v| f64::from(v)));
            let waveform_error = squared_norm(
                signal
                    .iter()
                    .zip(&output)
                    .map(|(&a, &b)| f64::from(a) - f64::from(b)),
            );
            let gain = signal
                .iter()
                .zip(&output)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
                / waveform_energy;
            let tail = squared_norm(output[length..].iter().map(|&v| f64::from(v)));
            eprintln!(
                "diagnostic_sweep: kind={kind} length={length} hop={hop} fmin={fmin} bins={bins} output_length={} relative_coefficient_residual={:.6e} waveform_gain={gain:.6e} waveform_snr_db={:.3} tail_energy_fraction={:.6e}",
                output.len(),
                (cost / energy).sqrt(),
                10.0 * (waveform_energy / waveform_error).log10(),
                tail / waveform_energy
            );
        }
    }
}

#[test]
fn inverse_gain_is_equivariant_over_finite_amplitude_scales() {
    let geometry = Geometry {
        sr: 2048,
        hop: 16,
        fmin: 128.0,
        bins: 24,
    };
    let signal = signal_for("mixture", 129, geometry);
    let coefficients = geometry.analyze(&signal);
    let base = geometry.invert(&coefficients);
    let base_energy = squared_norm(base.iter().map(|&v| f64::from(v)));
    for scale in [1e-30_f32, 1e-12, 1.0, 1e12, 1e30, -2.0] {
        let scaled = coefficients.mapv(|z| z * scale);
        let actual = geometry.invert(&scaled);
        let relative_error = (squared_norm(
            actual
                .iter()
                .zip(&base)
                .map(|(&a, &b)| f64::from(a) / f64::from(scale) - f64::from(b)),
        ) / base_energy)
            .sqrt();
        assert!(
            relative_error < 2e-6,
            "CG gain equivariance: scale={scale} relative_error={relative_error}"
        );
        eprintln!(
            "diagnostic_gain: scale={scale:.3e} normalized_waveform_relative_l2={relative_error:.6e}"
        );
    }
    let zero = Array2::zeros(coefficients.dim());
    assert!(geometry.invert(&zero).iter().all(|&value| value == 0.0));
}

#[test]
fn ill_conditioned_inconsistent_coefficients_do_not_worsen_zero_objective() {
    // A near-null direction can produce enormous f64 iterates whose final f32
    // rounding destroys coefficient cancellation. Waveform parity is undefined;
    // doing no worse than the zero initial guess is still a meaningful contract.
    for (sr, fmin) in [(32, 1.0), (4096, 128.0)] {
        let geometry = Geometry {
            sr,
            hop: 4,
            fmin,
            bins: 2,
        };
        for scale in [1.0_f32, 1e-30, 1e30] {
            let input = Array2::from_shape_fn((2, 5), |(i, j)| {
                Complex::new(
                    scale * ((17 * i + 3 * j) % 13) as f32 / 13.0,
                    scale * ((11 * i + 7 * j) % 17) as f32 / 17.0,
                )
            });
            let output = geometry.invert(&input);
            let cost = objective(geometry, &output, &input);
            let energy = coefficient_norm(&input);
            let peak = output
                .iter()
                .map(|&v| f64::from(v).abs())
                .fold(0.0_f64, f64::max);
            eprintln!(
                "diagnostic_ill_conditioned_inconsistent: sr={sr} scale={scale:.3e} relative_coefficient_residual={:.6e} waveform_peak={peak:.6e}",
                (cost / energy).sqrt()
            );
            assert!(
                cost <= energy * (1.0 + 2e-6),
                "representable output must not worsen zero-start objective: cost={cost} zero_cost={energy}"
            );
        }
    }
}

#[test]
fn finite_progress_is_retained_when_later_iterates_overflow() {
    // At this scale the near-null-direction iterates eventually exceed f32,
    // but the first useful iterates are representable and lower the objective.
    // A later failed conversion must not discard that useful finite result.
    let geometry = Geometry {
        sr: 32,
        hop: 4,
        fmin: 1.0,
        bins: 2,
    };
    let scale = 3e35_f32;
    let input = Array2::from_shape_fn((2, 5), |(i, j)| {
        Complex::new(
            scale * ((17 * i + 3 * j) % 13) as f32 / 13.0,
            scale * ((11 * i + 7 * j) % 17) as f32 / 17.0,
        )
    });
    let output = geometry.invert(&input);
    let cost = objective(geometry, &output, &input);
    let energy = coefficient_norm(&input);
    assert!(
        cost < energy,
        "Retained reconstruction must improve zero-start objective"
    );
    eprintln!(
        "diagnostic_later_overflow: relative_coefficient_residual={:.6e}",
        (cost / energy).sqrt()
    );
}
