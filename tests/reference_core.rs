//! Independent SciPy/librosa references; see the generator for exact conventions.
#[path = "common/reference.rs"]
mod reference;

use dasp_rs::{feat, mag, proc, types::AudioData};
use ndarray::Array2;
use reference::{check_real, load, real, signal};

fn row(values: Vec<f32>) -> Array2<f32> {
    Array2::from_shape_vec((1, values.len()), values).unwrap()
}

#[test]
fn lpc_and_autocorrelation_match_scipy_across_signals_orders_and_scales() {
    let data = load(include_str!("fixtures/core_reference.csv"));
    for name in ["noise", "voiced", "ar", "impulse"] {
        for scale in ["normal", "quiet", "loud"] {
            let key = format!("lpc_{name}_{scale}");
            let audio = AudioData::new(signal(&data, &(key.clone() + "_input")), 8000, 1).unwrap();
            for order in [1, 4, 12] {
                // f32 output rounding, plus conditioning of the voiced Toeplitz systems.
                check_real(
                    &format!("{key}_order{order}"),
                    &row(proc::lpc(&audio, order).unwrap()),
                    &data,
                    2e-6,
                    2e-6,
                );
            }
            // Include subnormal autocorrelation outputs without an amplitude-sized floor.
            check_real(
                &(key + "_autocorrelation"),
                &row(proc::autocorrelate(&audio, Some(13)).unwrap()),
                &data,
                1e-45,
                2e-6,
            );
        }
    }
}

#[test]
fn magnitude_scaling_and_weighting_match_librosa() {
    let data = load(include_str!("fixtures/core_reference.csv"));
    let amplitude = real(&data, "amplitude_input");
    let power = real(&data, "power_input");
    let db = real(&data, "db_input");
    check_real(
        "amplitude_db",
        &mag::amplitude_to_db(&amplitude)
            .ref_val(0.5)
            .amin(1e-5)
            .top_db(72.0)
            .compute()
            .unwrap(),
        &data,
        1e-5,
        2e-6,
    );
    check_real(
        "power_db",
        &mag::power_to_db(&power)
            .ref_val(0.25)
            .amin(1e-10)
            .top_db(72.0)
            .compute()
            .unwrap(),
        &data,
        1e-5,
        2e-6,
    );
    check_real(
        "db_amplitude",
        &mag::db_to_amplitude(&db, Some(0.5)).unwrap(),
        &data,
        1e-8,
        2e-6,
    );
    check_real(
        "db_power",
        &mag::db_to_power(&db, Some(0.25)).unwrap(),
        &data,
        1e-10,
        2e-6,
    );
    let frequencies = signal(&data, "weighting_input");
    for kind in ["A", "B", "C", "D"] {
        // librosa's D curve uses 1018.7^2 = 1037749.69 where this crate's
        // coefficient set uses 1037918.5, and other rounded poles also differ.
        // Retain this visible cross-implementation difference (under 0.03%),
        // rather than replacing coefficients just to fit the reference.
        let output = mag::frequency_weighting(&frequencies, Some(kind)).unwrap();
        let rtol = if kind == "D" { 3e-4 } else { 2e-5 };
        check_real(
            &format!("weighting_{kind}"),
            &row(output),
            &data,
            1e-7,
            rtol,
        );
    }
}

#[test]
fn pcen_matches_librosa_with_matching_initial_smoother_state() {
    let data = load(include_str!("fixtures/core_reference.csv"));
    for scale in ["normal", "quiet", "loud"] {
        let input = real(&data, &format!("pcen_{scale}_input"));
        for (sr, hop, gain, bias) in [(8000, 64, 0.98, 2.0), (22050, 256, 0.5, 0.0)] {
            let output = mag::pcen(&input)
                .sample_rate(sr)
                .hop_length(hop)
                .gain(gain)
                .bias(bias)
                .compute()
                .unwrap();
            check_real(&format!("pcen_{scale}_{sr}"), &output, &data, 1e-13, 3e-6);
        }
    }
}

#[test]
fn continuous_mu_law_matches_librosa() {
    let data = load(include_str!("fixtures/core_reference.csv"));
    let input = AudioData::new(signal(&data, "mu_input"), 8000, 1).unwrap();
    for mu in [1, 255] {
        check_real(
            &format!("mu_compress_{mu}"),
            &row(proc::mu_compress(&input).mu(mu as f32).compute().unwrap()),
            &data,
            1e-14,
            2e-6,
        );
        check_real(
            &format!("mu_expand_{mu}"),
            &row(proc::mu_expand(&input).mu(mu as f32).compute().unwrap()),
            &data,
            1e-14,
            2e-6,
        );
    }
}

#[test]
fn dtw_costs_and_paths_match_librosa() {
    let data = load(include_str!("fixtures/core_reference.csv"));
    for case in 0..3 {
        let x = real(&data, &format!("dtw_{case}_x"));
        let y = real(&data, &format!("dtw_{case}_y"));
        for (name, metric) in [
            ("euclidean", feat::DtwMetric::Euclidean),
            ("cityblock", feat::DtwMetric::Manhattan),
            ("cosine", feat::DtwMetric::Cosine),
        ] {
            let (cost, path) = feat::dtw(&x, &y).metric(metric).compute().unwrap();
            check_real(
                &format!("dtw_{case}_{name}_cost"),
                &row(vec![cost]),
                &data,
                1e-6,
                2e-6,
            );
            let path = Array2::from_shape_vec(
                (path.len(), 2),
                path.into_iter()
                    .flat_map(|(i, j)| [i as f32, j as f32])
                    .collect(),
            )
            .unwrap();
            check_real(&format!("dtw_{case}_{name}_path"), &path, &data, 0.0, 0.0);
        }
    }
}

#[test]
fn viterbi_scores_and_paths_match_librosa_after_initial_prior_adjustment() {
    let data = load(include_str!("fixtures/core_reference.csv"));
    for case in 0..3 {
        let emissions = real(&data, &format!("viterbi_{case}_emissions")).mapv(f32::ln);
        let transition = real(&data, &format!("viterbi_{case}_transition")).mapv(f32::ln);
        let (score, path) = feat::viterbi(&emissions, &transition).unwrap();
        check_real(
            &format!("viterbi_{case}_score"),
            &row(vec![score]),
            &data,
            1e-6,
            3e-6,
        );
        check_real(
            &format!("viterbi_{case}_path"),
            &row(path.into_iter().map(|v| v as f32).collect()),
            &data,
            0.0,
            0.0,
        );
    }
}
