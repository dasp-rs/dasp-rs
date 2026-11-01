//! Deterministic randomized contracts. Every failure includes seed/case geometry;
//! no external RNG, fixture generator, or dependency is required.
use dasp_rs::{feat, mag, proc, types::AudioData};
use ndarray::{Array1, Array2};
const SEED: u64 = 0x67d3_fa91_0b2c_4815;
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    fn size(&mut self, limit: usize) -> usize {
        self.next() as usize % limit
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16_777_216.0
    }
}
fn near(a: f64, b: f64, atol: f64, rtol: f64, context: &str) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= atol + rtol * b.abs(),
        "seed={SEED:x} {context}: {a} != {b}"
    );
}
#[test]
fn randomized_spectral_supplied_and_waveform_paths_agree() {
    let mut rng = Rng(SEED);
    for case in 0..36 {
        let n = 7 + rng.size(58);
        let hop = 1 + rng.size(n);
        let length = n * 3 + rng.size(n);
        let scale = [1e-9, 1.0, 1e9][case % 3];
        let y: Vec<f32> = (0..length)
            .map(|_| (2.0 * rng.unit() - 1.0) * scale)
            .collect();
        let audio = AudioData::new(y, 8000, 1).unwrap();
        let stft = proc::stft(&audio.samples)
            .n_fft(n)
            .hop_length(hop)
            .compute()
            .unwrap();
        let magnitude = stft.mapv(num_complex::Complex::norm);
        let power = stft.mapv(|z| z.norm().powi(2));
        let context = format!("case={case} n={n} hop={hop} scale={scale}");
        for (direct, supplied) in [
            (
                feat::spectral_centroid(&audio, None, Some(n), Some(hop)),
                feat::spectral_centroid(&audio, Some(&magnitude), Some(n), Some(hop)),
            ),
            (
                feat::spectral_bandwidth(&audio, None, Some(n), Some(hop), Some(3)),
                feat::spectral_bandwidth(&audio, Some(&magnitude), Some(n), Some(hop), Some(3)),
            ),
            (
                feat::spectral_rolloff(&audio, None, Some(n), Some(hop), Some(0.73)),
                feat::spectral_rolloff(&audio, Some(&magnitude), Some(n), Some(hop), Some(0.73)),
            ),
            (
                feat::spectral_flatness(&audio, None, Some(n), Some(hop)),
                feat::spectral_flatness(&audio, Some(&magnitude), Some(n), Some(hop)),
            ),
        ] {
            let direct = direct.unwrap();
            let supplied = supplied.unwrap();
            for (&a, &b) in direct.iter().zip(&supplied) {
                near(f64::from(a), f64::from(b), 0.0, 1e-6, &context);
            }
        }
        let fmin = rng.unit() * 300.0;
        let fmax = 2500.0 + rng.unit() * 1000.0;
        let bands = 1 + rng.size(8);
        let direct = feat::melspectrogram(
            &audio,
            None,
            Some(n),
            Some(hop),
            Some(bands),
            Some(fmin),
            Some(fmax),
        )
        .unwrap();
        let supplied = feat::melspectrogram(
            &audio,
            Some(&power),
            Some(n),
            Some(hop),
            Some(bands),
            Some(fmin),
            Some(fmax),
        )
        .unwrap();
        for (&a, &b) in direct.iter().zip(&supplied) {
            near(f64::from(a), f64::from(b), 0.0, 1e-6, &context);
        }
    }
}
#[test]
fn malformed_supplied_spectra_return_errors_instead_of_panicking_or_succeeding() {
    let audio = AudioData::new(vec![0.0; 64], 8000, 1).unwrap();
    let mut failures = Vec::new();
    for (case, spectrum) in [
        Array2::zeros((0, 2)),
        Array2::zeros((33, 0)),
        Array2::ones((32, 2)),
        Array2::ones((34, 2)),
        Array2::from_elem((33, 2), f32::NAN),
        Array2::from_elem((33, 2), f32::INFINITY),
        Array2::from_elem((33, 2), -1.0),
    ]
    .iter()
    .enumerate()
    {
        for feature in [
            "mel",
            "centroid",
            "bandwidth",
            "rolloff",
            "contrast",
            "flatness",
            "rms",
        ] {
            let result = std::panic::catch_unwind(|| match feature {
                "mel" => feat::melspectrogram(
                    &audio,
                    Some(spectrum),
                    Some(64),
                    Some(16),
                    Some(6),
                    None,
                    None,
                )
                .is_err(),
                "centroid" => {
                    feat::spectral_centroid(&audio, Some(spectrum), Some(64), Some(16)).is_err()
                }
                "bandwidth" => {
                    feat::spectral_bandwidth(&audio, Some(spectrum), Some(64), Some(16), None)
                        .is_err()
                }
                "rolloff" => {
                    feat::spectral_rolloff(&audio, Some(spectrum), Some(64), Some(16), None)
                        .is_err()
                }
                "contrast" => {
                    feat::spectral_contrast(&audio, Some(spectrum), Some(64), Some(16), Some(3))
                        .is_err()
                }
                "flatness" => {
                    feat::spectral_flatness(&audio, Some(spectrum), Some(64), Some(16)).is_err()
                }
                _ => feat::rms(&audio, Some(spectrum), Some(64), Some(16)).is_err(),
            });
            if !matches!(result, Ok(true)) {
                failures.push(format!(
                    "case={case} feature={feature} shape={:?} result={result:?}",
                    spectrum.dim()
                ));
            }
        }
    }
    assert_eq!(failures, Vec::<String>::new(), "seed={SEED:x}");
}
#[test]
fn malformed_mfcc_and_frequency_parameters_are_rejected() {
    let audio = AudioData::new(vec![0.0; 64], 8000, 1).unwrap();
    for value in [f32::NAN, f32::INFINITY, -1.0] {
        let matrix = Array2::from_elem((8, 3), value);
        assert!(
            feat::mfcc(&audio, Some(&matrix), Some(4), None, None).is_err(),
            "seed={SEED:x} MFCC value={value}"
        );
    }
    for matrix in [Array2::zeros((0, 3)), Array2::zeros((8, 0))] {
        assert!(
            feat::mfcc(&audio, Some(&matrix), Some(4), None, None).is_err(),
            "seed={SEED:x} MFCC shape={:?}",
            matrix.dim()
        );
    }
    for value in [f32::NAN, f32::INFINITY] {
        assert!(
            feat::melspectrogram(&audio, None, Some(64), Some(16), Some(6), Some(value), None)
                .is_err()
        );
        assert!(feat::spectral_rolloff(&audio, None, Some(64), Some(16), Some(value)).is_err());
    }
}
#[test]
fn spectral_rolloff_is_invariant_to_large_finite_magnitude_gain() {
    let mut rng = Rng(SEED + 1);
    let audio = AudioData::new(vec![0.0; 128], 8000, 1).unwrap();
    for case in 0..16 {
        let s = Array2::from_shape_fn((65, 3), |_| 0.2 + rng.unit());
        let loud = s.mapv(|x| x * 1e37);
        let a = feat::spectral_rolloff(&audio, Some(&s), Some(128), Some(32), Some(0.85)).unwrap();
        let b =
            feat::spectral_rolloff(&audio, Some(&loud), Some(128), Some(32), Some(0.85)).unwrap();
        assert_eq!(a, b, "seed={:x} case={case}", SEED + 1);
    }
}
#[test]
fn randomized_decibel_roundtrips_cover_reference_and_dynamic_range() {
    let mut rng = Rng(SEED + 2);
    for case in 0..32 {
        let s = Array2::from_shape_fn((2 + rng.size(4), 1 + rng.size(6)), |_| {
            10.0_f32.powf(-28.0 + 56.0 * rng.unit())
        });
        let reference = 10.0_f32.powf(-20.0 + 40.0 * rng.unit());
        let db = mag::amplitude_to_db(&s)
            .amin(1e-35)
            .ref_val(reference)
            .top_db(2000.0)
            .compute()
            .unwrap();
        let rebuilt = mag::db_to_amplitude(&db, Some(reference)).unwrap();
        let power_db = mag::power_to_db(&s)
            .amin(1e-35)
            .ref_val(reference)
            .top_db(2000.0)
            .compute()
            .unwrap();
        let power_back = mag::db_to_power(&power_db, Some(reference)).unwrap();
        for ((&x, &a), &p) in s.iter().zip(&rebuilt).zip(&power_back) {
            let context = format!("case={case} ref={reference}");
            near(f64::from(a), f64::from(x), 0.0, 3e-5, &context);
            near(f64::from(p), f64::from(x), 0.0, 3e-5, &context);
        }
    }
}
#[test]
fn scaling_rejects_unrepresentable_outputs_and_preserves_zero_pcen() {
    let db = Array2::from_elem((2, 3), 1000.0);
    assert!(
        mag::db_to_amplitude(&db, None).is_err(),
        "seed={SEED:x} amplitude overflow"
    );
    assert!(mag::db_to_power(&db, None).is_err());
    let silent = Array2::zeros((2, 3));
    let output = mag::pcen(&silent).gain(f32::MAX).compute().unwrap();
    assert!(output.iter().all(|&v| v == 0.0));
    let invalid = Array2::from_elem((2, 3), f32::NAN);
    assert!(mag::pcen(&invalid).compute().is_err());
}
fn exhaustive_dtw(cost: &Array2<f64>, i: usize, j: usize) -> f64 {
    let mut best = f64::INFINITY;
    if i + 1 == cost.nrows() && j + 1 == cost.ncols() {
        return cost[[i, j]];
    }
    for (di, dj) in [(1, 0), (0, 1), (1, 1)] {
        if i + di < cost.nrows() && j + dj < cost.ncols() {
            best = best.min(exhaustive_dtw(cost, i + di, j + dj));
        }
    }
    cost[[i, j]] + best
}
#[test]
fn randomized_dtw_matches_exhaustive_monotone_paths() {
    let mut rng = Rng(SEED + 3);
    for case in 0..48 {
        let nf = 1 + rng.size(3);
        let n = 1 + rng.size(4);
        let m = 1 + rng.size(4);
        let scale = [1e-20, 1.0, 1e20][case % 3];
        let a = Array2::from_shape_fn((nf, n), |_| (rng.unit() - 0.5) * scale);
        let b = Array2::from_shape_fn((nf, m), |_| (rng.unit() - 0.5) * scale);
        let costs = Array2::from_shape_fn((n, m), |(i, j)| {
            (0..nf)
                .map(|f| (f64::from(a[[f, i]]) - f64::from(b[[f, j]])).powi(2))
                .sum::<f64>()
                .sqrt()
        });
        let (score, path) = feat::dtw(&a, &b).compute().unwrap();
        let expected = exhaustive_dtw(&costs, 0, 0);
        near(
            f64::from(score),
            expected,
            0.0,
            3e-7,
            &format!("DTW case={case} shape={nf}x{n}x{m}"),
        );
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(n - 1, m - 1)));
        assert!(
            path.windows(2)
                .all(|w| matches!((w[1].0 - w[0].0, w[1].1 - w[0].1), (1, 0 | 1) | (0, 1)))
        );
        near(
            path.iter().map(|&ij| costs[ij]).sum(),
            expected,
            0.0,
            3e-7,
            &format!("DTW path case={case}"),
        );
    }
}
#[test]
fn randomized_viterbi_scores_match_all_small_state_sequences() {
    let mut rng = Rng(SEED + 4);
    for case in 0..64 {
        let states = 1 + rng.size(3);
        let frames = 1 + rng.size(5);
        let emission = Array2::from_shape_fn((states, frames), |_| {
            if rng.size(5) == 0 {
                f32::NEG_INFINITY
            } else {
                -(rng.size(32) as f32) / 8.0
            }
        });
        let transition = Array2::from_shape_fn((states, states), |_| {
            if rng.size(5) == 0 {
                f32::NEG_INFINITY
            } else {
                -(rng.size(32) as f32) / 8.0
            }
        });
        let score_path = |path: &[usize]| -> f64 {
            path.iter()
                .enumerate()
                .map(|(t, &s)| f64::from(emission[[s, t]]))
                .sum::<f64>()
                + path
                    .windows(2)
                    .map(|w| f64::from(transition[[w[0], w[1]]]))
                    .sum::<f64>()
        };
        let mut best = f64::NEG_INFINITY;
        for mut code in 0..states.pow(frames as u32) {
            let path: Vec<usize> = (0..frames)
                .map(|_| {
                    let s = code % states;
                    code /= states;
                    s
                })
                .collect();
            best = best.max(score_path(&path));
        }
        let (score, path) = feat::viterbi(&emission, &transition).unwrap();
        assert_eq!(
            f64::from(score),
            best,
            "seed={:x} case={case} states={states} frames={frames}",
            SEED + 4
        );
        assert_eq!(score_path(&path), best);
    }
}
#[test]
fn randomized_rhythm_silence_and_geometry_have_no_invented_activity() {
    let mut rng = Rng(SEED + 5);
    for case in 0..20 {
        let n = 1 + rng.size(65);
        let hop = 1 + rng.size(31);
        let y = vec![0.0; n * 3];
        let onset = feat::onset_strength(&y, 8000)
            .n_fft(n)
            .hop_length(hop)
            .compute()
            .unwrap();
        assert!(
            onset.iter().all(|&v| v == 0.0),
            "seed={SEED:x} case={case} n={n} hop={hop}"
        );
        let multiple = feat::onset_strength_multi(&y, 8000)
            .n_fft(n)
            .hop_length(hop)
            .n_bands(1 + rng.size(8))
            .compute()
            .unwrap();
        assert_eq!(multiple.ncols(), onset.len());
        assert!(multiple.iter().all(|&v| v == 0.0));
        let env = Array1::zeros(1 + rng.size(20));
        let gram = feat::tempogram(
            None,
            Some(8000),
            Some(&env),
            Some(hop),
            Some(1 + rng.size(15)),
        )
        .unwrap();
        assert!(gram.iter().all(|&v| v == 0.0));
    }
}

#[test]
fn pcen_zero_input_stays_zero_with_large_finite_gain() {
    let s = Array2::zeros((3, 5));
    let result = mag::pcen(&s).gain(f32::MAX).compute().unwrap();
    assert!(
        result.iter().all(|&x| x == 0.0),
        "seed={SEED:x} PCEN zero={result:?}"
    );
}
#[test]
fn sequence_empty_features_and_unrepresentable_distances_are_errors() {
    let empty = Array2::zeros((0, 3));
    assert!(
        feat::dtw(&empty, &empty).compute().is_err(),
        "seed={SEED:x} empty feature dimension"
    );
    let huge = Array2::from_elem((2, 3), 3e38);
    let opposite = -&huge;
    assert!(
        feat::dtw(&huge, &opposite).compute().is_err(),
        "seed={SEED:x} unrepresentable DTW distance"
    );
}
#[test]
fn invalid_scaling_and_sequence_inputs_never_succeed() {
    for value in [f32::NAN, f32::INFINITY] {
        let bad = Array2::from_elem((2, 3), value);
        let good = Array2::ones((2, 3));
        assert!(feat::dtw(&bad, &good).compute().is_err());
        assert!(feat::viterbi(&bad, &Array2::zeros((2, 2))).is_err());
        assert!(mag::amplitude_to_db(&bad).compute().is_err());
        assert!(mag::power_to_db(&bad).compute().is_err());
        assert!(mag::pcen(&bad).compute().is_err());
    }
    for shape in [(0, 2), (2, 0)] {
        let empty = Array2::zeros(shape);
        assert!(mag::amplitude_to_db(&empty).compute().is_err());
        assert!(feat::viterbi(&empty, &Array2::zeros((2, 2))).is_err());
    }
    let good = Array2::ones((2, 3));
    assert!(feat::viterbi(&good, &Array2::zeros((3, 2))).is_err());
    for parameter in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            mag::amplitude_to_db(&good)
                .amin(parameter)
                .compute()
                .is_err()
        );
        assert!(
            mag::power_to_db(&good)
                .ref_val(parameter)
                .compute()
                .is_err()
        );
    }
}

#[test]
fn randomized_softmasks_match_scaled_f64_power_ratios() {
    let mut rng = Rng(SEED + 6);
    for case in 0..32 {
        let shape = (1 + rng.size(5), 1 + rng.size(7));
        let a = Array2::from_shape_fn(shape, |_| {
            if rng.size(5) == 0 {
                0.0
            } else {
                10.0_f32.powf(-30.0 + 60.0 * rng.unit())
            }
        });
        let b = Array2::from_shape_fn(shape, |_| {
            if rng.size(5) == 0 {
                0.0
            } else {
                10.0_f32.powf(-30.0 + 60.0 * rng.unit())
            }
        });
        let power = [0.5, 1.0, 2.0, 4.0][case % 4];
        let mask = feat::softmask(&a, &b)
            .power(power)
            .split_zeros(true)
            .compute();
        for ((&a, &b), &actual) in a.iter().zip(&b).zip(&mask) {
            let ap = f64::from(a).powf(f64::from(power));
            let bp = f64::from(b).powf(f64::from(power));
            let expected = if ap + bp == 0.0 { 0.5 } else { ap / (ap + bp) };
            // A result smaller than f32's subnormal range is necessarily rounded to zero.
            near(
                f64::from(actual),
                expected,
                f64::from(f32::from_bits(1)),
                1e-7,
                &format!(
                    "softmask seed={:x} case={case} shape={shape:?} power={power}",
                    SEED + 6
                ),
            );
        }
    }
}
#[test]
fn randomized_normalization_reaches_unit_norm_on_both_axes() {
    let mut rng = Rng(SEED + 7);
    for case in 0..24 {
        let shape = (1 + rng.size(6), 1 + rng.size(7));
        let scale = [1e-30, 1.0, 1e30][case % 3];
        let input = Array2::from_shape_fn(shape, |_| (0.1 + rng.unit()) * scale);
        for axis in [0, 1] {
            for norm in [
                feat::FeatureNorm::L1,
                feat::FeatureNorm::L2,
                feat::FeatureNorm::LInf,
            ] {
                let normalized = feat::normalize(&input)
                    .axis(axis)
                    .norm(norm)
                    .threshold(0.0)
                    .compute();
                for vector in normalized.axis_iter(ndarray::Axis(1 - axis)) {
                    let actual = match norm {
                        feat::FeatureNorm::L1 => vector.iter().map(|&x| f64::from(x).abs()).sum(),
                        feat::FeatureNorm::L2 => vector
                            .iter()
                            .map(|&x| f64::from(x).powi(2))
                            .sum::<f64>()
                            .sqrt(),
                        feat::FeatureNorm::LInf => vector
                            .iter()
                            .map(|&x| f64::from(x).abs())
                            .fold(0.0, f64::max),
                    };
                    near(
                        actual,
                        1.0,
                        0.0,
                        1e-7,
                        &format!(
                            "normalize seed={:x} case={case} axis={axis} norm={norm:?} scale={scale}",
                            SEED + 7
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn unrepresentable_feature_dimensions_return_errors_without_panics() {
    let audio = AudioData::new(vec![0.0; 64], 8000, 1).unwrap();
    let s = Array2::ones((33, 2));
    let mfcc =
        std::panic::catch_unwind(|| feat::mfcc(&audio, Some(&s), Some(usize::MAX), None, None));
    let contrast = std::panic::catch_unwind(|| {
        feat::spectral_contrast(&audio, Some(&s), Some(64), Some(16), Some(usize::MAX))
    });
    assert!(
        matches!(mfcc, Ok(Err(_))),
        "seed={SEED:x} MFCC n_mfcc=usize::MAX result={mfcc:?}"
    );
    assert!(
        matches!(contrast, Ok(Err(_))),
        "seed={SEED:x} contrast n_bands=usize::MAX result={contrast:?}"
    );
    let mel = std::panic::catch_unwind(|| {
        feat::melspectrogram(
            &audio,
            Some(&s),
            Some(64),
            Some(16),
            Some(usize::MAX),
            None,
            None,
        )
    });
    assert!(
        matches!(mel, Ok(Err(_))),
        "seed={SEED:x} mel n_mels=usize::MAX result={mel:?}"
    );
}
