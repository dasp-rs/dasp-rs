//! Adversarial feature tests with analytic and exhaustive reference results.
use dasp_rs::feat::*;
use ndarray::{Array2, array};

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

#[test]
fn transition_helpers_preserve_impossibility_and_probability_mass() {
    for p in [0.0, 0.3, 1.0] {
        for n in 1..5 {
            for matrix in [
                transition_loop(n, p),
                transition_cycle(n, p),
                transition_acyclic(n, p),
            ] {
                for row in matrix.rows() {
                    close(row.iter().map(|v| v.exp()).sum(), 1.0);
                }
            }
        }
    }
    assert_eq!(transition_loop(2, 0.0)[[0, 0]], f32::NEG_INFINITY);
    assert_eq!(transition_cycle(2, 0.0)[[0, 1]], f32::NEG_INFINITY);
    assert_eq!(transition_cycle(2, 1.0)[[0, 0]], f32::NEG_INFINITY);
    assert_eq!(transition_acyclic(2, 0.0)[[0, 1]], f32::NEG_INFINITY);
}

#[test]
fn transition_local_unbounded_width_is_uniform() {
    assert_eq!(transition_local(3, usize::MAX), transition_uniform(3));
}

#[test]
fn viterbi_matches_exhaustive_paths_with_asymmetric_transitions() {
    let emission = array![[0.2_f32, 0.7, 0.1, 0.8], [0.8, 0.3, 0.9, 0.2]].mapv(f32::ln);
    let transition = array![[0.3_f32, 0.7], [0.9, 0.1]].mapv(f32::ln);
    let mut best = (f32::NEG_INFINITY, vec![]);
    for bits in 0..16 {
        let states: Vec<usize> = (0..4).map(|t| (bits >> t) & 1).collect();
        let score = (0..4).map(|t| emission[[states[t], t]]).sum::<f32>()
            + (1..4)
                .map(|t| transition[[states[t - 1], states[t]]])
                .sum::<f32>();
        if score > best.0 {
            best = (score, states);
        }
    }
    let actual = viterbi(&emission, &transition).unwrap();
    close(actual.0, best.0);
    assert_eq!(actual.1, best.1);
}

#[test]
fn viterbi_never_uses_zero_probability_posteriors() {
    let forced_one = array![
        [f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NEG_INFINITY, 0.0]
    ];
    let (score, _) = viterbi_binary(&[1.0, 0.0], &forced_one).unwrap();
    assert_eq!(score, f32::NEG_INFINITY);
    let prob = array![[0.0, 1.0], [1.0, 0.0]];
    let (score, _) = viterbi_discriminative(&prob, &forced_one, Some(&[0.5, 0.5])).unwrap();
    assert_eq!(score, f32::NEG_INFINITY);
}

#[test]
fn viterbi_rejects_invalid_values_but_accepts_impossible_events() {
    let transitions = transition_uniform(2);
    for p in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(viterbi_binary(&[p], &transitions).is_err());
        assert!(viterbi_discriminative(&array![[p], [0.5]], &transitions, None).is_err());
    }
    assert!(viterbi(&array![[f32::NAN], [0.0]], &transitions).is_err());
    assert!(viterbi(&array![[f32::NEG_INFINITY], [0.0]], &transitions).is_ok());
}

#[test]
fn softmask_is_scale_invariant_over_finite_f32_range() {
    for scale in [1e-30_f32, 1.0, 1e30] {
        let a = array![[3.0 * scale]];
        let b = array![[scale]];
        close(softmask(&a, &b).power(2.0).compute()[[0, 0]], 0.9);
    }
}

#[test]
fn normalization_handles_large_and_tiny_finite_vectors() {
    for scale in [1e-30_f32, 1.0, 1e30] {
        let x = array![[3.0 * scale], [4.0 * scale]];
        let y = normalize(&x).threshold(0.0).compute();
        close(y[[0, 0]], 0.6);
        close(y[[1, 0]], 0.8);
    }
    let x = array![[f32::MAX, f32::MAX]];
    let y = normalize(&x).axis(1).norm(FeatureNorm::L1).compute();
    close(y[[0, 0]], 0.5);
}

#[test]
fn stack_memory_large_delay_repeats_first_frame() {
    let x = array![[1.0, 2.0, 3.0]];
    let stacked = stack_memory(&x, Some(3), Some(usize::MAX));
    assert_eq!(
        stacked,
        array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );
}

#[test]
fn zero_crossing_rate_does_not_multiply_signs() {
    let y = [1e-30, -1e-30, 1e-30, -1e-30];
    close(zero_crossing_rate(&y).frame_length(4).compute()[0], 0.75);
}

#[test]
fn kurtosis_is_invariant_under_amplitude_scaling() {
    for scale in [1e-20_f32, 1.0, 1e20] {
        let y = [scale, -scale, scale, -scale];
        close(
            temporal_kurtosis(&y).frame_length(4).compute().unwrap()[0],
            -2.0,
        );
    }
}

#[test]
fn nmf_rank_one_reconstruction_is_scale_invariant() {
    for scale in [1e-20_f32, 1.0, 1e20] {
        let x = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]] * scale;
        let (w, h) = decompose(&x, 1).n_iter(50).compute().unwrap();
        let reconstructed = w.dot(&h);
        for (&a, &b) in reconstructed.iter().zip(x.iter()) {
            assert!(
                ((f64::from(a) - f64::from(b)) / f64::from(b)).abs() < 1e-4,
                "{a} != {b}"
            );
        }
    }
}

#[test]
fn nmf_rejects_negative_and_nonfinite_spectrograms() {
    for value in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(
            decompose(&Array2::from_elem((2, 2), value), 1)
                .compute()
                .is_err()
        );
    }
}

#[test]
fn cosine_and_euclidean_distances_survive_amplitude_scaling() {
    for scale in [1e-30_f32, 1.0, 1e30] {
        let x = array![[3.0 * scale], [4.0 * scale]];
        let y = &x * 0.5;
        close(
            dtw(&x, &y).metric(DtwMetric::Cosine).compute().unwrap().0,
            0.0,
        );
        close(cross_similarity(&x, &y).compute()[[0, 0]], 1.0);
        let (distance, _) = dtw(&x, &y).compute().unwrap();
        assert!((f64::from(distance) / f64::from(scale) - 2.5).abs() < 1e-5);
    }
}

#[test]
fn gaussian_affinity_respects_small_and_large_bandwidths() {
    for scale in [1e-20_f32, 1.0, 1e20] {
        let x = array![[0.0, scale]];
        let r = recurrence_matrix(&x)
            .metric(SimilarityMetric::Euclidean)
            .bandwidth(scale)
            .compute();
        close(r[[0, 1]], (-1.0_f32).exp());
    }
}

#[test]
fn dtw_matches_enumeration_of_all_monotone_paths() {
    fn enumerate(x: &[f32], y: &[f32], i: usize, j: usize) -> f32 {
        let local = (x[i] - y[j]).abs();
        if i + 1 == x.len() && j + 1 == y.len() {
            return local;
        }
        let mut best = f32::INFINITY;
        for (di, dj) in [(1, 0), (0, 1), (1, 1)] {
            if i + di < x.len() && j + dj < y.len() {
                best = best.min(enumerate(x, y, i + di, j + dj));
            }
        }
        local + best
    }
    for bits in 0..128 {
        let x: Vec<f32> = (0..3).map(|i| ((bits >> i) & 1) as f32).collect();
        let y: Vec<f32> = (3..7).map(|i| ((bits >> i) & 1) as f32).collect();
        let a = Array2::from_shape_vec((1, 3), x.clone()).unwrap();
        let b = Array2::from_shape_vec((1, 4), y.clone()).unwrap();
        let (cost, path) = dtw(&a, &b).compute().unwrap();
        close(cost, enumerate(&x, &y, 0, 0));
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(2, 3)));
        close(cost, path.iter().map(|&(i, j)| (x[i] - y[j]).abs()).sum());
        for w in path.windows(2) {
            assert!(matches!(
                (w[1].0 - w[0].0, w[1].1 - w[0].1),
                (1, 0 | 1) | (0, 1)
            ));
        }
    }
}

#[test]
fn agglomerative_cosine_clusters_are_scale_invariant() {
    // Frames 1 and 2 are identical and must merge before the orthogonal pair 0,1.
    for scale in [1e-30_f32, 1.0, 1e30] {
        let x = array![[scale, 0.0, 0.0], [0.0, scale, scale]];
        assert_eq!(agglomerative(&x, 2).compute(), vec![0, 1, 1]);
    }
}

#[test]
fn timelag_filter_huge_window_is_clipped_to_diagonal() {
    let x = array![[1.0, 2.0], [2.0, 3.0]];
    assert_eq!(
        timelag_filter(&x).n_window(usize::MAX).compute(),
        Array2::from_elem((2, 2), 2.0)
    );
}

#[test]
fn nmf_zero_matrix_and_extreme_constants_remain_finite() {
    for value in [0.0_f32, f32::from_bits(1), f32::MAX] {
        let x = Array2::from_elem((2, 3), value);
        let (w, h) = decompose(&x, 1).compute().unwrap();
        assert!(w.iter().chain(h.iter()).all(|v| v.is_finite() && *v >= 0.0));
        for i in 0..2 {
            for j in 0..3 {
                let reconstructed = f64::from(w[[i, 0]]) * f64::from(h[[0, j]]);
                let target = f64::from(value);
                assert!((reconstructed - target).abs() <= 1e-5 * target);
            }
        }
    }
}

#[test]
fn path_enhance_mean_preserves_large_constant_values() {
    let x = Array2::from_elem((2, 2), f32::MAX);
    assert_eq!(path_enhance(&x, 3).compute(), x);
}

#[test]
fn hard_mask_honors_zero_split_policy() {
    let zeros = Array2::zeros((1, 1));
    assert_eq!(
        softmask(&zeros, &zeros).power(f32::INFINITY).compute()[[0, 0]],
        0.0
    );
    assert_eq!(
        softmask(&zeros, &zeros)
            .power(f32::INFINITY)
            .split_zeros(true)
            .compute()[[0, 0]],
        0.5
    );
}

#[test]
fn softmask_fractional_power_preserves_extreme_ratios() {
    let a = array![[1e-30_f32]];
    let b = array![[1e30_f32]];
    let power = 0.01_f32;
    let ap = f64::from(a[[0, 0]]).powf(f64::from(power));
    let bp = f64::from(b[[0, 0]]).powf(f64::from(power));
    close(
        softmask(&a, &b).power(power).compute()[[0, 0]],
        (ap / (ap + bp)) as f32,
    );
}

#[test]
fn viterbi_obeys_forbidden_transitions_even_when_emissions_favor_them() {
    let emissions = array![[0.0, 0.0], [-1000.0, -1000.0]];
    // With no self-loops, every supported path changes state and pays -1000.
    let (score, path) = viterbi(&emissions, &transition_loop(2, 0.0)).unwrap();
    close(score, -1000.0);
    assert_ne!(path[0], path[1]);
    // Both observations require state zero: with no self-loop, no path exists.
    let impossible = array![[0.0, 0.0], [f32::NEG_INFINITY, f32::NEG_INFINITY]];
    assert_eq!(
        viterbi(&impossible, &transition_loop(2, 0.0)).unwrap().0,
        f32::NEG_INFINITY
    );
}

#[test]
#[should_panic(expected = "cross_similarity requires matching feature dimensions")]
fn cross_similarity_rejects_mismatched_features() {
    cross_similarity(&Array2::zeros((2, 3)), &Array2::zeros((1, 3))).compute();
}

#[test]
#[should_panic(expected = "cross_similarity requires matching feature dimensions")]
fn cross_similarity_validates_features_even_with_no_frames() {
    cross_similarity(&Array2::zeros((2, 0)), &Array2::zeros((1, 3))).compute();
}

#[test]
#[should_panic(expected = "softmask requires identical input shapes")]
fn softmask_rejects_larger_reference_instead_of_ignoring_extra_cells() {
    softmask(&Array2::ones((1, 1)), &Array2::ones((2, 1))).compute();
}

#[test]
#[should_panic(expected = "normalize axis must be 0 or 1")]
fn normalize_rejects_invalid_axis() {
    normalize(&Array2::ones((2, 2))).axis(2).compute();
}
