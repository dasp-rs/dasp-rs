//! Sequence analysis: Dynamic Time Warping and Viterbi decoding.

use ndarray::{Array2, ArrayView1};
use thiserror::Error;

/// Error conditions for sequence analysis operations.
#[derive(Error, Debug)]
pub enum SequenceError {
    /// Mismatched shapes or empty inputs.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

// ─── DTW ─────────────────────────────────────────────────────────────────────

/// Distance metric for Dynamic Time Warping.
#[derive(Debug, Clone, Copy)]
pub enum DtwMetric {
    /// Euclidean (L2) distance (default).
    Euclidean,
    /// Cosine distance: `1 − cosine_similarity`.
    Cosine,
    /// Manhattan (L1) distance.
    Manhattan,
}

/// Builder for [`dtw`].
#[derive(Debug, Clone)]
pub struct DtwBuilder<'a> {
    x: &'a Array2<f32>,
    y: &'a Array2<f32>,
    metric: DtwMetric,
}

impl DtwBuilder<'_> {
    /// Set the distance metric (default: [`DtwMetric::Euclidean`]).
    #[must_use]
    pub fn metric(mut self, v: DtwMetric) -> Self {
        self.metric = v;
        self
    }

    /// Compute DTW alignment cost and warping path.
    ///
    /// # Returns
    /// `(cost, path)` where `cost` is the total alignment cost and `path` is a
    /// list of `(i, j)` index pairs tracing the optimal alignment from
    /// `(0, 0)` to `(n_frames_x − 1, n_frames_y − 1)`.
    ///
    /// # Errors
    /// Returns an error if either input has zero frames, or if the two inputs
    /// have different feature dimensions.
    pub fn compute(self) -> Result<(f32, Vec<(usize, usize)>), SequenceError> {
        dtw_impl(self.x, self.y, self.metric)
    }
}

/// Aligns two feature sequences using Dynamic Time Warping.
///
/// Uses the standard three-move (diagonal, horizontal, vertical) DTW step pattern
/// with an `O(n · m)` cost matrix. The warping path is recovered by backtracking
/// from `(n − 1, m − 1)` to `(0, 0)`.
///
/// # Arguments
/// * `x` — Reference sequence, shape `(n_features, n_frames_x)`
/// * `y` — Query sequence, shape `(n_features, n_frames_y)`
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::dtw;
/// use ndarray::Array2;
/// let a: Array2<f32> = Array2::from_elem((12, 50), 1.0);
/// let b: Array2<f32> = Array2::from_elem((12, 45), 1.0);
/// let (cost, path) = dtw(&a, &b).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn dtw<'a>(x: &'a Array2<f32>, y: &'a Array2<f32>) -> DtwBuilder<'a> {
    DtwBuilder { x, y, metric: DtwMetric::Euclidean }
}

fn dtw_impl(
    x: &Array2<f32>,
    y: &Array2<f32>,
    metric: DtwMetric,
) -> Result<(f32, Vec<(usize, usize)>), SequenceError> {
    let n = x.shape()[1];
    let m = y.shape()[1];
    let nf = x.shape()[0];

    if n == 0 || m == 0 {
        return Err(SequenceError::InvalidInput("Input sequence has zero frames".into()));
    }
    if nf != y.shape()[0] {
        return Err(SequenceError::InvalidInput(format!(
            "Feature dimension mismatch: x={nf}, y={}",
            y.shape()[0]
        )));
    }

    // Build DP cost matrix
    let mut dp = vec![vec![f32::INFINITY; m]; n];
    dp[0][0] = col_dist(x.column(0), y.column(0), metric);
    for i in 1..n {
        dp[i][0] = dp[i - 1][0] + col_dist(x.column(i), y.column(0), metric);
    }
    for j in 1..m {
        dp[0][j] = dp[0][j - 1] + col_dist(x.column(0), y.column(j), metric);
    }
    for i in 1..n {
        for j in 1..m {
            let c = col_dist(x.column(i), y.column(j), metric);
            let prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = c + prev;
        }
    }

    let cost = dp[n - 1][m - 1];

    // Backtrack to recover path
    let mut path = Vec::new();
    let (mut i, mut j) = (n - 1, m - 1);
    path.push((i, j));
    while i > 0 || j > 0 {
        let (pi, pj) = match (i, j) {
            (0, _) => (0, j - 1),
            (_, 0) => (i - 1, 0),
            _ => {
                let candidates = [(i - 1, j - 1), (i - 1, j), (i, j - 1)];
                candidates
                    .iter()
                    .copied()
                    .min_by(|&(a, b), &(c, d)| dp[a][b].total_cmp(&dp[c][d]))
                    .unwrap()
            }
        };
        i = pi;
        j = pj;
        path.push((i, j));
    }
    path.reverse();

    Ok((cost, path))
}

// ─── Viterbi ─────────────────────────────────────────────────────────────────

/// Decodes the most probable state sequence from log-domain probabilities.
///
/// Implements the standard Viterbi algorithm in log space to avoid underflow.
///
/// # Arguments
/// * `log_prob` — Log-domain emission probabilities, shape `(n_states, n_frames)`.
///   `log_prob[[s, t]]` is `ln P(observation_t | state_s)`.
/// * `log_trans` — Log-domain transition matrix, shape `(n_states, n_states)`.
///   `log_trans[[i, j]]` is `ln P(to state j | from state i)`.
///
/// # Returns
/// `(log_likelihood, states)` where `log_likelihood` is the log-probability of
/// the most probable path and `states[t]` is the decoded state at frame `t`.
///
/// # Errors
/// Returns an error if inputs are empty or the transition matrix shape doesn't
/// match `(n_states, n_states)`.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::viterbi;
/// use ndarray::Array2;
/// let log_prob = Array2::from_elem((3, 10), -1.0_f32);
/// let log_trans = Array2::from_elem((3, 3), -1.1_f32);
/// let (ll, states) = viterbi(&log_prob, &log_trans)?;
/// assert_eq!(states.len(), 10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn viterbi(
    log_prob: &Array2<f32>,
    log_trans: &Array2<f32>,
) -> Result<(f32, Vec<usize>), SequenceError> {
    viterbi_impl(log_prob, log_trans)
}

fn viterbi_impl(
    log_prob: &Array2<f32>,
    log_trans: &Array2<f32>,
) -> Result<(f32, Vec<usize>), SequenceError> {
    let n_states = log_prob.shape()[0];
    let n_frames = log_prob.shape()[1];

    if n_states == 0 || n_frames == 0 {
        return Err(SequenceError::InvalidInput("Empty input".into()));
    }
    if log_trans.shape() != [n_states, n_states] {
        return Err(SequenceError::InvalidInput(format!(
            "log_trans shape {:?} does not match n_states={n_states}",
            log_trans.shape()
        )));
    }

    // delta[t][s] = best log-prob path to state s at frame t
    let mut delta = vec![vec![f32::NEG_INFINITY; n_states]; n_frames];
    // psi[t][s]   = predecessor state that achieves delta[t][s]
    let mut psi = vec![vec![0usize; n_states]; n_frames];

    for s in 0..n_states {
        delta[0][s] = log_prob[[s, 0]];
    }

    for t in 1..n_frames {
        for s in 0..n_states {
            let (best_prev, best_val) = (0..n_states)
                .map(|sp| (sp, delta[t - 1][sp] + log_trans[[sp, s]]))
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();
            delta[t][s] = log_prob[[s, t]] + best_val;
            psi[t][s] = best_prev;
        }
    }

    let last = (0..n_states)
        .max_by(|&a, &b| delta[n_frames - 1][a].total_cmp(&delta[n_frames - 1][b]))
        .unwrap();

    let log_likelihood = delta[n_frames - 1][last];

    let mut states = vec![0usize; n_frames];
    states[n_frames - 1] = last;
    for t in (0..n_frames - 1).rev() {
        states[t] = psi[t + 1][states[t + 1]];
    }

    Ok((log_likelihood, states))
}

// ─── Shared helper ────────────────────────────────────────────────────────────

fn col_dist(a: ArrayView1<f32>, b: ArrayView1<f32>, metric: DtwMetric) -> f32 {
    match metric {
        DtwMetric::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt(),
        DtwMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            let na = a.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
            let nb = b.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
            1.0 - dot / (na * nb)
        }
        DtwMetric::Manhattan => a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum(),
    }
}

// ─── Transition matrix helpers ────────────────────────────────────────────────

/// Builds a self-loop log-transition matrix.
///
/// Each state remains in itself with probability `prob` and transitions
/// uniformly to any other state with the remaining probability.
///
/// # Arguments
/// * `n_states` — Number of HMM states.
/// * `prob` — Self-loop probability (clamped to `[0, 1]`).
///
/// # Returns
/// Log-probability matrix of shape `(n_states, n_states)` suitable for use
/// with [`viterbi`] or [`viterbi_discriminative`].
///
/// # Examples
/// ```
/// use dasp_rs::feat::{transition_loop, viterbi};
/// use ndarray::Array2;
/// let log_t = transition_loop(3, 0.9);
/// assert!((log_t[[0, 0]] - 0.9_f32.ln()).abs() < 1e-5);
/// ```
pub fn transition_loop(n_states: usize, prob: f32) -> Array2<f32> {
    let prob = prob.clamp(0.0, 1.0);
    let off = if n_states > 1 { (1.0 - prob) / (n_states - 1) as f32 } else { 0.0 };
    let log_off = if off > 0.0 { off.ln() } else { f32::NEG_INFINITY };
    let mut t = Array2::from_elem((n_states, n_states), log_off);
    for i in 0..n_states {
        t[[i, i]] = prob.max(f32::MIN_POSITIVE).ln();
    }
    t
}

/// Builds a near-diagonal local log-transition matrix.
///
/// State `i` can only transition to states `j` with `|i − j| ≤ width`, with
/// equal probability among those neighbours.
///
/// # Arguments
/// * `n_states` — Number of HMM states.
/// * `width` — Maximum allowed transition distance.
///
/// # Examples
/// ```
/// use dasp_rs::feat::transition_local;
/// let log_t = transition_local(5, 1);
/// // State 2 can reach states 1, 2, 3 only
/// assert!(log_t[[2, 0]].is_infinite());
/// assert!(log_t[[2, 1]].is_finite());
/// ```
pub fn transition_local(n_states: usize, width: usize) -> Array2<f32> {
    let mut t = Array2::from_elem((n_states, n_states), f32::NEG_INFINITY);
    for i in 0..n_states {
        let lo = i.saturating_sub(width);
        let hi = (i + width + 1).min(n_states);
        let log_p = -((hi - lo) as f32).ln();
        for j in lo..hi {
            t[[i, j]] = log_p;
        }
    }
    t
}

/// Builds a uniform log-transition matrix.
///
/// All `n_states × n_states` entries equal `ln(1 / n_states)`.
///
/// # Examples
/// ```
/// use dasp_rs::feat::transition_uniform;
/// let log_t = transition_uniform(4);
/// assert!((log_t[[0, 0]] - (0.25_f32).ln()).abs() < 1e-5);
/// ```
pub fn transition_uniform(n_states: usize) -> Array2<f32> {
    let log_p = -(n_states as f32).ln();
    Array2::from_elem((n_states, n_states), log_p)
}

// ─── Discriminative Viterbi ───────────────────────────────────────────────────

/// Decodes the most probable state sequence from discriminative posterior probabilities.
///
/// Unlike [`viterbi`], which takes log-domain emission likelihoods, this
/// function accepts class-conditional **posterior** probabilities
/// `P(state | observation)` and internally converts them to approximate
/// log-likelihoods by dividing by state priors.
///
/// # Arguments
/// * `prob` — Posterior probability matrix, shape `(n_states, n_frames)`,
///   with values in `[0, 1]`.
/// * `log_trans` — Log-domain transition matrix, shape `(n_states, n_states)`.
/// * `p_state` — Optional state-prior vector of length `n_states`. When `None`,
///   priors are estimated as the time-averaged mean of `prob` across frames.
///
/// # Returns
/// `(log_likelihood, state_sequence)` — the same form as [`viterbi`].
///
/// # Errors
/// Returns an error if any input has incompatible shapes or zero size.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::{transition_loop, viterbi_discriminative};
/// use ndarray::Array2;
/// let prob = Array2::from_elem((3, 20), 1.0_f32 / 3.0);
/// let log_t = transition_loop(3, 0.8);
/// let (ll, states) = viterbi_discriminative(&prob, &log_t, None)?;
/// assert_eq!(states.len(), 20);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn viterbi_discriminative(
    prob: &Array2<f32>,
    log_trans: &Array2<f32>,
    p_state: Option<&[f32]>,
) -> Result<(f32, Vec<usize>), SequenceError> {
    let n_states = prob.shape()[0];
    let n_frames = prob.shape()[1];

    if n_states == 0 || n_frames == 0 {
        return Err(SequenceError::InvalidInput("Empty input".into()));
    }

    // Estimate priors if not provided
    let priors: Vec<f32> = match p_state {
        Some(p) => {
            if p.len() != n_states {
                return Err(SequenceError::InvalidInput(format!(
                    "p_state length {} does not match n_states {n_states}",
                    p.len()
                )));
            }
            p.to_vec()
        }
        None => (0..n_states)
            .map(|s| prob.row(s).iter().sum::<f32>() / n_frames as f32)
            .collect(),
    };

    // Convert posteriors to log-likelihoods: log P(obs|state) ≈ log P(state|obs) - log P(state)
    let log_prob = Array2::from_shape_fn((n_states, n_frames), |(s, t)| {
        let posterior = prob[[s, t]].max(f32::MIN_POSITIVE);
        let prior = priors[s].max(f32::MIN_POSITIVE);
        posterior.ln() - prior.ln()
    });

    viterbi_impl(&log_prob, log_trans)
}

// ─── More transition helpers ───────────────────────────────────────────────────

/// Builds a cyclic log-transition matrix.
///
/// Each state advances to the next with probability `prob` and stays put with
/// `1 − prob`. State `n − 1` wraps back to state `0`.
///
/// # Examples
/// ```
/// use dasp_rs::feat::transition_cycle;
/// let log_t = transition_cycle(4, 0.9);
/// assert!(log_t[[3, 0]].is_finite()); // wrap-around is reachable
/// assert!(log_t[[0, 2]].is_infinite()); // non-adjacent not reachable
/// ```
pub fn transition_cycle(n_states: usize, prob: f32) -> Array2<f32> {
    let prob = prob.clamp(0.0, 1.0);
    let stay = (1.0 - prob).max(f32::MIN_POSITIVE).ln();
    let advance = prob.max(f32::MIN_POSITIVE).ln();
    let mut t = Array2::from_elem((n_states, n_states), f32::NEG_INFINITY);
    for i in 0..n_states {
        t[[i, i]] = stay;
        t[[i, (i + 1) % n_states]] = advance;
    }
    t
}

/// Builds a left-to-right (acyclic) log-transition matrix.
///
/// Each state can only stay (`1 − prob`) or advance to the next state (`prob`).
/// The final state is absorbing (self-loop probability 1.0).
///
/// # Examples
/// ```
/// use dasp_rs::feat::transition_acyclic;
/// let log_t = transition_acyclic(3, 0.1);
/// // Last state is absorbing
/// assert!((log_t[[2, 2]] - 0.0_f32).abs() < 1e-5);
/// assert!(log_t[[2, 0]].is_infinite());
/// ```
pub fn transition_acyclic(n_states: usize, prob: f32) -> Array2<f32> {
    let prob = prob.clamp(0.0, 1.0);
    let stay = (1.0 - prob).max(f32::MIN_POSITIVE).ln();
    let advance = prob.max(f32::MIN_POSITIVE).ln();
    let mut t = Array2::from_elem((n_states, n_states), f32::NEG_INFINITY);
    for i in 0..n_states {
        t[[i, i]] = stay;
        if i + 1 < n_states {
            t[[i, i + 1]] = advance;
        }
    }
    // Absorbing end state
    if n_states > 0 {
        t[[n_states - 1, n_states - 1]] = 0.0; // log(1.0)
    }
    t
}

// ─── Binary Viterbi ───────────────────────────────────────────────────────────

/// Decodes a two-state sequence using an optimised O(n) binary Viterbi.
///
/// Accepts **posterior** probabilities `P(state=1 | obs)` in a 1-D slice and
/// a 2×2 log-transition matrix. Internally, `P(state=0 | obs) = 1 − prob[t]`.
///
/// # Arguments
/// * `prob` — Posterior probability of state 1, length `n_frames`, values in `[0, 1]`.
/// * `log_trans` — 2×2 log-transition matrix.
///
/// # Returns
/// `(log_likelihood, states)` where `states[t] ∈ {0, 1}`.
///
/// # Errors
/// Returns an error if `prob` is empty or `log_trans` is not 2×2.
///
/// # Examples
/// ```
/// use dasp_rs::feat::{transition_loop, viterbi_binary};
/// let prob = vec![0.1_f32, 0.1, 0.9, 0.9, 0.9];
/// let log_t = transition_loop(2, 0.9);
/// let (_, states) = viterbi_binary(&prob, &log_t)?;
/// assert_eq!(states[0], 0);
/// assert_eq!(states[2], 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn viterbi_binary(
    prob: &[f32],
    log_trans: &Array2<f32>,
) -> Result<(f32, Vec<usize>), SequenceError> {
    let n_frames = prob.len();
    if n_frames == 0 {
        return Err(SequenceError::InvalidInput("prob is empty".into()));
    }
    if log_trans.shape() != [2, 2] {
        return Err(SequenceError::InvalidInput(
            "log_trans must be 2×2 for binary Viterbi".into(),
        ));
    }

    // Build 2×n_frames log-prob from posterior P(state=1)
    let log_prob = Array2::from_shape_fn((2, n_frames), |(s, t)| {
        let p = prob[t].clamp(f32::MIN_POSITIVE, 1.0 - f32::MIN_POSITIVE);
        if s == 1 { p.ln() } else { (1.0 - p).ln() }
    });

    viterbi_impl(&log_prob, log_trans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // DTW tests

    #[test]
    fn test_dtw_identical_sequences() {
        let x = Array2::from_shape_fn((2, 5), |(i, j)| (i + j) as f32);
        let (cost, path) = dtw_impl(&x, &x, DtwMetric::Euclidean).unwrap();
        assert!(cost < 1e-5, "identical sequences should have zero cost");
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(4, 4)));
    }

    #[test]
    fn test_dtw_path_endpoints() {
        let x = Array2::from_elem((3, 10), 1.0_f32);
        let y = Array2::from_elem((3, 7), 1.0_f32);
        let (_, path) = dtw_impl(&x, &y, DtwMetric::Euclidean).unwrap();
        assert_eq!(path[0], (0, 0));
        assert_eq!(*path.last().unwrap(), (9, 6));
    }

    #[test]
    fn test_dtw_empty_error() {
        let empty: Array2<f32> = Array2::zeros((2, 0));
        let y = Array2::zeros((2, 5));
        assert!(dtw_impl(&empty, &y, DtwMetric::Euclidean).is_err());
    }

    #[test]
    fn test_dtw_dimension_mismatch_error() {
        let x = Array2::zeros((2, 5));
        let y = Array2::zeros((3, 5));
        assert!(dtw_impl(&x, &y, DtwMetric::Euclidean).is_err());
    }

    #[test]
    fn test_dtw_cosine_metric() {
        let x = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let y = arr2(&[[1.0_f32], [0.0]]);
        let (cost, _) = dtw_impl(&x, &y, DtwMetric::Cosine).unwrap();
        assert!(cost >= 0.0);
    }

    // Viterbi tests

    #[test]
    fn test_viterbi_trivial() {
        // One state → always stays in state 0
        let log_prob = Array2::from_elem((1, 5), 0.0_f32);
        let log_trans = Array2::from_elem((1, 1), 0.0_f32);
        let (_, states) = viterbi_impl(&log_prob, &log_trans).unwrap();
        assert_eq!(states, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_viterbi_follows_best_emission() {
        // State 0 is strongly preferred in frames 0–2, state 1 in frames 3–4
        let mut log_prob = Array2::from_elem((2, 5), -10.0_f32);
        for t in 0..3 {
            log_prob[[0, t]] = 0.0;
        }
        for t in 3..5 {
            log_prob[[1, t]] = 0.0;
        }
        let log_trans = Array2::from_elem((2, 2), -0.1_f32);
        let (_, states) = viterbi_impl(&log_prob, &log_trans).unwrap();
        assert_eq!(&states[0..3], &[0, 0, 0]);
        assert_eq!(&states[3..5], &[1, 1]);
    }

    #[test]
    fn test_viterbi_empty_error() {
        let log_prob: Array2<f32> = Array2::zeros((0, 5));
        let log_trans: Array2<f32> = Array2::zeros((0, 0));
        assert!(viterbi_impl(&log_prob, &log_trans).is_err());
    }

    #[test]
    fn test_viterbi_shape_mismatch_error() {
        let log_prob = Array2::from_elem((3, 10), -1.0_f32);
        let log_trans = Array2::from_elem((2, 2), -1.0_f32); // wrong n_states
        assert!(viterbi_impl(&log_prob, &log_trans).is_err());
    }
}
