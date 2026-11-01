//! Framewise YIN and the two-stage pYIN model of Mauch & Dixon (2014).
//! Uses fixed half-frame difference windows, a discretized Beta(2, 18)
//! threshold prior, and a banded voiced/unvoiced pitch HMM.

use super::tuning::TuningError;

const BINS_PER_OCTAVE: f64 = 120.0;

/// Probabilistic pitch estimates and voicing information, one value per complete frame.
#[derive(Debug, Clone)]
pub struct PyinResult {
    /// Viterbi-decoded fundamental frequency in Hz, or zero for unvoiced frames.
    pub f0: Vec<f32>,
    /// Voiced/unvoiced decisions from the decoded sequence.
    pub voiced_flag: Vec<bool>,
    /// Frame-local voiced probabilities before sequence decoding, in `[0, 1]`.
    /// These are model scores, not calibrated confidence in the final pitch.
    pub voiced_prob: Vec<f32>,
}

pub(super) struct Config {
    pub sr: u32,
    pub fmin: f32,
    pub fmax: f32,
    pub frame_length: usize,
    pub hop_length: Option<usize>,
}

struct Analysis {
    min_lag: usize,
    max_lag: usize,
    hop: usize,
    frames: usize,
}

impl Config {
    fn validate(&self, signal: &[f32]) -> Result<Analysis, TuningError> {
        if self.sr == 0
            || !self.fmin.is_finite()
            || !self.fmax.is_finite()
            || self.fmin <= 0.0
            || self.fmin >= self.fmax
            || self.fmax > self.sr as f32 / 2.0
        {
            return Err(TuningError::InvalidFrequencyRange(
                "Require 0 < fmin < fmax <= sr/2 and a positive sample rate".into(),
            ));
        }
        if self.frame_length < 4 || self.hop_length == Some(0) {
            return Err(TuningError::InvalidInput(
                "Frame length must be at least 4 and hop length must be positive".into(),
            ));
        }
        if signal.len() < self.frame_length {
            return Err(TuningError::InsufficientData(
                "Signal is shorter than one frame".into(),
            ));
        }
        // Bound periods before converting to usize or allocating. Reserve a
        // neighbouring lag for interpolation and retain a fixed comparison window.
        let max_period = (f64::from(self.sr) / f64::from(self.fmin)).ceil();
        if max_period > (self.frame_length - self.frame_length / 2 - 1) as f64 {
            return Err(TuningError::InsufficientData(
                "Frame length must accommodate two periods at fmin plus an interpolation neighbour"
                    .into(),
            ));
        }
        if signal.iter().any(|v| !v.is_finite()) {
            return Err(TuningError::InvalidInput(
                "Audio samples must be finite".into(),
            ));
        }
        let hop = self.hop_length.unwrap_or(self.frame_length / 4);
        Ok(Analysis {
            min_lag: (f64::from(self.sr) / f64::from(self.fmax)).floor().max(2.0) as usize,
            max_lag: max_period as usize,
            hop,
            frames: 1 + (signal.len() - self.frame_length) / hop,
        })
    }
}

// Constant window size prevents long lags from winning simply because they
// compare fewer samples. f64 accumulation preserves both very quiet and loud f32 audio.
fn normalized_difference(frame: &[f32], max_lag: usize) -> Vec<f64> {
    let window = frame.len() / 2;
    let mut cmnd = vec![1.0; max_lag + 2];
    let mut cumulative = 0.0;
    for lag in 1..cmnd.len() {
        let difference: f64 = (0..window)
            .map(|j| (f64::from(frame[j]) - f64::from(frame[j + lag])).powi(2))
            .sum();
        cumulative += difference;
        if cumulative > 0.0 {
            cmnd[lag] = difference * lag as f64 / cumulative;
        }
    }
    cmnd
}

fn troughs(cmnd: &[f64], analysis: &Analysis) -> Vec<usize> {
    (analysis.min_lag..=analysis.max_lag)
        .filter(|&lag| cmnd[lag] < cmnd[lag - 1] && cmnd[lag] <= cmnd[lag + 1])
        .collect()
}

fn frequency(cmnd: &[f64], lag: usize, config: &Config) -> f64 {
    let denominator = cmnd[lag - 1] - 2.0 * cmnd[lag] + cmnd[lag + 1];
    let shift = if denominator > 0.0 {
        (0.5 * (cmnd[lag - 1] - cmnd[lag + 1]) / denominator).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    (f64::from(config.sr) / (lag as f64 + shift))
        .clamp(f64::from(config.fmin), f64::from(config.fmax))
}

pub(super) fn yin(
    signal: &[f32],
    config: &Config,
    threshold: f32,
) -> Result<Vec<f32>, TuningError> {
    let analysis = config.validate(signal)?;
    if !threshold.is_finite() || threshold <= 0.0 || threshold >= 1.0 {
        return Err(TuningError::InvalidInput(
            "YIN threshold must be finite and in (0, 1)".into(),
        ));
    }
    Ok((0..analysis.frames)
        .map(|t| {
            let start = t * analysis.hop;
            let cmnd = normalized_difference(
                &signal[start..start + config.frame_length],
                analysis.max_lag,
            );
            troughs(&cmnd, &analysis)
                .into_iter()
                .find(|&lag| cmnd[lag] < f64::from(threshold))
                .map_or(0.0, |lag| frequency(&cmnd, lag, config) as f32)
        })
        .collect())
}

// Exact Beta(2,18) CDF. Its mean is 0.1; integrating bins avoids a dependency
// on a statistics library and ensures the 100 threshold masses sum to one.
fn beta_cdf(x: f64) -> f64 {
    1.0 - (1.0 - x).powi(18) * (1.0 + 18.0 * x)
}

fn candidate_probabilities(cmnd: &[f64], candidates: &[usize]) -> Vec<f64> {
    let mut probabilities = vec![0.0; candidates.len()];
    let Some((best, _)) = candidates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| cmnd[**a].total_cmp(&cmnd[**b]))
    else {
        return probabilities;
    };
    for i in 0..100 {
        let lower = f64::from(i) / 100.0;
        let upper = f64::from(i + 1) / 100.0;
        let mass = beta_cdf(upper) - beta_cdf(lower);
        if let Some(index) = candidates.iter().position(|&lag| cmnd[lag] < upper) {
            probabilities[index] += mass;
        } else {
            // Absolute-minimum fallback gets 1% of mass; the remainder is unvoiced.
            probabilities[best] += 0.01 * mass;
        }
    }
    probabilities
}

pub(super) fn pyin(
    signal: &[f32],
    config: &Config,
    max_transition_rate: f32,
    switch_probability: f32,
) -> Result<PyinResult, TuningError> {
    let analysis = config.validate(signal)?;
    if !max_transition_rate.is_finite()
        || max_transition_rate <= 0.0
        || !switch_probability.is_finite()
        || switch_probability <= 0.0
        || switch_probability >= 1.0
    {
        return Err(TuningError::InvalidInput(
            "Require a positive finite transition rate and switch probability in (0, 1)".into(),
        ));
    }
    let bins = (BINS_PER_OCTAVE * (f64::from(config.fmax) / f64::from(config.fmin)).log2()).floor()
        as usize
        + 1;
    let radius = (f64::from(max_transition_rate) * analysis.hop as f64 / f64::from(config.sr)
        * BINS_PER_OCTAVE)
        .floor()
        .min((bins - 1) as f64) as usize;
    let mut voiced_prob = Vec::with_capacity(analysis.frames);
    let observations = (0..analysis.frames).map(|t| {
        let start = t * analysis.hop;
        let cmnd = normalized_difference(
            &signal[start..start + config.frame_length],
            analysis.max_lag,
        );
        let candidates = troughs(&cmnd, &analysis);
        let probabilities = candidate_probabilities(&cmnd, &candidates);
        let mut observation = vec![0.0; 2 * bins];
        for (&lag, probability) in candidates.iter().zip(probabilities) {
            let pitch = frequency(&cmnd, lag, config);
            let bin = (BINS_PER_OCTAVE * (pitch / f64::from(config.fmin)).log2()).round() as usize;
            observation[bin.min(bins - 1)] += probability;
        }
        let voiced = observation[..bins].iter().sum::<f64>().clamp(0.0, 1.0);
        observation[bins..].fill((1.0 - voiced) / bins as f64);
        voiced_prob.push(voiced as f32);
        observation
    });
    let path = decode_path(
        observations,
        analysis.frames,
        bins,
        radius,
        f64::from(switch_probability),
    )?;
    let voiced_flag: Vec<bool> = path.iter().map(|&state| state < bins).collect();
    let f0 = path
        .iter()
        .map(|&state| {
            if state < bins {
                (f64::from(config.fmin) * 2.0_f64.powf(state as f64 / BINS_PER_OCTAVE)) as f32
            } else {
                0.0
            }
        })
        .collect();
    Ok(PyinResult {
        f0,
        voiced_flag,
        voiced_prob,
    })
}

// Banded Viterbi: retain two score vectors and a traceback, not a dense
// states-by-states transition matrix or all frame emissions.
fn decode_path(
    observations: impl Iterator<Item = Vec<f64>>,
    frames: usize,
    bins: usize,
    radius: usize,
    switch_probability: f64,
) -> Result<Vec<usize>, TuningError> {
    let states = 2 * bins;
    let size = frames
        .checked_mul(states)
        .ok_or_else(|| TuningError::InvalidInput("Pitch traceback size overflows".into()))?;
    let mut traceback = Vec::<u32>::new();
    traceback.try_reserve_exact(size).map_err(|e| {
        TuningError::ComputationFailed(format!("Cannot allocate pitch traceback: {e}"))
    })?;
    traceback.resize(size, 0);
    let mut incoming = vec![Vec::new(); bins];
    for from in 0..bins {
        let low = from.saturating_sub(radius);
        let high = (from + radius).min(bins - 1);
        let total: usize = (low..=high).map(|to| radius + 1 - from.abs_diff(to)).sum();
        for (to, predecessors) in incoming.iter_mut().enumerate().take(high + 1).skip(low) {
            let probability = (radius + 1 - from.abs_diff(to)) as f64 / total as f64;
            predecessors.push((from, probability.ln()));
        }
    }
    let stay = (1.0 - switch_probability).ln();
    let switch = switch_probability.ln();
    let mut previous = vec![0.0; states];
    let mut next = vec![0.0; states];
    for (t, observation) in observations.enumerate() {
        if t == 0 {
            // Uniform initial state prior (the common constant can be omitted).
            for (score, probability) in previous.iter_mut().zip(observation) {
                *score = probability.max(1e-300).ln();
            }
            continue;
        }
        for (bin, predecessors) in incoming.iter().enumerate() {
            let mut voiced = (0, f64::NEG_INFINITY);
            let mut unvoiced = (bins, f64::NEG_INFINITY);
            for &(from, log_pitch) in predecessors {
                if previous[from] + log_pitch > voiced.1 {
                    voiced = (from, previous[from] + log_pitch);
                }
                if previous[from + bins] + log_pitch > unvoiced.1 {
                    unvoiced = (from + bins, previous[from + bins] + log_pitch);
                }
            }
            let best_voiced = if voiced.1 + stay >= unvoiced.1 + switch {
                (voiced.0, voiced.1 + stay)
            } else {
                (unvoiced.0, unvoiced.1 + switch)
            };
            let best_unvoiced = if unvoiced.1 + stay >= voiced.1 + switch {
                (unvoiced.0, unvoiced.1 + stay)
            } else {
                (voiced.0, voiced.1 + switch)
            };
            for (state, best) in [(bin, best_voiced), (bin + bins, best_unvoiced)] {
                next[state] = best.1 + observation[state].max(1e-300).ln();
                traceback[t * states + state] = best.0 as u32;
            }
        }
        let maximum = next.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for score in &mut next {
            *score -= maximum;
        }
        std::mem::swap(&mut previous, &mut next);
    }
    let mut state = previous
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(index, _)| index);
    let mut path = vec![0; frames];
    for t in (0..frames).rev() {
        path[t] = state;
        state = traceback[t * states + state] as usize;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_retain_multiple_candidates_and_account_for_unvoiced_mass() {
        let cmnd = [1.0, 1.0, 0.08, 0.5, 0.02, 1.0];
        let probabilities = candidate_probabilities(&cmnd, &[2, 4]);
        assert!(probabilities.iter().all(|&p| p > 0.0));
        // Partition the prior analytically: thresholds above 0.08 choose the
        // first trough; thresholds in (0.02, 0.08] choose the second trough.
        assert!((probabilities[0] - (1.0 - beta_cdf(0.08))).abs() < 1e-12);
        assert!((probabilities[1] - (beta_cdf(0.08) - 0.99 * beta_cdf(0.02))).abs() < 1e-12);
        assert!(probabilities.iter().sum::<f64>() <= 1.0 + 1e-12);
        let fallback = candidate_probabilities(&[1.0, 2.0, 1.1, 2.0], &[2]);
        assert!((fallback[0] - 0.01).abs() < 1e-12);
        assert!((beta_cdf(1.0) - beta_cdf(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn switch_probability_controls_brief_voicing_changes() {
        let observations = vec![vec![0.99, 0.01], vec![0.4, 0.6], vec![0.99, 0.01]];
        assert_eq!(
            decode_path(observations.clone().into_iter(), 3, 1, 0, 0.01).unwrap(),
            [0, 0, 0]
        );
        assert_eq!(
            decode_path(observations.into_iter(), 3, 1, 0, 0.5).unwrap(),
            [0, 1, 0]
        );
    }

    #[test]
    fn long_sequences_do_not_underflow_the_path_scores() {
        let frames = 4096;
        let observations = (0..frames).map(|_| vec![0.9, 0.1]);
        let path = decode_path(observations, frames, 1, 0, 0.01).unwrap();
        assert_eq!(path, vec![0; frames]);
    }

    #[test]
    fn viterbi_recovers_a_secondary_candidate_using_context() {
        let observations: Vec<Vec<f64>> = vec![
            vec![0.98, 0.001, 0.009, 0.003, 0.003, 0.004],
            vec![0.40, 0.001, 0.589, 0.003, 0.003, 0.004],
            vec![0.98, 0.001, 0.009, 0.003, 0.003, 0.004],
        ];
        // The middle frame's independent winner is bin 2. A radius-one model
        // instead retains its bin-0 alternative and the continuous path.
        assert_eq!(
            decode_path(observations.into_iter(), 3, 3, 1, 0.01).unwrap(),
            [0, 0, 0]
        );
    }

    #[test]
    fn banded_viterbi_matches_exhaustive_path_enumeration() {
        // Dense transition matrices calculated by hand, including the different
        // normalization at pitch boundaries. Enumerate all paths independently
        // of the banded decoder, including forbidden jumps and one-bin ranges.
        let models = [
            (0, vec![vec![1.0]]),
            (0, vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            (
                1,
                vec![vec![2.0 / 3.0, 1.0 / 3.0], vec![1.0 / 3.0, 2.0 / 3.0]],
            ),
            (
                0,
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
            ),
            (
                1,
                vec![
                    vec![2.0 / 3.0, 1.0 / 3.0, 0.0],
                    vec![0.25, 0.5, 0.25],
                    vec![0.0, 1.0 / 3.0, 2.0 / 3.0],
                ],
            ),
            (
                2,
                vec![
                    vec![0.5, 1.0 / 3.0, 1.0 / 6.0],
                    vec![2.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0],
                    vec![1.0 / 6.0, 1.0 / 3.0, 0.5],
                ],
            ),
        ];
        for (radius, pitch) in models {
            let bins = pitch.len();
            let states = 2 * bins;
            let observations: Vec<Vec<f64>> = [
                [0.5, 0.1, 0.3, 0.1, 0.17, 0.23],
                [0.05, 0.45, 0.1, 0.4, 0.21, 0.15],
                [0.1, 0.2, 0.25, 0.45, 0.12, 0.33],
                [0.3, 0.13, 0.2, 0.25, 0.18, 0.17],
            ]
            .iter()
            .map(|row| row[..states].to_vec())
            .collect();
            for switch_probability in [0.01, 0.2, 0.8] {
                let path_probability = |path: &[usize]| {
                    let mut probability = observations[0][path[0]];
                    for t in 1..path.len() {
                        let voicing = if path[t] / bins == path[t - 1] / bins {
                            1.0 - switch_probability
                        } else {
                            switch_probability
                        };
                        probability *= observations[t][path[t]]
                            * pitch[path[t - 1] % bins][path[t] % bins]
                            * voicing;
                    }
                    probability
                };
                let mut best = 0.0_f64;
                for mut encoded in 0..states.pow(4) {
                    let mut path = [0; 4];
                    for state in &mut path {
                        *state = encoded % states;
                        encoded /= states;
                    }
                    best = best.max(path_probability(&path));
                }
                let path = decode_path(
                    observations.clone().into_iter(),
                    4,
                    bins,
                    radius,
                    switch_probability,
                )
                .unwrap();
                assert!(
                    (path_probability(&path) - best).abs() < 1e-14,
                    "bins={bins}, radius={radius}, switch={switch_probability}, path={path:?}"
                );
            }
        }
    }
}
