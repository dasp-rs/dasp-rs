use ndarray::Array2;
use num_complex::Complex;
use thiserror::Error;

// A fixed integer generator gives reproducible broadband initial energy. A
// single sinusoid leaves near-zero FFT bins whose phase is dominated by roundoff.
fn initial_signal(length: usize) -> Vec<f32> {
    let mut state = 0x1234_5678_u32;
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 * (1.0 / 16_777_216.0) - 0.5) * 0.02
        })
        .collect()
}

/// Custom error types for phase recovery operations.
#[derive(Error, Debug)]
pub enum PhaseRecoveryError {
    /// Computation failed during processing.
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Reconstructs a time-domain signal from an STFT magnitude spectrogram using the Griffin-Lim algorithm.
///
/// Iteratively refines a signal estimate by enforcing consistency with the given magnitude spectrogram.
/// Uses a centered, periodic Hann STFT with an inferred even FFT size. Output
/// length is `hop_length * (n_frames - 1)`; the original trailing partial hop
/// cannot be inferred from magnitudes alone. A single frame yields empty output.
/// Initialization uses deterministic broadband noise. Recovered phase is not
/// unique, and output need not match a differently initialized implementation.
///
/// # Arguments
/// * `s` - Magnitude spectrogram (shape: `[n_freqs, n_frames]`)
///
/// # Returns
/// Returns a builder that can be configured with method chaining.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::*;
/// use dasp_rs::types::*;
/// use ndarray::Array2;
/// let mag_spectrogram = Array2::from_shape_vec((513, 10), vec![1.0; 513 * 10]).unwrap();
/// let signal = griffinlim(&mag_spectrogram)
///     .n_iter(32)
///     .hop_length(256)
///     .compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn griffinlim(s: &Array2<f32>) -> GriffinLimBuilder<'_> {
    GriffinLimBuilder {
        s,
        n_iter: 32,
        hop_length: None, // Will be calculated from spectrogram
    }
}

/// Griffin-Lim builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct GriffinLimBuilder<'a> {
    s: &'a Array2<f32>,
    n_iter: usize,
    hop_length: Option<usize>,
}

impl GriffinLimBuilder<'_> {
    /// Set the number of iterations (default: 32).
    #[must_use]
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the hop length (default: calculated from spectrogram).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Compute the reconstructed signal with the configured parameters.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Vec<f32>, PhaseRecoveryError> {
        griffinlim_impl(self.s, self.n_iter, self.hop_length)
    }
}

/// Internal Griffin-Lim implementation.
fn griffinlim_impl(
    s: &Array2<f32>,
    n_iter: usize,
    hop_length: Option<usize>,
) -> Result<Vec<f32>, PhaseRecoveryError> {
    if s.shape()[0] < 2 || s.shape()[1] == 0 {
        return Err(PhaseRecoveryError::ComputationFailed(
            "Require at least two frequency bins and one frame".into(),
        ));
    }
    let n_fft = (s.shape()[0] - 1) * 2;
    let hop = hop_length.unwrap_or_else(|| (n_fft / 4).max(1));
    if hop == 0 || s.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(PhaseRecoveryError::ComputationFailed(
            "Require positive hop length and finite, non-negative magnitudes".into(),
        ));
    }
    // A centered even-length STFT has 1 + signal_len / hop frames.
    // Adding n_fft here would introduce extra frames on every iteration.
    let signal_len = hop
        .checked_mul(s.shape()[1] - 1)
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or_else(|| PhaseRecoveryError::ComputationFailed("Signal length overflows".into()))?;
    if signal_len == 0 {
        return Ok(Vec::new());
    }
    // Deterministic initialization keeps repeated reconstructions reproducible.
    let mut y = initial_signal(signal_len);
    for _ in 0..n_iter {
        let stft_y = crate::signal_processing::time_frequency::stft(&y)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| {
                PhaseRecoveryError::ComputationFailed(format!("STFT computation failed: {e}"))
            })?;
        // Preserve phase where defined. Zero-valued bins have arbitrary phase;
        // selecting zero radians avoids dividing by zero (including on silence).
        let new_stft = Array2::from_shape_fn(s.dim(), |index| {
            let value = stft_y[index];
            let norm = value.norm();
            let phase = if norm > 0.0 {
                value / norm
            } else {
                Complex::new(1.0, 0.0)
            };
            phase * s[index]
        });
        y = crate::signal_processing::time_frequency::istft(&new_stft)
            .hop_length(hop)
            .length(signal_len)
            .compute()
            .map_err(|e| PhaseRecoveryError::ComputationFailed(format!("ISTFT failed: {e}")))?;
    }
    Ok(y)
}

// ─── Griffin-Lim CQT ──────────────────────────────────────────────────────────

/// Builder for [`griffinlim_cqt`].
#[derive(Debug, Clone)]
pub struct GriffinLimCqtBuilder<'a> {
    c_mag: &'a Array2<f32>,
    sr: u32,
    n_iter: usize,
    hop_length: usize,
    fmin: f32,
}

impl GriffinLimCqtBuilder<'_> {
    /// Set the number of Griffin-Lim iterations (default: 32).
    #[must_use]
    pub fn n_iter(mut self, v: usize) -> Self {
        self.n_iter = v;
        self
    }

    /// Set the hop length in samples (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Set the minimum frequency in Hz (default: 32.70, C1).
    #[must_use]
    pub fn fmin(mut self, v: f32) -> Self {
        self.fmin = v;
        self
    }

    /// Reconstruct the time-domain signal.
    ///
    /// # Errors
    /// Returns an error if the magnitude spectrogram is empty or a CQT/iCQT
    /// operation fails.
    pub fn compute(self) -> Result<Vec<f32>, PhaseRecoveryError> {
        griffinlim_cqt_impl(self.c_mag, self.sr, self.n_iter, self.hop_length, self.fmin)
    }
}

/// Reconstructs a time-domain signal from a CQT magnitude spectrogram.
///
/// Uses the Griffin-Lim algorithm in the CQT domain: alternates between CQT
/// analysis (enforcing the target magnitude) and iCQT synthesis until the
/// signal converges. The centered analysis frame grid implies an output length
/// of `hop_length * (n_frames - 1)`; a single frame yields empty output.
/// Initialization uses the same deterministic broadband noise as [`griffinlim`].
///
/// # Arguments
/// * `c_mag` - CQT magnitude spectrogram of shape `(n_bins, n_frames)`
/// * `sr` - Sample rate in Hz
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::griffinlim_cqt;
/// use ndarray::Array2;
/// let mag = Array2::from_elem((84, 100), 0.1_f32);
/// let signal = griffinlim_cqt(&mag, 44100).n_iter(16).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn griffinlim_cqt(c_mag: &Array2<f32>, sr: u32) -> GriffinLimCqtBuilder<'_> {
    GriffinLimCqtBuilder {
        c_mag,
        sr,
        n_iter: 32,
        hop_length: 512,
        fmin: 32.70,
    }
}

fn griffinlim_cqt_impl(
    c_mag: &Array2<f32>,
    sr: u32,
    n_iter: usize,
    hop_length: usize,
    fmin: f32,
) -> Result<Vec<f32>, PhaseRecoveryError> {
    use crate::signal_processing::time_frequency::{cqt, icqt, magphase};

    let n_bins = c_mag.shape()[0];
    let n_frames = c_mag.shape()[1];

    if n_bins == 0 || n_frames == 0 {
        return Err(PhaseRecoveryError::ComputationFailed(
            "Empty CQT magnitude spectrogram".into(),
        ));
    }

    if sr == 0
        || hop_length == 0
        || !fmin.is_finite()
        || fmin <= 0.0
        || fmin > sr as f32 / 2.0
        || c_mag.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(PhaseRecoveryError::ComputationFailed(
            "Require valid sample rate/hop/fmin and finite non-negative magnitudes".into(),
        ));
    }
    // Centered CQT analysis uses the same frame grid as STFT. Preserve that
    // geometry every iteration instead of accumulating one new frame each time.
    let n_samples = (n_frames - 1)
        .checked_mul(hop_length)
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or_else(|| PhaseRecoveryError::ComputationFailed("Signal length overflows".into()))?;
    if n_samples == 0 || n_iter == 0 {
        // Skipping iteration must not skip the transform's passband validation.
        // A one-sample zero probe uses bounded support regardless of fmin.
        cqt(&[0.0], sr)
            .hop_length(hop_length)
            .fmin(fmin)
            .n_bins(n_bins)
            .compute()
            .map_err(|e| PhaseRecoveryError::ComputationFailed(e.to_string()))?;
    }
    if n_samples == 0 {
        return Ok(Vec::new());
    }

    // Deterministic pseudo-random initialization avoids phase-lock artifacts.
    let mut y = initial_signal(n_samples);

    for _ in 0..n_iter {
        let c = cqt(&y, sr)
            .hop_length(hop_length)
            .fmin(fmin)
            .n_bins(n_bins)
            .compute()
            .map_err(|e| PhaseRecoveryError::ComputationFailed(e.to_string()))?;

        let (_mag, phase) = magphase(&c, None);

        // Replace magnitude with the target while keeping the current phase.
        let (nk, nf) = (n_bins.min(phase.shape()[0]), n_frames.min(phase.shape()[1]));
        let mut c_new = Array2::from_elem(c_mag.raw_dim(), Complex::new(0.0_f32, 0.0));
        for k in 0..nk {
            for t in 0..nf {
                c_new[[k, t]] = phase[[k, t]] * c_mag[[k, t]];
            }
        }

        y = icqt(&c_new)
            .sample_rate(sr)
            .hop_length(hop_length)
            .fmin(fmin)
            .compute()
            .map_err(|e| PhaseRecoveryError::ComputationFailed(e.to_string()))?;
        y.resize(n_samples, 0.0);
    }

    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_griffinlim_cqt_empty() {
        let empty = Array2::zeros((0, 0));
        assert!(griffinlim_cqt_impl(&empty, 44100, 4, 512, 32.7).is_err());
    }

    #[test]
    fn test_griffinlim_cqt_smoke() {
        let mag = Array2::from_elem((12, 20), 0.1_f32);
        let result = griffinlim_cqt_impl(&mag, 44100, 2, 512, 32.7);
        assert!(
            result.is_ok(),
            "griffinlim_cqt should succeed: {:?}",
            result.err()
        );
        let y = result.unwrap();
        assert_eq!(y.len(), 19 * 512);
    }
}
