use ndarray::Array2;
use num_complex::Complex;
use thiserror::Error;

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
fn griffinlim_impl(s: &Array2<f32>, n_iter: usize, hop_length: Option<usize>) -> Result<Vec<f32>, PhaseRecoveryError> {
    let n_fft = (s.shape()[0] - 1) * 2;
    let hop = hop_length.unwrap_or(n_fft / 4).max(1);
    let signal_len = hop * (s.shape()[1] - 1) + n_fft;
    let mut y = crate::signal_generation::generators::tone(440.0, 44100)
        .duration(signal_len as f32 / 44100.0)
        .compute();
    for _ in 0..n_iter {
        let stft_y = crate::signal_processing::time_frequency::stft(&y)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| PhaseRecoveryError::ComputationFailed(format!("STFT computation failed: {e}")))?;
        let (mut mag, mut phase) = crate::signal_processing::time_frequency::magphase(&stft_y, None);
        // Griffin-Lim: replace magnitude with target magnitude, keep phase from current estimate
        for ((i, j), m) in mag.indexed_iter_mut() {
            *m = s[[i, j]];  // Use given magnitude directly (not sqrt, as s is already magnitude)
            let p = &mut phase[[i, j]];
            if m.abs() > 1e-10 {
                *p /= p.norm();  // Normalize phase to unit magnitude
            }
        }
        // Create complex spectrogram: magnitude * phase (element-wise)
        let new_stft = mag.mapv(|x| Complex::new(x, 0.0)) * phase;
        y = crate::signal_processing::time_frequency::istft(&new_stft)
            .hop_length(hop)
            .length(signal_len)
            .compute();
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
/// signal converges.
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
    GriffinLimCqtBuilder { c_mag, sr, n_iter: 32, hop_length: 512, fmin: 32.70 }
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

    let n_samples = n_frames * hop_length;

    // Deterministic pseudo-random initialization avoids phase-lock artifacts.
    let mut y: Vec<f32> = (0..n_samples)
        .map(|i| (i as f32 * 0.031_415_93).sin() * 0.01)
        .collect();

    for _ in 0..n_iter {
        let c = cqt(&y, sr)
            .hop_length(hop_length)
            .fmin(fmin)
            .n_bins(n_bins)
            .compute()
            .map_err(|e| PhaseRecoveryError::ComputationFailed(e.to_string()))?;

        let (_mag, phase) = magphase(&c, None);

        // Replace magnitude with the target while keeping the current phase.
        let (nk, nf) = (
            n_bins.min(phase.shape()[0]),
            n_frames.min(phase.shape()[1]),
        );
        let mut c_new = Array2::from_elem(phase.raw_dim(), Complex::new(0.0_f32, 0.0));
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
        assert!(result.is_ok(), "griffinlim_cqt should succeed: {:?}", result.err());
        let y = result.unwrap();
        assert!(!y.is_empty());
    }
}
