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
