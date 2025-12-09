use ndarray::{Array1, Array2};
use thiserror::Error;

/// Custom error types for signal manipulation operations.
#[derive(Error, Debug)]
pub enum ManipulationError {
    /// Invalid input parameters or data.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Stacks delayed copies of a 2D array for temporal context.
///
/// # Arguments
/// * `data` - Input 2D array (features × time)
/// * `n_steps` - Optional number of delayed copies (defaults to 2)
/// * `delay` - Optional delay between steps in frames (defaults to 1)
///
/// # Returns
/// Returns a 2D array of shape `(n_features * n_steps, n_frames)` containing stacked features.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let stacked = stack_memory(&data, None, None);
/// ```
pub fn stack_memory(
    data: &Array2<f32>,
    n_steps: Option<usize>,
    delay: Option<usize>,
) -> Array2<f32> {
    let n_steps = n_steps.unwrap_or(2);
    let delay = delay.unwrap_or(1);
    let n_frames = data.shape()[1];
    let n_features = data.shape()[0];
    let mut stacked = Array2::zeros((n_features * n_steps, n_frames));
    for step in 0..n_steps {
        let offset = step * delay;
        for t in 0..n_frames {
            let src_t = (t as isize - offset as isize).max(0) as usize;
            for f in 0..n_features {
                stacked[[f + step * n_features, t]] = data[[f, src_t]];
            }
        }
    }
    stacked
}

/// Computes temporal kurtosis from audio signal.
///
/// Kurtosis measures the "tailedness" of the distribution in each frame.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
///
/// # Returns
/// Returns a builder that can be configured with method chaining.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4, 0.5];
/// let kurtosis = temporal_kurtosis(&y)
///     .frame_length(2048)
///     .hop_length(512)
///     .compute()?;
/// ```
pub fn temporal_kurtosis(y: &[f32]) -> TemporalKurtosisBuilder {
    TemporalKurtosisBuilder {
        y,
        frame_length: 2048,
        hop_length: 512,
    }
}

/// Temporal kurtosis builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct TemporalKurtosisBuilder<'a> {
    y: &'a [f32],
    frame_length: usize,
    hop_length: usize,
}

impl<'a> TemporalKurtosisBuilder<'a> {
    /// Set the frame length (default: 2048).
    pub fn frame_length(mut self, frame_length: usize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Set the hop length (default: 512).
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Compute temporal kurtosis with the configured parameters.
    pub fn compute(self) -> Result<Array1<f32>, ManipulationError> {
        temporal_kurtosis_impl(self.y, self.frame_length, self.hop_length)
    }
}

/// Internal temporal kurtosis implementation.
fn temporal_kurtosis_impl(
    y: &[f32],
    frame_length: usize,
    hop_length: usize,
) -> Result<Array1<f32>, ManipulationError> {
    let frame_len = frame_length;
    let hop = hop_length;
    let n_frames = (y.len() - frame_len) / hop + 1;
    let mut kurtosis = Array1::zeros(n_frames);
    for i in 0..n_frames {
        let start = i * hop;
        let frame = &y[start..(start + frame_len).min(y.len())];
        let mean = frame.iter().sum::<f32>() / frame.len() as f32;
        let m2 = frame.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / frame.len() as f32;
        let m4 = frame.iter().map(|&x| (x - mean).powi(4)).sum::<f32>() / frame.len() as f32;
        kurtosis[i] = if m2 > 1e-10 { m4 / m2.powi(2) - 3.0 } else { 0.0 };
    }
    Ok(kurtosis)
}

/// Computes zero-crossing rate from an audio signal.
///
/// Measures the rate at which the signal changes sign in each frame.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
///
/// # Returns
/// Returns a builder that can be configured with method chaining.
///
/// # Examples
/// ```
/// let y = vec![1.0, -1.0, 2.0, -2.0, 1.0];
/// let zcr = zero_crossing_rate(&y)
///     .frame_length(2048)
///     .hop_length(512)
///     .compute();
/// ```
pub fn zero_crossing_rate(y: &[f32]) -> ZeroCrossingRateBuilder {
    ZeroCrossingRateBuilder {
        y,
        frame_length: 2048,
        hop_length: 512,
    }
}

/// Zero-crossing rate builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct ZeroCrossingRateBuilder<'a> {
    y: &'a [f32],
    frame_length: usize,
    hop_length: usize,
}

impl<'a> ZeroCrossingRateBuilder<'a> {
    /// Set the frame length (default: 2048).
    pub fn frame_length(mut self, frame_length: usize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Set the hop length (default: 512).
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Compute zero-crossing rate with the configured parameters.
    pub fn compute(self) -> Array1<f32> {
        zero_crossing_rate_impl(self.y, self.frame_length, self.hop_length)
    }
}

/// Internal zero-crossing rate implementation.
fn zero_crossing_rate_impl(
    y: &[f32],
    frame_length: usize,
    hop_length: usize,
) -> Array1<f32> {
    let frame_len = frame_length;
    let hop = hop_length;
    let n_frames = (y.len() - frame_len) / hop + 1;
    let mut zcr = Array1::zeros(n_frames);
    for i in 0..n_frames {
        let start = i * hop;
        let slice = &y[start..(start + frame_len).min(y.len())];
        let count = slice.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        zcr[i] = count as f32 / frame_len as f32;
    }
    zcr
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_stack_memory() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = stack_memory(&data, Some(2), Some(1));
        let expected = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [1.0, 1.0, 2.0],
            [4.0, 4.0, 5.0]
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_temporal_kurtosis_y() {
        let y = vec![1.0, -1.0, 1.0, -1.0];
        let result = temporal_kurtosis(&y)
            .frame_length(4)
            .hop_length(4)
            .compute()
            .unwrap();
        let expected = array![-2.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let y = vec![1.0, -1.0, 2.0, -2.0, 1.0];
        let result = zero_crossing_rate(&y)
            .frame_length(2)
            .hop_length(1)
            .compute();
        let expected = array![0.5, 0.5, 0.5, 0.5];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_zero_crossing_rate_full() {
        let y = vec![1.0, -1.0, 2.0, -2.0, 1.0];
        let result = zero_crossing_rate(&y)
            .frame_length(5)
            .hop_length(5)
            .compute();
        let expected = array![0.8];
        assert_eq!(result, expected);
    }
}