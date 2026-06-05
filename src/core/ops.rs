use crate::core::AudioData;
use log::warn;
use rayon::prelude::*;
use thiserror::Error;

/// Enumerates error conditions for signal operation failures in DSP workflows.
///
/// Provides detailed diagnostics for operations on audio signals, tailored for debugging
/// and error recovery in production-grade audio processing pipelines.
///
/// # Error Variants
/// - `LengthMismatch`: Occurs in binary operations (e.g., `subtract_signals`, `multiply_signals`)
///   when input signals have different sample counts.
/// - `DivisionByZero`: Occurs in `divide_signals` or `scalar_operation` with `Divide` operation
///   when dividing by zero.
/// - `InvalidInput`: Occurs when inputs are invalid (e.g., empty signal array, mismatched metadata,
///   zero sample rate, or zero channels).
/// - `ComputationFailed`: Occurs when numerical errors (e.g., overflow, NaN) are detected.
///
/// # Notes
/// - Functions assume input `AudioData` instances have valid `samples` (non-empty unless specified).
/// - `AudioData` construction ensures non-zero `sample_rate` and `channels`, validated by
///   `AudioData::new`.
/// - Division-by-zero warnings are logged using the `log` crate. Users must configure a logging
///   backend (e.g., `env_logger`) to see these warnings.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum SignalOpError {
    #[error("Sample length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),

    #[error("Division by zero at sample index {0}")]
    DivisionByZero(usize),

    #[error("Invalid input parameter: {0}")]
    InvalidInput(String),

    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Formats a vector of indices into a comma-separated string.
#[allow(dead_code)]
fn format_indices(indices: &[usize]) -> String {
    indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<String>>()
        .join(", ")
}

/// Validates that two audio signals have compatible metadata (sample rate and channels).
#[allow(dead_code)]
fn validate_metadata(signal1: &AudioData, signal2: &AudioData) -> Result<(), SignalOpError> {
    if signal1.sample_rate != signal2.sample_rate || signal1.channels != signal2.channels {
        return Err(SignalOpError::InvalidInput(format!(
            "Metadata mismatch: expected {} Hz, {} channels; got {} Hz, {} channels",
            signal1.sample_rate, signal1.channels, signal2.sample_rate, signal2.channels
        )));
    }
    Ok(())
}

/// Mixes multiple audio signals by averaging their samples in parallel.
///
/// Computes the sample-wise mean of a slice of `AudioData` signals, producing a new
/// `AudioData` instance. All signals must share identical sample lengths, sample rates,
/// and channel counts. Parallelized using `rayon` for multi-core efficiency.
///
/// # Parameters
/// - `signals`: Slice of `AudioData` signals to mix.
///
/// # Returns
/// - `Ok(AudioData)`: Mixed signal with averaged samples.
/// - `Err(SignalOpError)`: Failure due to empty input, length mismatch, metadata inconsistency,
///   or invalid parameters (e.g., zero sample rate or channels).
///
/// # Examples
/// ```
/// use dasp_rs::{types::AudioData, ops::mix_signals};
/// let s1 = AudioData::new(vec![2.0, 4.0], 44100, 1)?;
/// let s2 = AudioData::new(vec![4.0, 6.0], 44100, 1)?;
/// let mixed = mix_signals(&[s1, s2])?;
/// assert_eq!(mixed.samples, vec![3.0, 5.0]);
///
/// // Mix stereo signals
/// let s1 = AudioData::new(vec![2.0, 1.0, 4.0, 2.0], 44100, 2)?;
/// let s2 = AudioData::new(vec![1.0, 2.0, 2.0, 4.0], 44100, 2)?;
/// let mixed = mix_signals(&[s1, s2])?;
/// assert_eq!(mixed.samples, vec![1.5, 1.5, 3.0, 3.0]);
/// assert_eq!(mixed.channels, 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(dead_code)]
pub fn mix_signals(signals: &[AudioData]) -> Result<AudioData, SignalOpError> {
    if signals.is_empty() {
        return Err(SignalOpError::InvalidInput(
            "Signal array is empty".to_string(),
        ));
    }

    let length = signals[0].samples.len();
    let sample_rate = signals[0].sample_rate;
    let channels = signals[0].channels;

    if length == 0 {
        return Err(SignalOpError::InvalidInput(
            "Signal samples are empty".to_string(),
        ));
    }

    let mismatches: Vec<_> = signals.iter().enumerate().skip(1)
        .filter_map(|(i, signal)| {
            if signal.samples.len() != length {
                Some(format!("Signal {}: length mismatch ({} vs {})", i, signal.samples.len(), length))
            } else if signal.sample_rate != sample_rate || signal.channels != channels {
                Some(format!(
                    "Signal {}: metadata mismatch (expected {} Hz, {} channels; got {} Hz, {} channels)",
                    i, sample_rate, channels, signal.sample_rate, signal.channels
                ))
            } else {
                None
            }
        })
        .collect();

    if !mismatches.is_empty() {
        return Err(SignalOpError::InvalidInput(mismatches.join("; ")));
    }

    let mixed_samples: Vec<f32> = (0..length)
        .into_par_iter()
        .map(|i| {
            let sum: f32 = signals.par_iter().map(|s| s.samples[i]).sum();
            sum / signals.len() as f32
        })
        .collect();

    let output = AudioData::new(mixed_samples, sample_rate, channels)
        .map_err(|e| SignalOpError::InvalidInput(e.to_string()))?;
    Ok(output)
}

/// Subtracts one audio signal from another with parallel sample processing.
///
/// Performs sample-wise subtraction (`signal1 - signal2`), producing a new `AudioData`.
/// Signals must have identical sample lengths, sample rates, and channel counts.
///
/// # Parameters
/// - `signal1`: Base signal (minuend).
/// - `signal2`: Signal to subtract (subtrahend).
///
/// # Returns
/// - `Ok(AudioData)`: Resulting difference signal.
/// - `Err(SignalOpError)`: Failure due to length mismatch, metadata mismatch, or computation errors.
///
/// # Examples
/// ```
/// use dasp_rs::{types::AudioData, ops::subtract_signals};
/// let s1 = AudioData::new(vec![3.0, 5.0], 44100, 1)?;
/// let s2 = AudioData::new(vec![1.0, 2.0], 44100, 1)?;
/// let result = subtract_signals(&s1, &s2)?;
/// assert_eq!(result.samples, vec![2.0, 3.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(dead_code)]
pub fn subtract_signals(
    signal1: &AudioData,
    signal2: &AudioData,
) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(
            signal1.samples.len(),
            signal2.samples.len(),
        ));
    }
    if signal1.samples.is_empty() {
        return Err(SignalOpError::InvalidInput(
            "Signal samples are empty".to_string(),
        ));
    }
    validate_metadata(signal1, signal2)?;

    let non_finite: Vec<_> = signal1
        .samples
        .par_iter()
        .zip(&signal2.samples)
        .enumerate()
        .filter_map(|(i, (&s1, &s2))| {
            let result = s1 - s2;
            if !result.is_finite() {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if !non_finite.is_empty() {
        return Err(SignalOpError::ComputationFailed(format!(
            "Non-finite results at indices {}",
            format_indices(&non_finite)
        )));
    }

    let samples: Vec<f32> = signal1
        .samples
        .par_iter()
        .zip(&signal2.samples)
        .map(|(&s1, &s2)| s1 - s2)
        .collect();

    let output = AudioData::new(samples, signal1.sample_rate, signal1.channels)
        .map_err(|e| SignalOpError::InvalidInput(e.to_string()))?;
    Ok(output)
}

/// Multiplies two audio signals sample-wise in parallel (e.g., amplitude modulation).
///
/// Computes the product of corresponding samples from `signal1` and `signal2`, producing
/// a new `AudioData`. Suitable for modulation effects. Signals must match in length,
/// sample rate, and channels.
///
/// # Parameters
/// - `signal1`: First signal (carrier or base).
/// - `signal2`: Second signal (modulator).
///
/// # Returns
/// - `Ok(AudioData)`: Product signal.
/// - `Err(SignalOpError)`: Failure due to length mismatch, metadata mismatch, or computation errors.
///
/// # Examples
/// ```
/// use dasp_rs::{types::AudioData, ops::multiply_signals};
/// let s1 = AudioData::new(vec![2.0, 3.0], 44100, 1)?;
/// let s2 = AudioData::new(vec![2.0, 2.0], 44100, 1)?;
/// let result = multiply_signals(&s1, &s2)?;
/// assert_eq!(result.samples, vec![4.0, 6.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(dead_code)]
pub fn multiply_signals(
    signal1: &AudioData,
    signal2: &AudioData,
) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(
            signal1.samples.len(),
            signal2.samples.len(),
        ));
    }
    if signal1.samples.is_empty() {
        return Err(SignalOpError::InvalidInput(
            "Signal samples are empty".to_string(),
        ));
    }
    validate_metadata(signal1, signal2)?;

    let non_finite: Vec<_> = signal1
        .samples
        .par_iter()
        .zip(&signal2.samples)
        .enumerate()
        .filter_map(|(i, (&s1, &s2))| {
            let result = s1 * s2;
            if !result.is_finite() {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if !non_finite.is_empty() {
        return Err(SignalOpError::ComputationFailed(format!(
            "Non-finite results at indices {}",
            format_indices(&non_finite)
        )));
    }

    let samples: Vec<f32> = signal1
        .samples
        .par_iter()
        .zip(&signal2.samples)
        .map(|(&s1, &s2)| s1 * s2)
        .collect();

    let output = AudioData::new(samples, signal1.sample_rate, signal1.channels)
        .map_err(|e| SignalOpError::InvalidInput(e.to_string()))?;
    Ok(output)
}

/// Divides one audio signal by another with parallel processing and zero handling.
///
/// Performs sample-wise division (`signal1 / signal2`), producing a new `AudioData`.
/// Handles division by zero by clamping to 0.0 and logging a warning. Signals must match
/// in length, sample rate, and channels.
///
/// # Parameters
/// - `signal1`: Numerator signal.
/// - `signal2`: Denominator signal.
///
/// # Returns
/// - `Ok(AudioData)`: Quotient signal.
/// - `Err(SignalOpError)`: Failure due to length mismatch, metadata mismatch, or computation errors.
///
/// # Examples
/// ```
/// use dasp_rs::{types::AudioData, ops::divide_signals};
/// let s1 = AudioData::new(vec![6.0, 8.0], 44100, 1)?;
/// let s2 = AudioData::new(vec![2.0, 4.0], 44100, 1)?;
/// let result = divide_signals(&s1, &s2)?;
/// assert_eq!(result.samples, vec![3.0, 2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(dead_code)]
pub fn divide_signals(
    signal1: &AudioData,
    signal2: &AudioData,
) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(
            signal1.samples.len(),
            signal2.samples.len(),
        ));
    }
    if signal1.samples.is_empty() {
        return Err(SignalOpError::InvalidInput(
            "Signal samples are empty".to_string(),
        ));
    }
    validate_metadata(signal1, signal2)?;

    let non_finite: Vec<_> = signal1
        .samples
        .par_iter()
        .zip(&signal2.samples)
        .enumerate()
        .filter_map(|(i, (&s1, &s2))| {
            if s2 == 0.0 {
                warn!("Division by zero at index {}, clamping to 0.0", i);
                return None;
            }
            let result = s1 / s2;
            if !result.is_finite() {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if !non_finite.is_empty() {
        return Err(SignalOpError::ComputationFailed(format!(
            "Non-finite results at indices {}",
            format_indices(&non_finite)
        )));
    }

    let samples: Vec<f32> = signal1
        .samples
        .par_iter()
        .zip(&signal2.samples)
        .map(|(&s1, &s2)| if s2 == 0.0 { 0.0 } else { s1 / s2 })
        .collect();

    let output = AudioData::new(samples, signal1.sample_rate, signal1.channels)
        .map_err(|e| SignalOpError::InvalidInput(e.to_string()))?;
    Ok(output)
}

/// Supported scalar operations for `scalar_operation`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum ScalarOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

/// Applies a scalar operation to an audio signal in parallel.
///
/// Performs element-wise addition, subtraction, multiplication, or division between
/// a signalÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢s samples and a scalar value, producing a new `AudioData`. Division by zero
/// is explicitly rejected.
///
/// # Parameters
/// - `signal`: Input signal.
/// - `scalar`: Scalar value for operation.
/// - `op`: Operation type (`ScalarOp::Add`, `ScalarOp::Subtract`, etc.).
///
/// # Returns
/// - `Ok(AudioData)`: Resulting signal.
/// - `Err(SignalOpError)`: Failure due to division by zero, computation errors, or invalid parameters.
///
/// # Examples
/// ```
/// use dasp_rs::{types::AudioData, ops::{scalar_operation, ScalarOp}};
/// let s = AudioData::new(vec![2.0, 3.0], 44100, 1)?;
/// let result = scalar_operation(&s, 2.0, ScalarOp::Multiply)?;
/// assert_eq!(result.samples, vec![4.0, 6.0]);
///
/// // Process individual channels
/// let stereo = AudioData::new(vec![2.0, 1.0, 4.0, 2.0], 44100, 2)?;
/// let channels = stereo.split_channels()?;
/// let scaled: Vec<Vec<f32>> = channels.into_iter()
///     .map(|ch| {
///         let audio = AudioData::new(ch, 44100, 1)?;
///         Ok(scalar_operation(&audio, 2.0, ScalarOp::Multiply)?.samples)
///     })
///     .collect::<Result<_, Box<dyn std::error::Error>>>()?;
/// assert_eq!(scaled, vec![vec![4.0, 8.0], vec![2.0, 4.0]]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(dead_code)]
pub fn scalar_operation(
    signal: &AudioData,
    scalar: f32,
    op: ScalarOp,
) -> Result<AudioData, SignalOpError> {
    if signal.samples.is_empty() {
        return Err(SignalOpError::InvalidInput(
            "Signal samples are empty".to_string(),
        ));
    }

    let non_finite: Vec<_> = signal
        .samples
        .par_iter()
        .enumerate()
        .filter_map(|(i, &s)| {
            let result = match op {
                ScalarOp::Add => s + scalar,
                ScalarOp::Subtract => s - scalar,
                ScalarOp::Multiply => s * scalar,
                ScalarOp::Divide => {
                    if scalar == 0.0 {
                        return Some(i); // Will trigger DivisionByZero later
                    }
                    s / scalar
                }
            };
            if !result.is_finite() {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if !non_finite.is_empty() {
        if matches!(op, ScalarOp::Divide) && scalar == 0.0 {
            return Err(SignalOpError::DivisionByZero(non_finite[0]));
        }
        return Err(SignalOpError::ComputationFailed(format!(
            "Non-finite results at indices {}",
            format_indices(&non_finite)
        )));
    }

    let samples: Vec<f32> = signal
        .samples
        .par_iter()
        .map(|&s| {
            match op {
                ScalarOp::Add => s + scalar,
                ScalarOp::Subtract => s - scalar,
                ScalarOp::Multiply => s * scalar,
                ScalarOp::Divide => s / scalar, // Already checked for zero
            }
        })
        .collect();

    let output = AudioData::new(samples, signal.sample_rate, signal.channels)
        .map_err(|e| SignalOpError::InvalidInput(e.to_string()))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::AudioError;

    fn test_signal(samples: Vec<f32>, sample_rate: u32, channels: u16) -> AudioData {
        AudioData::new(samples, sample_rate, channels).unwrap()
    }

    #[test]
    fn test_mix_signals_basic() {
        let s1 = test_signal(vec![1.0, 2.0, 3.0], 44100, 1);
        let s2 = test_signal(vec![2.0, 4.0, 6.0], 44100, 1);
        let mixed = mix_signals(&[s1, s2]).unwrap();
        assert_eq!(mixed.samples, vec![1.5, 3.0, 4.5]);
        assert_eq!(mixed.sample_rate, 44100);
        assert_eq!(mixed.channels, 1);
    }

    #[test]
    fn test_mix_signals_empty() {
        let result = mix_signals(&[]);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_mix_signals_empty_samples() {
        let s1 = AudioData::new(vec![], 44100, 1).unwrap();
        let result = mix_signals(&[s1]);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_mix_signals_length_mismatch() {
        let s1 = test_signal(vec![1.0, 2.0], 44100, 1);
        let s2 = test_signal(vec![2.0, 4.0, 6.0], 44100, 1);
        let result = mix_signals(&[s1, s2]);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_mix_signals_metadata_mismatch() {
        let s1 = test_signal(vec![1.0, 2.0], 44100, 1);
        let s2 = test_signal(vec![2.0, 4.0], 48000, 1);
        let result = mix_signals(&[s1, s2]);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_mix_signals_concurrent_large() {
        let signal = test_signal(vec![1.0; 1_000_000], 44100, 1);
        let signals = vec![signal; 10];
        let result = mix_signals(&signals).unwrap();
        assert_eq!(result.samples.len(), 1_000_000);
        assert!(result.samples.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_subtract_signals() {
        let s1 = test_signal(vec![2.0, 4.0, 6.0], 44100, 1);
        let s2 = test_signal(vec![1.0, 2.0, 3.0], 44100, 1);
        let result = subtract_signals(&s1, &s2).unwrap();
        assert_eq!(result.samples, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_subtract_signals_empty() {
        let s1 = AudioData::new(vec![], 44100, 1).unwrap();
        let s2 = AudioData::new(vec![], 44100, 1).unwrap();
        let result = subtract_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_subtract_signals_mismatch() {
        let s1 = test_signal(vec![2.0, 4.0], 44100, 1);
        let s2 = test_signal(vec![1.0, 2.0, 3.0], 44100, 1);
        let result = subtract_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::LengthMismatch(2, 3))));
    }

    #[test]
    fn test_multiply_signals() {
        let s1 = test_signal(vec![1.0, 2.0, 3.0], 44100, 1);
        let s2 = test_signal(vec![2.0, 2.0, 2.0], 44100, 1);
        let result = multiply_signals(&s1, &s2).unwrap();
        assert_eq!(result.samples, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_multiply_signals_empty() {
        let s1 = AudioData::new(vec![], 44100, 1).unwrap();
        let s2 = AudioData::new(vec![], 44100, 1).unwrap();
        let result = multiply_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_multiply_signals_non_finite() {
        let s1 = test_signal(vec![f32::MAX, 2.0], 44100, 1);
        let s2 = test_signal(vec![2.0, 2.0], 44100, 1);
        let result = multiply_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::ComputationFailed(_))));
    }

    #[test]
    fn test_divide_signals() {
        let s1 = test_signal(vec![4.0, 6.0, 8.0], 44100, 1);
        let s2 = test_signal(vec![2.0, 0.0, 4.0], 44100, 1);
        let result = divide_signals(&s1, &s2).unwrap();
        assert_eq!(result.samples, vec![2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_divide_signals_empty() {
        let s1 = AudioData::new(vec![], 44100, 1).unwrap();
        let s2 = AudioData::new(vec![], 44100, 1).unwrap();
        let result = divide_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_divide_signals_non_finite() {
        let s1 = test_signal(vec![f32::MAX, 1.0], 44100, 1);
        let s2 = test_signal(vec![0.001, 1.0], 44100, 1);
        let result = divide_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::ComputationFailed(_))));
    }

    #[test]
    fn test_scalar_operation_add() {
        let s = test_signal(vec![1.0, 2.0, 3.0], 44100, 1);
        let result = scalar_operation(&s, 1.0, ScalarOp::Add).unwrap();
        assert_eq!(result.samples, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_operation_multiply() {
        let s = test_signal(vec![1.0, 2.0, 3.0], 44100, 1);
        let result = scalar_operation(&s, 2.0, ScalarOp::Multiply).unwrap();
        assert_eq!(result.samples, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scalar_operation_empty() {
        let s = AudioData::new(vec![], 44100, 1).unwrap();
        let result = scalar_operation(&s, 1.0, ScalarOp::Add);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_scalar_operation_divide_by_zero() {
        let s = test_signal(vec![1.0, 2.0], 44100, 1);
        let result = scalar_operation(&s, 0.0, ScalarOp::Divide);
        assert!(matches!(result, Err(SignalOpError::DivisionByZero(_))));
    }

    #[test]
    fn test_invalid_metadata() {
        let s1 = test_signal(vec![1.0, 2.0], 44100, 1);
        let s2 = test_signal(vec![1.0, 2.0], 44100, 2);
        let result = subtract_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_zero_sample_rate() {
        let result = AudioData::new(vec![1.0, 2.0], 0, 1);
        assert!(matches!(result, Err(AudioError::InvalidInput(_))));
    }

    #[test]
    fn test_zero_channels() {
        let result = AudioData::new(vec![1.0, 2.0], 44100, 0);
        assert!(matches!(result, Err(AudioError::InvalidInput(_))));
    }
}
