use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use thiserror::Error;
use crate::core::AudioError;

/// Custom error type for resampling operations.
///
/// # Variants
/// * `RubatoError(String)` - Wraps errors from the `rubato` resampling library with a descriptive message.
#[derive(Error, Debug)]
pub enum ResampleError {
    #[error("Resampling failed: {0}")]
    RubatoError(String),
}

/// Resamples audio data from one sample rate to another using sinc interpolation.
///
/// # Arguments
/// * `samples` - Input audio samples as a slice of `f32`
/// * `orig_sr` - Original sample rate in Hz
/// * `target_sr` - Target sample rate in Hz
///
/// # Returns
/// Returns a `Result` containing a `Vec<f32>` with the resampled audio data,
/// or an `AudioError` if resampling fails.
///
/// # Errors
/// * `AudioError::ResampleError(ResampleError::RubatoError)` - If resampler initialization or processing fails.
///
/// # Notes
/// - If `orig_sr` equals `target_sr`, returns a clone of the input samples.
/// - If `samples` is empty, returns an empty vector.
/// - Uses sinc interpolation with fixed input length, Blackman-Harris window, and linear interpolation.
///
/// # Examples
/// ```
/// let samples = vec![0.1, 0.2, 0.3, 0.4];
/// let resampled = resample(&samples, 44100, 22050).unwrap();
/// ```
pub fn resample(samples: &[f32], orig_sr: u32, target_sr: u32) -> Result<Vec<f32>, AudioError> {
    if orig_sr == target_sr {
        return Ok(samples.to_vec());
    }

    if samples.is_empty() {
        return Ok(vec![]);
    }

    let ratio = target_sr as f64 / orig_sr as f64;

    let sinc_len = 256;
    let f_cutoff = 0.95;
    let oversampling_factor = 256;
    let interpolation_params = SincInterpolationParameters {
        sinc_len,
        f_cutoff,
        oversampling_factor,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        1.0,
        interpolation_params,
        samples.len(),
        1,
    ).map_err(|e: rubato::ResamplerConstructionError| ResampleError::RubatoError(format!("Resampler initialization failed: {}", e)))?;

    let input = vec![samples.to_vec()];

    let output = resampler.process(&input, None)
        .map_err(|e| ResampleError::RubatoError(format!("Resampling failed: {}", e)))?;
    Ok(output[0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_clone_when_sample_rates_match() {
        let samples = vec![0.1, 0.2, 0.3];
        let out = resample(&samples, 44_100, 44_100).unwrap();
        assert_eq!(out, samples);
    }

    #[test]
    fn resamples_empty_input_to_empty() {
        let out = resample(&[], 48_000, 44_100).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn downsample_basic_signal() {
        let samples = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample(&samples, 4, 2).unwrap();
        // Should roughly halve length; allow tolerance for sinc padding
        assert!(out.len() >= 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}