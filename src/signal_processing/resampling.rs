use crate::core::AudioError;
use rubato::{FftFixedIn, Resampler};
use thiserror::Error;

/// Custom error type for resampling operations.
#[derive(Error, Debug)]
pub enum ResampleError {
    /// Wraps errors from the `rubato` resampling library with a descriptive message.
    #[error("Resampling failed: {0}")]
    RubatoError(String),
}

/// Resamples a mono audio clip from one sample rate to another using FFT-based band-limited interpolation.
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
/// * `AudioError::InvalidInput` - If either sample rate is zero or output length overflows.
/// * `AudioError::ResampleError(ResampleError::RubatoError)` - If resampler initialization or processing fails.
///
/// # Notes
/// - Sample rates must be positive, including for empty input.
/// - Output length is `ceil(samples.len() * target_sr / orig_sr)`.
/// - The filter is flushed with zeros and its startup delay is removed.
/// - If `orig_sr` equals `target_sr`, returns a clone of the input samples.
/// - If `samples` is empty, returns an empty vector.
/// - Uses a synchronous FFT resampler with fixed-size input blocks.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let samples = vec![0.1, 0.2, 0.3, 0.4];
/// let resampled = resample(&samples, 44100, 22050).unwrap();
/// ```
pub fn resample(samples: &[f32], orig_sr: u32, target_sr: u32) -> Result<Vec<f32>, AudioError> {
    if orig_sr == 0 || target_sr == 0 {
        return Err(AudioError::InvalidInput(
            "Sample rates must be positive".into(),
        ));
    }
    if orig_sr == target_sr {
        return Ok(samples.to_vec());
    }

    if samples.is_empty() {
        return Ok(vec![]);
    }

    // Rates are fixed for the whole clip, so use the synchronous resampler.
    // Fixed-size blocks keep temporary storage independent of clip length.
    let chunk_size = 1024;
    let mut resampler =
        FftFixedIn::<f32>::new(orig_sr as usize, target_sr as usize, chunk_size, 2, 1).map_err(
            |e: rubato::ResamplerConstructionError| {
                ResampleError::RubatoError(format!("Resampler initialization failed: {e}"))
            },
        )?;

    let output_len = usize::try_from(
        (samples.len() as u128 * u128::from(target_sr)).div_ceil(u128::from(orig_sr)),
    )
    .map_err(|_| AudioError::InvalidInput("Resampled length overflows usize".into()))?;
    let delay = resampler.output_delay();
    let required = output_len
        .checked_add(delay)
        .ok_or_else(|| AudioError::InvalidInput("Resampled length overflows usize".into()))?;
    let mut output = Vec::new();
    for chunk in samples.chunks(chunk_size) {
        let block = resampler
            .process_partial(Some(&[chunk]), None)
            .map_err(|e| ResampleError::RubatoError(format!("Resampling failed: {e}")))?;
        output.extend_from_slice(&block[0]);
    }
    // Even an exact input block leaves a filter tail to drain. The final partial
    // block is already zero-padded by rubato; continue until the full clip is available.
    while output.len() < required {
        let block = resampler
            .process_partial::<&[f32]>(None, None)
            .map_err(|e| ResampleError::RubatoError(format!("Resampler flush failed: {e}")))?;
        output.extend_from_slice(&block[0]);
    }
    Ok(output[delay..required].to_vec())
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
        let samples: Vec<f32> = (0..512)
            .map(|n| (2.0 * std::f32::consts::PI * 4.0 * n as f32 / 512.0).sin())
            .collect();
        let out = resample(&samples, 4, 2).unwrap();
        assert!(out.len() >= 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
