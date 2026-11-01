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
/// - Inputs above unit peak use f64 FFT processing with reusable block buffers,
///   preventing overflow without erasing distant quiet regions by normalization.
///   Unrepresentable requested output (including interpolation overshoot) returns
///   a computation error.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let samples = vec![0.1, 0.2, 0.3, 0.4];
/// let resampled = resample(&samples, 44100, 22050).unwrap();
/// ```
pub fn resample(samples: &[f32], orig_sr: u32, target_sr: u32) -> Result<Vec<f32>, AudioError> {
    // Accumulate both flags in one pass. Non-short-circuit boolean operations
    // let the compiler vectorize validation on long clips.
    let (finite, extended_range) = samples.iter().fold((true, false), |(finite, extended), v| {
        (finite & v.is_finite(), extended | (v.abs() > 1.0))
    });
    if !finite {
        return Err(AudioError::InvalidInput("Samples must be finite".into()));
    }
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
    // Keep both internal FFT lengths even. rubato centers its FIR at the
    // integer input midpoint and reports integer output delay; odd lengths can
    // otherwise leave a fractional-sample timing error after delay removal.
    let mut a = orig_sr;
    let mut b = target_sr;
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let quantum = 2 * (orig_sr / a) as usize;
    let chunk_size = 2 * 512_usize.div_ceil(quantum) * quantum;
    let output_len = usize::try_from(
        (samples.len() as u128 * u128::from(target_sr)).div_ceil(u128::from(orig_sr)),
    )
    .map_err(|_| AudioError::InvalidInput("Resampled length overflows usize".into()))?;
    if extended_range {
        return resample_extended_range(samples, orig_sr, target_sr, chunk_size, output_len);
    }
    let mut resampler =
        FftFixedIn::<f32>::new(orig_sr as usize, target_sr as usize, chunk_size, 2, 1).map_err(
            |e: rubato::ResamplerConstructionError| {
                ResampleError::RubatoError(format!("Resampler initialization failed: {e}"))
            },
        )?;

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
    for value in &output[delay..required] {
        if !value.is_finite() {
            return Err(AudioError::ComputationFailed(
                "Resampled output exceeds finite f32 range".into(),
            ));
        }
    }
    Ok(output[delay..required].to_vec())
}

/// Keep the exceptional high-amplitude path in f64 throughout filtering. A
/// global f32 normalization would underflow quiet samples elsewhere in a clip.
/// The FIR state and scratch buffers are bounded by the rate-dependent block
/// size; only the requested f32 result grows with the clip duration.
fn resample_extended_range(
    samples: &[f32],
    orig_sr: u32,
    target_sr: u32,
    chunk_size: usize,
    output_len: usize,
) -> Result<Vec<f32>, AudioError> {
    let mut resampler =
        FftFixedIn::<f64>::new(orig_sr as usize, target_sr as usize, chunk_size, 2, 1).map_err(
            |e| ResampleError::RubatoError(format!("Resampler initialization failed: {e}")),
        )?;
    let mut remaining_delay = resampler.output_delay();
    let mut input = vec![0.0_f64; chunk_size];
    let mut block = resampler.output_buffer_allocate(true);
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| AudioError::InvalidInput("Resampled output cannot be allocated".into()))?;
    let mut consumed = 0;
    while consumed < samples.len() || output.len() < output_len {
        input.fill(0.0);
        let take = chunk_size.min(samples.len() - consumed);
        for (target, &sample) in input.iter_mut().zip(&samples[consumed..consumed + take]) {
            *target = f64::from(sample);
        }
        consumed += take;
        // Supplying a pre-padded full block avoids process_partial's temporary
        // allocation, both for the last partial input and the zero-filled drain.
        let (_, written) = resampler
            .process_into_buffer(&[input.as_slice()], &mut block, None)
            .map_err(|e| ResampleError::RubatoError(format!("Resampling failed: {e}")))?;
        let skip = remaining_delay.min(written);
        remaining_delay -= skip;
        let keep = (output_len - output.len()).min(written - skip);
        for &sample in &block[0][skip..skip + keep] {
            let value = sample as f32;
            if !value.is_finite() {
                return Err(AudioError::ComputationFailed(
                    "Resampled output exceeds finite f32 range".into(),
                ));
            }
            output.push(value);
        }
    }
    Ok(output)
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
        assert_eq!(out, Vec::<f32>::new());
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
