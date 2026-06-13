use crate::core::io::AudioData;
use thiserror::Error;

/// Custom error types for signal mixing operations.
///
/// This enum defines errors specific to combining multiple audio signals, such as
/// stereo mixing, multi-channel mixing, and dry/wet blending.
#[derive(Error, Debug)]
pub enum MixingError {
    /// Error when signal lengths do not match.
    #[error("Signal lengths mismatch: expected {0}, found {1}")]
    LengthMismatch(usize, usize),

    /// Error when the number of input signals is invalid for the operation.
    #[error("Invalid number of signals: {0}")]
    InvalidSignalCount(String),

    /// Error when input parameters (e.g., mix factor, channels) are invalid.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Error when an input signal has an incompatible format (e.g., not mono).
    #[error("Incompatible signal format: {0}")]
    IncompatibleFormat(String),
}

/// Combines two mono signals into a stereo signal.
///
/// This function takes two mono audio signals and interleaves them into a single
/// stereo signal (left channel from the first signal, right channel from the second).
/// Both signals must have the same length and sample rate, and be mono (1 channel).
///
/// # Arguments
/// * `left` - The mono signal for the left channel.
/// * `right` - The mono signal for the right channel.
///
/// # Returns
/// Returns `Result<AudioData, MixingError>` containing the stereo signal or an error.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let left = AudioData { samples: vec![0.1, 0.2, 0.3], sample_rate: 44100, channels: 1 };
/// let right = AudioData { samples: vec![0.4, 0.5, 0.6], sample_rate: 44100, channels: 1 };
/// let stereo = stereo_mix(&left, &right)?;
/// assert_eq!(stereo.samples, vec![0.1, 0.4, 0.2, 0.5, 0.3, 0.6]);
/// assert_eq!(stereo.channels, 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn stereo_mix(left: &AudioData, right: &AudioData) -> Result<AudioData, MixingError> {
    if left.channels != 1 || right.channels != 1 {
        return Err(MixingError::IncompatibleFormat(
            "Both signals must be mono".to_string(),
        ));
    }
    if left.samples.len() != right.samples.len() {
        return Err(MixingError::LengthMismatch(
            left.samples.len(),
            right.samples.len(),
        ));
    }
    if left.sample_rate != right.sample_rate {
        return Err(MixingError::InvalidParameter(
            "Sample rates must match".to_string(),
        ));
    }

    let mut samples = Vec::with_capacity(left.samples.len() * 2);
    for (l, r) in left.samples.iter().zip(right.samples.iter()) {
        samples.push(*l);
        samples.push(*r);
    }

    Ok(AudioData {
        samples,
        sample_rate: left.sample_rate,
        channels: 2,
    })
}

/// Combines multiple signals into a multi-channel output.
///
/// This function takes a vector of mono signals and combines them into a single
/// multi-channel signal with the specified number of channels. The number of signals
/// must match the target channel count (e.g., 6 for 5.1 surround).
/// All signals must be mono, have the same length, and same sample rate.
///
/// # Arguments
/// * `signals` - A vector of mono audio signals.
/// * `channels` - The target number of channels (e.g., 6 for 5.1).
///
/// # Returns
/// Returns `Result<AudioData, MixingError>` containing the multi-channel signal or an error.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let left = AudioData { samples: vec![0.1, 0.2], sample_rate: 44100, channels: 1 };
/// let right = AudioData { samples: vec![0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let stereo = multi_channel_mix(&[&left, &right], 2)?;
/// assert_eq!(stereo.samples, vec![0.1, 0.3, 0.2, 0.4]);
/// assert_eq!(stereo.channels, 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn multi_channel_mix(signals: &[&AudioData], channels: u16) -> Result<AudioData, MixingError> {
    if signals.is_empty() {
        return Err(MixingError::InvalidSignalCount(
            "At least one signal is required".to_string(),
        ));
    }
    if signals.len() != channels as usize {
        return Err(MixingError::InvalidSignalCount(format!(
            "Number of signals ({}) must match target channels ({})",
            signals.len(),
            channels
        )));
    }

    let length = signals[0].samples.len();
    let sample_rate = signals[0].sample_rate;
    for &signal in signals {
        if signal.channels != 1 {
            return Err(MixingError::IncompatibleFormat(
                "All signals must be mono".to_string(),
            ));
        }
        if signal.samples.len() != length {
            return Err(MixingError::LengthMismatch(length, signal.samples.len()));
        }
        if signal.sample_rate != sample_rate {
            return Err(MixingError::InvalidParameter(
                "Sample rates must match".to_string(),
            ));
        }
    }

    let mut samples = Vec::with_capacity(length * channels as usize);
    for i in 0..length {
        for &signal in signals {
            samples.push(signal.samples[i]);
        }
    }

    Ok(AudioData {
        samples,
        sample_rate,
        channels,
    })
}

/// Blends a processed (wet) signal with the original (dry) signal.
///
/// This function combines the original signal with a processed version using a mix factor.
/// A `wet_mix` of 0.0 returns the dry signal, 1.0 returns the wet signal, and values
/// in between blend them proportionally. Signals must have the same length, sample rate,
/// and channels.
///
/// # Arguments
/// * `dry` - The original (unprocessed) signal.
/// * `wet` - The processed signal.
/// * `wet_mix` - The mix factor (0.0 = fully dry, 1.0 = fully wet).
///
/// # Returns
/// Returns `Result<AudioData, MixingError>` containing the blended signal or an error.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let dry = AudioData { samples: vec![1.0, 1.0], sample_rate: 44100, channels: 1 };
/// let wet = AudioData { samples: vec![2.0, 2.0], sample_rate: 44100, channels: 1 };
/// let mixed = dry_wet_mix(&dry, &wet, 0.5)?;
/// assert_eq!(mixed.samples, vec![1.5, 1.5]); // (1.0 * 0.5) + (2.0 * 0.5)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn dry_wet_mix(dry: &AudioData, wet: &AudioData, wet_mix: f32) -> Result<AudioData, MixingError> {
    if !(0.0..=1.0).contains(&wet_mix) {
        return Err(MixingError::InvalidParameter(
            "Wet mix must be between 0.0 and 1.0".to_string(),
        ));
    }
    if dry.samples.len() != wet.samples.len() {
        return Err(MixingError::LengthMismatch(
            dry.samples.len(),
            wet.samples.len(),
        ));
    }
    if dry.sample_rate != wet.sample_rate || dry.channels != wet.channels {
        return Err(MixingError::InvalidParameter(
            "Sample rate and channels must match".to_string(),
        ));
    }

    let dry_mix = 1.0 - wet_mix;
    let samples: Vec<f32> = dry
        .samples
        .iter()
        .zip(&wet.samples)
        .map(|(&d, &w)| d * dry_mix + w * wet_mix)
        .collect();

    Ok(AudioData {
        samples,
        sample_rate: dry.sample_rate,
        channels: dry.channels,
    })
}
