use crate::core::io::AudioData;
use thiserror::Error;

/// Custom error types for amplitude processing operations.
///
/// This enum defines errors specific to amplitude manipulation functions like
/// amplification, attenuation, and normalization.
#[derive(Error, Debug)]
pub enum AmplitudeError {
    /// Error when an invalid gain factor is provided (e.g., negative or zero where not allowed).
    #[error("Invalid gain factor: {0}")]
    InvalidGain(String),

    /// Error when the input signal is invalid (e.g., empty or zero amplitude).
    #[error("Invalid signal: {0}")]
    InvalidSignal(String),
}

/// Amplifies an audio signal by a specified gain factor.
///
/// This function increases the amplitude of the signal by multiplying each sample
/// by the gain factor. A gain > 1.0 increases amplitude; values <= 0 are invalid.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `gain` - The amplification factor (must be positive).
///
/// # Returns
/// Returns `Result<AudioData, AmplitudeError>` containing the amplified signal or an error.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let signal = AudioData { samples: vec![0.5, 1.0, 0.5], sample_rate: 44100, channels: 1 };
/// let amplified = amplify(&signal, 2.0)?;
/// assert_eq!(amplified.samples, vec![1.0, 2.0, 1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn amplify(signal: &AudioData, gain: f32) -> Result<AudioData, AmplitudeError> {
    if gain <= 0.0 {
        return Err(AmplitudeError::InvalidGain(
            "Gain must be positive".to_string(),
        ));
    }

    let samples = signal.samples.iter().map(|&s| s * gain).collect();
    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: signal.channels,
    })
}

/// Attenuates an audio signal by a specified gain factor.
///
/// This function decreases the amplitude of the signal by multiplying each sample
/// by the gain factor. A gain between 0.0 and 1.0 reduces amplitude; values < 0 are invalid.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `gain` - The attenuation factor (must be non-negative, typically < 1.0).
///
/// # Returns
/// Returns `Result<AudioData, AmplitudeError>` containing the attenuated signal or an error.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let signal = AudioData { samples: vec![1.0, 2.0, 1.0], sample_rate: 44100, channels: 1 };
/// let attenuated = attenuate(&signal, 0.5)?;
/// assert_eq!(attenuated.samples, vec![0.5, 1.0, 0.5]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn attenuate(signal: &AudioData, gain: f32) -> Result<AudioData, AmplitudeError> {
    if gain < 0.0 {
        return Err(AmplitudeError::InvalidGain(
            "Gain must be non-negative".to_string(),
        ));
    }

    let samples = signal.samples.iter().map(|&s| s * gain).collect();
    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: signal.channels,
    })
}

/// Normalizes an audio signal to a target peak or RMS level.
///
/// This function scales the signal so its peak amplitude or RMS (root mean square)
/// level matches the target value. Useful for ensuring consistent loudness.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `target` - The target level (e.g., 1.0 for full scale).
///
/// # Returns
/// Returns a builder; defaults to [`NormalizeMode::Peak`].
///
/// # Examples
/// ```
/// use dasp_rs::proc::{normalize, NormalizeMode};
/// use dasp_rs::types::AudioData;
/// let signal = AudioData { samples: vec![0.2, 0.4, 0.2], sample_rate: 44100, channels: 1 };
/// let normalized = normalize(&signal, 1.0).mode(NormalizeMode::Peak).compute()?;
/// assert_eq!(normalized.samples, vec![0.5, 1.0, 0.5]); // Scaled to peak 1.0
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn normalize(signal: &AudioData, target: f32) -> NormalizeBuilder<'_> {
    NormalizeBuilder { signal, target, mode: NormalizeMode::Peak }
}

/// Normalization strategy for [`normalize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalizeMode {
    /// Scale so the peak absolute sample equals the target.
    #[default]
    Peak,
    /// Scale so the RMS level equals the target.
    Rms,
}

/// Builder for [`normalize`].
#[derive(Debug, Clone)]
pub struct NormalizeBuilder<'a> {
    signal: &'a AudioData,
    target: f32,
    mode: NormalizeMode,
}

impl NormalizeBuilder<'_> {
    /// Set the normalization mode (default: [`NormalizeMode::Peak`]).
    #[must_use]
    pub fn mode(mut self, mode: NormalizeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Apply normalization.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<AudioData, AmplitudeError> {
        normalize_impl(self.signal, self.target, self.mode)
    }
}

fn normalize_impl(
    signal: &AudioData,
    target: f32,
    mode: NormalizeMode,
) -> Result<AudioData, AmplitudeError> {
    if target <= 0.0 {
        return Err(AmplitudeError::InvalidGain(
            "Target level must be positive".to_string(),
        ));
    }
    if signal.samples.is_empty() {
        return Err(AmplitudeError::InvalidSignal(
            "Signal cannot be empty".to_string(),
        ));
    }

    let gain = match mode {
        NormalizeMode::Peak => {
            let max_amplitude = signal
                .samples
                .iter()
                .map(|s| s.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            if max_amplitude == 0.0 {
                return Err(AmplitudeError::InvalidSignal(
                    "Signal has no amplitude to normalize".to_string(),
                ));
            }
            target / max_amplitude
        }
        NormalizeMode::Rms => {
            let rms = (signal.samples.iter().map(|&s| s * s).sum::<f32>() / signal.samples.len() as f32)
                .sqrt();
            if rms == 0.0 {
                return Err(AmplitudeError::InvalidSignal(
                    "Signal has no RMS level to normalize".to_string(),
                ));
            }
            target / rms
        }
    };

    let samples = signal.samples.iter().map(|&s| s * gain).collect();
    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: signal.channels,
    })
}
