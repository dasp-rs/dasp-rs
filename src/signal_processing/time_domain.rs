use crate::core::io::{AudioData, AudioError};
use ndarray::Array1;
use thiserror::Error;

/// Custom error types for time-domain signal operations.
///
/// This enum defines errors specific to manipulating signals in the time domain,
/// including delays, reversals, cropping, padding, and advanced operations like LPC.
#[derive(Error, Debug)]
pub enum TimeDomainError {
    /// Error when a time parameter (e.g., delay, start time) is invalid.
    #[error("Invalid time parameter: {0}")]
    InvalidTime(String),

    /// Error when cropping parameters exceed the signal length.
    #[error("Invalid crop range: start {0}, duration {1}, signal length {2}")]
    InvalidCropRange(usize, usize, usize),

    /// Error when the signal length is invalid for the operation.
    #[error("Invalid signal length: {0}")]
    InvalidLength(String),

    /// Wraps an AudioError from the core module (e.g., for LPC).
    #[error("Audio processing error: {0}")]
    Audio(#[from] AudioError),
}

/// Introduces a time delay to an audio signal.
///
/// This function shifts the signal forward in time by adding silence (zeros) at the
/// beginning and optionally trimming the end to maintain the original length.
/// The delay is specified in seconds and converted to samples based on the sample rate.
///
/// # Arguments
/// * `signal` - The input audio signal to delay.
/// * `delay_seconds` - The delay duration in seconds (must be non-negative).
/// * `preserve_length` - If true, trims the end to keep the original length; if false, extends it.
///
/// # Returns
/// Returns `Result<AudioData, TimeDomainError>` containing the delayed signal or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidTime` - If `delay_seconds` is negative.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::delay;
/// let signal = AudioData { samples: vec![1.0, 2.0, 3.0], sample_rate: 3, channels: 1 };
/// let delayed = delay(&signal, 1.0, true).unwrap(); // 1s = 3 samples
/// assert_eq!(delayed.samples, vec![0.0, 0.0, 1.0]);
///
/// let extended = delay(&signal, 1.0, false).unwrap();
/// assert_eq!(extended.samples, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
/// ```
pub fn delay(
    signal: &AudioData,
    delay_seconds: f32,
    preserve_length: bool,
) -> Result<AudioData, TimeDomainError> {
    if delay_seconds < 0.0 {
        return Err(TimeDomainError::InvalidTime(
            "Delay must be non-negative".to_string(),
        ));
    }

    let delay_samples = (delay_seconds * signal.sample_rate as f32).round() as usize;
    let original_length = signal.samples.len();
    let mut samples = Vec::with_capacity(if preserve_length {
        original_length
    } else {
        original_length + delay_samples
    });

    samples.extend(vec![0.0; delay_samples]);
    if preserve_length {
        let take_samples = original_length.saturating_sub(delay_samples);
        samples.extend_from_slice(&signal.samples[..take_samples]);
    } else {
        samples.extend_from_slice(&signal.samples);
    }

    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: signal.channels,
    })
}

/// Reverses the order of samples in an audio signal.
///
/// This function creates a time-reversed version of the signal by reversing the
/// order of its samples. The sample rate and channels remain unchanged.
///
/// # Arguments
/// * `signal` - The input audio signal to reverse.
///
/// # Returns
/// Returns `Result<AudioData, TimeDomainError>` containing the reversed signal or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidLength` - If the signal is empty.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::time_reversal;
/// let signal = AudioData { samples: vec![1.0, 2.0, 3.0], sample_rate: 44100, channels: 1 };
/// let reversed = time_reversal(&signal).unwrap();
/// assert_eq!(reversed.samples, vec![3.0, 2.0, 1.0]);
/// ```
pub fn time_reversal(signal: &AudioData) -> Result<AudioData, TimeDomainError> {
    if signal.samples.is_empty() {
        return Err(TimeDomainError::InvalidLength(
            "Signal cannot be empty".to_string(),
        ));
    }

    let samples: Vec<f32> = signal.samples.iter().rev().copied().collect();
    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: signal.channels,
    })
}

/// Extracts a segment of an audio signal.
///
/// This function crops the signal to a specified start time and duration, both in
/// seconds. The start time and duration are converted to sample indices based on
/// the sample rate, and the signal is trimmed accordingly.
///
/// # Arguments
/// * `signal` - The input audio signal to crop.
/// * `start_seconds` - The start time of the segment in seconds (non-negative).
/// * `duration_seconds` - The duration of the segment in seconds (non-negative).
///
/// # Returns
/// Returns `Result<AudioData, TimeDomainError>` containing the cropped signal or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidTime` - If `start_seconds` or `duration_seconds` is negative.
/// * `TimeDomainError::InvalidCropRange` - If `start_seconds` exceeds the signal duration.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::time_crop;
/// let signal = AudioData { samples: vec![1.0, 2.0, 3.0, 4.0], sample_rate: 2, channels: 1 };
/// let cropped = time_crop(&signal, 0.5, 1.0).unwrap(); // 0.5s = 1 sample, 1s = 2 samples
/// assert_eq!(cropped.samples, vec![2.0, 3.0]);
/// ```
pub fn time_crop(
    signal: &AudioData,
    start_seconds: f32,
    duration_seconds: f32,
) -> Result<AudioData, TimeDomainError> {
    if start_seconds < 0.0 || duration_seconds < 0.0 {
        return Err(TimeDomainError::InvalidTime(
            "Start time and duration must be non-negative".to_string(),
        ));
    }

    let start_samples = (start_seconds * signal.sample_rate as f32).round() as usize;
    let duration_samples = (duration_seconds * signal.sample_rate as f32).round() as usize;
    let end_samples = start_samples.saturating_add(duration_samples);

    if start_samples >= signal.samples.len() {
        return Err(TimeDomainError::InvalidCropRange(
            start_samples,
            duration_samples,
            signal.samples.len(),
        ));
    }

    let end = end_samples.min(signal.samples.len());
    let samples = signal.samples[start_samples..end].to_vec();

    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: signal.channels,
    })
}

/// Adds silence (zero-padding) to the beginning or end of an audio signal.
///
/// This function extends the signal by adding zeros either at the start, end, or both,
/// specified in seconds. The padding duration is converted to samples based on the
/// sample rate.
///
/// # Arguments
/// * `signal` - The input audio signal to pad.
/// * `start_padding_seconds` - Duration of silence to add at the start (non-negative).
/// * `end_padding_seconds` - Duration of silence to add at the end (non-negative).
///
/// # Returns
/// Returns `Result<AudioData, TimeDomainError>` containing the padded signal or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidTime` - If `start_padding_seconds` or `end_padding_seconds` is negative.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::zero_padding;
/// let signal = AudioData { samples: vec![1.0, 2.0], sample_rate: 2, channels: 1 };
/// let padded = zero_padding(&signal, 0.5, 1.0).unwrap(); // 0.5s = 1 sample, 1s = 2 samples
/// assert_eq!(padded.samples, vec![0.0, 1.0, 2.0, 0.0, 0.0]);
/// ```
pub fn zero_padding(
    signal: &AudioData,
    start_padding_seconds: f32,
    end_padding_seconds: f32,
) -> Result<AudioData, TimeDomainError> {
    if start_padding_seconds < 0.0 || end_padding_seconds < 0.0 {
        return Err(TimeDomainError::InvalidTime(
            "Padding durations must be non-negative".to_string(),
        ));
    }

    let start_samples = (start_padding_seconds * signal.sample_rate as f32).round() as usize;
    let end_samples = (end_padding_seconds * signal.sample_rate as f32).round() as usize;
    let mut samples = Vec::with_capacity(signal.samples.len() + start_samples + end_samples);

    samples.extend(vec![0.0; start_samples]);
    samples.extend_from_slice(&signal.samples);
    samples.extend(vec![0.0; end_samples]);

    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: signal.channels,
    })
}

/// Computes the autocorrelation of a signal.
///
/// This function calculates the autocorrelation of the signal for lags from 0 to
/// `max_size - 1`, providing insight into the signal's self-similarity over time.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `max_size` - Optional maximum lag size in samples (defaults to signal length if None).
///
/// # Returns
/// Returns `Result<Vec<f32>, TimeDomainError>` containing the autocorrelation values or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidLength` - If the signal is empty.
/// * `TimeDomainError::InvalidTime` - If `max_size` exceeds the signal length.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::autocorrelate;
/// let signal = AudioData { samples: vec![1.0, 2.0, 3.0], sample_rate: 44100, channels: 1 };
/// let autocorr = autocorrelate(&signal, Some(2)).unwrap();
/// assert_eq!(autocorr, vec![14.0, 8.0]); // [1*1 + 2*2 + 3*3, 1*2 + 2*3]
/// ```
pub fn autocorrelate(
    signal: &AudioData,
    max_size: Option<usize>,
) -> Result<Vec<f32>, TimeDomainError> {
    if signal.samples.is_empty() {
        return Err(TimeDomainError::InvalidLength(
            "Signal cannot be empty".to_string(),
        ));
    }

    let max_lag = max_size.unwrap_or(signal.samples.len());
    if max_lag > signal.samples.len() {
        return Err(TimeDomainError::InvalidTime(format!(
            "Max lag {} exceeds signal length {}",
            max_lag,
            signal.samples.len()
        )));
    }

    let mut result = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        let mut sum = 0.0;
        for i in 0..(signal.samples.len() - lag) {
            sum += signal.samples[i] * signal.samples[i + lag];
        }
        result.push(sum);
    }
    Ok(result)
}

/// Computes Linear Predictive Coding (LPC) coefficients using the autocorrelation method.
///
/// This function estimates the LPC coefficients, useful for modeling the signal as an
/// autoregressive process, commonly used in speech processing.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `order` - The LPC order (number of coefficients to compute, excluding the leading 1.0).
///
/// # Returns
/// Returns `Result<Vec<f32>, TimeDomainError>` containing the LPC coefficients or an error.
///
/// # Errors
/// * `TimeDomainError::Audio(AudioError::InvalidRange)` - If signal length is less than or equal to `order`.
/// * `TimeDomainError::InvalidTime` - If a division by zero occurs during computation.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::lpc;
/// let signal = AudioData { samples: vec![1.0, 2.0, 3.0, 4.0], sample_rate: 44100, channels: 1 };
/// let coeffs = lpc(&signal, 2).unwrap();
/// assert_eq!(coeffs.len(), 3); // Includes leading 1.0
/// ```
pub fn lpc(signal: &AudioData, order: usize) -> Result<Vec<f32>, TimeDomainError> {
    if signal.samples.len() <= order {
        return Err(TimeDomainError::Audio(AudioError::InvalidRange));
    }

    let r = autocorrelate(signal, Some(order + 1))?;
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut e = r[0];

    for i in 1..=order {
        let mut k = 0.0;
        for j in 0..i {
            k += a[j] * r[i - j];
        }
        k = -k / e;
        if e == 0.0 {
            return Err(TimeDomainError::InvalidTime(
                "Division by zero in LPC computation".to_string(),
            ));
        }
        for j in 0..i {
            a[j] -= k * a[i - 1 - j];
        }
        a[i] = k;
        e *= 1.0 - k * k;
    }
    Ok(a)
}

/// Detects zero crossings in a signal.
///
/// This function identifies points where the signal crosses a threshold, useful for
/// analyzing periodicity or detecting transitions.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `threshold` - Optional threshold value for zero crossing (defaults to 0.0 if None).
/// * `pad` - Optional flag to pad with a zero crossing at index 0 if none are found (defaults to false).
///
/// # Returns
/// Returns `Result<Vec<usize>, TimeDomainError>` containing the indices of zero crossings or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidLength` - If the signal is empty.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::zero_crossings;
/// let signal = AudioData { samples: vec![1.0, -1.0, 2.0, -2.0], sample_rate: 44100, channels: 1 };
/// let crossings = zero_crossings(&signal, None, None).unwrap();
/// assert_eq!(crossings, vec![1, 3]);
/// ```
pub fn zero_crossings(
    signal: &AudioData,
    threshold: Option<f32>,
    pad: Option<bool>,
) -> Result<Vec<usize>, TimeDomainError> {
    if signal.samples.is_empty() {
        return Err(TimeDomainError::InvalidLength(
            "Signal cannot be empty".to_string(),
        ));
    }

    let thresh = threshold.unwrap_or(0.0);
    let mut crossings = Vec::new();
    let mut prev_sign = signal.samples[0] >= thresh;
    for (i, &sample) in signal.samples.iter().enumerate().skip(1) {
        let sign = sample >= thresh;
        if sign != prev_sign {
            crossings.push(i);
        }
        prev_sign = sign;
    }
    if pad.unwrap_or(false) && crossings.is_empty() {
        crossings.push(0);
    }
    Ok(crossings)
}

/// Applies μ-law compression to a signal.
///
/// This function compresses the dynamic range of the signal using the μ-law algorithm,
/// often used in telephony to improve signal-to-noise ratio.
///
/// # Arguments
/// * `signal` - The input audio signal to compress.
/// * `mu` - Optional μ-law parameter (defaults to 255.0 if None).
/// * `quantize` - Optional flag to quantize the output to 8-bit levels (defaults to false).
///
/// # Returns
/// Returns `Result<Vec<f32>, TimeDomainError>` containing the compressed signal or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidLength` - If the signal is empty.
/// * `TimeDomainError::InvalidTime` - If `mu` is not positive.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::mu_compress;
/// let signal = AudioData { samples: vec![0.5, -0.5], sample_rate: 44100, channels: 1 };
/// let compressed = mu_compress(&signal, None, None).unwrap();
/// assert!(compressed[0] > 0.0 && compressed[1] < 0.0);
/// ```
pub fn mu_compress(
    signal: &AudioData,
    mu: Option<f32>,
    quantize: Option<bool>,
) -> Result<Vec<f32>, TimeDomainError> {
    if signal.samples.is_empty() {
        return Err(TimeDomainError::InvalidLength(
            "Signal cannot be empty".to_string(),
        ));
    }

    let mu_val = mu.unwrap_or(255.0);
    if mu_val <= 0.0 {
        return Err(TimeDomainError::InvalidTime(
            "μ value must be positive".to_string(),
        ));
    }

    // Precompute constant: ln(1 + μ) - used for every sample
    let ln_one_plus_mu = (1.0 + mu_val).ln();
    let compressed = signal
        .samples
        .iter()
        .map(|&v| {
            let sign = if v >= 0.0 { 1.0 } else { -1.0 };
            // Standard μ-law compression: sign * ln(1 + μ|x|) / ln(1 + μ)
            let compressed = sign * (1.0 + mu_val * v.abs()).ln() / ln_one_plus_mu;
            if quantize.unwrap_or(false) {
                (compressed * 255.0).round() / 255.0
            } else {
                compressed
            }
        })
        .collect();
    Ok(compressed)
}

/// Applies μ-law expansion to a compressed signal.
///
/// This function expands a μ-law compressed signal back to its original dynamic range.
///
/// # Arguments
/// * `signal` - The input compressed audio signal.
/// * `mu` - Optional μ-law parameter (defaults to 255.0 if None).
/// * `quantize` - Optional flag (unused, included for symmetry with `mu_compress`).
///
/// # Returns
/// Returns `Result<Vec<f32>, TimeDomainError>` containing the expanded signal or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidLength` - If the signal is empty.
/// * `TimeDomainError::InvalidTime` - If `mu` is not positive.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::mu_expand;
/// let signal = AudioData { samples: vec![0.5, -0.5], sample_rate: 44100, channels: 1 };
/// let expanded = mu_expand(&signal, None, None).unwrap();
/// assert!(expanded[0] > 0.0 && expanded[1] < 0.0);
/// ```
pub fn mu_expand(
    signal: &AudioData,
    mu: Option<f32>,
    _quantize: Option<bool>,
) -> Result<Vec<f32>, TimeDomainError> {
    if signal.samples.is_empty() {
        return Err(TimeDomainError::InvalidLength(
            "Signal cannot be empty".to_string(),
        ));
    }

    let mu_val = mu.unwrap_or(255.0);
    if mu_val <= 0.0 {
        return Err(TimeDomainError::InvalidTime(
            "μ value must be positive".to_string(),
        ));
    }

    let expanded = signal
        .samples
        .iter()
        .map(|&v| {
            let sign = if v >= 0.0 { 1.0 } else { -1.0 };
            // Standard μ-law expansion: sign * ((1 + μ)^|y| - 1) / μ
            sign * ((1.0 + mu_val).powf(v.abs()) - 1.0) / mu_val
        })
        .collect();
    Ok(expanded)
}

/// Computes the logarithmic energy of framed audio.
///
/// This function calculates the log energy of the signal in overlapping frames,
/// useful for feature extraction in audio analysis.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `frame_length` - Optional frame length in samples (defaults to 2048 if None).
/// * `hop_length` - Optional hop length in samples (defaults to `frame_length / 4` if None).
///
/// # Returns
/// Returns `Result<Array1<f32>, TimeDomainError>` containing the log energy for each frame or an error.
///
/// # Errors
/// * `TimeDomainError::InvalidLength` - If the signal is empty.
/// * `TimeDomainError::InvalidTime` - If `frame_length` or `hop_length` is zero or exceeds signal length.
///
/// # Examples
/// ```
/// use dasp_rs::io::core::AudioData;
/// use dasp_rs::signal_processing::time_domain::log_energy;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4, 0.5], sample_rate: 44100, channels: 1 };
/// let energy = log_energy(&signal, Some(2), Some(1)).unwrap();
/// assert_eq!(energy.len(), 4); // (5 - 2) / 1 + 1
/// ```
pub fn log_energy(
    signal: &AudioData,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, TimeDomainError> {
    if signal.samples.is_empty() {
        return Err(TimeDomainError::InvalidLength(
            "Signal cannot be empty".to_string(),
        ));
    }

    let frame_len = frame_length.unwrap_or(2048);
    if frame_len == 0 {
        return Err(TimeDomainError::InvalidTime(
            "Frame length must be positive".to_string(),
        ));
    }
    if frame_len > signal.samples.len() {
        return Err(TimeDomainError::InvalidTime(format!(
            "Frame length {} exceeds signal length {}",
            frame_len,
            signal.samples.len()
        )));
    }

    let hop = hop_length.unwrap_or(frame_len / 4);
    if hop == 0 {
        return Err(TimeDomainError::InvalidTime(
            "Hop length must be positive".to_string(),
        ));
    }

    let n_frames = (signal.samples.len() - frame_len) / hop + 1;
    let mut energy = Array1::zeros(n_frames);

    for i in 0..n_frames {
        let start = i * hop;
        let frame = &signal.samples[start..(start + frame_len).min(signal.samples.len())];
        let e = frame.iter().map(|&x| x.powi(2)).sum::<f32>();
        energy[i] = (e + 1e-10).ln();
    }

    Ok(energy)
}
