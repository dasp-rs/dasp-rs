use ndarray::Array2;
use thiserror::Error;

use crate::signal_processing::time_frequency::stft;
use crate::AudioError;
use crate::fft_frequencies;

/// Errors specific to pitch and tuning operations.
#[derive(Error, Debug)]
pub enum TuningError {
    /// Invalid frequency range (e.g., fmin >= fmax or out of Nyquist bounds).
    #[error("Invalid frequency range: {0}")]
    InvalidFrequencyRange(String),

    /// Insufficient data for the requested operation.
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Computation failed (e.g., STFT or frequency bin mismatch).
    #[error("Computation failed: {0}")]
    ComputationFailed(String),

    /// Invalid input parameters (e.g., missing required inputs).
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl From<TuningError> for AudioError {
    fn from(err: TuningError) -> Self {
        match err {
            TuningError::InvalidFrequencyRange(msg) => AudioError::InvalidInput(msg),
            TuningError::InsufficientData(msg) => AudioError::InsufficientData(msg),
            TuningError::ComputationFailed(msg) => AudioError::ComputationFailed(msg),
            TuningError::InvalidInput(msg) => AudioError::InvalidInput(msg),
        }
    }
}

/// Computes pitch estimates using the pYIN algorithm (probabilistic YIN).
///
/// The pYIN algorithm extends the YIN algorithm by incorporating probabilistic modeling
/// to improve robustness in noisy conditions. It estimates pitch by analyzing
/// cumulative mean normalized difference functions across frames.
///
/// # Arguments
/// * `signal` - Audio time series as a slice of `f32` samples.
/// * `fmin` - Minimum frequency in Hz (must be positive).
/// * `fmax` - Maximum frequency in Hz (must be less than Nyquist frequency).
/// * `sample_rate` - Sample rate in Hz (defaults to 44100 if `None`).
/// * `frame_length` - Frame length in samples (defaults to 2048 if `None`).
///
/// # Returns
/// A `Result` containing a `Vec<f32>` of pitch estimates in Hz for each frame.
/// Returns 0.0 for frames where no valid pitch is detected.
/// Errors if inputs are invalid or insufficient.
///
/// # Errors
/// * `TuningError::InvalidFrequencyRange` - If `fmin >= fmax`, `fmin < 0`, or `fmax > sample_rate/2`.
/// * `TuningError::InsufficientData` - If signal length is less than frame length.
///
/// # Example
/// ```
/// use dasp_rs::tuning::pyin;
/// let signal = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // Short signal
/// let pitches = pyin(&signal, 50.0, 500.0, None, Some(4)).unwrap();
/// assert_eq!(pitches.len(), 1); // Single frame due to short signal
/// ```
pub fn pyin(
    signal: &[f32],
    fmin: f32,
    fmax: f32,
    sample_rate: Option<u32>,
    frame_length: Option<usize>,
) -> Result<Vec<f32>, TuningError> {
    let sr = sample_rate.unwrap_or(44_100);
    let frame_len = frame_length.unwrap_or(2048);
    validate_inputs(fmin, fmax, sr, frame_len, signal.len())?;

    let hop_length = frame_len / 4;
    let n_frames = calculate_n_frames(signal.len(), frame_len, hop_length);
    let lag_min = (sr as f32 / fmax).round() as usize;
    let lag_max = (sr as f32 / fmin).round() as usize;

    let mut pitches = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * hop_length;
        let frame = &signal[start..(start + frame_len).min(signal.len())];
        let pitch = compute_pyin_frame(frame, lag_min, lag_max, sr, fmin, fmax)?;
        pitches.push(pitch);
    }

    Ok(pitches)
}

/// Computes pitch estimates using the YIN algorithm.
///
/// The YIN algorithm detects pitch by computing the cumulative mean normalized
/// difference function and finding the lag with the minimum value, corresponding
/// to the fundamental period.
///
/// # Arguments
/// * `signal` - Audio time series as a slice of `f32` samples.
/// * `fmin` - Minimum frequency in Hz (must be positive).
/// * `fmax` - Maximum frequency in Hz (must be less than Nyquist frequency).
/// * `sample_rate` - Sample rate in Hz (defaults to 44100 if `None`).
/// * `frame_length` - Frame length in samples (defaults to 2048 if `None`).
///
/// # Returns
/// A `Result` containing a `Vec<f32>` of pitch estimates in Hz for each frame.
/// Returns 0.0 for frames where no valid pitch is detected.
/// Errors if inputs are invalid or insufficient.
///
/// # Errors
/// * `TuningError::InvalidFrequencyRange` - If `fmin >= fmax`, `fmin < 0`, or `fmax > sample_rate/2`.
/// * `TuningError::InsufficientData` - If signal length is less than frame length.
///
/// # Example
/// ```
/// use dasp_rs::tuning::yin;
/// let signal = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
/// let pitches = yin(&signal, 50.0, 500.0, None, Some(4)).unwrap();
/// assert_eq!(pitches.len(), 1);
/// ```
pub fn yin(
    signal: &[f32],
    fmin: f32,
    fmax: f32,
    sample_rate: Option<u32>,
    frame_length: Option<usize>,
) -> Result<Vec<f32>, TuningError> {
    let sr = sample_rate.unwrap_or(44_100);
    let frame_len = frame_length.unwrap_or(2048);
    validate_inputs(fmin, fmax, sr, frame_len, signal.len())?;

    let hop_length = frame_len / 4;
    let n_frames = calculate_n_frames(signal.len(), frame_len, hop_length);
    let lag_min = (sr as f32 / fmax).round() as usize;
    let lag_max = (sr as f32 / fmin).round() as usize;

    let mut pitches = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * hop_length;
        let frame = &signal[start..(start + frame_len).min(signal.len())];
        let pitch = compute_yin_frame(frame, lag_min, lag_max, sr, fmin, fmax)?;
        pitches.push(pitch);
    }

    Ok(pitches)
}

/// Estimates tuning deviation in cents from a reference pitch.
///
/// Analyzes a signal or spectrogram to compute the average deviation from
/// standard pitch (A440) in cents, weighted by magnitude.
///
/// # Arguments
/// * `signal` - Optional audio time series as a slice of `f32` samples.
/// * `sample_rate` - Sample rate in Hz (defaults to 44100 if `None`).
/// * `spectrogram` - Optional pre-computed magnitude spectrogram as `Array2<f32>`.
/// * `n_fft` - FFT window size in samples (defaults to 2048 if `None`).
///
/// # Returns
/// A `Result` containing the tuning deviation in cents.
/// Returns 0.0 if no valid pitches are detected.
/// Errors if inputs are invalid or computation fails.
///
/// # Errors
/// * `TuningError::InsufficientData` - If signal length is less than `n_fft`.
/// * `TuningError::InvalidInput` - If neither `signal` nor `spectrogram` is provided.
/// * `TuningError::ComputationFailed` - If STFT or piptrack computation fails.
///
/// # Example
/// ```
/// use dasp_rs::tuning::estimate_tuning;
/// let signal = vec![0.1, 0.2, 0.3, 0.4];
/// let tuning = estimate_tuning(Some(&signal), None, None, Some(4)).unwrap();
/// assert_eq!(tuning, 0.0); // No valid pitches in short signal
/// ```
pub fn estimate_tuning(
    signal: Option<&[f32]>,
    sample_rate: Option<u32>,
    spectrogram: Option<&Array2<f32>>,
    n_fft: Option<usize>,
) -> Result<f32, TuningError> {
    let sr = sample_rate.unwrap_or(44_100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = n_fft / 4;

    let s = compute_spectrogram(signal, spectrogram, n_fft, hop_length)?;
    let (pitches, mags) = piptrack(signal, Some(sr), Some(&s), Some(n_fft), Some(hop_length))?;

    let mut total_deviation = 0.0;
    let mut total_weight = 0.0;

    for t in 0..pitches.shape()[1] {
        for f in 0..pitches.shape()[0] {
            let freq = pitches[[f, t]];
            let mag = mags[[f, t]];
            if freq > 0.0 && mag > 1e-6 {
                let ref_freq = 440.0 * 2.0f32.powf((f32::log2(freq / 440.0)).floor());
                let deviation = 1200.0 * f32::log2(freq / ref_freq);
                total_deviation += deviation * mag;
                total_weight += mag;
            }
        }
    }

    Ok(if total_weight > 1e-6 { total_deviation / total_weight } else { 0.0 })
}

/// Estimates tuning deviation from a list of frequencies.
///
/// Computes the average tuning deviation in cents from a list of pitch frequencies,
/// relative to standard pitch (A440).
///
/// # Arguments
/// * `frequencies` - Slice of pitch frequencies in Hz.
/// * `resolution` - Tuning resolution in cents (defaults to 1.0 if `None`).
///
/// # Returns
/// A `Result` containing the average tuning deviation in cents.
/// Returns 0.0 if no valid frequencies are provided.
/// Errors if resolution is non-positive.
///
/// # Errors
/// * `TuningError::InvalidInput` - If resolution is less than or equal to 0.
///
/// # Example
/// ```
/// use dasp_rs::tuning::pitch_tuning;
/// let freqs = vec![440.0, 442.0, 438.0];
/// let tuning = pitch_tuning(&freqs, None).unwrap();
/// assert!(tuning.abs() < 10.0); // Deviation within reasonable range
/// ```
pub fn pitch_tuning(frequencies: &[f32], resolution: Option<f32>) -> Result<f32, TuningError> {
    let resolution = resolution.unwrap_or(1.0);
    if resolution <= 0.0 {
        return Err(TuningError::InvalidInput(
            "Resolution must be positive".to_string(),
        ));
    }

    let valid_freqs: Vec<f32> = frequencies.iter().filter(|&&f| f > 0.0).copied().collect();
    if valid_freqs.is_empty() {
        return Ok(0.0);
    }

    let mut total_deviation = 0.0;
    for &freq in &valid_freqs {
        let ref_freq = 440.0 * 2.0f32.powf((f32::log2(freq / 440.0)).floor());
        let cents = 1200.0 * f32::log2(freq / ref_freq);
        total_deviation += (cents % resolution)
            - if cents % resolution > resolution / 2.0 {
                resolution
            } else {
                0.0
            };
    }

    Ok(total_deviation / valid_freqs.len() as f32)
}

/// Tracks pitch using peak interpolation in a spectrogram.
///
/// Identifies pitch by finding the maximum magnitude in each spectrogram frame
/// and interpolating the corresponding frequency bin to estimate the true peak.
///
/// # Arguments
/// * `signal` - Optional audio time series as a slice of `f32` samples.
/// * `sample_rate` - Sample rate in Hz (defaults to 44100 if `None`).
/// * `spectrogram` - Optional pre-computed magnitude spectrogram as `Array2<f32>`.
/// * `n_fft` - FFT window size in samples (defaults to 2048 if `None`).
/// * `hop_length` - Hop length in samples (defaults to n_fft/4 if `None`).
///
/// # Returns
/// A `Result` containing a tuple of:
/// * `pitches` - 2D array of pitch estimates in Hz (`Array2<f32>`).
/// * `magnitudes` - 2D array of corresponding peak magnitudes (`Array2<f32>`).
/// Errors if inputs are invalid or computation fails.
///
/// # Errors
/// * `TuningError::InsufficientData` - If signal length is less than `n_fft`.
/// * `TuningError::InvalidInput` - If neither `signal` nor `spectrogram` is provided.
/// * `TuningError::ComputationFailed` - If STFT or frequency bin computation fails.
///
/// # Example
/// ```
/// use dasp_rs::tuning::piptrack;
/// let signal = vec![0.1, 0.2, 0.3, 0.4];
/// let (pitches, mags) = piptrack(Some(&signal), None, None, Some(4), None).unwrap();
/// assert_eq!(pitches.shape(), &[3, 1]); // n_fft/2 + 1, n_frames
/// ```
pub fn piptrack(
    signal: Option<&[f32]>,
    sample_rate: Option<u32>,
    spectrogram: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<(Array2<f32>, Array2<f32>), TuningError> {
    let sr = sample_rate.unwrap_or(44_100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(n_fft / 4);

    let s = compute_spectrogram(signal, spectrogram, n_fft, hop_length)?;
    let freqs = fft_frequencies(Some(sr), Some(n_fft));
    if freqs.len() != s.shape()[0] {
        return Err(TuningError::ComputationFailed(
            "Frequency bins mismatch with spectrogram".to_string(),
        ));
    }

    let mut pitches = Array2::zeros(s.dim());
    let mut mags = Array2::zeros(s.dim());

    for t in 0..s.shape()[1] {
        let frame = s.column(t);
        if let Some(max_idx) = frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
        {
            let peak_mag = frame[max_idx];
            if peak_mag > 1e-6 {
                let left = if max_idx > 0 { frame[max_idx - 1] } else { peak_mag };
                let right = if max_idx < frame.len() - 1 {
                    frame[max_idx + 1]
                } else {
                    peak_mag
                };
                let delta = (left - right) / (2.0 * (left - 2.0 * peak_mag + right) + 1e-6);
                pitches[[max_idx, t]] = freqs[max_idx] + delta * (freqs[1] - freqs[0]);
                mags[[max_idx, t]] = peak_mag;
            }
        }
    }

    Ok((pitches, mags))
}

// Helper functions to reduce code duplication and improve maintainability.

fn validate_inputs(
    fmin: f32,
    fmax: f32,
    sample_rate: u32,
    frame_length: usize,
    signal_len: usize,
) -> Result<(), TuningError> {
    if fmin >= fmax || fmin < 0.0 || fmax > sample_rate as f32 / 2.0 {
        return Err(TuningError::InvalidFrequencyRange(
            "fmin must be less than fmax and within Nyquist bounds".to_string(),
        ));
    }
    if signal_len < frame_length {
        return Err(TuningError::InsufficientData(
            "Signal length is less than frame length".to_string(),
        ));
    }
    Ok(())
}

fn calculate_n_frames(signal_len: usize, frame_length: usize, hop_length: usize) -> usize {
    if signal_len < frame_length {
        1
    } else {
        (signal_len - frame_length) / hop_length + 1
    }
}

fn compute_diff_function(frame: &[f32], lag_max: usize) -> Vec<f32> {
    let mut diff = vec![0.0; lag_max];
    for tau in 0..lag_max {
        let sum: f32 = (0..frame.len().saturating_sub(tau))
            .map(|j| {
                let d = frame[j] - frame[j + tau];
                d * d
            })
            .sum();
        diff[tau] = sum;
    }
    diff
}

fn compute_cmnd(diff: &[f32], lag_max: usize) -> Vec<f32> {
    let mut cmnd = vec![1.0; lag_max];
    let mut running_sum = 0.0;
    for tau in 1..lag_max {
        running_sum += diff[tau];
        cmnd[tau] = if running_sum > 1e-6 {
            diff[tau] * tau as f32 / running_sum
        } else {
            1.0
        };
    }
    cmnd
}

fn find_min_cmnd(cmnd: &[f32], lag_min: usize, lag_max: usize) -> (usize, f32) {
    let mut min_idx = lag_min;
    let mut min_val = cmnd[lag_min];
    for tau in lag_min + 1..lag_max {
        if cmnd[tau] < min_val {
            min_val = cmnd[tau];
            min_idx = tau;
        }
    }
    (min_idx, min_val)
}

fn compute_pyin_frame(
    frame: &[f32],
    lag_min: usize,
    lag_max: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> Result<f32, TuningError> {
    let diff = compute_diff_function(frame, lag_max);
    let cmnd = compute_cmnd(&diff, lag_max);
    let (min_idx, min_val) = find_min_cmnd(&cmnd, lag_min, lag_max);

    let pitch = if min_val < 0.1 && min_idx > 1 && min_idx < lag_max - 1 {
        let a = cmnd[min_idx - 1];
        let b = cmnd[min_idx];
        let c = cmnd[min_idx + 1];
        let delta = (a - c) / (2.0 * (a - 2.0 * b + c) + 1e-6);
        sample_rate as f32 / (min_idx as f32 + delta)
    } else {
        0.0
    };

    Ok(if pitch >= fmin && pitch <= fmax { pitch } else { 0.0 })
}

fn compute_yin_frame(
    frame: &[f32],
    lag_min: usize,
    lag_max: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> Result<f32, TuningError> {
    let diff = compute_diff_function(frame, lag_max);
    let cmnd = compute_cmnd(&diff, lag_max);
    let (min_idx, min_val) = find_min_cmnd(&cmnd, lag_min, lag_max);

    let pitch = if min_val < 0.5 && min_idx > 0 {
        sample_rate as f32 / min_idx as f32
    } else {
        0.0
    };

    Ok(if pitch >= fmin && pitch <= fmax { pitch } else { 0.0 })
}

fn compute_spectrogram(
    signal: Option<&[f32]>,
    spectrogram: Option<&Array2<f32>>,
    n_fft: usize,
    hop_length: usize,
) -> Result<Array2<f32>, TuningError> {
    match (signal, spectrogram) {
        (Some(y), None) => {
            if y.len() < n_fft {
                return Err(TuningError::InsufficientData(
                    "Signal length is less than n_fft".to_string(),
                ));
            }
            stft(y, Some(n_fft), Some(hop_length), None)
                .map_err(|e| {
                    TuningError::ComputationFailed(format!("STFT computation failed: {}", e))
                })
                .map(|s| s.mapv(|x| x.norm()))
        }
        (None, Some(s)) => Ok(s.to_owned()),
        _ => Err(TuningError::InvalidInput(
            "Must provide either signal or spectrogram".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_pyin_invalid_inputs() {
        let signal = vec![0.1, 0.2, 0.3];
        // Invalid frequency range
        assert!(matches!(
            pyin(&signal, 500.0, 50.0, None, None),
            Err(TuningError::InvalidFrequencyRange(_))
        ));
        // Signal too short
        assert!(matches!(
            pyin(&signal, 50.0, 500.0, None, Some(10)),
            Err(TuningError::InsufficientData(_))
        ));
    }

    #[test]
    fn test_yin_short_signal() {
        let signal = vec![0.1, 0.2, 0.3, 0.4];
        let pitches = yin(&signal, 50.0, 500.0, None, Some(4)).unwrap();
        assert_eq!(pitches, vec![0.0]);
    }

    #[test]
    fn test_estimate_tuning_no_input() {
        assert!(matches!(
            estimate_tuning(None, None, None, None),
            Err(TuningError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_pitch_tuning_invalid_resolution() {
        let freqs = vec![440.0];
        assert!(matches!(
            pitch_tuning(&freqs, Some(0.0)),
            Err(TuningError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_piptrack_spectrogram() {
        let s = Array2::from_elem((3, 2), 1.0);
        let (pitches, mags) = piptrack(None, None, Some(&s), Some(4), None).unwrap();
        assert_eq!(pitches.shape(), &[3, 2]);
        assert_eq!(mags.shape(), &[3, 2]);
    }

    #[test]
    fn test_piptrack_invalid_spectrogram() {
        let signal = vec![0.1, 0.2, 0.3];
        assert!(matches!(
            piptrack(Some(&signal), None, None, Some(10), None),
            Err(TuningError::InsufficientData(_))
        ));
    }
}