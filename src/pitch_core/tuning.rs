use ndarray::Array2;
use thiserror::Error;

use crate::signal_processing::time_frequency::stft;
use crate::core::AudioError;
use crate::utils::frequency::fft_frequencies_impl;

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
            TuningError::InvalidFrequencyRange(msg) | TuningError::InvalidInput(msg) => {
                Self::InvalidInput(msg)
            }
            TuningError::InsufficientData(msg) => Self::InsufficientData(msg),
            TuningError::ComputationFailed(msg) => Self::ComputationFailed(msg),
        }
    }
}

/// Builder for the pYIN pitch tracker (probabilistic YIN).
#[derive(Debug, Clone)]
pub struct PyinBuilder<'a> {
    signal: &'a [f32],
    fmin: f32,
    fmax: f32,
    sample_rate: u32,
    frame_length: usize,
    hop_length: Option<usize>,
}

impl PyinBuilder<'_> {
    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Set the frame length in samples (default: 2048).
    #[must_use]
    pub fn frame_length(mut self, frame_length: usize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Set the hop length in samples (default: `frame_length / 4`).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Compute per-frame pitch estimates in Hz.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Vec<f32>, TuningError> {
        pyin_impl(
            self.signal,
            self.fmin,
            self.fmax,
            self.sample_rate,
            self.frame_length,
            self.hop_length,
        )
    }
}

/// Estimates pitch using the pYIN algorithm (probabilistic YIN).
///
/// Returns a builder. `fmin`/`fmax` (Hz) are required; sample rate, frame
/// length, and hop length have defaults (44100, 2048, `frame_length / 4`).
///
/// # Example
/// ```no_run
/// use dasp_rs::pitch::pyin;
/// let signal = vec![0.0_f32; 4096];
/// let pitches = pyin(&signal, 50.0, 500.0).sample_rate(44100).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn pyin(signal: &[f32], fmin: f32, fmax: f32) -> PyinBuilder<'_> {
    PyinBuilder {
        signal,
        fmin,
        fmax,
        sample_rate: 44_100,
        frame_length: 2048,
        hop_length: None,
    }
}

fn pyin_impl(
    signal: &[f32],
    fmin: f32,
    fmax: f32,
    sr: u32,
    frame_len: usize,
    hop_length: Option<usize>,
) -> Result<Vec<f32>, TuningError> {
    validate_inputs(fmin, fmax, sr, frame_len, signal.len())?;

    let hop_length = hop_length.unwrap_or(frame_len / 4).max(1);
    let n_frames = calculate_n_frames(signal.len(), frame_len, hop_length);
    let lag_min = (sr as f32 / fmax).round() as usize;
    let lag_max = (sr as f32 / fmin).round() as usize;

    let mut pitches = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * hop_length;
        let frame = &signal[start..(start + frame_len).min(signal.len())];
        let pitch = compute_pyin_frame(frame, lag_min, lag_max, sr, fmin, fmax);
        pitches.push(pitch);
    }

    Ok(pitches)
}

/// Builder for the YIN pitch tracker.
#[derive(Debug, Clone)]
pub struct YinBuilder<'a> {
    signal: &'a [f32],
    fmin: f32,
    fmax: f32,
    sample_rate: u32,
    frame_length: usize,
    hop_length: Option<usize>,
}

impl YinBuilder<'_> {
    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Set the frame length in samples (default: 2048).
    #[must_use]
    pub fn frame_length(mut self, frame_length: usize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Set the hop length in samples (default: `frame_length / 4`).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Compute per-frame pitch estimates in Hz.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Vec<f32>, TuningError> {
        yin_impl(
            self.signal,
            self.fmin,
            self.fmax,
            self.sample_rate,
            self.frame_length,
            self.hop_length,
        )
    }
}

/// Estimates pitch using the YIN algorithm.
///
/// Returns a builder. `fmin`/`fmax` (Hz) are required; sample rate, frame
/// length, and hop length have defaults (44100, 2048, `frame_length / 4`).
///
/// # Example
/// ```no_run
/// use dasp_rs::pitch::yin;
/// let signal = vec![0.0_f32; 4096];
/// let pitches = yin(&signal, 50.0, 500.0).sample_rate(44100).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn yin(signal: &[f32], fmin: f32, fmax: f32) -> YinBuilder<'_> {
    YinBuilder {
        signal,
        fmin,
        fmax,
        sample_rate: 44_100,
        frame_length: 2048,
        hop_length: None,
    }
}

fn yin_impl(
    signal: &[f32],
    fmin: f32,
    fmax: f32,
    sr: u32,
    frame_len: usize,
    hop_length: Option<usize>,
) -> Result<Vec<f32>, TuningError> {
    validate_inputs(fmin, fmax, sr, frame_len, signal.len())?;

    let hop_length = hop_length.unwrap_or(frame_len / 4).max(1);
    let n_frames = calculate_n_frames(signal.len(), frame_len, hop_length);
    let lag_min = (sr as f32 / fmax).round() as usize;
    let lag_max = (sr as f32 / fmin).round() as usize;

    let mut pitches = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * hop_length;
        let frame = &signal[start..(start + frame_len).min(signal.len())];
        let pitch = compute_yin_frame(frame, lag_min, lag_max, sr, fmin, fmax);
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
/// * `hop_length` - Hop length in samples between frames (defaults to `n_fft / 4` if `None`).
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
/// ```no_run
/// use dasp_rs::pitch::estimate_tuning;
/// let signal = vec![0.0_f32; 4096];
/// let tuning = estimate_tuning(&signal).n_fft(2048).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn estimate_tuning(signal: &[f32]) -> TuningBuilder<'_> {
    TuningBuilder {
        signal,
        spectrogram: None,
        sample_rate: 44_100,
        n_fft: 2048,
        hop_length: None,
    }
}

/// Builder for [`estimate_tuning`].
#[derive(Debug, Clone)]
pub struct TuningBuilder<'a> {
    signal: &'a [f32],
    spectrogram: Option<&'a Array2<f32>>,
    sample_rate: u32,
    n_fft: usize,
    hop_length: Option<usize>,
}

impl<'a> TuningBuilder<'a> {
    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length (default: `n_fft / 4`).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Use a precomputed magnitude spectrogram instead of computing one.
    #[must_use]
    pub fn spectrogram(mut self, spectrogram: &'a Array2<f32>) -> Self {
        self.spectrogram = Some(spectrogram);
        self
    }

    /// Compute the tuning deviation in cents.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<f32, TuningError> {
        estimate_tuning_impl(
            Some(self.signal),
            Some(self.sample_rate),
            self.spectrogram,
            Some(self.n_fft),
            self.hop_length,
        )
    }
}

fn estimate_tuning_impl(
    signal: Option<&[f32]>,
    sample_rate: Option<u32>,
    spectrogram: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<f32, TuningError> {
    let sr = sample_rate.unwrap_or(44_100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(n_fft / 4).max(1);

    let s = compute_spectrogram(signal, spectrogram, n_fft, hop_length)?;
    // The spectrogram is already computed; pass it (not the signal) to piptrack.
    let (pitches, mags) = piptrack_impl(None, Some(sr), Some(&s), Some(n_fft), Some(hop_length))?;

    let mut total_deviation = 0.0;
    let mut total_weight = 0.0;

    for t in 0..pitches.shape()[1] {
        for f in 0..pitches.shape()[0] {
            let freq = pitches[[f, t]];
            let mag = mags[[f, t]];
            if freq > 0.0 && mag > 1e-6 {
                // Find the nearest lower semitone in equal temperament tuning
                // n_semitones = 12 * log2(freq / 440.0), then floor to get nearest lower semitone
                let ref_freq = 440.0 * 2.0f32.powf((12.0 * f32::log2(freq / 440.0)).floor() / 12.0);
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
/// use dasp_rs::pitch::pitch_tuning;
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
        // Find the nearest lower semitone in equal temperament tuning
        // n_semitones = 12 * log2(freq / 440.0), then floor to get nearest lower semitone
        let ref_freq = 440.0 * 2.0f32.powf((12.0 * f32::log2(freq / 440.0)).floor() / 12.0);
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
/// * `hop_length` - Hop length in samples (defaults to `n_fft/4` if `None`).
///
/// # Returns
/// A `Result` containing a tuple of:
/// * `pitches` - 2D array of pitch estimates in Hz (`Array2<f32>`).
/// * `magnitudes` - 2D array of corresponding peak magnitudes (`Array2<f32>`).
///
/// # Errors
/// * `TuningError::InsufficientData` - If signal length is less than `n_fft`.
/// * `TuningError::InvalidInput` - If neither `signal` nor `spectrogram` is provided.
/// * `TuningError::ComputationFailed` - If STFT or frequency bin computation fails.
///
/// # Example
/// ```no_run
/// use dasp_rs::pitch::piptrack;
/// let signal = vec![0.0_f32; 4096];
/// let (pitches, mags) = piptrack(&signal).n_fft(2048).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn piptrack(signal: &[f32]) -> PiptrackBuilder<'_> {
    PiptrackBuilder {
        signal,
        spectrogram: None,
        sample_rate: 44_100,
        n_fft: 2048,
        hop_length: None,
    }
}

/// Builder for [`piptrack`].
#[derive(Debug, Clone)]
pub struct PiptrackBuilder<'a> {
    signal: &'a [f32],
    spectrogram: Option<&'a Array2<f32>>,
    sample_rate: u32,
    n_fft: usize,
    hop_length: Option<usize>,
}

impl<'a> PiptrackBuilder<'a> {
    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length (default: `n_fft / 4`).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Use a precomputed magnitude spectrogram instead of the signal.
    #[must_use]
    pub fn spectrogram(mut self, spectrogram: &'a Array2<f32>) -> Self {
        self.spectrogram = Some(spectrogram);
        self
    }

    /// Compute `(pitches, magnitudes)`.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<(Array2<f32>, Array2<f32>), TuningError> {
        let signal = if self.spectrogram.is_some() { None } else { Some(self.signal) };
        piptrack_impl(
            signal,
            Some(self.sample_rate),
            self.spectrogram,
            Some(self.n_fft),
            self.hop_length,
        )
    }
}

fn piptrack_impl(
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
    let freqs = fft_frequencies_impl(sr, n_fft);
    if freqs.len() != s.shape()[0] {
        return Err(TuningError::ComputationFailed(
            "Frequency bins mismatch with spectrogram".to_string(),
        ));
    }

    let mut pitches = Array2::zeros(s.dim());
    let mut mags = Array2::zeros(s.dim());
    // Precompute frequency bin width (constant for all frames)
    let freq_bin_width = if freqs.len() > 1 { freqs[1] - freqs[0] } else { 0.0 };

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
                pitches[[max_idx, t]] = freqs[max_idx] + delta * freq_bin_width;
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
    cmnd[lag_min..lag_max]
        .iter()
        .enumerate()
        .map(|(i, &v)| (lag_min + i, v))
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((lag_min, cmnd[lag_min]))
}

fn compute_pyin_frame(
    frame: &[f32],
    lag_min: usize,
    lag_max: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> f32 {
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

    if pitch >= fmin && pitch <= fmax { pitch } else { 0.0 }
}

fn compute_yin_frame(
    frame: &[f32],
    lag_min: usize,
    lag_max: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> f32 {
    let diff = compute_diff_function(frame, lag_max);
    let cmnd = compute_cmnd(&diff, lag_max);
    let (min_idx, min_val) = find_min_cmnd(&cmnd, lag_min, lag_max);

    let pitch = if min_val < 0.5 && min_idx > 0 {
        sample_rate as f32 / min_idx as f32
    } else {
        0.0
    };

    if pitch >= fmin && pitch <= fmax { pitch } else { 0.0 }
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
            stft(y)
                .n_fft(n_fft)
                .hop_length(hop_length)
                .compute()
                .map_err(|e| {
                    TuningError::ComputationFailed(format!("STFT computation failed: {e}"))
                })
                .map(|s| s.mapv(num_complex::Complex::norm))
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
            pyin(&signal, 500.0, 50.0).compute(),
            Err(TuningError::InvalidFrequencyRange(_))
        ));
        // Signal too short
        assert!(matches!(
            pyin(&signal, 50.0, 500.0).frame_length(10).compute(),
            Err(TuningError::InsufficientData(_))
        ));
    }

    #[test]
    fn test_yin_short_signal() {
        let signal = vec![0.1, 0.2, 0.3, 0.4];
        let pitches = yin(&signal, 50.0, 500.0).frame_length(4).compute().unwrap();
        assert_eq!(pitches, vec![0.0]);
    }

    #[test]
    fn test_estimate_tuning_short_signal_returns_zero() {
        let signal = vec![0.0_f32; 4096];
        let tuning = estimate_tuning(&signal).n_fft(2048).compute().unwrap();
        assert!(tuning.abs() < f32::EPSILON);
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
        let (pitches, mags) = piptrack(&[]).n_fft(4).spectrogram(&s).compute().unwrap();
        assert_eq!(pitches.shape(), &[3, 2]);
        assert_eq!(mags.shape(), &[3, 2]);
    }

    #[test]
    fn test_piptrack_invalid_spectrogram() {
        let signal = vec![0.1, 0.2, 0.3];
        assert!(matches!(
            piptrack(&signal).n_fft(10).compute(),
            Err(TuningError::InsufficientData(_))
        ));
    }
}
