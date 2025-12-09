use ndarray::{Array1, Array2, Axis};
use crate::signal_processing::time_frequency::stft;
use thiserror::Error;

/// Tempo analysis builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct TempoBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    win_length: usize,
}

impl<'a> TempoBuilder<'a> {
    /// Set the hop length (default: 512).
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the window length (default: 2048).
    pub fn win_length(mut self, win_length: usize) -> Self {
        self.win_length = win_length;
        self
    }

    /// Compute tempo.
    pub fn compute(self) -> Result<f32, RhythmError> {
        tempo_impl(Some(self.y), Some(self.sr), None, Some(self.hop_length))
    }
}

/// Computes tempo from audio signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Sample rate in Hz
///
/// # Returns
/// Returns a builder that can be configured with method chaining.
///
/// # Examples
/// ```
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// let bpm = tempo(&y, 44100)
///     .hop_length(512)
///     .compute()?;
/// ```
pub fn tempo(y: &[f32], sr: u32) -> TempoBuilder {
    TempoBuilder {
        y,
        sr,
        hop_length: 512,
        win_length: 2048,
    }
}

// Old RhythmBuilder removed - use direct functions instead

/// Custom error types for rhythm analysis operations.
#[derive(Error, Debug)]
pub enum RhythmError {
    /// Invalid input parameters or data.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Computation failed during processing.
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Estimates the tempo (beats per minute) from audio or onset envelope.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `onset_envelope` - Optional pre-computed onset strength envelope
/// * `hop_length` - Optional hop length in samples (defaults to 512)
///
/// # Returns
/// Returns a single `f32` value representing the estimated tempo in BPM.
///
/// # Errors
/// Returns an error if `y` is None and `onset_envelope` is None, or if STFT computation fails when `y` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// // Clean, ergonomic API
/// let bpm = tempo(&y, 44100)
///     .hop_length(512)
///     .compute()?;
/// ```
pub fn tempo_impl(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
) -> Result<f32, RhythmError> {
    let sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let onset_owned = if onset_envelope.is_none() {
        let y = y.ok_or(RhythmError::InvalidInput("Audio signal required when onset_envelope is None".to_string()))?;
        let s = stft(y)
            .hop_length(hop)
            .compute()
            .map_err(|e| RhythmError::ComputationFailed(format!("STFT computation failed: {}", e)))?
            .mapv(|x| x.norm());
        s.map_axis(Axis(0), |row| row.iter().map(|&x| x.max(0.0)).sum::<f32>())
    } else {
        onset_envelope.unwrap().to_owned()
    };
    let onset = &onset_owned;
        let tempogram = tempogram_impl(None, Some(sr), Some(onset), hop_length, None)?;
    Ok(tempogram.axis_iter(Axis(1)).map(|col| {
        let max_val = col.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let max_idx = col.iter().position(|&x| x == *max_val).unwrap_or(0);
        crate::utils::frequency::tempo_frequencies(tempogram.shape()[0], Some(hop), Some(sr))[max_idx]
    }).sum::<f32>() / tempogram.shape()[1] as f32)
}

/// Computes a tempogram (local autocorrelation of onset strength).
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `onset_envelope` - Optional pre-computed onset strength envelope
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `win_length` - Optional window length for autocorrelation (defaults to 384)
///
/// # Returns
/// Returns a 2D array of shape `(win_length/2 + 1, n_frames)` representing the tempogram.
///
/// # Panics
/// Panics if `y` is None and `onset_envelope` is None, or if STFT computation fails when `y` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// // Clean, ergonomic API
/// let tgram = tempogram(&y, 44100)
///     .hop_length(512)
///     .win_length(384)
///     .compute()?;
/// ```
pub fn tempogram_impl(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
) -> Result<Array2<f32>, RhythmError> {
    let _sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let win = win_length.unwrap_or(384);
    let onset_owned = if onset_envelope.is_none() {
        let y = y.ok_or(RhythmError::InvalidInput("Audio signal required when onset_envelope is None".to_string()))?;
        let s = stft(y)
            .hop_length(hop)
            .compute()
            .map_err(|e| RhythmError::ComputationFailed(format!("STFT computation failed: {}", e)))?
            .mapv(|x| x.norm());
        s.map_axis(Axis(0), |row| row.iter().map(|&x| x.max(0.0)).sum::<f32>())
    } else {
        onset_envelope.unwrap().to_owned()
    };
    let onset = &onset_owned;
    let mut tempogram = Array2::zeros((win / 2 + 1, onset.len()));
    for t in 0..onset.len() {
        for lag in 0..(win / 2 + 1) {
            let past = (t as isize - lag as isize).max(0) as usize;
            tempogram[[lag, t]] = onset[t] * onset[past];
        }
    }
    Ok(tempogram)
}

/// Computes a tempogram with harmonic ratio analysis.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `onset_envelope` - Optional pre-computed onset strength envelope
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `ratios` - Optional array of tempo ratios to analyze (defaults to [2.0, 3.0, 4.0])
///
/// # Returns
/// Returns a 2D array of shape `(n_ratios, n_frames)` representing the ratio tempogram.
///
/// # Panics
/// Panics if `y` is None and `onset_envelope` is None, or if STFT computation fails when `y` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// // Using the builder pattern (recommended)
/// let ratio_tgram = RhythmBuilder::new()
///     .signal(&y)
///     .sr(44100)
///     .hop_length(512)
///     .ratios(&[2.0, 3.0, 4.0])
///     .tempogram_ratio()?;
/// 
/// // Or using the direct function (legacy)
/// let ratio_tgram = tempogram_ratio(Some(&y), Some(44100), None, Some(512), Some(&[2.0, 3.0, 4.0]))?;
/// ```
pub fn tempogram_ratio_impl(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
    ratios: Option<&[f32]>,
) -> Result<Array2<f32>, RhythmError> {
        let tempogram = tempogram_impl(y, sr, onset_envelope, hop_length, None)?;
    let ratios = ratios.unwrap_or(&[2.0, 3.0, 4.0]);
    let mut ratio_map = Array2::zeros((ratios.len(), tempogram.shape()[1]));
    for (r_idx, &r) in ratios.iter().enumerate() {
        for t in 0..tempogram.shape()[1] {
            let mut sum = 0.0;
            for f in 0..tempogram.shape()[0] {
                let target_f = f as f32 * r;
                let bin = target_f.round() as usize;
                if bin < tempogram.shape()[0] {
                    sum += tempogram[[bin, t]];
                }
            }
            ratio_map[[r_idx, t]] = sum;
        }
    }
    Ok(ratio_map)
}