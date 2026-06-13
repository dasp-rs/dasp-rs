use crate::core::io::AudioData;
use crate::features::phase_recovery::griffinlim;
use crate::utils::frequency::{fft_frequencies_impl, mel_frequencies_impl};
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use thiserror::Error;

/// Error conditions for MFCC processing and reconstruction.
///
/// Enumerates specific failure modes in MFCC delta computation and spectrogram/audio
/// reconstruction, tailored for DSP pipeline diagnostics.
#[derive(Error, Debug)]
pub enum MfccError {
    /// Input dimensions are invalid (e.g., empty matrix).
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Input parameters are invalid (e.g., negative width, even width).
    #[error("Invalid parameter: {0}")]
    InvalidInput(String),

    /// Numerical computation failure (e.g., overflow in reconstruction).
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Computes the first-order delta coefficients of MFCCs.
///
/// # Parameters
/// - `mfcc`: Input MFCC matrix, shape `(n_mfcc, n_frames)`.
/// - `width`: Optional window width for delta computation; defaults to 9.
/// - `axis`: Optional time axis; -1 (frames) or 0 (mfcc), defaults to -1.
///
/// # Returns
/// - `Ok(Array2<f32>)`: Delta coefficients, same shape as input MFCCs.
/// - `Err(MfccError)`: Failure due to invalid input or dimensions.
///
/// # Constraints
/// - `width` must be a positive odd integer.
/// - `mfcc` must have at least `width` elements along the time axis.
pub fn compute_delta(mfcc: &Array2<f32>) -> ComputeDeltaBuilder<'_> {
    ComputeDeltaBuilder { mfcc, width: 9, axis: -1 }
}

/// Builder for [`compute_delta`].
#[derive(Debug, Clone)]
pub struct ComputeDeltaBuilder<'a> {
    mfcc: &'a Array2<f32>,
    width: usize,
    axis: isize,
}

impl ComputeDeltaBuilder<'_> {
    /// Set the window width (must be a positive odd integer; default: 9).
    #[must_use]
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set the time axis (-1 for frames, 0 for coefficients; default: -1).
    #[must_use]
    pub fn axis(mut self, axis: isize) -> Self {
        self.axis = axis;
        self
    }

    /// Compute the delta coefficients.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<f32>, MfccError> {
        compute_delta_impl(self.mfcc, Some(self.width), Some(self.axis))
    }
}

fn compute_delta_impl(
    mfcc: &Array2<f32>,
    width: Option<usize>,
    axis: Option<isize>,
) -> Result<Array2<f32>, MfccError> {
    let width = width.unwrap_or(9);
    let axis = axis.unwrap_or(-1);

    if width == 0 || width % 2 == 0 {
        return Err(MfccError::InvalidInput(
            "Width must be a positive odd integer".to_string(),
        ));
    }
    let ax = usize::from(axis < 0);
    let (n_mfcc, n_frames) = if ax == 1 {
        mfcc.dim()
    } else {
        (mfcc.shape()[1], mfcc.shape()[0])
    };
    if n_frames == 0 || n_mfcc == 0 {
        return Err(MfccError::InvalidDimensions(
            "MFCC matrix is empty".to_string(),
        ));
    }
    if n_frames < width {
        return Err(MfccError::InvalidDimensions(format!(
            "Time axis length {n_frames} less than width {width}"
        )));
    }

    let half_width = width / 2;
    let weights: Vec<f32> = (-(half_width as isize)..=half_width as isize)
        .map(|i| i as f32)
        .collect();
    let norm = weights.iter().map(|x| x.powi(2)).sum::<f32>();
    if norm == 0.0 {
        return Err(MfccError::ComputationFailed(
            "Normalization factor is zero".to_string(),
        ));
    }

    let mut delta = Array2::zeros(mfcc.dim());
    delta
        .axis_iter_mut(Axis(ax))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            let row = mfcc.index_axis(Axis(ax), i);
            for j in 0..row.len() {
                let mut sum = 0.0;
                for (w_idx, &w) in weights.iter().enumerate() {
                    let offset = w_idx as isize - half_width as isize;
                    let idx = (j as isize + offset).clamp(0, row.len() as isize - 1) as usize;
                    sum += w * row[idx];
                }
                slice[j] = sum / norm;
            }
        });

    Ok(delta)
}

/// Converts mel spectrogram to STFT magnitude spectrogram.
///
/// Reconstructs an STFT magnitude spectrogram from a mel spectrogram using inverse mel
/// filterbank weighting, ensuring energy preservation.
///
/// # Parameters
/// - `m`: Mel spectrogram, shape `(n_mels, n_frames)`.
/// - `sr`: Optional sample rate in Hz; defaults to 44100.
/// - `n_fft`: Optional FFT size; defaults to 2048.
/// - `power`: Optional power of input spectrogram; defaults to 2.0.
///
/// # Returns
/// - `Ok(Array2<f32>)`: STFT magnitude spectrogram, shape `(n_fft/2 + 1, n_frames)`.
/// - `Err(MfccError)`: Failure due to invalid dimensions or parameters.
pub fn mel_to_stft(m: &Array2<f32>) -> MelToStftBuilder<'_> {
    MelToStftBuilder { m, sr: 44100, n_fft: 2048, power: 2.0 }
}

/// Builder for [`mel_to_stft`].
#[derive(Debug, Clone)]
pub struct MelToStftBuilder<'a> {
    m: &'a Array2<f32>,
    sr: u32,
    n_fft: usize,
    power: f32,
}

impl MelToStftBuilder<'_> {
    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sr = sr;
        self
    }

    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the magnitude exponent (default: 2.0).
    #[must_use]
    pub fn power(mut self, power: f32) -> Self {
        self.power = power;
        self
    }

    /// Approximate an STFT magnitude spectrogram from a mel spectrogram.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<f32>, MfccError> {
        mel_to_stft_impl(self.m, Some(self.sr), Some(self.n_fft), Some(self.power))
    }
}

fn mel_to_stft_impl(
    m: &Array2<f32>,
    sr: Option<u32>,
    n_fft: Option<usize>,
    power: Option<f32>,
) -> Result<Array2<f32>, MfccError> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let power = power.unwrap_or(2.0);
    if m.is_empty() {
        return Err(MfccError::InvalidDimensions(
            "Mel spectrogram is empty".to_string(),
        ));
    }
    if n_fft < 2 {
        return Err(MfccError::InvalidInput(
            "n_fft must be at least 2".to_string(),
        ));
    }
    if power <= 0.0 {
        return Err(MfccError::InvalidInput(
            "Power must be positive".to_string(),
        ));
    }

    let n_mels = m.shape()[0];
    let n_frames = m.shape()[1];
    let mel_f = mel_frequencies_impl(n_mels + 2, 0.0, sr as f32 / 2.0);
    let fft_f = fft_frequencies_impl(sr, n_fft);
    let n_bins = n_fft / 2 + 1;

    let mut s = Array2::zeros((n_bins, n_frames));
    s.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(t, mut col)| {
            for mel in 0..n_mels {
                let f_low = mel_f[mel];
                let f_center = mel_f[mel + 1];
                let f_high = mel_f[mel + 2];
                for (bin, &f) in fft_f.iter().enumerate().take(n_bins) {
                    let weight = if f >= f_low && f <= f_high {
                        if f <= f_center {
                            (f - f_low) / (f_center - f_low)
                        } else {
                            (f_high - f) / (f_high - f_center)
                        }
                    } else {
                        0.0
                    }
                    .max(0.0);
                    col[bin] += m[[mel, t]].max(0.0) * weight;
                }
            }
        });

    Ok(s.mapv(|x: f32| x.powf(1.0 / power)))
}

/// Converts mel spectrogram to audio waveform.
///
/// Reconstructs a time-domain audio signal from a mel spectrogram using STFT magnitude
/// estimation and Griffin-Lim phase recovery.
///
/// # Parameters
/// - `m`: Mel spectrogram, shape `(n_mels, n_frames)`.
/// - `sr`: Optional sample rate in Hz; defaults to 44100.
/// - `n_fft`: Optional FFT size; defaults to 2048.
/// - `hop_length`: Optional hop length; defaults to `n_fft / 4`.
///
/// # Returns
/// - `Ok(AudioData)`: Reconstructed audio waveform with metadata.
/// - `Err(MfccError)`: Failure due to invalid input or reconstruction errors.
///
/// # Complexity
/// - O(M * F * B + G) where G is Griffin-Lim complexity, parallelized in `mel_to_stft`.
pub fn mel_to_audio(m: &Array2<f32>) -> MelToAudioBuilder<'_> {
    MelToAudioBuilder { m, sr: 44100, n_fft: 2048, hop_length: None }
}

/// Builder for [`mel_to_audio`].
#[derive(Debug, Clone)]
pub struct MelToAudioBuilder<'a> {
    m: &'a Array2<f32>,
    sr: u32,
    n_fft: usize,
    hop_length: Option<usize>,
}

impl MelToAudioBuilder<'_> {
    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sr = sr;
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

    /// Reconstruct audio from a mel spectrogram (via Griffin-Lim).
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<AudioData, MfccError> {
        mel_to_audio_impl(self.m, Some(self.sr), Some(self.n_fft), self.hop_length)
    }
}

fn mel_to_audio_impl(
    m: &Array2<f32>,
    sr: Option<u32>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<AudioData, MfccError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let sr = sr.unwrap_or(44100);
    if hop == 0 {
        return Err(MfccError::InvalidInput(
            "Hop length must be positive".to_string(),
        ));
    }

    let s = mel_to_stft_impl(m, Some(sr), Some(n_fft), None)?;
    let samples = griffinlim(&s)
        .hop_length(hop)
        .compute()
        .map_err(|e| MfccError::ComputationFailed(format!("Griffin-Lim failed: {e}")))?;
    if samples.is_empty() {
        return Err(MfccError::ComputationFailed(
            "Griffin-Lim returned empty samples".to_string(),
        ));
    }
    if samples.iter().any(|&x| !x.is_finite()) {
        return Err(MfccError::ComputationFailed(
            "Non-finite samples in reconstruction".to_string(),
        ));
    }
    AudioData::new(samples, sr, 1).map_err(|e| MfccError::ComputationFailed(e.to_string()))
}

/// Converts MFCCs back to mel spectrogram using inverse DCT.
///
/// Reconstructs a mel spectrogram from MFCCs via inverse discrete cosine transform (type II).
///
/// # Parameters
/// - `mfcc`: MFCC matrix, shape `(n_mfcc, n_frames)`.
/// - `n_mels`: Optional number of mel bins; defaults to 128.
/// - `dct_type`: Optional DCT type (1, 2, 3, 4); defaults to 2.
///
/// # Returns
/// - `Ok(Array2<f32>)`: Mel spectrogram, shape `(n_mels, n_frames)`.
/// - `Err(MfccError)`: Failure due to invalid dimensions or DCT type.
///
/// # Complexity
/// - O(M * F * K) where M is mel bins, F is frames, K is MFCC coefficients, parallelized over frames.
pub fn mfcc_to_mel(mfcc: &Array2<f32>) -> MfccToMelBuilder<'_> {
    MfccToMelBuilder { mfcc, n_mels: 128, dct_type: 2 }
}

/// Builder for [`mfcc_to_mel`].
#[derive(Debug, Clone)]
pub struct MfccToMelBuilder<'a> {
    mfcc: &'a Array2<f32>,
    n_mels: usize,
    dct_type: i32,
}

impl MfccToMelBuilder<'_> {
    /// Set the number of mel bins (default: 128).
    #[must_use]
    pub fn n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Set the DCT type (default: 2).
    #[must_use]
    pub fn dct_type(mut self, dct_type: i32) -> Self {
        self.dct_type = dct_type;
        self
    }

    /// Invert MFCCs back to a mel spectrogram.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<f32>, MfccError> {
        mfcc_to_mel_impl(self.mfcc, Some(self.n_mels), Some(self.dct_type))
    }
}

fn mfcc_to_mel_impl(
    mfcc: &Array2<f32>,
    n_mels: Option<usize>,
    dct_type: Option<i32>,
) -> Result<Array2<f32>, MfccError> {
    let n_mels = n_mels.unwrap_or(128);
    let dct_type = dct_type.unwrap_or(2);
    if mfcc.is_empty() {
        return Err(MfccError::InvalidDimensions(
            "MFCC matrix is empty".to_string(),
        ));
    }
    if ![1, 2, 3, 4].contains(&dct_type) {
        return Err(MfccError::InvalidInput(format!(
            "Unsupported DCT type: {dct_type}"
        )));
    }

    let n_frames = mfcc.shape()[1];
    let n_mfcc = mfcc.shape()[0];
    let mut mel = Array2::zeros((n_mels, n_frames));
    mel.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(t, mut col)| {
            for n in 0..n_mels {
                let mut sum = 0.0;
                for k in 0..n_mfcc {
                    let scale = if k == 0 {
                        1.0 / (n_mels as f32).sqrt()
                    } else {
                        (2.0 / n_mels as f32).sqrt()
                    };
                    let theta = std::f32::consts::PI * k as f32 * (n as f32 + 0.5) / n_mels as f32;
                    sum += scale * mfcc[[k, t]] * theta.cos();
                }
                col[n] = sum.max(0.0);
            }
        });
    Ok(mel.mapv(f32::exp))
}

/// Converts MFCCs to audio waveform.
///
/// Reconstructs a time-domain audio signal from MFCCs via mel spectrogram and STFT.
///
/// # Parameters
/// - `mfcc`: MFCC matrix, shape `(n_mfcc, n_frames)`.
/// - `n_mels`: Optional number of mel bins; defaults to 128.
/// - `sr`: Optional sample rate in Hz; defaults to 44100.
/// - `n_fft`: Optional FFT size; defaults to 2048.
/// - `hop_length`: Optional hop length; defaults to `n_fft / 4`.
///
/// # Returns
/// - `Ok(AudioData)`: Reconstructed audio waveform with metadata.
/// - `Err(MfccError)`: Failure due to invalid input or reconstruction errors.
pub fn mfcc_to_audio(mfcc: &Array2<f32>) -> MfccToAudioBuilder<'_> {
    MfccToAudioBuilder { mfcc, n_mels: 128, sr: 44100, n_fft: 2048, hop_length: None }
}

/// Builder for [`mfcc_to_audio`].
#[derive(Debug, Clone)]
pub struct MfccToAudioBuilder<'a> {
    mfcc: &'a Array2<f32>,
    n_mels: usize,
    sr: u32,
    n_fft: usize,
    hop_length: Option<usize>,
}

impl MfccToAudioBuilder<'_> {
    /// Set the number of mel bins (default: 128).
    #[must_use]
    pub fn n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sr = sr;
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

    /// Reconstruct audio directly from MFCCs.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<AudioData, MfccError> {
        let mel = mfcc_to_mel_impl(self.mfcc, Some(self.n_mels), Some(2))?;
        mel_to_audio_impl(&mel, Some(self.sr), Some(self.n_fft), self.hop_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compute_delta_invalid_width() {
        let mfcc = array![[0.1, 0.2], [0.3, 0.4]];
        let result = compute_delta(&mfcc).width(2).compute(); // Invalid even width
        assert!(matches!(result, Err(MfccError::InvalidInput(_))));
    }

    #[test]
    fn test_compute_delta_empty_input() {
        let mfcc = array![[]]; // Empty input
        let result = compute_delta(&mfcc).width(3).compute();
        assert!(matches!(result, Err(MfccError::InvalidDimensions(_))));
    }

    #[test]
    fn test_compute_delta_insufficient_frames() {
        let mfcc = array![[0.1, 0.2], [0.3, 0.4]]; // Only 2 frames
        let result = compute_delta(&mfcc).width(5).compute(); // Width 5 requires at least 5 frames
        assert!(matches!(result, Err(MfccError::InvalidDimensions(_))));
    }

    #[test]
    fn test_mfcc_to_mel() {
        let mfcc = array![[0.1, 0.2], [0.3, 0.4]];
        let mel = mfcc_to_mel(&mfcc).n_mels(4).compute().unwrap();
        assert_eq!(mel.shape(), &[4, 2]);
        assert!(mel[[0, 0]] > 0.0);
    }

    #[test]
    fn test_invalid_input() {
        let empty = array![[]];
        assert!(matches!(
            compute_delta(&empty).compute(),
            Err(MfccError::InvalidDimensions(_))
        ));
        assert!(matches!(
            mel_to_stft(&empty).compute(),
            Err(MfccError::InvalidDimensions(_))
        ));
    }
}
