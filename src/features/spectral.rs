use ndarray::{s, stack, Array1, Array2, Axis};
use rayon::prelude::*;
use crate::signal_processing::time_frequency::{stft, cqt};
use crate::signal_processing::time_domain::{autocorrelate, log_energy};
use crate::utils::frequency::hz_to_midi;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use thiserror::Error;
use crate::core::io::{AudioError, AudioData};
use crate::utils::frequency::{fft_frequencies_impl, mel_frequencies_impl};

/// Chroma STFT builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct ChromaStftBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    norm: f32,
}

impl ChromaStftBuilder<'_> {
    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the normalization factor (default: 1.0).
    #[must_use]
    pub fn norm(mut self, norm: f32) -> Self {
        self.norm = norm;
        self
    }

    /// Compute chroma features using STFT.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<f32>, SpectralError> {
        chroma_stft_impl(self.y, self.sr, None, Some(self.norm), Some(self.n_fft), Some(self.hop_length))
    }
}

/// Mel spectrogram builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct MelSpectrogramBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    fmax: f32,
}

impl MelSpectrogramBuilder<'_> {
    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the number of mel bins (default: 128).
    #[must_use]
    pub fn n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Set the maximum frequency (default: `sample_rate` / 2).
    #[must_use]
    pub fn fmax(mut self, fmax: f32) -> Self {
        self.fmax = fmax;
        self
    }

    /// Compute mel spectrogram.
    ///
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<f32>, SpectralError> {
        melspectrogram_impl(self.y, self.sr, None, Some(self.n_fft), Some(self.hop_length), Some(self.n_mels), Some(self.fmax))
    }
}

/// Spectral analysis builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct SpectralBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
    norm: f32,
    bins_per_octave: usize,
    frame_length: usize,
    harm_win: usize,
    perc_win: usize,
    n_bands: usize,
    n_formants: usize,
    order: usize,
    roll_percent: f32,
    p: i32,
}

impl SpectralBuilder<'_> {
    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the window length (default: 2048).
    #[must_use]
    pub fn win_length(mut self, win_length: usize) -> Self {
        self.win_length = win_length;
        self
    }

    /// Set the number of mel bins (default: 128).
    #[must_use]
    pub fn n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Set the minimum frequency (default: 0.0).
    #[must_use]
    pub fn fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Set the maximum frequency (default: `sample_rate` / 2).
    #[must_use]
    pub fn fmax(mut self, fmax: f32) -> Self {
        self.fmax = fmax;
        self
    }

    /// Set the normalization factor (default: 1.0).
    #[must_use]
    pub fn norm(mut self, norm: f32) -> Self {
        self.norm = norm;
        self
    }

    /// Compute chroma features using STFT.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn chroma_stft(self) -> Result<Array2<f32>, SpectralError> {
        chroma_stft_impl(self.y, self.sr, None, Some(self.norm), Some(self.n_fft), Some(self.hop_length))
    }

    /// Compute mel spectrogram.
    ///
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn melspectrogram(self) -> Result<Array2<f32>, SpectralError> {
        melspectrogram_impl(self.y, self.sr, None, Some(self.n_fft), Some(self.hop_length), Some(self.n_mels), Some(self.fmax))
    }

    /// Compute MFCC features.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn mfcc(self) -> Result<Array2<f32>, SpectralError> {
        mfcc_impl(self.y, self.sr, None, None, Some(self.n_fft), Some(self.hop_length))
    }

    /// Compute spectral centroid.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn spectral_centroid(self) -> Result<Array1<f32>, SpectralError> {
        spectral_centroid_impl(self.y, self.sr, None, Some(self.n_fft), Some(self.hop_length))
    }

    /// Compute spectral bandwidth.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn spectral_bandwidth(self) -> Result<Array1<f32>, SpectralError> {
        spectral_bandwidth_impl(self.y, self.sr, None, Some(self.n_fft), Some(self.hop_length), Some(self.p))
    }

    /// Compute spectral rolloff.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn spectral_rolloff(self) -> Result<Array1<f32>, SpectralError> {
        spectral_rolloff_impl(self.y, self.sr, None, Some(self.n_fft), Some(self.hop_length), Some(self.roll_percent))
    }

    /// Compute spectral flatness.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn spectral_flatness(self) -> Result<Array1<f32>, SpectralError> {
        spectral_flatness_impl(self.y, self.sr, None, Some(self.n_fft), Some(self.hop_length))
    }

    /// Compute spectral flux.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn spectral_flux(self) -> Result<Array1<f32>, SpectralError> {
        spectral_flux_impl(self.y, self.sr, None, Some(self.n_fft), Some(self.hop_length))
    }

    /// Compute spectral entropy.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn spectral_entropy(self) -> Result<Array1<f32>, SpectralError> {
        spectral_entropy_impl(self.y, self.sr, None, Some(self.n_fft), Some(self.hop_length))
    }

    /// Set the bins per octave for CQT-based chroma (default: 12).
    #[must_use]
    pub fn bins_per_octave(mut self, v: usize) -> Self {
        self.bins_per_octave = v;
        self
    }

    /// Set the frame length for frame-based features (default: 2048).
    #[must_use]
    pub fn frame_length(mut self, v: usize) -> Self {
        self.frame_length = v;
        self
    }

    /// Set the harmonic median-filter window for HPSS (default: 31).
    #[must_use]
    pub fn harm_win(mut self, v: usize) -> Self {
        self.harm_win = v;
        self
    }

    /// Set the percussive median-filter window for HPSS (default: 31).
    #[must_use]
    pub fn perc_win(mut self, v: usize) -> Self {
        self.perc_win = v;
        self
    }

    /// Set the number of sub-bands (default: 6).
    #[must_use]
    pub fn n_bands(mut self, v: usize) -> Self {
        self.n_bands = v;
        self
    }

    /// Set the number of formants (default: 3).
    #[must_use]
    pub fn n_formants(mut self, v: usize) -> Self {
        self.n_formants = v;
        self
    }

    /// Set the polynomial order for `poly_features` (default: 1).
    #[must_use]
    pub fn order(mut self, v: usize) -> Self {
        self.order = v;
        self
    }

    /// Set the roll-off fraction for `spectral_rolloff` (default: 0.85).
    #[must_use]
    pub fn roll_percent(mut self, v: f32) -> Self {
        self.roll_percent = v;
        self
    }

    /// Set the norm exponent for `spectral_bandwidth` (default: 2).
    #[must_use]
    pub fn p(mut self, v: i32) -> Self {
        self.p = v;
        self
    }

    /// Compute CQT-based chroma.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn chroma_cqt(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        let fmin = if self.fmin > 0.0 { Some(self.fmin) } else { None };
        chroma_cqt(&audio, None, Some(self.hop_length), fmin, Some(self.bins_per_octave))
    }

    /// Compute CENS chroma.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn chroma_cens(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        let fmin = if self.fmin > 0.0 { Some(self.fmin) } else { None };
        chroma_cens(
            &audio,
            None,
            Some(self.hop_length),
            fmin,
            Some(self.bins_per_octave),
            Some(self.win_length),
        )
    }

    /// Compute spectral contrast.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn spectral_contrast(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        spectral_contrast(&audio, None, Some(self.n_fft), Some(self.hop_length), Some(self.n_bands))
    }

    /// Compute polynomial coefficients of the spectrum.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn poly_features(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        poly_features(&audio, None, Some(self.n_fft), Some(self.hop_length), Some(self.order))
    }

    /// Compute tonal centroid features (tonnetz).
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn tonnetz(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        tonnetz(&audio, None)
    }

    /// Compute a pitch-class chromagram.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn pitch_chroma(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        pitch_chroma(&audio, None, Some(self.n_fft), Some(self.hop_length))
    }

    /// Compute sub-band spectral centroids.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn spectral_subband_centroids(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        spectral_subband_centroids(
            &audio,
            None,
            Some(self.n_fft),
            Some(self.hop_length),
            Some(self.n_bands),
        )
    }

    /// Compute root-mean-square energy per frame.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn rms(self) -> Result<Array1<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        rms(&audio, None, Some(self.frame_length), Some(self.hop_length))
    }

    /// Separate harmonic and percussive components.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn hpss(self) -> Result<(Array2<f32>, Array2<f32>), SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        hpss(
            &audio,
            None,
            Some(self.n_fft),
            Some(self.hop_length),
            Some(self.harm_win),
            Some(self.perc_win),
        )
    }

    /// Estimate fundamental frequency by autocorrelation.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn pitch_autocorr(self) -> Result<Array1<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        let fmin = if self.fmin > 0.0 { Some(self.fmin) } else { None };
        pitch_autocorr(&audio, Some(self.frame_length), Some(self.hop_length), fmin, None)
    }

    /// Compute voice-activity-detection features.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn vad_features(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        vad_features(&audio, Some(self.frame_length), Some(self.hop_length), Some(self.n_fft))
    }

    /// Estimate formant frequencies.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn formant_frequencies(self) -> Result<Array2<f32>, SpectralError> {
        let audio = AudioData::new(self.y.to_vec(), self.sr, 1)?;
        formant_frequencies(
            &audio,
            Some(self.n_formants),
            Some(self.frame_length),
            Some(self.hop_length),
        )
    }
}

/// Creates a spectral-feature builder over a signal.
///
/// Ergonomic, chainable entry point for the spectral feature family
/// (`mfcc`, `melspectrogram`, `spectral_centroid`, …) instead of positional
/// optional arguments. Terminal methods (`.mfcc()`, `.melspectrogram()`, …)
/// run the chosen analysis.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::spectral;
/// let y = vec![0.0_f32; 4096];
/// let cc = spectral(&y, 44100).n_fft(2048).hop_length(512).mfcc()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn spectral(y: &[f32], sr: u32) -> SpectralBuilder<'_> {
    SpectralBuilder {
        y,
        sr,
        n_fft: 2048,
        hop_length: 512,
        win_length: 2048,
        n_mels: 128,
        fmin: 0.0,
        fmax: sr as f32 / 2.0,
        norm: 1.0,
        bins_per_octave: 12,
        frame_length: 2048,
        harm_win: 31,
        perc_win: 31,
        n_bands: 6,
        n_formants: 3,
        order: 1,
        roll_percent: 0.85,
        p: 2,
    }
}

/// Computes chroma features using STFT.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Sample rate in Hz
///
/// # Returns
/// Returns a builder that can be configured with method chaining.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::*;
/// use dasp_rs::types::*;
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// let chroma = chroma_stft(&y, 44100)
///     .n_fft(2048)
///     .hop_length(512)
///     .compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn chroma_stft(y: &[f32], sr: u32) -> ChromaStftBuilder<'_> {
    ChromaStftBuilder {
        y,
        sr,
        n_fft: 2048,
        hop_length: 512,
        norm: 1.0,
    }
}

/// Custom error types for spectral signal processing operations.
///
/// This enum defines errors specific to spectral feature extraction and analysis.
#[derive(Error, Debug)]
pub enum SpectralError {
    /// Error when a parameter (e.g., frame length, hop length) is invalid.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Error when input dimensions or sizes are insufficient or mismatched.
    #[error("Invalid input size: {0}")]
    InvalidSize(String),

    /// Error during numerical computations (e.g., matrix solving, eigenvalue decomposition).
    #[error("Numerical error: {0}")]
    Numerical(String),

    /// Wraps an `AudioError` from the core module (e.g., from time-domain functions).
    #[error("Audio processing error: {0}")]
    Audio(#[from] AudioError),

    /// A variant for `TimeDomainError`
    #[error("Time-domain processing error: {0}")]
    TimeDomain(String),

    /// Wraps a time-frequency processing error (e.g., from STFT or CQT).
    #[error("Time-frequency error: {0}")]
    TimeFrequency(String),
}

/// Computes chroma features using Short-Time Fourier Transform (STFT).
///
/// Maps spectral energy to 12 pitch classes based on a magnitude spectrogram.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `s` - Optional pre-computed magnitude spectrogram.
/// * `norm` - Optional normalization factor.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
/// * `tuning` - Optional tuning adjustment in semitones (currently unused).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing a 2D array of shape `(12, n_frames)`
/// with chroma features, or an error.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::chroma_stft;
/// let y = vec![0.0; 2048];
/// let chroma = chroma_stft(&y, 44100).n_fft(2048).hop_length(512).compute()?;
/// assert_eq!(chroma.shape(), &[12, 1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub(crate) fn chroma_stft_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    norm: Option<f32>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    
    if n_fft == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft and hop_length must be positive".into(),
        ));
    }
    if y.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".into(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(y)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(|x| x.norm().powi(2)),
    };

    let n_bins = s.shape()[0];
    let freqs = fft_frequencies_impl(sr, n_fft);

    let pitch_classes: Vec<usize> = (1..n_bins)
        .map(|bin| {
            let midi = hz_to_midi(&[freqs[bin]])[0];
            (midi.round() as isize).rem_euclid(12) as usize
        })
        .collect();

    if let Some(norm_val) = norm {
        if norm_val <= 0.0 {
            return Err(SpectralError::InvalidParameter(
                "Normalization factor must be positive".into(),
            ));
        }
    }

    let n_frames = s.shape()[1];

    let chroma_cols: Vec<Array1<f32>> = (0..n_frames)
        .into_par_iter()
        .map(|frame| {
            let mut temp = Array1::zeros(12);
            for bin in 1..n_bins {
                let pitch_class = pitch_classes[bin - 1];
                temp[pitch_class] += s[[bin, frame]];
            }
            if let Some(norm_val) = norm {
                temp /= norm_val;
            }
            temp
        })
        .collect();

    let views: Vec<_> = chroma_cols.iter().map(ndarray::ArrayBase::view).collect();
    let chroma = stack(Axis(1), views.as_slice()).map_err(|e| {
        SpectralError::Numerical(format!("Failed to stack chroma columns: {e}"))
    })?;

    Ok(chroma)
}

/// Computes chroma features using Constant-Q Transform (CQT).
///
/// Maps CQT spectral energy to 12 pitch classes.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `c` - Optional pre-computed CQT spectrogram.
/// * `hop_length` - Optional hop length (defaults to 512).
/// * `fmin` - Optional minimum frequency (defaults to 32.70 Hz, C1).
/// * `bins_per_octave` - Optional bins per octave (defaults to 12).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing a 2D array of shape `(12, n_frames)`
/// with chroma features, or an error.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::chroma_cqt;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let chroma = chroma_cqt(&signal, None, None, None, None).unwrap();
/// assert_eq!(chroma.shape(), &[12, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn chroma_cqt(
    signal: &AudioData,
    c: Option<&Array2<f32>>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    bins_per_octave: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let hop = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let bpo = bins_per_octave.unwrap_or(12);
    
    if hop == 0 {
        return Err(SpectralError::InvalidParameter("hop_length must be positive".into()));
    }
    if fmin <= 0.0 {
        return Err(SpectralError::InvalidParameter("fmin must be positive".into()));
    }
    if bpo == 0 {
        return Err(SpectralError::InvalidParameter("bins_per_octave must be positive".into()));
    }

    let nyquist = signal.sample_rate as f32 / 2.0;
    if fmin >= nyquist {
        return Err(SpectralError::InvalidParameter("fmin must be less than Nyquist frequency".into()));
    }
    let max_bin = (nyquist / fmin).log2() * bpo as f32;
    let n_bins = max_bin.floor() as usize + 1;

    let c: Array2<f32> = if let Some(c_mag) = c { Ok::<Array2<f32>, SpectralError>(c_mag.to_owned()) } else {
        let cqt_result = cqt(&signal.samples, signal.sample_rate)
            .hop_length(hop)
            .fmin(fmin)
            .n_bins(n_bins)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?;
        Ok(cqt_result.mapv(num_complex::Complex::norm))
    }?;

    let mut pitch_classes = Vec::with_capacity(n_bins);
    for bin in 0..n_bins {
        let freq = fmin * 2.0f32.powf(bin as f32 / bpo as f32);
        if !freq.is_finite() || freq <= 0.0 {
            return Err(SpectralError::Numerical("Invalid frequency computed for bin".into()));
        }
        let midi = hz_to_midi(&[freq])[0];
        if !midi.is_finite() {
            return Err(SpectralError::Numerical("Invalid MIDI value computed".into()));
        }
        let pitch_class = (midi.round() as usize) % 12;
        pitch_classes.push(pitch_class);
    }

    let n_frames = c.shape()[1];
    let chroma_cols: Vec<Array1<f32>> = (0..n_frames)
        .into_par_iter()
        .map(|frame| {
            let mut chroma_frame = Array1::zeros(12);
            for bin in 0..n_bins {
                let pc = pitch_classes[bin];
                chroma_frame[pc] += c[[bin, frame]];
            }
            chroma_frame
        })
        .collect();

    let views: Vec<_> = chroma_cols.iter().map(ndarray::ArrayBase::view).collect();
    let chroma = stack(Axis(1), views.as_slice())
        .map_err(|e| SpectralError::Numerical(e.to_string()))?;

    Ok(chroma)
}

/// Computes Chroma Energy Normalized Statistics (CENS) features.
///
/// Normalizes chroma features over a window to emphasize energy distribution.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `C` - Optional pre-computed CQT spectrogram.
/// * `hop_length` - Optional hop length (defaults to 512).
/// * `fmin` - Optional minimum frequency (defaults to 32.70 Hz).
/// * `bins_per_octave` - Optional bins per octave (defaults to 12).
/// * `win_length` - Optional window length for normalization (defaults to 41).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing a 2D array of shape `(12, n_frames)`
/// with CENS features, or an error.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::chroma_cens;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let cens = chroma_cens(&signal, None, None, None, None, None).unwrap();
/// assert_eq!(cens.shape(), &[12, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn chroma_cens(
    signal: &AudioData,
    c: Option<&Array2<f32>>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    bins_per_octave: Option<usize>,
    win_length: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let win = win_length.unwrap_or(41);
    if win == 0 {
        return Err(SpectralError::InvalidParameter(
            "win_length must be positive".to_string(),
        ));
    }
    let chroma = chroma_cqt(signal, c, hop_length, fmin, bins_per_octave)?;
    let half_win = win / 2;
    let mut cens = Array2::zeros(chroma.dim());
    for t in 0..chroma.shape()[1] {
        let start = t.saturating_sub(half_win);
        let end = (t + half_win + 1).min(chroma.shape()[1]);
        let slice = chroma.slice(s![.., start..end]);
        let norm = slice
            .mapv(|x| x.powi(2))
            .sum_axis(Axis(1))
            .mapv(f32::sqrt);
        for p in 0..12 {
            cens[[p, t]] = if norm[p] > 1e-6 {
                chroma[[p, t]] / norm[p]
            } else {
                0.0
            };
        }
    }
    Ok(cens)
}

/// Computes a mel spectrogram.
///
/// Projects spectral energy onto mel-frequency bands.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed magnitude spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
/// * `n_mels` - Optional number of mel bands (defaults to 128).
/// * `fmin` - Optional minimum frequency (defaults to 0 Hz).
/// * `fmax` - Optional maximum frequency (defaults to sr/2).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing a 2D array of shape `(n_mels, n_frames)`
/// with mel spectrogram, or an error.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::melspectrogram;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let mel = melspectrogram(&signal, None, None, None, None, None, None).unwrap();
/// assert_eq!(mel.shape(), &[128, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn melspectrogram(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_mels: Option<usize>,
    fmin: Option<f32>,
    fmax: Option<f32>,
) -> Result<Array2<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let n_mels = n_mels.unwrap_or(128);
    let fmin = fmin.unwrap_or(0.0);
    let fmax = fmax.unwrap_or(signal.sample_rate as f32 / 2.0);
    if n_fft == 0 || hop == 0 || n_mels == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft, hop_length, and n_mels must be positive".to_string(),
        ));
    }
    if fmin < 0.0 || fmax <= fmin || fmax > signal.sample_rate as f32 / 2.0 {
        return Err(SpectralError::InvalidParameter(
            "fmin and fmax must satisfy 0 <= fmin < fmax <= sr/2".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(|x| x.norm().powi(2)),
    };

        let mel_f = mel_frequencies_impl(n_mels, fmin, fmax);
    let mut mel_s = Array2::zeros((n_mels, s.shape()[1]));
    let fft_f = fft_frequencies_impl(signal.sample_rate, n_fft);
    for m in 0..n_mels {
        let f_low = if m == 0 { fmin } else { mel_f[m - 1] };
        let f_center = mel_f[m];
        let f_high = mel_f.get(m + 1).copied().unwrap_or(fmax);
        for (bin, &f) in fft_f.iter().enumerate() {
            let weight = if f >= f_low && f <= f_high {
                if f <= f_center {
                    (f - f_low) / (f_center - f_low)
                } else {
                    (f_high - f) / (f_high - f_center)
                }
            } else {
                0.0
            };
            for t in 0..s.shape()[1] {
                mel_s[[m, t]] += s[[bin, t]] * weight.max(0.0);
            }
        }
    }
    Ok(mel_s)
}

/// Internal wrapper for melspectrogram that takes &[f32] and sample rate.
fn melspectrogram_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_mels: Option<usize>,
    fmax: Option<f32>,
) -> Result<Array2<f32>, SpectralError> {
    let signal = AudioData::new(y.to_vec(), sr, 1)
        .map_err(SpectralError::Audio)?;
    melspectrogram(&signal, s, n_fft, hop_length, n_mels, None, fmax)
}

/// Computes Mel-frequency cepstral coefficients (MFCCs).
///
/// Extracts MFCCs from a mel spectrogram using DCT.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_mfcc` - Optional number of MFCCs (defaults to 20).
/// * `dct_type` - Optional DCT type (defaults to 2; only 2 is supported).
/// * `norm` - Optional normalization type ("ortho" or None).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing a 2D array of shape `(n_mfcc, n_frames)`
/// with MFCCs, or an error.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::mfcc;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let mfcc = mfcc(&signal, None, None, None, None).unwrap();
/// assert_eq!(mfcc.shape(), &[20, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn mfcc(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_mfcc: Option<usize>,
    dct_type: Option<i32>,
    norm: Option<&str>,
) -> Result<Array2<f32>, SpectralError> {
    let n_mfcc = n_mfcc.unwrap_or(20);
    let dct_type = dct_type.unwrap_or(2);
    if n_mfcc == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_mfcc must be positive".to_string(),
        ));
    }
    if dct_type != 2 {
        return Err(SpectralError::InvalidParameter(
            "Only DCT type 2 is supported".to_string(),
        ));
    }
    if let Some(n) = norm {
        if n != "ortho" {
            return Err(SpectralError::InvalidParameter(
                "norm must be 'ortho' or None".to_string(),
            ));
        }
    }

    let s = if let Some(s) = s { s.to_owned() } else {
        let temp_signal = AudioData::new(signal.samples.clone(), signal.sample_rate, signal.channels)
            .map_err(SpectralError::Audio)?;
        melspectrogram(&temp_signal, None, None, None, None, None, None)?
    };
    let log_s = s.mapv(|x| x.max(1e-10).ln());
    let n_mels = s.shape()[0] as f32;
    let pi_over_n_mels = std::f32::consts::PI / n_mels;
    // Precompute scale factors (constant for each k)
    let scale_k0 = f32::sqrt(1.0 / n_mels);  // sqrt(2/N) * 1/sqrt(2) = sqrt(1/N)
    let scale_k_other = f32::sqrt(2.0 / n_mels);  // sqrt(2/N) * 1
    let mut mfcc = Array2::zeros((n_mfcc, s.shape()[1]));
    for t in 0..s.shape()[1] {
        for k in 0..n_mfcc {
            let mut sum = 0.0;
            let k_pi = k as f32 * pi_over_n_mels;
            for n in 0..s.shape()[0] {
                sum += log_s[[n, t]] * (k_pi * (n as f32 + 0.5)).cos();
            }
            // DCT Type II: X[k] = sqrt(2/N) * c[k] * sum, where c[0] = 1/sqrt(2), c[k>0] = 1
            let scale = if k == 0 { scale_k0 } else { scale_k_other };
            mfcc[[k, t]] = sum * scale;
        }
    }
    if norm == Some("ortho") {
        // Orthonormal DCT already has correct normalization, no additional scaling needed
        // (The scale factor above already includes the orthonormal normalization)
    }
    Ok(mfcc)
}

/// Internal wrapper for mfcc that takes &[f32] and sample rate.
fn mfcc_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    _norm: Option<f32>,
    _n_fft: Option<usize>,
    _hop_length: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let signal = AudioData::new(y.to_vec(), sr, 1)
        .map_err(SpectralError::Audio)?;
    mfcc(&signal, s, None, None, None)
}

/// Computes root mean square (RMS) energy.
///
/// Calculates RMS energy per frame from either the signal or a spectrogram.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `frame_length` - Optional frame length (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `frame_length/4`).
///
/// # Returns
/// Returns `Result<Array1<f32>, SpectralError>` containing RMS values per frame, or an error.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::rms;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let rms = rms(&signal, None, None, None).unwrap();
/// assert_eq!(rms.len(), 1);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn rms(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    if frame_len == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "frame_length and hop_length must be positive".to_string(),
        ));
    }

    if let Some(s) = s { Ok(s.map_axis(Axis(0), |row| {
        f32::sqrt(row.iter().map(|x| x.powi(2)).sum::<f32>() / row.len() as f32)
    })) } else {
        if signal.samples.len() < frame_len {
            return Err(SpectralError::InvalidSize(
                "Signal length must be at least frame_length".to_string(),
            ));
        }
        let n_frames = (signal.samples.len() - frame_len) / hop + 1;
        let mut rms = Array1::zeros(n_frames);
        for i in 0..n_frames {
            let start = i * hop;
            let slice = &signal.samples[start..(start + frame_len).min(signal.samples.len())];
            rms[i] = f32::sqrt(slice.iter().map(|x| x.powi(2)).sum::<f32>() / slice.len() as f32);
        }
        Ok(rms)
    }
}

/// Computes spectral centroid frequencies.
///
/// Represents the "center of mass" of the spectrum per frame.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
///
/// # Returns
/// Returns `Result<Array1<f32>, SpectralError>` containing centroid frequencies per frame, or an error.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::spectral_centroid;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let centroid = spectral_centroid(&signal, None, None, None).unwrap();
/// assert_eq!(centroid.len(), 1);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn spectral_centroid(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    if n_fft == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft and hop_length must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };

    let freqs = Array1::from_vec(fft_frequencies_impl(signal.sample_rate, n_fft));
    Ok(s.axis_iter(Axis(1))
        .map(|frame| {
            let total = frame.sum();
            if total > 1e-6 {
                frame.dot(&freqs) / total
            } else {
                0.0
            }
        })
        .collect())
}

/// Internal wrapper for `spectral_centroid` that takes &[f32] and sample rate.
fn spectral_centroid_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let signal = AudioData::new(y.to_vec(), sr, 1)
        .map_err(SpectralError::Audio)?;
    spectral_centroid(&signal, s, n_fft, hop_length)
}

/// Computes spectral bandwidth.
///
/// Measures the spread of the spectrum around the centroid per frame.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
/// * `p` - Optional power for bandwidth calculation (defaults to 2).
///
/// # Returns
/// Returns `Result<Array1<f32>, SpectralError>` containing bandwidth values per frame, or an error.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::spectral_bandwidth;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let bandwidth = spectral_bandwidth(&signal, None, None, None, None).unwrap();
/// assert_eq!(bandwidth.len(), 1);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn spectral_bandwidth(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    p: Option<i32>,
) -> Result<Array1<f32>, SpectralError> {
    let p = p.unwrap_or(2);
    if p <= 0 {
        return Err(SpectralError::InvalidParameter(
            "p must be positive".to_string(),
        ));
    }
        let temp_signal = AudioData::new(signal.samples.clone(), signal.sample_rate, signal.channels)
            .map_err(SpectralError::Audio)?;
        let centroid = spectral_centroid(&temp_signal, None, None, None)?;
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };
    let freqs = fft_frequencies_impl(signal.sample_rate, n_fft);
    Ok(s.axis_iter(Axis(1))
        .zip(centroid.iter())
        .map(|(frame, &c)| {
            let total = frame.sum();
            if total > 1e-6 {
                let dev = frame
                    .iter()
                    .zip(freqs.iter())
                    .map(|(&s, &f)| s * (f - c).powi(p))
                    .fold(0.0, |acc, x| acc + x)
                    / total;
                dev.powf(1.0 / p as f32)
            } else {
                0.0
            }
        })
        .collect())
}

/// Computes spectral contrast across frequency bands.
///
/// Calculates the difference between peaks and valleys in subbands.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
/// * `n_bands` - Optional number of frequency bands (defaults to 6).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing contrast values of shape `(n_bands + 1, n_frames)`.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::spectral_contrast;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let contrast = spectral_contrast(&signal, None, None, None, None).unwrap();
/// assert_eq!(contrast.shape(), &[7, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn spectral_contrast(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_bands: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let n_bands = n_bands.unwrap_or(6);
    if n_fft == 0 || hop == 0 || n_bands == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft, hop_length, and n_bands must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };

    let freqs = fft_frequencies_impl(signal.sample_rate, n_fft);
    let band_edges = Array1::logspace(
        2.0,
        0.0,
        f32::log2(signal.sample_rate as f32 / 2.0),
        n_bands + 1,
    );
    let mut contrast = Array2::zeros((n_bands + 1, s.shape()[1]));
    for t in 0..s.shape()[1] {
        for b in 0..=n_bands {
            let f_low = if b == 0 { 0.0 } else { band_edges[b - 1] };
            let f_high = band_edges[b];
            let slice = s.slice(s![.., t]);
            let band: Vec<f32> = slice
                .iter()
                .zip(freqs.iter())
                .filter(|&(_, &f)| f >= f_low && f <= f_high)
                .map(|(&s, _)| s)
                .collect();
            if !band.is_empty() {
                let mut sorted = band;
                sorted.sort_by(f32::total_cmp);
                let peak = sorted[sorted.len() - 1];
                let valley = sorted[0];
                contrast[[b, t]] = peak - valley;
            }
        }
    }
    Ok(contrast)
}

/// Computes spectral flatness.
///
/// Measures the uniformity of the spectrum per frame (geometric mean / arithmetic mean).
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
///
/// # Returns
/// Returns `Result<Array1<f32>, SpectralError>` containing flatness values per frame.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::spectral_flatness;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let flatness = spectral_flatness(&signal, None, None, None).unwrap();
/// assert_eq!(flatness.len(), 1);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn spectral_flatness(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    if n_fft == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft and hop_length must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(|x| x.norm().max(1e-10)),
    };

    Ok(s.axis_iter(Axis(1))
        .map(|frame| {
            let log_frame = frame.mapv(f32::ln);
            let geo_mean = log_frame.sum() / frame.len() as f32;
            let arith_mean = frame.sum() / frame.len() as f32;
            f32::exp(geo_mean) / arith_mean
        })
        .collect())
}

/// Internal wrapper for `spectral_bandwidth` that takes &[f32] and sample rate.
fn spectral_bandwidth_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    p: Option<i32>,
) -> Result<Array1<f32>, SpectralError> {
    let signal = AudioData::new(y.to_vec(), sr, 1)
        .map_err(SpectralError::Audio)?;
    spectral_bandwidth(&signal, s, n_fft, hop_length, p)
}

/// Computes spectral roll-off frequency.
///
/// Finds the frequency below which a specified percentage of total spectral energy lies.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
/// * `roll_percent` - Optional roll-off percentage (defaults to 0.85).
///
/// # Returns
/// Returns `Result<Array1<f32>, SpectralError>` containing roll-off frequencies per frame.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::spectral_rolloff;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let rolloff = spectral_rolloff(&signal, None, None, None, None).unwrap();
/// assert_eq!(rolloff.len(), 1);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn spectral_rolloff(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    roll_percent: Option<f32>,
) -> Result<Array1<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let roll_percent = roll_percent.unwrap_or(0.85);
    if n_fft == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft and hop_length must be positive".to_string(),
        ));
    }
    if roll_percent <= 0.0 || roll_percent > 1.0 {
        return Err(SpectralError::InvalidParameter(
            "roll_percent must be between 0 and 1".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };

    let freqs = fft_frequencies_impl(signal.sample_rate, n_fft);
    Ok(s.axis_iter(Axis(1))
        .map(|frame| {
            let total_energy = frame.sum();
            let target_energy = total_energy * roll_percent;
            let mut cum_energy = 0.0;
            for (f, &s) in freqs.iter().zip(frame.iter()) {
                cum_energy += s;
                if cum_energy >= target_energy {
                    return *f;
                }
            }
            freqs[freqs.len() - 1]
        })
        .collect())
}

/// Internal wrapper for `spectral_rolloff` that takes &[f32] and sample rate.
fn spectral_rolloff_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    roll_percent: Option<f32>,
) -> Result<Array1<f32>, SpectralError> {
    let signal = AudioData::new(y.to_vec(), sr, 1)
        .map_err(SpectralError::Audio)?;
    spectral_rolloff(&signal, s, n_fft, hop_length, roll_percent)
}

/// Internal wrapper for `spectral_flatness` that takes &[f32] and sample rate.
fn spectral_flatness_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let signal = AudioData::new(y.to_vec(), sr, 1)
        .map_err(SpectralError::Audio)?;
    spectral_flatness(&signal, s, n_fft, hop_length)
}

/// Computes polynomial fit coefficients for spectral features.
///
/// Fits a polynomial to each frame's spectral magnitude.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
/// * `order` - Optional polynomial order (defaults to 1).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing coefficients of shape `(order + 1, n_frames)`.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::poly_features;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let coeffs = poly_features(&signal, None, None, None, None).unwrap();
/// assert_eq!(coeffs.shape(), &[2, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn poly_features(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    order: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let order = order.unwrap_or(1);
    if n_fft == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft and hop_length must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };

    let mut coeffs = Array2::zeros((order + 1, s.shape()[1]));
    let x = Array1::linspace(0.0, s.shape()[0] as f32 - 1.0, s.shape()[0]);
    for t in 0..s.shape()[1] {
        let y_t = s.slice(s![.., t]).to_owned();
        let poly = polyfit(&x, &y_t, order);
        for (i, &c) in poly.iter().enumerate() {
            coeffs[[i, t]] = c;
        }
    }
    Ok(coeffs)
}

/// Computes Tonnetz features from chroma.
///
/// Projects chroma features onto a 6-dimensional tonal space.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `chroma` - Optional pre-computed chroma features.
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing Tonnetz features of shape `(6, n_frames)`.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::tonnetz;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let tonnetz = tonnetz(&signal, None).unwrap();
/// assert_eq!(tonnetz.shape(), &[6, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn tonnetz(
    signal: &AudioData,
    chroma: Option<&Array2<f32>>,
) -> Result<Array2<f32>, SpectralError> {
        let chroma_stft_result = chroma_stft(&signal.samples, signal.sample_rate)
            .compute()?;
    let chroma = chroma.unwrap_or(&chroma_stft_result);
    if chroma.shape()[0] != 12 {
        return Err(SpectralError::InvalidSize(
            "Chroma must have 12 pitch classes".to_string(),
        ));
    }
    let transform = ndarray::arr2(&[
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Fifths
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Minor thirds
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Major thirds
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Minor sevenths
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Major seconds
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Tritones
    ]);
    Ok(transform.dot(chroma))
}

/// Fits a polynomial to data points.
///
/// Helper function for polynomial feature extraction.
///
/// # Arguments
/// * `x` - X-coordinates.
/// * `y` - Y-coordinates.
/// * `order` - Polynomial order.
///
/// # Returns
/// Returns a vector of polynomial coefficients, or zeros if solving fails.
fn polyfit(x: &Array1<f32>, y: &Array1<f32>, order: usize) -> Vec<f32> {
    let n = order + 1;
    let rows = x.len();
    // Vandermonde matrix (rows x n), column j = x^j.
    let a = DMatrix::from_fn(rows, n, |i, j| x[i].powi(j as i32));
    let b = DVector::from_iterator(rows, y.iter().copied());
    // Least-squares solve via SVD (handles over-/under-determined systems).
    a.svd(true, true).solve(&b, 1e-9_f32).map_or_else(
        |_| vec![0.0; n],
        |coeffs| coeffs.iter().copied().collect(),
    )
}

/// Computes spectral flux.
///
/// Measures the change in spectral magnitude between consecutive frames.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
///
/// # Returns
/// Returns `Result<Array1<f32>, SpectralError>` containing flux values per frame.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::spectral_flux;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let flux = spectral_flux(&signal, None, None, None).unwrap();
/// assert_eq!(flux.len(), 1);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn spectral_flux(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    if n_fft == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft and hop_length must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };

    let mut flux = Array1::zeros(s.shape()[1]);
    for t in 1..s.shape()[1] {
        let diff = &s.slice(s![.., t]) - &s.slice(s![.., t - 1]);
        flux[t] = diff.mapv(|x| x.powi(2)).sum().sqrt();
    }
    Ok(flux)
}

/// Internal wrapper for `spectral_flux` that takes &[f32] and sample rate.
fn spectral_flux_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let signal = AudioData::new(y.to_vec(), sr, 1)
        .map_err(SpectralError::Audio)?;
    spectral_flux(&signal, s, n_fft, hop_length)
}

/// Computes spectral entropy.
///
/// Calculates the entropy of the normalized spectrum per frame.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
///
/// # Returns
/// Returns `Result<Array1<f32>, SpectralError>` containing entropy values per frame.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::spectral_entropy;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let entropy = spectral_entropy(&signal, None, None, None).unwrap();
/// assert_eq!(entropy.len(), 1);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn spectral_entropy(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    if n_fft == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft and hop_length must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };

    Ok(s.axis_iter(Axis(1))
        .map(|frame| {
            let sum = frame.sum();
            if sum <= 1e-10 {
                0.0
            } else {
                let p = frame.mapv(|x| x / sum);
                -p.mapv(|x| if x > 1e-10 { x * x.ln() } else { 0.0 }).sum()
            }
        })
        .collect())
}

/// Internal wrapper for `spectral_entropy` that takes &[f32] and sample rate.
fn spectral_entropy_impl(
    y: &[f32],
    sr: u32,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, SpectralError> {
    let signal = AudioData::new(y.to_vec(), sr, 1)
        .map_err(SpectralError::Audio)?;
    spectral_entropy(&signal, s, n_fft, hop_length)
}

/// Computes pitch chroma features.
///
/// Normalizes spectral energy across 12 pitch classes per frame.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing normalized pitch chroma features of shape `(12, n_frames)`.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::pitch_chroma;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let chroma = pitch_chroma(&signal, None, None, None).unwrap();
/// assert_eq!(chroma.shape(), &[12, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn pitch_chroma(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    if n_fft == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft and hop_length must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };

    let freqs = fft_frequencies_impl(signal.sample_rate, n_fft);
    let mut chroma = Array2::zeros((12, s.shape()[1]));
    for t in 0..s.shape()[1] {
        let frame = s.column(t);
        for (bin, &f) in freqs.iter().enumerate() {
            if frame[bin] > 0.0 {
                let midi = hz_to_midi(&[f])[0];
                let pitch_class = midi.round() as usize % 12;
                chroma[[pitch_class, t]] += frame[bin];
            }
        }
    }
    for t in 0..chroma.shape()[1] {
        let sum = chroma.column(t).sum();
        if sum > 1e-6 {
            chroma.column_mut(t).mapv_inplace(|x| x / sum);
        }
    }
    Ok(chroma)
}

/// Applies cepstral mean and variance normalization (CMVN).
///
/// Normalizes features by subtracting the mean and optionally dividing by the standard deviation.
///
/// # Arguments
/// * `features` - Input feature matrix.
/// * `axis` - Optional axis for normalization (-1 for time, 0 for features; defaults to -1).
/// * `variance` - Optional flag to normalize variance (defaults to true).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing the normalized feature matrix.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::cmvn;
/// use ndarray::Array2;
/// let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let normalized = cmvn(&features).compute().unwrap();
/// assert_eq!(normalized.shape(), &[2, 3]);
/// ```
pub fn cmvn(features: &Array2<f32>) -> CmvnBuilder<'_> {
    CmvnBuilder { features, axis: -1, variance: true }
}

/// Builder for [`cmvn`] (cepstral mean/variance normalization).
#[derive(Debug, Clone)]
pub struct CmvnBuilder<'a> {
    features: &'a Array2<f32>,
    axis: isize,
    variance: bool,
}

impl CmvnBuilder<'_> {
    /// Set the normalization axis (-1 for frames, 0 for features; default: -1).
    #[must_use]
    pub fn axis(mut self, axis: isize) -> Self {
        self.axis = axis;
        self
    }

    /// Enable variance normalization in addition to mean (default: true).
    #[must_use]
    pub fn variance(mut self, variance: bool) -> Self {
        self.variance = variance;
        self
    }

    /// Apply normalization.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<f32>, SpectralError> {
        cmvn_impl(self.features, Some(self.axis), Some(self.variance))
    }
}

fn cmvn_impl(
    features: &Array2<f32>,
    axis: Option<isize>,
    variance: Option<bool>,
) -> Result<Array2<f32>, SpectralError> {
    let axis = axis.unwrap_or(-1);
    let do_variance = variance.unwrap_or(true);
    let ax = usize::from(axis < 0);

    if features.shape()[ax] < 2 {
        return Err(SpectralError::InvalidSize(
            "Feature dimension too small for normalization".to_string(),
        ));
    }

    let mut normalized = features.to_owned();
    let means = normalized
        .mean_axis(Axis(ax))
        .ok_or_else(|| SpectralError::Numerical("Failed to compute mean".to_string()))?;
    for i in 0..normalized.shape()[1 - ax] {
        for j in 0..normalized.shape()[ax] {
            let idx = if ax == 1 { [j, i] } else { [i, j] };
            normalized[idx] -= means[i];
        }
    }

    if do_variance {
        let variances = normalized
            .mapv(|x| x.powi(2))
            .mean_axis(Axis(ax))
            .ok_or_else(|| SpectralError::Numerical("Failed to compute variance".to_string()))?;
        let std_devs = variances.mapv(|x| (x + 1e-10).sqrt());
        for i in 0..normalized.shape()[1 - ax] {
            for j in 0..normalized.shape()[ax] {
                let idx = if ax == 1 { [j, i] } else { [i, j] };
                normalized[idx] /= std_devs[if ax == 1 { j } else { i }];
            }
        }
    }

    Ok(normalized)
}

/// Performs Harmonic-Percussive Source Separation (HPSS).
///
/// Separates the spectrogram into harmonic and percussive components using median filtering.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
/// * `harm_win` - Optional window size for harmonic component (defaults to 31).
/// * `perc_win` - Optional window size for percussive component (defaults to 31).
///
/// # Returns
/// Returns a tuple `(harmonic, percussive)` containing two `Array2<f32>` with separated components.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::hpss;
/// let signal = AudioData { samples: vec![0.0; 2048], sample_rate: 44100, channels: 1 };
/// let (harmonic, percussive) = hpss(&signal, None, None, None, None, None)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn hpss(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    harm_win: Option<usize>,
    perc_win: Option<usize>,
) -> Result<(Array2<f32>, Array2<f32>), SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let harm_win: usize = harm_win.unwrap_or(31);
    let perc_win: usize = perc_win.unwrap_or(31);
    if n_fft == 0 || hop == 0 || harm_win == 0 || perc_win == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft, hop_length, harm_win, and perc_win must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(|x| x.norm().powi(2)),
    };

    let mut harmonic = Array2::zeros(s.dim());
    for f in 0..s.shape()[0] {
        let row = s.index_axis(Axis(0), f);
        for t in 0..s.shape()[1] {
            let start = t.saturating_sub(harm_win / 2);
            let end = (t + harm_win / 2 + 1).min(s.shape()[1]);
            let mut slice: Vec<f32> = row.slice(s![start..end]).to_vec();
            slice.sort_by(f32::total_cmp);
            harmonic[[f, t]] = slice[slice.len() / 2];
        }
    }

    let mut percussive = Array2::zeros(s.dim());
    for t in 0..s.shape()[1] {
        let col = s.index_axis(Axis(1), t);
        for f in 0..s.shape()[0] {
            let start = f.saturating_sub(perc_win / 2);
            let end = (f + perc_win / 2 + 1).min(s.shape()[0]);
            let mut slice: Vec<f32> = col.slice(s![start..end]).to_vec();
            slice.sort_by(f32::total_cmp);
            percussive[[f, t]] = slice[slice.len() / 2];
        }
    }

    // Compute total once and reuse for both masks
    let total = &harmonic + &percussive;
    let total_safe = total.mapv(|x| if x > 0.0 { x } else { 1.0 });
    let harm_mask = &harmonic / &total_safe;
    let perc_mask = &percussive / &total_safe;
    Ok((s.to_owned() * &harm_mask, s.to_owned() * &perc_mask))
}

/// Estimates pitch using autocorrelation.
///
/// Detects pitch by finding peaks in the autocorrelation function.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `frame_length` - Optional frame length (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `frame_length/4`).
/// * `fmin` - Optional minimum frequency (defaults to 50 Hz).
/// * `fmax` - Optional maximum frequency (defaults to 500 Hz).
///
/// # Returns
/// Returns `Result<Array1<f32>, SpectralError>` containing pitch estimates in Hz per frame.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::pitch_autocorr;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let pitch = pitch_autocorr(&signal, None, None, None, None).unwrap();
/// assert_eq!(pitch.len(), 1);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn pitch_autocorr(
    signal: &AudioData,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    fmax: Option<f32>,
) -> Result<Array1<f32>, SpectralError> {
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    let fmin = fmin.unwrap_or(50.0);
    let fmax = fmax.unwrap_or(500.0);
    if frame_len == 0 || hop == 0 {
        return Err(SpectralError::InvalidParameter(
            "frame_length and hop_length must be positive".to_string(),
        ));
    }
    if fmin <= 0.0 || fmax <= fmin || fmax > signal.sample_rate as f32 / 2.0 {
        return Err(SpectralError::InvalidParameter(
            "fmin and fmax must satisfy 0 < fmin < fmax <= sr/2".to_string(),
        ));
    }
    if signal.samples.len() < frame_len {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least frame_length".to_string(),
        ));
    }

    let n_frames = (signal.samples.len() - frame_len) / hop + 1;
    let mut pitch = Array1::zeros(n_frames);

    for i in 0..n_frames {
        let start = i * hop;
        let frame = &signal.samples[start..(start + frame_len).min(signal.samples.len())];
        let frame_audio = AudioData {
            samples: frame.to_vec(),
            sample_rate: signal.sample_rate,
            channels: signal.channels,
        };
        let autocorr = autocorrelate(&frame_audio, Some(frame_len))
            .map_err(|e| SpectralError::TimeDomain(e.to_string()))?;
        let lag_min = (signal.sample_rate as f32 / fmax).round() as usize;
        let lag_max = (signal.sample_rate as f32 / fmin).round() as usize;
        let slice = &autocorr[lag_min..lag_max.min(autocorr.len())];
        let max_idx = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map_or(0, |(i, _)| i)
            + lag_min;
        pitch[i] = if max_idx > 0 {
            signal.sample_rate as f32 / max_idx as f32
        } else {
            0.0
        };
    }

    Ok(pitch)
}

/// Computes features for voice activity detection (VAD).
///
/// Extracts log energy, zero-crossing rate, and spectral flatness.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `frame_length` - Optional frame length (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `frame_length/4`).
/// * `n_fft` - Optional FFT window size (defaults to 2048).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing features of shape `(3, n_frames)`.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::vad_features;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let vad = vad_features(&signal, None, None, None).unwrap();
/// assert_eq!(vad.shape(), &[3, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn vad_features(
    signal: &AudioData,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    let n_fft = n_fft.unwrap_or(2048);
    if frame_len == 0 || hop == 0 || n_fft == 0 {
        return Err(SpectralError::InvalidParameter(
            "frame_length, hop_length, and n_fft must be positive".to_string(),
        ));
    }
    if signal.samples.len() < frame_len || signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least max(frame_length, n_fft)".to_string(),
        ));
    }

    let n_frames = (signal.samples.len() - frame_len) / hop + 1;
    let energy = log_energy(signal)
        .frame_length(frame_len)
        .hop_length(hop)
        .compute()
        .map_err(|e| SpectralError::TimeDomain(e.to_string()))?;
        let zcr = crate::features::zero_crossing_rate(&signal.samples)
            .frame_length(frame_len)
            .hop_length(hop)
            .compute();
    let s = stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
        .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
        .mapv(num_complex::Complex::norm);
    let flatness = s.axis_iter(Axis(1))
        .map(|frame| {
            // STFT frames always hold n_fft/2 + 1 bins, so `mean` is only None
            // for an impossible zero-bin frame; fall back to silence in that case.
            let geo_mean = frame.mapv(|x| x.max(1e-10).ln()).mean().map_or(0.0, f32::exp);
            let arith_mean = frame.mean().unwrap_or(0.0);
            if arith_mean > 1e-10 {
                geo_mean / arith_mean
            } else {
                0.0
            }
        })
        .collect::<Array1<f32>>();

    let mut features = Array2::zeros((3, n_frames));
    for i in 0..n_frames {
        features[[0, i]] = energy[i];
        features[[1, i]] = zcr[i];
        features[[2, i]] = flatness[i];
    }
    Ok(features)
}

/// Computes spectral subband centroids.
///
/// Calculates the centroid frequency for each subband per frame.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `S` - Optional pre-computed spectrogram.
/// * `n_fft` - Optional FFT window size (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `n_fft/4`).
/// * `n_bands` - Optional number of subbands (defaults to 4).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing subband centroids of shape `(n_bands, n_frames)`.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::spectral_subband_centroids;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let centroids = spectral_subband_centroids(&signal, None, None, None, None).unwrap();
/// assert_eq!(centroids.shape(), &[4, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn spectral_subband_centroids(
    signal: &AudioData,
    s: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_bands: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let n_bands = n_bands.unwrap_or(4);
    if n_fft == 0 || hop == 0 || n_bands == 0 {
        return Err(SpectralError::InvalidParameter(
            "n_fft, hop_length, and n_bands must be positive".to_string(),
        ));
    }
    if signal.samples.len() < n_fft {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least n_fft".to_string(),
        ));
    }

    let s = match s {
        Some(s) => s.to_owned(),
        None => stft(&signal.samples)
            .n_fft(n_fft)
            .hop_length(hop)
            .compute()
            .map_err(|e| SpectralError::TimeFrequency(e.to_string()))?
            .mapv(num_complex::Complex::norm),
    };

    let freqs = fft_frequencies_impl(signal.sample_rate, n_fft);
    let band_edges = Array1::linspace(0.0, signal.sample_rate as f32 / 2.0, n_bands + 1);
    let mut centroids = Array2::zeros((n_bands, s.shape()[1]));
    for t in 0..s.shape()[1] {
        for b in 0..n_bands {
            let f_low = band_edges[b];
            let f_high = band_edges[b + 1];
            let subband: Vec<(f32, f32)> = freqs
                .iter()
                .zip(s.column(t))
                .filter(|(f, _)| **f >= f_low && **f < f_high)
                .map(|(f, s)| (*f, *s))
                .collect();
            if subband.is_empty() {
                centroids[[b, t]] = f32::midpoint(f_low, f_high);
            } else {
                let total_energy = subband.iter().map(|(_, s)| s).sum::<f32>();
                centroids[[b, t]] = if total_energy > 1e-10 {
                    subband.iter().map(|(f, s)| f * s).sum::<f32>() / total_energy
                } else {
                    f32::midpoint(f_low, f_high)
                };
            }
        }
    }
    Ok(centroids)
}

/// Estimates formant frequencies using LPC.
///
/// Extracts resonant frequencies from the vocal tract model.
///
/// # Arguments
/// * `signal` - The input audio signal.
/// * `n_formants` - Optional number of formants to extract (defaults to 3).
/// * `frame_length` - Optional frame length (defaults to 2048).
/// * `hop_length` - Optional hop length (defaults to `frame_length/4`).
///
/// # Returns
/// Returns `Result<Array2<f32>, SpectralError>` containing formant frequencies of shape `(n_formants, n_frames)`.
///
/// # Examples
/// ```ignore
/// use dasp_rs::types::AudioData;
/// use dasp_rs::feat::formant_frequencies;
/// let signal = AudioData { samples: vec![0.1, 0.2, 0.3, 0.4], sample_rate: 44100, channels: 1 };
/// let formants = formant_frequencies(&signal, None, None, None).unwrap();
/// assert_eq!(formants.shape(), &[3, 1]);
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn formant_frequencies(
    signal: &AudioData,
    n_formants: Option<usize>,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array2<f32>, SpectralError> {
    let n_formants = n_formants.unwrap_or(3);
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    let order = (2.0 * signal.sample_rate as f32 / 1000.0).round() as usize + 2;
    if frame_len == 0 || hop == 0 || n_formants == 0 {
        return Err(SpectralError::InvalidParameter(
            "frame_length, hop_length, and n_formants must be positive".to_string(),
        ));
    }
    if signal.samples.len() < frame_len {
        return Err(SpectralError::InvalidSize(
            "Signal length must be at least frame_length".to_string(),
        ));
    }

    let n_frames = (signal.samples.len() - frame_len) / hop + 1;
    let mut formants = Array2::zeros((n_formants, n_frames));

    for i in 0..n_frames {
        let start = i * hop;
        let frame_slice = &signal.samples[start..(start + frame_len).min(signal.samples.len())];
        let frame = AudioData {
            samples: frame_slice.to_vec(),
            sample_rate: signal.sample_rate,
            channels: signal.channels,
        };
        let lpc_coeffs = lpc(&frame, order)?;
        let roots = polynomial_roots(&lpc_coeffs)?;
        let mut freqs: Vec<f32> = roots
            .iter()
            .filter_map(|r| {
                if r.im.abs() > 1e-6 {
                    let freq = r.arg().abs() * signal.sample_rate as f32 / (2.0 * std::f32::consts::PI);
                    if freq > 50.0 && freq < signal.sample_rate as f32 / 2.0 {
                        Some(freq)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        freqs.sort_by(f32::total_cmp);
        for (j, &f) in freqs.iter().take(n_formants).enumerate() {
            formants[[j, i]] = f;
        }
    }
    Ok(formants)
}

/// Computes Linear Predictive Coding (LPC) coefficients.
///
/// Helper function for formant estimation.
///
/// # Arguments
/// * `frame` - Audio frame as `AudioData`.
/// * `order` - LPC order.
///
/// # Returns
/// Returns `Result<Vec<f32>, SpectralError>` containing LPC coefficients.
fn lpc(frame: &AudioData, order: usize) -> Result<Vec<f32>, SpectralError> {
    if frame.samples.len() < order {
        return Err(SpectralError::InvalidSize(
            "Frame length must be at least LPC order".to_string(),
        ));
    }
    let autocorr = autocorrelate(frame, Some(order + 1))
        .map_err(|e| SpectralError::TimeDomain(e.to_string()))?;
    if autocorr[0] <= 1e-10 {
        return Err(SpectralError::Numerical(
            "Frame energy too low for LPC".to_string(),
        ));
    }

    let mut a = vec![1.0; order + 1];
    let mut e = autocorr[0];
    let mut tmp = vec![0.0; order + 1];

    for i in 1..=order {
        let mut lambda = 0.0;
        for j in 0..i {
            lambda -= a[j] * autocorr[i - j];
        }
        lambda /= e;
        for j in 0..i {
            tmp[j] = a[j] + lambda * a[i - 1 - j];
        }
        a[..i].copy_from_slice(&tmp[..i]);
        a[i] = lambda;
        e *= 1.0 - lambda * lambda;
        if e <= 1e-10 {
            return Err(SpectralError::Numerical(
                "LPC instability detected".to_string(),
            ));
        }
    }
    Ok(a)
}

/// Computes roots of a polynomial.
///
/// Helper function for formant estimation.
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients (highest degree first).
///
/// # Returns
/// Returns `Result<Vec<Complex<f32>>, SpectralError>` containing complex roots.
fn polynomial_roots(coeffs: &[f32]) -> Result<Vec<Complex<f32>>, SpectralError> {
    if coeffs.len() <= 1 {
        return Ok(vec![]);
    }

    let n = coeffs.len() - 1;
    let a_n = coeffs[n];
    if a_n.abs() < 1e-10 {
        return Err(SpectralError::Numerical(
            "Leading coefficient too small".to_string(),
        ));
    }
    // Build the companion matrix; its eigenvalues are the polynomial roots.
    let mut companion = DMatrix::<f32>::zeros(n, n);
    for i in 0..n - 1 {
        companion[(i + 1, i)] = 1.0;
    }
    for i in 0..n {
        companion[(i, n - 1)] = -coeffs[n - 1 - i] / a_n;
    }

    let eigenvalues = companion.complex_eigenvalues();
    Ok(eigenvalues.iter().map(|c| Complex::new(c.re, c.im)).collect())
}
