use rustfft::FftPlanner;
use num_complex::Complex;
use ndarray::{Array1, Array2, s};
use crate::{utils::frequency::fft_frequencies, core::AudioError};
use std::f32::consts::{PI, SQRT_2};

/// STFT builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct StftBuilder<'a> {
    y: &'a [f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
}

impl<'a> StftBuilder<'a> {
    /// Set the FFT size (default: 2048).
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

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

    /// Compute the STFT with the configured parameters.
    pub fn compute(self) -> Result<Array2<Complex<f32>>, AudioError> {
        stft_impl(self.y, self.n_fft, self.hop_length, self.win_length)
    }
}

/// Computes the Short-Time Fourier Transform (STFT) of a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
///
/// # Returns
/// Returns a builder that can be configured with method chaining.
///
/// # Examples
/// ```
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// // Clean, ergonomic API with method chaining
/// let spectrogram = stft(&y)
///     .n_fft(1024)
///     .hop_length(256)
///     .compute()?;
/// 
/// // Or with defaults
/// let spectrogram = stft(&y).compute()?;
/// ```
pub fn stft(y: &[f32]) -> StftBuilder {
    StftBuilder {
        y,
        n_fft: 2048,
        hop_length: 512,
        win_length: 2048,
    }
}

/// Internal STFT implementation.
fn stft_impl(
    y: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let n = n_fft;
    let hop = hop_length.max(1);
    let win = win_length;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = vec![Complex::new(0.0, 0.0); n];
    let mut spectrogram = Vec::new();

    if y.len() < n {
        let mut padded = vec![0.0; n];
        padded[..y.len()].copy_from_slice(y);
        buffer[..n].copy_from_slice(&padded.iter().map(|&x| Complex::new(x * hamming(0, win), 0.0)).collect::<Vec<_>>());
        fft.process(&mut buffer);
        spectrogram.push(buffer.clone());
    } else {
        for i in (0..y.len()).step_by(hop) {
            let end = std::cmp::min(i + n, y.len());
            buffer.fill(Complex::new(0.0, 0.0));
            for (j, &sample) in y[i..end].iter().enumerate() {
                buffer[j] = Complex::new(sample * hamming(j, win), 0.0);
            }
            fft.process(&mut buffer);
            spectrogram.push(buffer.clone());
        }
    }

    let n_frames = spectrogram.len();
    Ok(Array2::from_shape_vec((n / 2 + 1, n_frames), spectrogram.into_iter().flat_map(|v| v.into_iter().take(n / 2 + 1)).collect())?)
}

/// Computes the inverse Short-Time Fourier Transform (iSTFT) to reconstruct a signal.
///
/// # Arguments
/// * `stft_matrix` - STFT spectrogram as an `Array2<Complex<f32>>`
/// * `hop_length` - Optional hop length in samples (defaults to n_fft/4, minimum 1)
/// * `win_length` - Optional window length in samples (defaults to n_fft)
/// * `length` - Optional output signal length in samples (defaults to maximum possible length)
///
/// # Returns
/// Returns a `Vec<f32>` containing the reconstructed time-domain signal.
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let stft_data = arr2(&[[Complex::new(1.0, 0.0)], [Complex::new(0.5, 0.0)]]);
/// let signal = istft(&stft_data, None, None, None);
/// ```
pub fn istft(
    stft_matrix: &Array2<Complex<f32>>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    length: Option<usize>,
) -> Vec<f32> {
    let n_fft = (stft_matrix.shape()[0] - 1) * 2;
    let hop = hop_length.unwrap_or(n_fft / 4).max(1);
    let win = win_length.unwrap_or(n_fft);
    let n_frames = stft_matrix.shape()[1];
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n_fft);

    let max_len = hop * (n_frames - 1) + n_fft;
    let target_len = length.unwrap_or(max_len);
    let mut signal = vec![0.0; max_len];
    let mut window_sum = vec![0.0; max_len];
    let window = hamming_vec(win);

    for (frame_idx, frame) in stft_matrix.axis_iter(ndarray::Axis(1)).enumerate() {
        let mut buffer: Vec<Complex<f32>> = frame.to_vec();
        buffer.extend(vec![Complex::new(0.0, 0.0); n_fft - buffer.len()]);
        fft.process(&mut buffer);
        let start = frame_idx * hop;
        for (i, &val) in buffer.iter().enumerate().take(win) {
            if start + i < signal.len() {
                signal[start + i] += val.re * window[i];
                window_sum[start + i] += window[i];
            }
        }
    }

    for (i, &sum) in window_sum.iter().enumerate() {
        if sum > 1e-6 {
            signal[i] /= sum;
        }
    }

    signal.resize(target_len, 0.0);
    signal
}

/// Computes the Hamming window value at a given sample index.
///
/// # Arguments
/// * `n` - Sample index
/// * `win_length` - Total window length
///
/// # Returns
/// Returns a `f32` representing the Hamming window coefficient.
///
/// # Examples
/// ```
/// let value = hamming(0, 10);
/// assert!(value > 0.0 && value <= 1.0);
/// ```
fn hamming(n: usize, win_length: usize) -> f32 {
    0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / (win_length - 1) as f32).cos()
}

/// Generates a Hamming window vector.
///
/// # Arguments
/// * `win_length` - Length of the window
///
/// # Returns
/// Returns a `Vec<f32>` containing the Hamming window coefficients.
///
/// # Examples
/// ```
/// let window = hamming_vec(5);
/// assert_eq!(window.len(), 5);
/// ```
fn hamming_vec(win_length: usize) -> Vec<f32> {
    (0..win_length).map(|n| hamming(n, win_length)).collect()
}

/// Separates magnitude and phase from a complex spectrogram.
///
/// # Arguments
/// * `D` - Input spectrogram as an `Array2<Complex<f32>>`
/// * `power` - Optional power to raise the magnitude (defaults to 1.0)
///
/// # Returns
/// Returns a tuple `(magnitude, phase)` where:
/// - `magnitude` is an `Array2<f32>` of magnitude values
/// - `phase` is an `Array2<Complex<f32>>` of unit-magnitude phase values
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let spectrogram = arr2(&[[Complex::new(3.0, 4.0)]]);
/// let (mag, phase) = magphase(&spectrogram, None);
/// assert_eq!(mag[[0, 0]], 5.0); // sqrt(3^2 + 4^2)
/// ```
pub fn magphase(d: &Array2<Complex<f32>>, power: Option<f32>) -> (Array2<f32>, Array2<Complex<f32>>) {
    let power_val = power.unwrap_or(1.0);
    let magnitude = d.mapv(|x| x.norm().powf(power_val));
    let phase = d.mapv(|x| x / x.norm());
    (magnitude, phase)
}

/// Computes a reassigned spectrogram for improved time-frequency resolution.
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
/// let reassigned = reassigned_spectrogram(&y, 44100)
///     .n_fft(2048)
///     .compute()?;
/// ```
pub fn reassigned_spectrogram(y: &[f32], sr: u32) -> ReassignedSpectrogramBuilder {
    ReassignedSpectrogramBuilder {
        y,
        sr,
        n_fft: 2048,
    }
}

/// Reassigned spectrogram builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct ReassignedSpectrogramBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    n_fft: usize,
}

impl<'a> ReassignedSpectrogramBuilder<'a> {
    /// Set the FFT size (default: 2048).
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Compute the reassigned spectrogram with the configured parameters.
    pub fn compute(self) -> Result<Array2<f32>, AudioError> {
        reassigned_spectrogram_impl(self.y, self.sr, self.n_fft)
    }
}

/// Internal reassigned spectrogram implementation.
fn reassigned_spectrogram_impl(
    y: &[f32],
    sr: u32,
    n_fft: usize,
) -> Result<Array2<f32>, AudioError> {
    let hop_length = n_fft / 4;

    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), n_fft)));
    }

    let s = stft(y)
        .n_fft(n_fft)
        .hop_length(hop_length)
        .compute()
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let s_time = stft_with_derivative(y, Some(n_fft), Some(hop_length), true)?;
    let s_freq = stft_with_derivative(y, Some(n_fft), Some(hop_length), false)?;

    let mut reassigned = Array2::zeros(s.dim());
    let freqs = fft_frequencies(Some(sr), Some(n_fft));
    let times = Array1::linspace(0.0, (y.len() as f32 - 1.0) / sr as f32, s.shape()[1]);
    // Precompute constants for efficiency
    let sr_f = sr as f32;
    let hop_f = hop_length as f32;
    let n_fft_f = n_fft as f32;
    let time_scale = sr_f / hop_f;
    let freq_scale = sr_f / n_fft_f;

    for t in 0..s.shape()[1] {
        for f in 0..s.shape()[0] {
            let mag = s[[f, t]].norm();
            if mag > 1e-6 {
                let dphi_dt = s_time[[f, t]].im / mag;
                let t_reassigned = times[t] - dphi_dt * hop_f / sr_f;
                let dphi_df = s_freq[[f, t]].im / mag;
                let f_reassigned = freqs[f] + dphi_df * freq_scale;

                let t_idx = ((t_reassigned * time_scale).round() as usize).min(s.shape()[1] - 1);
                // Use binary search for frequency lookup (more efficient for sorted array)
                let f_idx = freqs.binary_search_by(|&x| {
                    x.partial_cmp(&f_reassigned).unwrap_or(std::cmp::Ordering::Less)
                })
                .unwrap_or_else(|i| i)
                .min(s.shape()[0] - 1);
                reassigned[[f_idx, t_idx]] += mag;
            }
        }
    }

    Ok(reassigned)
}

/// Computes the Constant-Q Transform (CQT) of a signal.
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
/// let cqt = cqt(&y, 44100)
///     .hop_length(512)
///     .fmin(32.70)
///     .compute()?;
/// ```
pub fn cqt(y: &[f32], sr: u32) -> CqtBuilder {
    CqtBuilder {
        y,
        sr,
        hop_length: 512,
        fmin: 32.70,
        n_bins: 84,
    }
}

/// CQT builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct CqtBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_bins: usize,
}

impl<'a> CqtBuilder<'a> {
    /// Set the hop length (default: 512).
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the minimum frequency (default: 32.70 Hz).
    pub fn fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Set the number of frequency bins (default: 84).
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Compute the CQT with the configured parameters.
    pub fn compute(self) -> Result<Array2<Complex<f32>>, AudioError> {
        cqt_impl(self.y, self.sr, self.hop_length, self.fmin, self.n_bins)
    }
}

/// Internal CQT implementation.
fn cqt_impl(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_bins: usize,
) -> Result<Array2<Complex<f32>>, AudioError> {
    // Parameters are already provided directly
    let bins_per_octave = 12;

    if y.len() < hop_length {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), hop_length)));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let s_stft = stft(y)
        .n_fft(n_fft)
        .hop_length(hop_length)
        .compute()
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let n_frames = s_stft.shape()[1];
    let mut s_cqt = Array2::zeros((n_bins, n_frames));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    for k in 0..n_bins {
        let fk = fmin * 2.0f32.powf(k as f32 / bins_per_octave as f32);
        let n = (sr as f32 / fk).round() as usize;
        let mut kernel = Array1::zeros(n_fft);
        let window = hann_window(n);
        for i in 0..n {
            let phase = 2.0 * PI * fk * i as f32 / sr as f32;
            kernel[i] = Complex::new(window[i] * phase.cos(), window[i] * phase.sin()) / n as f32;
        }
        fft.process(&mut kernel.to_vec());

        for t in 0..n_frames {
            let stft_frame = s_stft.slice(s![.., t]);
            s_cqt[[k, t]] = stft_frame.iter().zip(kernel.iter()).map(|(&s, &k)| s * k.conj()).sum();
        }
    }

    Ok(s_cqt)
}

/// Computes the inverse Constant-Q Transform (iCQT) to reconstruct a signal.
///
/// # Arguments
/// * `C` - CQT spectrogram as an `Array2<Complex<f32>>`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
///
/// # Returns
/// Returns a `Result` containing a `Vec<f32>` of the reconstructed signal,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let cqt_data = arr2(&[[Complex::new(1.0, 0.0)]]);
/// let signal = icqt(&cqt_data, None, None, None).unwrap();
/// ```
pub fn icqt(
    c: &Array2<Complex<f32>>,
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
) -> Result<Vec<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_bins = c.shape()[0];
    let n_frames = c.shape()[1];
    let bins_per_octave = 12;

    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let n_samples = n_frames * hop_length;
    let mut y = vec![0.0; n_samples];
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    for k in 0..n_bins {
        let fk = fmin * 2.0f32.powf(k as f32 / bins_per_octave as f32);
        let n = (sr as f32 / fk).round() as usize;
        let window = hann_window(n);
        let mut kernel = Array1::zeros(n_fft);
        for i in 0..n {
            let phase = 2.0 * PI * fk * i as f32 / sr as f32;
            kernel[i] = Complex::new(window[i] * phase.cos(), window[i] * phase.sin()) / n as f32;
        }
        ifft.process(&mut kernel.to_vec());

        for t in 0..n_frames {
            let mut frame = vec![Complex::new(c[[k, t]].re, c[[k, t]].im) * Complex::conj(&kernel[0]); n_fft];
            ifft.process(&mut frame);
            let start = t * hop_length;
            for i in 0..n.min(n_samples - start) {
                y[start + i] += frame[i].re * window[i];
            }
        }
    }

    let mut overlap = vec![0.0; n_samples];
    for t in 0..n_frames {
        let start = t * hop_length;
        for i in 0..n_fft.min(n_samples - start) {
            overlap[start + i] += hann_window(n_fft)[i].powi(2);
        }
    }
    for i in 0..n_samples {
        if overlap[i] > 1e-6 {
            y[i] /= overlap[i];
        }
    }

    Ok(y)
}

/// Computes a hybrid Constant-Q Transform (CQT) combining STFT and CQT properties.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` representing the hybrid CQT,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `n_fft`.
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
/// * `AudioError::ComputationFailed` - If STFT computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let hybrid = hybrid_cqt(&signal, None, None, None).unwrap();
/// ```
pub fn hybrid_cqt(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let n_bins = 84;

    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), n_fft)));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let s_stft = stft(y)
        .n_fft(n_fft)
        .hop_length(hop_length)
        .compute()
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let mut s_hybrid = Array2::zeros((n_bins, s_stft.shape()[1]));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    for k in 0..n_bins {
        let fk = fmin * 2.0f32.powf(k as f32 / 12.0);
        let n = (sr as f32 / fk).round() as usize;
        let mut kernel = Array1::zeros(n_fft);
        let window = hann_window(n);
        for i in 0..n {
            let phase = 2.0 * PI * fk * i as f32 / sr as f32;
            kernel[i] = Complex::new(window[i] * phase.cos(), window[i] * phase.sin()) / n as f32;
        }
        fft.process(&mut kernel.to_vec());

        for t in 0..s_stft.shape()[1] {
            s_hybrid[[k, t]] = s_stft.slice(s![.., t]).iter().zip(kernel.iter()).map(|(&s, &k)| s * k.conj()).sum();
        }
    }

    Ok(s_hybrid)
}

/// Computes a pseudo Constant-Q Transform (CQT) using STFT bin mapping.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` representing the pseudo CQT,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `n_fft`.
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
/// * `AudioError::ComputationFailed` - If STFT computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let pseudo = pseudo_cqt(&signal, None, None, None).unwrap();
/// ```
pub fn pseudo_cqt(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let n_bins = 84;

    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), n_fft)));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let s_stft = stft(y)
        .n_fft(n_fft)
        .hop_length(hop_length)
        .compute()
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let mut s_pseudo = Array2::zeros((n_bins, s_stft.shape()[1]));
    let freqs = fft_frequencies(Some(sr), Some(n_fft));

    for t in 0..s_stft.shape()[1] {
        for k in 0..n_bins {
            let fk = fmin * 2.0f32.powf(k as f32 / 12.0);
            let idx = freqs.iter().position(|&f| f >= fk).unwrap_or(0);
            s_pseudo[[k, t]] = s_stft[[idx.min(s_stft.shape()[0] - 1), t]];
        }
    }

    Ok(s_pseudo)
}

/// Computes the Variable-Q Transform (VQT) of a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
/// * `n_bins` - Optional number of frequency bins (defaults to 84)
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` representing the VQT,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `hop_length`.
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
/// * `AudioError::ComputationFailed` - If STFT computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let vqt_result = vqt(&signal, None, None, None, None).unwrap();
/// ```
pub fn vqt(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    n_bins: Option<usize>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_bins = n_bins.unwrap_or(84);
    let gamma = 24.0;

    if y.len() < hop_length {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), hop_length)));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let s_stft = stft(y)
        .n_fft(n_fft)
        .hop_length(hop_length)
        .compute()
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let mut s_vqt = Array2::zeros((n_bins, s_stft.shape()[1]));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    for k in 0..n_bins {
        let fk = fmin * 2.0f32.powf(k as f32 / 12.0);
        let q = gamma / (2.0f32.powf(1.0 / 12.0) - 1.0);
        let n = (sr as f32 * q / fk).round() as usize;
        let mut kernel = Array1::zeros(n_fft);
        let window = hann_window(n);
        for i in 0..n {
            let phase = 2.0 * PI * fk * i as f32 / sr as f32;
            kernel[i] = Complex::new(window[i] * phase.cos(), window[i] * phase.sin()) / n as f32;
        }
        fft.process(&mut kernel.to_vec());

        for t in 0..s_stft.shape()[1] {
            s_vqt[[k, t]] = s_stft.slice(s![.., t]).iter().zip(kernel.iter()).map(|(&s, &k)| s * k.conj()).sum();
        }
    }

    Ok(s_vqt)
}

/// Computes the Fourier Modulation Transform (FMT) of a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `t_min` - Optional minimum time period in seconds (defaults to 0.005)
/// * `n_fmt` - Optional number of modulation frequencies (defaults to 5)
/// * `kind` - Optional transform kind ("cos" or others, defaults to "cos")
/// * `beta` - Optional power for magnitude scaling (defaults to 2.0)
///
/// # Returns
/// Returns a `Result` containing an `Array2<f32>` representing the FMT spectrogram,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `hop_length`.
/// * `AudioError::InvalidInput` - If `t_min` is not positive.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let fmt_result = fmt(&signal, None, None, None, None).unwrap();
/// ```
pub fn fmt(
    y: &[f32],
    t_min: Option<f32>,
    n_fmt: Option<usize>,
    kind: Option<&str>,
    beta: Option<f32>,
) -> Result<Array2<f32>, AudioError> {
    let sr = 44100;
    let t_min = t_min.unwrap_or(0.005);
    let n_fmt = n_fmt.unwrap_or(5);
    let _kind = kind.unwrap_or("cos");
    let beta = beta.unwrap_or(2.0);
    let hop_length = (sr as f32 * t_min).round() as usize;

    if y.len() < hop_length {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), hop_length)));
    }
    if t_min <= 0.0 {
        return Err(AudioError::InvalidInput("t_min must be positive".to_string()));
    }

    let n_frames = (y.len() - hop_length) / hop_length + 1;
    let mut s = Array2::zeros((n_fmt, n_frames));
    let window = hann_window(hop_length);

    for t in 0..n_frames {
        let start = t * hop_length;
        let frame = &y[start..(start + hop_length).min(y.len())];
        for k in 0..n_fmt {
            let freq = (k + 1) as f32 / t_min;
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for (i, &sample) in frame.iter().enumerate() {
                let phase = 2.0 * PI * freq * i as f32 / sr as f32;
                let w = window[i];
                sum_re += sample * w * phase.cos();
                sum_im += sample * w * phase.sin();
            }
            let mag = Complex::new(sum_re, sum_im).norm() / hop_length as f32;
            s[[k, t]] = mag.powf(beta);
        }
    }

    Ok(s)
}

/// Generates a Hann window vector.
///
/// # Arguments
/// * `n` - Length of the window
///
/// # Returns
/// Returns a `Vec<f32>` containing the Hann window coefficients.
///
/// # Examples
/// ```
/// let window = hann_window(5);
/// assert_eq!(window.len(), 5);
/// ```
fn hann_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())).collect()
}

/// Computes STFT with time or frequency derivative for reassignment.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length in samples (defaults to n_fft/4)
/// * `time_derivative` - If true, computes time derivative; if false, frequency derivative
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` with derivative information,
/// or an `AudioError` if computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 2048];
/// let deriv = stft_with_derivative(&signal, None, None, true).unwrap();
/// ```
fn stft_with_derivative(
    y: &[f32],
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    time_derivative: bool,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(n_fft / 4);
    let n_frames = (y.len() - n_fft) / hop_length + 1;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut s = Array2::zeros((n_fft / 2 + 1, n_frames));
    let window = hann_window(n_fft);
    let deriv_window = if time_derivative {
        (0..n_fft).map(|i| i as f32 * window[i]).collect::<Vec<_>>()
    } else {
        (0..n_fft).map(|i| window[i] * (2.0 * PI * i as f32 / n_fft as f32).sin()).collect::<Vec<_>>()
    };

    for t in 0..n_frames {
        let start = t * hop_length;
        let frame = &y[start..(start + n_fft).min(y.len())];
        let mut buffer = frame.iter().zip(deriv_window.iter()).map(|(&x, &w)| Complex::new(x * w, 0.0)).collect::<Vec<_>>();
        buffer.resize(n_fft, Complex::new(0.0, 0.0));
        fft.process(&mut buffer);
        for f in 0..n_fft / 2 + 1 {
            s[[f, t]] = buffer[f];
        }
    }
    Ok(s)
}

/// Designs a Butterworth bandpass filter.
///
/// # Arguments
/// * `lowcut` - Lower cutoff frequency in Hz
/// * `highcut` - Upper cutoff frequency in Hz
/// * `fs` - Sampling frequency in Hz
/// * `order` - Optional filter order (defaults to 2)
///
/// # Returns
/// Returns a `Result` containing a tuple `(b, a)` of numerator and denominator coefficients,
/// or an `AudioError` if frequencies are invalid.
///
/// # Errors
/// * `AudioError::InvalidInput` - If `lowcut` <= 0, `highcut` <= `lowcut`, or `highcut` >= `fs/2`.
///
/// # Examples
/// ```
/// let (b, a) = butterworth_bandpass(100.0, 1000.0, 44100.0, None).unwrap();
/// ```
fn butterworth_bandpass(lowcut: f32, highcut: f32, fs: f32, order: Option<usize>) -> Result<(Vec<f32>, Vec<f32>), AudioError> {
    if lowcut <= 0.0 || highcut <= lowcut || highcut >= fs / 2.0 {
        return Err(AudioError::InvalidInput(format!(
            "Invalid frequencies: lowcut={} must be > 0, highcut={} must be > lowcut and < fs/2={}",
            lowcut, highcut, fs / 2.0
        )));
    }

    let order = order.unwrap_or(2);
    let n = order as i32;

    // Bilinear transform pre-warping: ω_analog = 2*fs * tan(π * f / fs)
    let w_low = 2.0 * fs * (PI * lowcut / fs).tan();
    let w_high = 2.0 * fs * (PI * highcut / fs).tan();
    let w0 = (w_high * w_low).sqrt();  // Geometric mean (center frequency)
    let bw = w_high - w_low;  // Bandwidth

    // Calculate poles for bandpass Butterworth filter in s-domain
    // Standard Butterworth pole angles: θ_k = π(2k+1)/(2n) for k = 0..n-1
    let mut poles = Vec::new();
    for k in 0..n {
        // Corrected: removed erroneous +n term from original formula
        let theta = PI * (2.0 * k as f32 + 1.0) / (2.0 * n as f32);
        // Bandpass poles: real part from bandwidth, imaginary from center frequency
        let real = -bw / 2.0 * theta.sin();
        let imag = w0 * theta.cos();
        poles.push(Complex::new(real, imag));
        poles.push(Complex::new(real, -imag));
    }

    let mut z_poles = Vec::new();
    let fs2 = 2.0 * fs;
    for p in poles {
        let pz = (fs2 + p) / (fs2 - p);
        z_poles.push(pz);
    }

    let mut b = vec![1.0];
    let mut a = vec![1.0];
    for p in z_poles.iter() {
        b = convolve(&b, &[1.0, -p.re]);
        a = convolve(&a, &[1.0, -p.re]);
    }
    for _ in 0..n {
        b = convolve(&b, &[1.0, 0.0]);
    }

    let w_center = 2.0 * PI * (lowcut + highcut) / 2.0 / fs;
    let gain = evaluate_filter(&b, &a, w_center).norm();
    for b_k in b.iter_mut() {
        *b_k /= gain;
    }

    Ok((b, a))
}

/// Convolves two vectors.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
/// Returns a `Vec<f32>` containing the convolution result.
///
/// # Examples
/// ```
/// let result = convolve(&[1.0, 2.0], &[3.0, 4.0]);
/// assert_eq!(result, vec![3.0, 10.0, 8.0]);
/// ```
fn convolve(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] += a[i] * b[j];
        }
    }
    result
}

/// Evaluates a digital filter's frequency response at a given frequency.
///
/// # Arguments
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `w` - Frequency in radians/sample
///
/// # Returns
/// Returns a `Complex<f32>` representing the filter's response.
///
/// # Examples
/// ```
/// let response = evaluate_filter(&[1.0], &[1.0, -0.5], 0.1);
/// ```
fn evaluate_filter(b: &[f32], a: &[f32], w: f32) -> Complex<f32> {
    let mut num = Complex::new(0.0, 0.0);
    let mut den = Complex::new(0.0, 0.0);
    for (k, &bk) in b.iter().enumerate() {
        let phase = -w * k as f32;
        num += Complex::new(bk * phase.cos(), bk * phase.sin());
    }
    for (k, &ak) in a.iter().enumerate() {
        let phase = -w * k as f32;
        den += Complex::new(ak * phase.cos(), ak * phase.sin());
    }
    num / den
}

/// Computes the Instantaneous Impulse Response Transform (IIRT) using bandpass filtering.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `win_length` - Optional window length in samples (defaults to 2048)
/// * `hop_length` - Optional hop length in samples (defaults to win_length/4)
///
/// # Returns
/// Returns a `Result` containing an `Array2<f32>` representing the IIRT spectrogram,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `win_length`.
/// * `AudioError::InvalidInput` - If bandpass filter frequencies are invalid.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 4096];
/// let iirt_result = iirt(&signal, None, None, None).unwrap();
/// ```
pub fn iirt(
    y: &[f32],
    sr: Option<u32>,
    win_length: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array2<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let win_length = win_length.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(win_length / 4);
    let n_bands = 12;

    if y.len() < win_length {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), win_length)));
    }

    let n_frames = (y.len() - win_length) / hop_length + 1;
    let mut s = Array2::zeros((n_bands, n_frames));
    let fmin = 32.70;

    for b in 0..n_bands {
        let fc = fmin * 2.0f32.powf(b as f32);
        let bw = fc / SQRT_2;
        let (b_coeffs, a_coeffs) = butterworth_bandpass(fc - bw / 2.0, fc + bw / 2.0, sr as f32, Some(4))?;
        
        for t in 0..n_frames {
            let start = t * hop_length;
            let frame = &y[start..(start + win_length).min(y.len())];
            let filtered = filter(frame, &b_coeffs, &a_coeffs);
            s[[b, t]] = filtered.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt() / win_length as f32;
        }
    }

    Ok(s)
}

/// Applies an IIR filter to a signal.
///
/// # Arguments
/// * `x` - Input signal as a slice of `f32`
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
///
/// # Returns
/// Returns a `Vec<f32>` containing the filtered signal.
///
/// # Examples
/// ```
/// let signal = vec![1.0, 2.0, 3.0];
/// let filtered = filter(&signal, &[1.0, 0.0, 0.0], &[1.0, -0.5, 0.0]);
/// ```
fn filter(x: &[f32], b: &[f32], a: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for n in 0..x.len() {
        y[n] = b[0] * x[n] + b[1] * x.get(n - 1).unwrap_or(&0.0) + b[2] * x.get(n - 2).unwrap_or(&0.0)
            - a[1] * y.get(n - 1).unwrap_or(&0.0) - a[2] * y.get(n - 2).unwrap_or(&0.0);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn stft_and_istft_round_trip_lengths() {
        let y = vec![1.0, 0.0, -1.0, 0.0];
        let spec = stft(&y)
            .n_fft(8)
            .hop_length(4)
            .win_length(8)
            .compute()
            .expect("stft");
        assert_eq!(spec.shape(), &[5, 1]); // n_fft/2 + 1 rows, single frame

        let recon = istft(&spec, Some(4), Some(8), Some(y.len()));
        assert_eq!(recon.len(), y.len());
        assert!(recon.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn magphase_returns_unit_phase_and_expected_magnitude() {
        let d = arr2(&[[Complex::new(3.0, 4.0)]]);
        let (mag, phase) = magphase(&d, None);
        assert!(approx_eq(mag[[0, 0]], 5.0, 1e-6));
        let ph = phase[[0, 0]];
        assert!(approx_eq(ph.norm(), 1.0, 1e-6));
    }

    #[test]
    fn reassigned_spectrogram_errors_on_short_signal() {
        let y = vec![0.0; 4];
        let result = reassigned_spectrogram(&y, 44_100).n_fft(8).compute();
        assert!(matches!(result, Err(AudioError::InsufficientData(_))));
    }

    #[test]
    fn filter_and_convolution_helpers_behave() {
        let conv = super::convolve(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(conv, vec![3.0, 10.0, 8.0]);

        let filtered = filter(&[1.0, 2.0, 3.0], &[1.0, 0.0, 0.0], &[1.0, -0.5, 0.0]);
        assert_eq!(filtered.len(), 3);
        assert!(filtered.iter().all(|v| v.is_finite()));
    }
}