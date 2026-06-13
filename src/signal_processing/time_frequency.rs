use rustfft::FftPlanner;
use num_complex::Complex;
use ndarray::{Array1, Array2, s};
use crate::{utils::frequency::fft_frequencies_impl, core::AudioError};
use std::f32::consts::{PI, SQRT_2};

/// Analysis window function.
///
/// Windows are *periodic* (DFT-even), the standard convention for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Window {
    /// Periodic Hann window (the default).
    #[default]
    Hann,
    /// Periodic Hamming window.
    Hamming,
}

/// Generates a periodic window of the given length.
pub fn window_vec(window: Window, len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }
    let denom = len as f32;
    (0..len)
        .map(|n| {
            let x = 2.0 * PI * n as f32 / denom;
            match window {
                Window::Hann => 0.5 - 0.5 * x.cos(),
                Window::Hamming => 0.54 - 0.46 * x.cos(),
            }
        })
        .collect()
}

/// Centers a `win_length` window inside an `n_fft` buffer (zero-padded or truncated).
fn fft_window(window: Window, win_length: usize, n_fft: usize) -> Vec<f32> {
    let win = window_vec(window, win_length);
    let mut out = vec![0.0f32; n_fft];
    if win_length <= n_fft {
        let lpad = (n_fft - win_length) / 2;
        out[lpad..lpad + win_length].copy_from_slice(&win);
    } else {
        let start = (win_length - n_fft) / 2;
        out.copy_from_slice(&win[start..start + n_fft]);
    }
    out
}

/// Reflect-pads a signal by `pad` samples on each side (`NumPy` `mode="reflect"`).
///
/// Falls back to zero padding when the signal is shorter than `pad`.
fn reflect_pad(y: &[f32], pad: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(y.len() + 2 * pad);
    if y.len() > pad {
        for i in (1..=pad).rev() {
            out.push(y[i]);
        }
        out.extend_from_slice(y);
        for i in 1..=pad {
            out.push(y[y.len() - 1 - i]);
        }
    } else {
        out.extend(std::iter::repeat_n(0.0, pad));
        out.extend_from_slice(y);
        out.extend(std::iter::repeat_n(0.0, pad));
    }
    out
}

/// STFT builder.
#[derive(Debug, Clone)]
pub struct StftBuilder<'a> {
    y: &'a [f32],
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Window,
    center: bool,
}

impl StftBuilder<'_> {
    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length (default: `win_length / 4`).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Set the window length (default: `n_fft`).
    #[must_use]
    pub fn win_length(mut self, win_length: usize) -> Self {
        self.win_length = Some(win_length);
        self
    }

    /// Set the analysis window (default: [`Window::Hann`]).
    #[must_use]
    pub fn window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Enable or disable centering (default: `true`).
    ///
    /// When enabled, the signal is reflect-padded by `n_fft / 2` so that frame `t`
    /// is centered at sample `t * hop_length`.
    #[must_use]
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Compute the STFT with the configured parameters.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<Complex<f32>>, AudioError> {
        let win_length = self.win_length.unwrap_or(self.n_fft);
        let hop_length = self.hop_length.unwrap_or(win_length / 4).max(1);
        stft_impl(self.y, self.n_fft, hop_length, win_length, self.window, self.center)
    }
}

/// Computes the Short-Time Fourier Transform (STFT) of a signal.
///
/// Returns a builder. Defaults: a periodic Hann window, `win_length = n_fft`,
/// `hop_length = win_length / 4`, and `center = true` (reflect padding). The result
/// has shape `(n_fft / 2 + 1, n_frames)`.
///
/// # Examples
/// ```
/// use dasp_rs::proc::stft;
/// let y = vec![0.0_f32; 4096];
/// let spectrogram = stft(&y).n_fft(1024).hop_length(256).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn stft(y: &[f32]) -> StftBuilder<'_> {
    StftBuilder {
        y,
        n_fft: 2048,
        hop_length: None,
        win_length: None,
        window: Window::Hann,
        center: true,
    }
}

/// Internal STFT implementation.
fn stft_impl(
    y: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    window: Window,
    center: bool,
) -> Result<Array2<Complex<f32>>, AudioError> {
    if n_fft == 0 {
        return Err(AudioError::InvalidInput("n_fft must be positive".into()));
    }
    let hop = hop_length.max(1);
    let win = fft_window(window, win_length, n_fft);

    let padded = if center {
        reflect_pad(y, n_fft / 2)
    } else {
        y.to_vec()
    };

    if padded.len() < n_fft {
        return Err(AudioError::InsufficientData(format!(
            "Signal too short: {} < n_fft {}",
            padded.len(),
            n_fft
        )));
    }

    let n_frames = 1 + (padded.len() - n_fft) / hop;
    let n_bins = n_fft / 2 + 1;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
    let mut out = Array2::zeros((n_bins, n_frames));

    for t in 0..n_frames {
        let start = t * hop;
        for j in 0..n_fft {
            buffer[j] = Complex::new(padded[start + j] * win[j], 0.0);
        }
        fft.process(&mut buffer);
        for b in 0..n_bins {
            out[[b, t]] = buffer[b];
        }
        buffer.fill(Complex::new(0.0, 0.0));
    }

    Ok(out)
}

/// iSTFT builder.
#[derive(Debug, Clone)]
pub struct IstftBuilder<'a> {
    stft_matrix: &'a Array2<Complex<f32>>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Window,
    center: bool,
    length: Option<usize>,
}

impl IstftBuilder<'_> {
    /// Set the hop length (default: `win_length / 4`).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Set the window length (default: `n_fft`).
    #[must_use]
    pub fn win_length(mut self, win_length: usize) -> Self {
        self.win_length = Some(win_length);
        self
    }

    /// Set the synthesis window (default: [`Window::Hann`]).
    #[must_use]
    pub fn window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Set whether the source STFT was centered (default: `true`).
    #[must_use]
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set the target output length in samples.
    #[must_use]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    /// Reconstruct the time-domain signal.
    pub fn compute(self) -> Vec<f32> {
        istft_impl(
            self.stft_matrix,
            self.hop_length,
            self.win_length,
            self.window,
            self.center,
            self.length,
        )
    }
}

/// Inverse Short-Time Fourier Transform (iSTFT).
///
/// Returns a builder. Defaults mirror [`stft`]: a periodic Hann window,
/// `win_length = n_fft`, `hop_length = win_length / 4`, and `center = true`.
///
/// # Examples
/// ```
/// use dasp_rs::proc::stft;
/// let y = vec![0.0_f32; 4096];
/// let spec = stft(&y).n_fft(1024).hop_length(256).compute()?;
/// let recon = dasp_rs::proc::istft(&spec).hop_length(256).length(y.len()).compute();
/// assert_eq!(recon.len(), y.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn istft(stft_matrix: &Array2<Complex<f32>>) -> IstftBuilder<'_> {
    IstftBuilder {
        stft_matrix,
        hop_length: None,
        win_length: None,
        window: Window::Hann,
        center: true,
        length: None,
    }
}

/// Internal iSTFT implementation (overlap-add with window-squared normalization).
fn istft_impl(
    stft_matrix: &Array2<Complex<f32>>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Window,
    center: bool,
    length: Option<usize>,
) -> Vec<f32> {
    let n_bins = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];
    let n_fft = (n_bins - 1) * 2;
    let win_length = win_length.unwrap_or(n_fft);
    let hop = hop_length.unwrap_or(win_length / 4).max(1);
    let win = fft_window(window, win_length, n_fft);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n_fft);
    let inv_n = 1.0 / n_fft as f32;

    let full_len = if n_frames == 0 { 0 } else { hop * (n_frames - 1) + n_fft };
    let mut signal = vec![0.0f32; full_len];
    let mut window_sum = vec![0.0f32; full_len];

    let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
    for (frame_idx, frame) in stft_matrix.axis_iter(ndarray::Axis(1)).enumerate() {
        // Rebuild the conjugate-symmetric full spectrum.
        for b in 0..n_bins {
            buffer[b] = frame[b];
        }
        for b in n_bins..n_fft {
            buffer[b] = frame[n_fft - b].conj();
        }
        fft.process(&mut buffer);
        let start = frame_idx * hop;
        for j in 0..n_fft {
            signal[start + j] += buffer[j].re * inv_n * win[j];
            window_sum[start + j] += win[j] * win[j];
        }
        buffer.fill(Complex::new(0.0, 0.0));
    }

    for (s, &w) in signal.iter_mut().zip(window_sum.iter()) {
        if w > 1e-8 {
            *s /= w;
        }
    }

    let start_off = if center { n_fft / 2 } else { 0 };
    let target = length.unwrap_or_else(|| {
        if center {
            full_len.saturating_sub(n_fft)
        } else {
            full_len
        }
    });
    let mut out = vec![0.0f32; target];
    for (i, o) in out.iter_mut().enumerate() {
        let idx = start_off + i;
        if idx < signal.len() {
            *o = signal[idx];
        }
    }
    out
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
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// use ndarray::arr2;
/// use num_complex::Complex;
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
/// ```no_run
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// let reassigned = reassigned_spectrogram(&y, 44100)
///     .n_fft(2048)
///     .compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn reassigned_spectrogram(y: &[f32], sr: u32) -> ReassignedSpectrogramBuilder<'_> {
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

impl ReassignedSpectrogramBuilder<'_> {
    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Compute the reassigned spectrogram with the configured parameters.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
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
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {e}")))?;
    let s_time = stft_with_derivative(y, Some(n_fft), Some(hop_length), true)?;
    let s_freq = stft_with_derivative(y, Some(n_fft), Some(hop_length), false)?;

    let mut reassigned = Array2::zeros(s.dim());
    let freqs = fft_frequencies_impl(sr, n_fft);
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
/// ```no_run
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// let cqt = cqt(&y, 44100)
///     .hop_length(512)
///     .fmin(32.70)
///     .compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn cqt(y: &[f32], sr: u32) -> CqtBuilder<'_> {
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

impl CqtBuilder<'_> {
    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the minimum frequency (default: 32.70 Hz).
    #[must_use]
    pub fn fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Set the number of frequency bins (default: 84).
    #[must_use]
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Compute the CQT with the configured parameters.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
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
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {e}")))?;
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
/// use dasp_rs::proc::icqt;
/// use ndarray::arr2;
/// use num_complex::Complex;
/// let cqt_data = arr2(&[[Complex::new(1.0, 0.0)]]);
/// let signal = icqt(&cqt_data).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn icqt(c: &Array2<Complex<f32>>) -> IcqtBuilder<'_> {
    IcqtBuilder { c, sr: 44100, hop_length: 512, fmin: 32.70 }
}

/// Builder for [`icqt`].
#[derive(Debug, Clone)]
pub struct IcqtBuilder<'a> {
    c: &'a Array2<Complex<f32>>,
    sr: u32,
    hop_length: usize,
    fmin: f32,
}

impl IcqtBuilder<'_> {
    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sr = sr;
        self
    }

    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the minimum frequency in Hz (default: 32.70).
    #[must_use]
    pub fn fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Reconstruct the time-domain signal.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Vec<f32>, AudioError> {
        icqt_impl(self.c, Some(self.sr), Some(self.hop_length), Some(self.fmin))
    }
}

fn icqt_impl(
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
/// ```no_run
/// use dasp_rs::proc::hybrid_cqt;
/// let signal = vec![0.0_f32; 4096];
/// let hybrid = hybrid_cqt(&signal, 44100).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn hybrid_cqt(y: &[f32], sr: u32) -> HybridCqtBuilder<'_> {
    HybridCqtBuilder { y, sr, hop_length: 512, fmin: 32.70 }
}

/// Builder for [`hybrid_cqt`].
#[derive(Debug, Clone)]
pub struct HybridCqtBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
}

impl HybridCqtBuilder<'_> {
    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the minimum frequency in Hz (default: 32.70).
    #[must_use]
    pub fn fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Compute the hybrid CQT.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<Complex<f32>>, AudioError> {
        hybrid_cqt_impl(self.y, Some(self.sr), Some(self.hop_length), Some(self.fmin))
    }
}

fn hybrid_cqt_impl(
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
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {e}")))?;
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
/// ```no_run
/// use dasp_rs::proc::pseudo_cqt;
/// let signal = vec![0.0_f32; 4096];
/// let pseudo = pseudo_cqt(&signal, 44100).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn pseudo_cqt(y: &[f32], sr: u32) -> PseudoCqtBuilder<'_> {
    PseudoCqtBuilder { y, sr, hop_length: 512, fmin: 32.70 }
}

/// Builder for [`pseudo_cqt`].
#[derive(Debug, Clone)]
pub struct PseudoCqtBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
}

impl PseudoCqtBuilder<'_> {
    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the minimum frequency in Hz (default: 32.70).
    #[must_use]
    pub fn fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Compute the pseudo CQT.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<Complex<f32>>, AudioError> {
        pseudo_cqt_impl(self.y, Some(self.sr), Some(self.hop_length), Some(self.fmin))
    }
}

fn pseudo_cqt_impl(
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
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {e}")))?;
    let mut s_pseudo = Array2::zeros((n_bins, s_stft.shape()[1]));
    let freqs = fft_frequencies_impl(sr, n_fft);

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
/// ```no_run
/// use dasp_rs::proc::vqt;
/// let signal = vec![0.0_f32; 4096];
/// let vqt_result = vqt(&signal, 44100).n_bins(84).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn vqt(y: &[f32], sr: u32) -> VqtBuilder<'_> {
    VqtBuilder { y, sr, hop_length: 512, fmin: 32.70, n_bins: 84 }
}

/// Builder for [`vqt`].
#[derive(Debug, Clone)]
pub struct VqtBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_bins: usize,
}

impl VqtBuilder<'_> {
    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the minimum frequency in Hz (default: 32.70).
    #[must_use]
    pub fn fmin(mut self, fmin: f32) -> Self {
        self.fmin = fmin;
        self
    }

    /// Set the number of frequency bins (default: 84).
    #[must_use]
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Compute the variable-Q transform.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<Complex<f32>>, AudioError> {
        vqt_impl(self.y, Some(self.sr), Some(self.hop_length), Some(self.fmin), Some(self.n_bins))
    }
}

fn vqt_impl(
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
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {e}")))?;
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
/// ```no_run
/// use dasp_rs::proc::fmt;
/// let signal = vec![0.0_f32; 4096];
/// let fmt_result = fmt(&signal).n_fmt(5).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn fmt(y: &[f32]) -> FmtBuilder<'_> {
    FmtBuilder { y, t_min: 0.005, n_fmt: 5, kind: "cos", beta: 2.0 }
}

/// Builder for [`fmt`] (Fast Mellin Transform).
#[derive(Debug, Clone)]
pub struct FmtBuilder<'a> {
    y: &'a [f32],
    t_min: f32,
    n_fmt: usize,
    kind: &'a str,
    beta: f32,
}

impl<'a> FmtBuilder<'a> {
    /// Set the minimum time constant in seconds (default: 0.005).
    #[must_use]
    pub fn t_min(mut self, t_min: f32) -> Self {
        self.t_min = t_min;
        self
    }

    /// Set the number of Mellin bins (default: 5).
    #[must_use]
    pub fn n_fmt(mut self, n_fmt: usize) -> Self {
        self.n_fmt = n_fmt;
        self
    }

    /// Set the window kind (default: "cos").
    #[must_use]
    pub fn kind(mut self, kind: &'a str) -> Self {
        self.kind = kind;
        self
    }

    /// Set the beta shape parameter (default: 2.0).
    #[must_use]
    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Compute the Fast Mellin Transform.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<f32>, AudioError> {
        fmt_impl(self.y, Some(self.t_min), Some(self.n_fmt), Some(self.kind), Some(self.beta))
    }
}

fn fmt_impl(
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
fn hann_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())).collect()
}

/// Computes STFT with time or frequency derivative for reassignment.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length in samples (defaults to `n_fft/4`)
/// * `time_derivative` - If true, computes time derivative; if false, frequency derivative
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` with derivative information,
/// or an `AudioError` if computation fails.
fn stft_with_derivative(
    y: &[f32],
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    time_derivative: bool,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(n_fft / 4);
    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!(
            "Signal length {} is shorter than n_fft {n_fft}",
            y.len()
        )));
    }
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
        for f in 0..=(n_fft / 2) {
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
fn butterworth_bandpass(lowcut: f32, highcut: f32, fs: f32, order: Option<usize>) -> Result<(Vec<f32>, Vec<f32>), AudioError> {
    if lowcut <= 0.0 || highcut <= lowcut || highcut >= fs / 2.0 {
        return Err(AudioError::InvalidInput(format!(
            "Invalid frequencies: lowcut={} must be > 0, highcut={} must be > lowcut and < fs/2={}",
            lowcut, highcut, fs / 2.0
        )));
    }

    let order = order.unwrap_or(2);
    let n = order as i32;

    // Bilinear transform pre-warping: Ï‰_analog = 2*fs * tan(Ï€ * f / fs)
    let w_low = 2.0 * fs * (PI * lowcut / fs).tan();
    let w_high = 2.0 * fs * (PI * highcut / fs).tan();
    let w0 = (w_high * w_low).sqrt();  // Geometric mean (center frequency)
    let bw = w_high - w_low;  // Bandwidth

    // Calculate poles for bandpass Butterworth filter in s-domain
    // Standard Butterworth pole angles: Î¸_k = Ï€(2k+1)/(2n) for k = 0..n-1
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
    for p in &z_poles {
        b = convolve(&b, &[1.0, -p.re]);
        a = convolve(&a, &[1.0, -p.re]);
    }
    for _ in 0..n {
        b = convolve(&b, &[1.0, 0.0]);
    }

    let w_center = 2.0 * PI * (lowcut + highcut) / 2.0 / fs;
    let gain = evaluate_filter(&b, &a, w_center).norm();
    for b_k in &mut b {
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
/// * `hop_length` - Optional hop length in samples (defaults to `win_length/4`)
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
/// ```no_run
/// use dasp_rs::proc::iirt;
/// let signal = vec![0.0_f32; 8192];
/// let iirt_result = iirt(&signal, 44100).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn iirt(y: &[f32], sr: u32) -> IirtBuilder<'_> {
    IirtBuilder { y, sr, win_length: 2048, hop_length: None }
}

/// Builder for [`iirt`].
#[derive(Debug, Clone)]
pub struct IirtBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    win_length: usize,
    hop_length: Option<usize>,
}

impl IirtBuilder<'_> {
    /// Set the window length (default: 2048).
    #[must_use]
    pub fn win_length(mut self, win_length: usize) -> Self {
        self.win_length = win_length;
        self
    }

    /// Set the hop length (default: `win_length / 4`).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Compute the IIR (semitone filterbank) transform.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Array2<f32>, AudioError> {
        iirt_impl(self.y, Some(self.sr), Some(self.win_length), self.hop_length)
    }
}

fn iirt_impl(
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
fn filter(x: &[f32], b: &[f32], a: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for n in 0..x.len() {
        let x1 = if n >= 1 { x[n - 1] } else { 0.0 };
        let x2 = if n >= 2 { x[n - 2] } else { 0.0 };
        let y1 = if n >= 1 { y[n - 1] } else { 0.0 };
        let y2 = if n >= 2 { y[n - 2] } else { 0.0 };
        y[n] = b[0] * x[n] + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2;
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
    fn stft_shape_and_centered_frame_count() {
        // Centered STFT: n_frames == 1 + len / hop.
        let y = vec![0.0_f32; 1024];
        let spec = stft(&y).n_fft(256).hop_length(64).compute().expect("stft");
        assert_eq!(spec.shape()[0], 256 / 2 + 1);
        assert_eq!(spec.shape()[1], 1 + 1024 / 64);
    }

    #[test]
    fn stft_istft_round_trip_reconstructs_signal() {
        // Hann window with hop = n_fft/4 satisfies the COLA constraint, so the
        // interior of the signal reconstructs almost exactly.
        let y: Vec<f32> = (0..2048)
            .map(|n| (2.0 * PI * 7.0 * n as f32 / 256.0).sin())
            .collect();
        let spec = stft(&y).n_fft(256).hop_length(64).compute().expect("stft");
        let recon = istft(&spec).hop_length(64).length(y.len()).compute();
        assert_eq!(recon.len(), y.len());

        let (mut err, mut cnt) = (0.0_f32, 0usize);
        for i in 256..(y.len() - 256) {
            err += (recon[i] - y[i]).abs();
            cnt += 1;
        }
        assert!((err / cnt as f32) < 1e-3, "mean abs reconstruction error too high");
    }

    #[test]
    fn stft_locates_sinusoid_peak_bin() {
        // A pure tone at bin k should produce its maximum magnitude at row k.
        let n_fft = 256;
        let k = 10usize;
        let y: Vec<f32> = (0..4096)
            .map(|n| (2.0 * PI * k as f32 * n as f32 / n_fft as f32).sin())
            .collect();
        let spec = stft(&y).n_fft(n_fft).hop_length(64).compute().expect("stft");
        let frame = spec.column(spec.shape()[1] / 2);
        let peak = (0..frame.len())
            .max_by(|&a, &b| frame[a].norm().partial_cmp(&frame[b].norm()).unwrap())
            .unwrap();
        assert_eq!(peak, k);
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
