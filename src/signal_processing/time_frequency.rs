use crate::{core::AudioError, utils::frequency::fft_frequencies_impl};
use ndarray::Array2;
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

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
/// Repeats the reflection for short signals; a singleton extends its sole value.
fn reflect_pad(y: &[f32], pad: usize) -> Vec<f32> {
    if y.len() == 1 {
        return vec![y[0]; 1 + 2 * pad];
    }
    let period = 2 * (y.len() - 1);
    let offset = (period - pad % period) % period;
    (0..y.len() + 2 * pad)
        .map(|i| {
            let position = (i + offset) % period;
            y[if position < y.len() {
                position
            } else {
                period - position
            }]
        })
        .collect()
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
        let hop_length = self.hop_length.unwrap_or_else(|| (win_length / 4).max(1));
        stft_impl(
            self.y,
            self.n_fft,
            hop_length,
            win_length,
            self.window,
            self.center,
        )
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
    if n_fft == 0 || win_length == 0 || win_length > n_fft || hop_length == 0 {
        return Err(AudioError::InvalidInput(
            "Require n_fft > 0, 0 < win_length <= n_fft, and hop_length > 0".into(),
        ));
    }
    if y.is_empty() || y.iter().any(|v| !v.is_finite()) {
        return Err(AudioError::InvalidInput(
            "Signal must be nonempty and finite".into(),
        ));
    }
    let max_elements = isize::MAX as usize / std::mem::size_of::<Complex<f32>>();
    let padded_len = y
        .len()
        .checked_add(if center { 2 * (n_fft / 2) } else { 0 })
        .filter(|&len| len <= max_elements);
    if n_fft > max_elements || padded_len.is_none() {
        return Err(AudioError::InvalidInput("STFT dimensions overflow".into()));
    }
    if let Some(len) = padded_len {
        let frames = 1 + len.saturating_sub(n_fft) / hop_length;
        if (n_fft / 2 + 1)
            .checked_mul(frames)
            .is_none_or(|n| n > max_elements)
        {
            return Err(AudioError::InvalidInput("STFT dimensions overflow".into()));
        }
    }
    let hop = hop_length;
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
            if !buffer[b].re.is_finite() || !buffer[b].im.is_finite() {
                return Err(AudioError::ComputationFailed(
                    "STFT exceeds finite f32 range".into(),
                ));
            }
            out[[b, t]] = buffer[b];
        }
        buffer.fill(Complex::new(0.0, 0.0));
    }

    Ok(out)
}

/// iSTFT builder.
#[derive(Debug, Clone)]
pub struct IstftBuilder<'a> {
    n_fft: Option<usize>,
    stft_matrix: &'a Array2<Complex<f32>>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Window,
    center: bool,
    length: Option<usize>,
}

impl IstftBuilder<'_> {
    /// Set the original FFT size. Required for odd-length FFTs; otherwise inferred
    /// as `2 * (n_bins - 1)`.
    #[must_use]
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = Some(n_fft);
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
    ///
    /// # Errors
    /// Rejects empty or incompatible spectra, nonfinite coefficients, invalid
    /// FFT/window/hop sizes, and overflowing reconstruction lengths.
    pub fn compute(self) -> Result<Vec<f32>, AudioError> {
        istft_impl(
            self.stft_matrix,
            self.n_fft,
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
/// Set [`IstftBuilder::n_fft`] explicitly for odd FFT sizes. Without a target
/// length, both centering pads are removed from the overlap-add output.
///
/// # Examples
/// ```
/// use dasp_rs::proc::stft;
/// let y = vec![0.0_f32; 4096];
/// let spec = stft(&y).n_fft(1024).hop_length(256).compute()?;
/// let recon = dasp_rs::proc::istft(&spec).hop_length(256).length(y.len()).compute()?;
/// assert_eq!(recon.len(), y.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn istft(stft_matrix: &Array2<Complex<f32>>) -> IstftBuilder<'_> {
    IstftBuilder {
        stft_matrix,
        n_fft: None,
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
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Window,
    center: bool,
    length: Option<usize>,
) -> Result<Vec<f32>, AudioError> {
    let n_bins = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];
    if n_bins < 2 || n_frames == 0 {
        return Err(AudioError::InvalidInput(
            "ISTFT requires at least two frequency bins and one frame".into(),
        ));
    }
    let n_fft = n_fft.unwrap_or((n_bins - 1) * 2);
    let win_length = win_length.unwrap_or(n_fft);
    let hop = hop_length.unwrap_or_else(|| (win_length / 4).max(1));
    if n_fft < 2 || n_fft / 2 + 1 != n_bins {
        return Err(AudioError::InvalidInput(
            "FFT size does not match spectrum".into(),
        ));
    }
    if win_length == 0 || win_length > n_fft || hop == 0 {
        return Err(AudioError::InvalidInput(
            "Require 0 < win_length <= n_fft and hop_length > 0".into(),
        ));
    }
    if stft_matrix
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(AudioError::InvalidInput("Spectrum must be finite".into()));
    }
    let full_len = hop
        .checked_mul(n_frames - 1)
        .and_then(|n| n.checked_add(n_fft))
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| AudioError::InvalidInput("Reconstruction length overflows".into()))?;
    let start_off = if center { n_fft / 2 } else { 0 };
    let target = length.unwrap_or_else(|| full_len.saturating_sub(2 * start_off));
    if target > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(AudioError::InvalidInput("Output length overflows".into()));
    }
    let win: Vec<f64> = fft_window(window, win_length, n_fft)
        .into_iter()
        .map(f64::from)
        .collect();

    // The unnormalized inverse and overlapping frames can exceed f32 even when
    // the normalized result is representable. Keep intermediates in f64.
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(n_fft);
    let inv_n = 1.0 / n_fft as f64;

    let mut signal = vec![0.0_f64; full_len];
    let mut window_sum = vec![0.0_f64; full_len];

    let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];
    for (frame_idx, frame) in stft_matrix.axis_iter(ndarray::Axis(1)).enumerate() {
        // Rebuild the conjugate-symmetric full spectrum.
        for b in 0..n_bins {
            buffer[b] = Complex::new(f64::from(frame[b].re), f64::from(frame[b].im));
        }
        for b in n_bins..n_fft {
            let value = frame[n_fft - b];
            buffer[b] = Complex::new(f64::from(value.re), -f64::from(value.im));
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
        // Small positive weights are valid near Hann endpoints. An absolute
        // epsilon discards recoverable samples instead of normalizing them.
        if w > f64::MIN_POSITIVE {
            *s /= w;
        }
    }

    let mut out = vec![0.0f32; target];
    for (i, o) in out.iter_mut().enumerate() {
        let idx = start_off.saturating_add(i);
        if idx < signal.len() {
            *o = signal[idx] as f32;
            if !o.is_finite() {
                return Err(AudioError::ComputationFailed(
                    "ISTFT exceeds finite f32 range".into(),
                ));
            }
        }
    }
    Ok(out)
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
pub fn magphase(
    d: &Array2<Complex<f32>>,
    power: Option<f32>,
) -> (Array2<f32>, Array2<Complex<f32>>) {
    let power_val = power.unwrap_or(1.0);
    let magnitude = d.mapv(|x| x.norm().powf(power_val));
    // The phase of zero is arbitrary; use zero radians so silence remains
    // finite and magnitude * phase reconstructs every input coefficient.
    let phase = d.mapv(|x| {
        let magnitude = x.norm();
        if magnitude == 0.0 {
            Complex::new(1.0, 0.0)
        } else {
            x / magnitude
        }
    });
    (magnitude, phase)
}

/// Computes a reassigned spectrogram for improved time-frequency resolution.
///
/// Returns a magnitude raster, not frequency/time coordinate arrays. Periodic
/// Hann analysis and its analytic derivative use reflect-centered frames with
/// hop `max(n_fft / 4, 1)`. Every nonzero coefficient contributes its magnitude
/// to the nearest frequency bin and time frame; estimates outside the grid are
/// clamped. Half-grid ties round upward, treating a `64 * f64::EPSILON` relative
/// neighborhood as an exact tie to stabilize transform roundoff.
///
/// Unlike earlier upward-frequency rasterization, nearest-grid assignment has
/// no systematic upward bias. Quiet coefficients are retained. Intermediate
/// transforms and accumulation use `f64`; unrepresentable `f32` output is an
/// error. Reassignment assumes one localized component per coefficient, so
/// overlapping components and reflected boundaries can produce mixed estimates.
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
    ReassignedSpectrogramBuilder { y, sr, n_fft: 2048 }
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
    if sr == 0 || n_fft < 2 {
        return Err(AudioError::InvalidInput(
            "Require positive sample rate and n_fft >= 2".into(),
        ));
    }
    let hop_length = (n_fft / 4).max(1);

    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!(
            "Signal too short: {} < {}",
            y.len(),
            n_fft
        )));
    }

    if y.iter().any(|v| !v.is_finite()) {
        return Err(AudioError::InvalidInput("Signal must be finite".into()));
    }
    let padded_len = y
        .len()
        .checked_add(2 * (n_fft / 2))
        .ok_or_else(|| AudioError::InvalidInput("Reassignment padding length overflows".into()))?;
    let frames = 1 + (padded_len - n_fft) / hop_length;
    if (n_fft / 2 + 1)
        .checked_mul(frames)
        .is_none_or(|n| n > isize::MAX as usize / std::mem::size_of::<Complex<f64>>())
    {
        return Err(AudioError::InvalidInput(
            "Reassignment dimensions overflow".into(),
        ));
    }
    let s = stft_with_derivative(y, n_fft, hop_length, ReassignmentWindow::Analysis);
    let s_time = stft_with_derivative(y, n_fft, hop_length, ReassignmentWindow::TimeWeighted);
    let s_freq = stft_with_derivative(y, n_fft, hop_length, ReassignmentWindow::Derivative);
    let mut reassigned = Array2::<f64>::zeros(s.dim());

    for t in 0..s.shape()[1] {
        for f in 0..s.shape()[0] {
            let mag = s[[f, t]].norm();
            if mag > 0.0 {
                // Work directly in grid coordinates: sample rate cancels from
                // both ratios, avoiding additional frequency/time rounding.
                let time_offset = (s_time[[f, t]] / s[[f, t]]).re;
                let t_reassigned = t as f64 + time_offset / hop_length as f64;
                let frequency_offset = (s_freq[[f, t]] / s[[f, t]]).im;
                let f_reassigned =
                    f as f64 - frequency_offset * n_fft as f64 / std::f64::consts::TAU;
                if !t_reassigned.is_finite() || !f_reassigned.is_finite() {
                    return Err(AudioError::ComputationFailed(
                        "Nonfinite reassignment coordinate".into(),
                    ));
                }
                let t_idx = nearest_reassignment_bin(t_reassigned, s.ncols());
                let f_idx = nearest_reassignment_bin(f_reassigned, s.nrows());
                reassigned[[f_idx, t_idx]] += mag;
            }
        }
    }

    let output = reassigned.mapv(|v| v as f32);
    if output.iter().any(|v| !v.is_finite()) {
        return Err(AudioError::ComputationFailed(
            "Reassigned magnitude exceeds finite f32 range".into(),
        ));
    }
    Ok(output)
}

/// Nearest grid coordinate, with stable upward half-grid ties.
fn nearest_reassignment_bin(coordinate: f64, size: usize) -> usize {
    let value = coordinate.clamp(0.0, (size - 1) as f64);
    let half_grid = (2.0 * value).round() / 2.0;
    let value = if (value - half_grid).abs() <= 64.0 * f64::EPSILON * value.abs().max(1.0) {
        half_grid
    } else {
        value
    };
    (value.round() as usize).min(size - 1)
}

/// Computes the Constant-Q Transform (CQT) of a signal.
///
/// Uses 12 bins per octave and centered periodic-Hann wavelets, with relative
/// bandwidth `(2^(2/12)-1)/(2^(2/12)+1)`, L1 normalization, and square-root
/// filter-length scaling. Frames are centered at `t * hop_length`, with zero
/// padding and shape `(n_bins, 1 + y.len() / hop_length)`.
///
/// This direct filterbank follows `librosa.filters.wavelet` conventions; it does
/// not reproduce the resampling and sparsification of `librosa.cqt`. Compared to
/// versions using one-period kernels, bandwidth, gain, phase, and edge padding
/// are corrected. All filter cutoff estimates must lie below Nyquist.
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

/// Builder for [`cqt`].
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

/// Compute the common CQT analysis size without saturating casts or overflow.
fn cqt_fft_size(sr: u32, hop: usize, fmin: f32) -> Result<usize, AudioError> {
    if sr == 0 || hop == 0 || !fmin.is_finite() || fmin <= 0.0 || fmin > sr as f32 / 2.0 {
        return Err(AudioError::InvalidInput(
            "Require positive sample rate/hop and finite 0 < fmin <= sr/2".into(),
        ));
    }
    let size = 2.0 * f64::from(sr) / f64::from(fmin);
    if size > f64::from(u32::MAX) {
        return Err(AudioError::InvalidInput("CQT FFT size overflows".into()));
    }
    (size.ceil() as u32)
        .checked_next_power_of_two()
        .map(|n| n as usize)
        .ok_or_else(|| AudioError::InvalidInput("CQT FFT size overflows".into()))
}

// Geometric relative bandwidth used by librosa.filters.wavelet_lengths (12 bins/octave).
fn wavelet_alpha() -> f64 {
    let ratio = 2.0_f64.powf(2.0 / 12.0);
    (ratio - 1.0) / (ratio + 1.0)
}

struct Wavelet {
    values: Vec<Complex<f64>>,
    // Offset within an even, center-padded frame. Odd filters have their extra
    // sample on the left, matching librosa.util.pad_center.
    left: usize,
}

fn wavelet_bank(
    sr: u32,
    hop: usize,
    fmin: f32,
    bins: usize,
    gamma: f64,
    samples: usize,
    frames: usize,
) -> Result<Vec<Wavelet>, AudioError> {
    if sr == 0 || hop == 0 || bins == 0 || !fmin.is_finite() || fmin <= 0.0 {
        return Err(AudioError::InvalidInput(
            "Require positive sample rate, hop, bins, and finite positive fmin".into(),
        ));
    }
    let last_center = frames
        .checked_sub(1)
        .and_then(|last| last.checked_mul(hop))
        .filter(|&last| samples > 0 && last <= samples)
        .ok_or_else(|| {
            AudioError::InvalidInput("Wavelet frame geometry overflows or is empty".into())
        })?;
    let alpha = wavelet_alpha();
    let highest = f64::from(fmin) * 2.0_f64.powf((bins - 1) as f64 / 12.0);
    // Hann equivalent noise bandwidth is 1.5 bins. Use half that bandwidth
    // as the cutoff margin, following librosa.filters.wavelet_lengths.
    if !highest.is_finite() || highest * (1.0 + 0.75 * alpha) + gamma / 2.0 > f64::from(sr) / 2.0 {
        return Err(AudioError::InvalidInput(
            "Wavelet passband exceeds Nyquist".into(),
        ));
    }
    let longest = f64::from(sr) / (alpha * f64::from(fmin) + gamma);
    if !longest.is_finite() || longest > f64::from(u32::MAX) {
        return Err(AudioError::InvalidInput("Wavelet length overflows".into()));
    }
    let mut bank = Vec::with_capacity(bins);
    for k in 0..bins {
        let frequency = f64::from(fmin) * 2.0_f64.powf(k as f64 / 12.0);
        let length = f64::from(sr) / (alpha * frequency + gamma);
        // np.arange(-length // 2, length // 2) uses floor division, including
        // its negative start. Preserve this phase convention for odd lengths.
        let first = (-length / 2.0).floor();
        let size = ((length / 2.0).floor() - first) as usize;
        let left = size.div_ceil(2);
        // Across all frames, only this union of full-kernel taps can overlap
        // actual samples. Preserve their original indices for phase/windowing.
        // In particular, a tiny input must not allocate an entire low-frequency
        // wavelet that may span hundreds of millions of zero-padded samples.
        let start = left.saturating_sub(last_center);
        let end = left
            .checked_add(samples)
            .ok_or_else(|| AudioError::InvalidInput("Wavelet support geometry overflows".into()))?
            .min(size);
        let stored_len = end
            .checked_sub(start)
            .filter(|&len| len <= isize::MAX as usize / std::mem::size_of::<Complex<f64>>())
            .ok_or_else(|| AudioError::InvalidInput("Wavelet storage geometry overflows".into()))?;
        let mut values = Vec::with_capacity(stored_len);
        // The full periodic Hann L1 norm is N/2 (unity for N=1). Normalizing
        // by the retained taps would incorrectly amplify short/edge frames.
        let normalizer = if size == 1 { 1.0 } else { size as f64 / 2.0 };
        let scale = length.sqrt() / normalizer;
        for i in start..end {
            // A singleton periodic Hann window is unity (SciPy/librosa),
            // not the zero obtained by substituting i=0 in the general formula.
            let window = if size == 1 {
                1.0
            } else {
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos()
            };
            let phase = 2.0 * std::f64::consts::PI * frequency * (first + i as f64) / f64::from(sr);
            values.push(Complex::from_polar(window, phase) * scale);
        }
        bank.push(Wavelet {
            values,
            left: left - start,
        });
    }
    Ok(bank)
}

fn wavelet_analysis(
    y: &[f64],
    bank: &[Wavelet],
    hop: usize,
    frames: usize,
) -> Array2<Complex<f64>> {
    let mut out = Array2::zeros((bank.len(), frames));
    for (k, wavelet) in bank.iter().enumerate() {
        for t in 0..frames {
            let center = t * hop;
            let begin = wavelet.left.saturating_sub(center);
            let end = wavelet
                .values
                .len()
                .min(y.len().saturating_add(wavelet.left).saturating_sub(center));
            let mut value = Complex::new(0.0, 0.0);
            for i in begin..end {
                value += wavelet.values[i].conj() * y[center + i - wavelet.left];
            }
            out[[k, t]] = value;
        }
    }
    out
}

// Real adjoint of the complex analysis operator, with the same zero padding
// and sample origins. This is also used by the least-squares inverse below.
fn wavelet_adjoint(
    c: &Array2<Complex<f64>>,
    bank: &[Wavelet],
    hop: usize,
    samples: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; samples];
    for (k, wavelet) in bank.iter().enumerate() {
        for t in 0..c.ncols() {
            let center = t * hop;
            let begin = wavelet.left.saturating_sub(center);
            let end = wavelet
                .values
                .len()
                .min(samples.saturating_add(wavelet.left).saturating_sub(center));
            for i in begin..end {
                out[center + i - wavelet.left] += (wavelet.values[i] * c[[k, t]]).re;
            }
        }
    }
    out
}

fn wavelet_transform(
    y: &[f32],
    sr: u32,
    hop: usize,
    fmin: f32,
    bins: usize,
    gamma: f64,
) -> Result<Array2<Complex<f32>>, AudioError> {
    if y.is_empty() || y.iter().any(|v| !v.is_finite()) {
        return Err(AudioError::InvalidInput(
            "Signal must be nonempty and finite".into(),
        ));
    }
    let frames = y
        .len()
        .checked_div(hop)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| {
            AudioError::InvalidInput("Require positive hop and nonoverflowing frame count".into())
        })?;
    if bins
        .checked_mul(frames)
        .is_none_or(|n| n > isize::MAX as usize / std::mem::size_of::<Complex<f64>>())
    {
        return Err(AudioError::InvalidInput(
            "Wavelet output dimensions overflow".into(),
        ));
    }
    let bank = wavelet_bank(sr, hop, fmin, bins, gamma, y.len(), frames)?;
    let input: Vec<f64> = y.iter().copied().map(f64::from).collect();
    let output = wavelet_analysis(&input, &bank, hop, frames)
        .mapv(|z| Complex::new(z.re as f32, z.im as f32));
    if output
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(AudioError::ComputationFailed(
            "Wavelet transform exceeds finite f32 range".into(),
        ));
    }
    Ok(output)
}

/// Internal CQT implementation.
fn cqt_impl(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_bins: usize,
) -> Result<Array2<Complex<f32>>, AudioError> {
    wavelet_transform(y, sr, hop_length, fmin, n_bins, 0.0)
}

/// Approximately inverts the centered wavelet CQT by real least squares.
///
/// Runs up to 32 conjugate-gradient iterations from zero using the exact analysis
/// operator and its adjoint. Returns `n_frames * hop_length` samples, preserving
/// the historical output length. The original length cannot be inferred from a
/// CQT matrix; trim the result when known. Frequencies outside the analyzed bands
/// and information lost to the hop cannot be recovered. This is not an exact
/// inverse and does not implement librosa's recursive iCQT. Cost grows with the
/// total wavelet support, frame count, and iteration count. Each candidate is
/// checked after f32 conversion; the solver retains improving finite candidates.
/// Ill-conditioned or inconsistent spectra can still amplify waveform magnitude
/// while reducing coefficient residual: this guard is not regularization.
///
/// # Arguments
/// * `c` - CQT spectrogram as an `Array2<Complex<f32>>`.
///
/// Configure sample rate, hop length, and minimum frequency on the returned builder.
///
/// # Returns
/// Returns a builder whose `compute()` method produces the reconstructed signal.
///
/// # Errors
/// The builder rejects empty/nonfinite spectra, invalid analysis parameters,
/// passbands above Nyquist, overflowing lengths, and nonfinite reconstruction.
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
    IcqtBuilder {
        c,
        sr: 44100,
        hop_length: 512,
        fmin: 32.70,
    }
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
        icqt_impl(
            self.c,
            Some(self.sr),
            Some(self.hop_length),
            Some(self.fmin),
        )
    }
}

fn icqt_impl(
    c: &Array2<Complex<f32>>,
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
) -> Result<Vec<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    if c.is_empty() || c.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(AudioError::InvalidInput(
            "CQT spectrum must be nonempty and finite".into(),
        ));
    }
    let samples = c
        .ncols()
        .checked_mul(hop)
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| AudioError::InvalidInput("CQT output length overflows".into()))?;
    let bank = wavelet_bank(sr, hop, fmin, c.nrows(), 0.0, samples, c.ncols())?;
    let target = c.mapv(|z| Complex::new(f64::from(z.re), f64::from(z.im)));
    let mut coefficient_residual = target;
    let mut gradient = wavelet_adjoint(&coefficient_residual, &bank, hop, samples);
    let mut direction = gradient.clone();
    let mut iterate = vec![0.0; samples];
    let mut best = vec![0.0_f32; samples];
    let zero_cost: f64 = coefficient_residual.iter().map(Complex::norm_sqr).sum();
    let mut best_cost = zero_cost;
    let mut squared: f64 = gradient.iter().map(|v| v * v).sum();
    let initial = squared;
    // CGLS solves the real least-squares problem without forming A* A. Computing
    // curvature as ||A p||² avoids cancellation in p dot (A* A p). The complex
    // coefficient residual is updated directly, then mapped back by the adjoint.
    for _ in 0..32 {
        if squared <= initial * 1e-20 || squared == 0.0 {
            break;
        }
        let projected = wavelet_analysis(&direction, &bank, hop, c.ncols());
        let curvature: f64 = projected.iter().map(Complex::norm_sqr).sum();
        if curvature <= 0.0 || !curvature.is_finite() {
            break;
        }
        let step = squared / curvature;
        for i in 0..samples {
            iterate[i] += step * direction[i];
        }
        let candidate: Vec<f32> = iterate.iter().map(|&v| v as f32).collect();
        if candidate.iter().any(|v| !v.is_finite()) {
            if best_cost < zero_cost {
                break;
            }
            return Err(AudioError::ComputationFailed(
                "Inverse CQT exceeds finite f32 range".into(),
            ));
        }
        // Near-null directions can have huge coefficients that cancel in f64
        // but not after conversion to the public f32 waveform. Verify the actual
        // representable candidate, retaining zero as the initial fallback.
        let represented: Vec<f64> = candidate.iter().copied().map(f64::from).collect();
        let rebuilt = wavelet_analysis(&represented, &bank, hop, c.ncols());
        let cost: f64 = rebuilt
            .iter()
            .zip(c)
            .map(|(actual, target)| {
                (*actual - Complex::new(f64::from(target.re), f64::from(target.im))).norm_sqr()
            })
            .sum();
        if !cost.is_finite() || cost >= best_cost {
            break;
        }
        best = candidate;
        best_cost = cost;
        for (residual, projected) in coefficient_residual.iter_mut().zip(&projected) {
            *residual -= step * projected;
        }
        gradient = wavelet_adjoint(&coefficient_residual, &bank, hop, samples);
        let next: f64 = gradient.iter().map(|v| v * v).sum();
        let beta = next / squared;
        for i in 0..samples {
            direction[i] = gradient[i] + beta * direction[i];
        }
        squared = next;
    }
    Ok(best)
}

/// Computes the same direct filterbank as [`cqt`] with 84 bins.
///
/// Retained as a compatibility entry point; no recursive/hybrid acceleration is
/// performed, and this is not librosa's magnitude-only hybrid CQT.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Sample rate in Hz.
///
/// Configure hop length and minimum frequency on the returned builder.
///
/// # Returns
/// Returns a builder whose `compute()` method produces the CQT matrix.
///
/// # Errors
/// The builder rejects empty/nonfinite signals, invalid analysis parameters,
/// passbands above Nyquist, and outputs outside finite `f32` range.
///
/// # Examples
/// ```no_run
/// use dasp_rs::proc::hybrid_cqt;
/// let signal = vec![0.0_f32; 4096];
/// let hybrid = hybrid_cqt(&signal, 44100).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn hybrid_cqt(y: &[f32], sr: u32) -> HybridCqtBuilder<'_> {
    HybridCqtBuilder {
        y,
        sr,
        hop_length: 512,
        fmin: 32.70,
    }
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
        hybrid_cqt_impl(
            self.y,
            Some(self.sr),
            Some(self.hop_length),
            Some(self.fmin),
        )
    }
}

fn hybrid_cqt_impl(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    cqt_impl(
        y,
        sr.unwrap_or(44100),
        hop_length.unwrap_or(512),
        fmin.unwrap_or(32.70),
        84,
    )
}

/// Samples a complex STFT at 84 logarithmically spaced frequencies.
///
/// Each channel takes the first FFT bin at or above its requested frequency.
/// This compatibility approximation has STFT bandwidth and normalization; it is
/// not the wavelet magnitude projection implemented by `librosa.pseudo_cqt`.
/// Requested channel centers must not exceed Nyquist.
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
    PseudoCqtBuilder {
        y,
        sr,
        hop_length: 512,
        fmin: 32.70,
    }
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
        pseudo_cqt_impl(
            self.y,
            Some(self.sr),
            Some(self.hop_length),
            Some(self.fmin),
        )
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
    let n_fft = cqt_fft_size(sr, hop_length, fmin)?;
    let n_bins = 84;
    if f64::from(fmin) * 2.0_f64.powf((n_bins - 1) as f64 / 12.0) > f64::from(sr) / 2.0 {
        return Err(AudioError::InvalidInput(
            "Pseudo-CQT frequencies exceed Nyquist".into(),
        ));
    }

    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!(
            "Signal too short: {} < {}",
            y.len(),
            n_fft
        )));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput(
            "fmin must be positive".to_string(),
        ));
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

/// Computes a centered Variable-Q Transform (VQT) of a signal.
///
/// Uses the same filter, phase, normalization, and padding conventions as [`cqt`],
/// with bandwidth `alpha * frequency + 24 Hz`. The fixed 24 Hz offset differs
/// from librosa's default ERB-derived offset; compare with `librosa.vqt(gamma=24)`.
/// Direct full-rate analysis omits recursive resampling and sparsification.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Sample rate in Hz.
///
/// Configure hop length, minimum frequency, and bin count on the returned builder.
///
/// # Returns
/// Returns a builder whose `compute()` method produces the VQT matrix.
///
/// # Errors
/// The builder rejects empty/nonfinite signals, invalid analysis parameters,
/// passbands above Nyquist, and outputs outside finite `f32` range.
///
/// # Examples
/// ```no_run
/// use dasp_rs::proc::vqt;
/// let signal = vec![0.0_f32; 4096];
/// let vqt_result = vqt(&signal, 44100).n_bins(84).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn vqt(y: &[f32], sr: u32) -> VqtBuilder<'_> {
    VqtBuilder {
        y,
        sr,
        hop_length: 512,
        fmin: 32.70,
        n_bins: 84,
    }
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
        vqt_impl(
            self.y,
            Some(self.sr),
            Some(self.hop_length),
            Some(self.fmin),
            Some(self.n_bins),
        )
    }
}

fn vqt_impl(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    n_bins: Option<usize>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    wavelet_transform(
        y,
        sr.unwrap_or(44100),
        hop_length.unwrap_or(512),
        fmin.unwrap_or(32.70),
        n_bins.unwrap_or(84),
        24.0,
    )
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
    FmtBuilder {
        y,
        t_min: 0.005,
        n_fmt: 5,
        kind: "cos",
        beta: 2.0,
    }
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
        fmt_impl(
            self.y,
            Some(self.t_min),
            Some(self.n_fmt),
            Some(self.kind),
            Some(self.beta),
        )
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
        return Err(AudioError::InsufficientData(format!(
            "Signal too short: {} < {}",
            y.len(),
            hop_length
        )));
    }
    if t_min <= 0.0 {
        return Err(AudioError::InvalidInput(
            "t_min must be positive".to_string(),
        ));
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
pub(crate) fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()))
        .collect()
}

#[derive(Clone, Copy)]
enum ReassignmentWindow {
    Analysis,
    TimeWeighted,
    Derivative,
}

/// Consistent periodic-Hann transforms for validated reassignment inputs.
/// Double arithmetic avoids overflow/underflow of the complex ratios for finite
/// f32 signals. The time origin is sample `n_fft/2`, including odd FFT lengths.
fn stft_with_derivative(
    y: &[f32],
    n_fft: usize,
    hop_length: usize,
    kind: ReassignmentWindow,
) -> Array2<Complex<f64>> {
    let padded = reflect_pad(y, n_fft / 2);
    let n_frames = (padded.len() - n_fft) / hop_length + 1;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut s = Array2::zeros((n_fft / 2 + 1, n_frames));
    let window: Vec<f64> = (0..n_fft)
        .map(|i| {
            let phase = std::f64::consts::TAU * i as f64 / n_fft as f64;
            let hann = 0.5 - 0.5 * phase.cos();
            match kind {
                ReassignmentWindow::Analysis => hann,
                ReassignmentWindow::TimeWeighted => (i as f64 - (n_fft / 2) as f64) * hann,
                ReassignmentWindow::Derivative => std::f64::consts::PI / n_fft as f64 * phase.sin(),
            }
        })
        .collect();
    let mut buffer = vec![Complex::new(0.0, 0.0); n_fft];

    for t in 0..n_frames {
        let start = t * hop_length;
        let frame = &padded[start..start + n_fft];
        for ((slot, &sample), &weight) in buffer.iter_mut().zip(frame).zip(&window) {
            *slot = Complex::new(f64::from(sample) * weight, 0.0);
        }
        fft.process(&mut buffer);
        for f in 0..=(n_fft / 2) {
            s[[f, t]] = buffer[f];
        }
    }
    s
}

/// Designs a Butterworth bandpass as real second-order sections in f64.
/// `order` is the lowpass prototype order; the bandpass order is twice that.
fn butterworth_bandpass(
    lowcut: f64,
    highcut: f64,
    fs: f64,
    order: Option<usize>,
) -> Result<Vec<[f64; 6]>, AudioError> {
    let order = order.unwrap_or(2);
    if !lowcut.is_finite()
        || !highcut.is_finite()
        || !fs.is_finite()
        || fs <= 0.0
        || lowcut <= 0.0
        || highcut <= lowcut
        || highcut >= fs / 2.0
        || order == 0
        || order > isize::MAX as usize / std::mem::size_of::<[f64; 6]>()
    {
        return Err(AudioError::InvalidInput(
            "Require finite 0 < lowcut < highcut < fs/2 and positive filter order".into(),
        ));
    }
    let pi = std::f64::consts::PI;
    let fs2 = 2.0 * fs;
    let low = fs2 * (pi * lowcut / fs).tan();
    let high = fs2 * (pi * highcut / fs).tan();
    let center = (low * high).sqrt();
    let bandwidth = high - low;
    let center_radians = 2.0 * (center / fs2).atan();
    let mut sections = Vec::with_capacity(order);
    for k in 0..order {
        // Unit Butterworth poles lie in the left half-plane. The lowpass to
        // bandpass substitution solves s² - bandwidth*p*s + center² = 0;
        // independently scaling real/imaginary parts does not implement it.
        let angle = pi * (2.0 * k as f64 + 1.0 + order as f64) / (2.0 * order as f64);
        let half = Complex::from_polar(bandwidth / 2.0, angle);
        let root = (half * half - Complex::new(center * center, 0.0)).sqrt();
        for analog in [half + root, half - root] {
            if analog.im <= 0.0 {
                continue;
            }
            let pole = (fs2 + analog) / (fs2 - analog);
            if !pole.re.is_finite() || !pole.im.is_finite() || pole.norm_sqr() >= 1.0 {
                return Err(AudioError::ComputationFailed(
                    "Butterworth poles are not representable inside the unit circle".into(),
                ));
            }
            // Each section gets one zero at DC and one at Nyquist. Distribute
            // the overall gain so each section has unit gain at warped center.
            let mut section = [1.0, 0.0, -1.0, 1.0, -2.0 * pole.re, pole.norm_sqr()];
            let gain = evaluate_filter(&[section], center_radians).norm();
            if !gain.is_finite() || gain <= 0.0 {
                return Err(AudioError::ComputationFailed(
                    "Butterworth section gain is not finite and positive".into(),
                ));
            }
            section[0] /= gain;
            section[2] /= gain;
            sections.push(section);
        }
    }
    if sections.len() != order {
        return Err(AudioError::ComputationFailed(
            "Butterworth pole pairing failed".into(),
        ));
    }
    // Smaller pole radii first reduces intermediate resonant amplification.
    sections.sort_by(|a, b| a[5].total_cmp(&b[5]));
    Ok(sections)
}

/// Evaluates the complete SOS transfer function without expanding polynomials.
fn evaluate_filter(sections: &[[f64; 6]], radians: f64) -> Complex<f64> {
    let z = Complex::from_polar(1.0, -radians);
    let z2 = z * z;
    sections.iter().fold(Complex::new(1.0, 0.0), |response, s| {
        response * (s[0] + s[1] * z + s[2] * z2) / (s[3] + s[4] * z + s[5] * z2)
    })
}

/// Computes a causal twelve-band semitone IIR filterbank and frame RMS.
///
/// Band `b` has center `32.7 * 2^(b/12)` Hz for `b = 0..11` (C1 through B1).
/// Half-semitone cutoff frequencies define fourth-order Butterworth prototypes,
/// producing eighth-order bandpasses represented by four f64 second-order
/// sections. All twelve passbands must fit strictly below Nyquist.
///
/// Filters process the complete input continuously from zero initial state.
/// Frame `t` starts at `t * hop_length`; no centering or padding is applied.
/// Each output is `sqrt(mean(filtered_frame²))`, with shape
/// `(12, 1 + (y.len() - win_length) / hop_length)`. Narrow low-frequency filters
/// can take seconds to settle: early frames retain the causal startup transient.
///
/// This corrects older octave-spaced bands, unstable f32 polynomial filtering,
/// state resets at every frame, and an extra square-root-window gain penalty.
/// It is not librosa's multirate, zero-phase, mean-square-power IIRT.
///
/// # Arguments
/// * `y` - Nonempty finite input signal.
/// * `sr` - Sample rate in Hz.
///
/// Configure window and hop lengths on the returned builder.
///
/// # Errors
/// The builder rejects zero sample rate/window/hop, nonfinite input, passbands
/// above Nyquist, overflowing dimensions, and inputs shorter than one window.
///
/// # Examples
/// ```
/// use dasp_rs::proc::iirt;
/// let signal = vec![0.0_f32; 8192];
/// let bands = iirt(&signal, 44100).compute()?;
/// assert_eq!(bands.nrows(), 12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn iirt(y: &[f32], sr: u32) -> IirtBuilder<'_> {
    IirtBuilder {
        y,
        sr,
        win_length: 2048,
        hop_length: None,
    }
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

    /// Set the hop length (default: `max(1, win_length / 4)`).
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
        iirt_impl(
            self.y,
            Some(self.sr),
            Some(self.win_length),
            self.hop_length,
        )
    }
}

fn iirt_impl(
    y: &[f32],
    sr: Option<u32>,
    win_length: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array2<f32>, AudioError> {
    const BANDS: usize = 12;
    let sr = sr.unwrap_or(44100);
    let win_length = win_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or_else(|| (win_length / 4).max(1));
    if sr == 0 || win_length == 0 || hop == 0 || y.is_empty() || y.iter().any(|v| !v.is_finite()) {
        return Err(AudioError::InvalidInput(
            "Require positive sample rate/window/hop and nonempty finite input".into(),
        ));
    }
    if y.len() < win_length {
        return Err(AudioError::InsufficientData(format!(
            "Signal too short: {} < {}",
            y.len(),
            win_length
        )));
    }
    let frames = 1 + (y.len() - win_length) / hop;
    if BANDS
        .checked_mul(frames)
        .is_none_or(|n| n > isize::MAX as usize / std::mem::size_of::<f32>())
    {
        return Err(AudioError::InvalidInput(
            "IIR output dimensions overflow".into(),
        ));
    }
    let edge_ratio = 2.0_f64.powf(1.0 / 24.0);
    let highest = 32.7 * 2.0_f64.powf((BANDS - 1) as f64 / 12.0) * edge_ratio;
    if highest >= f64::from(sr) / 2.0 {
        return Err(AudioError::InvalidInput(
            "The twelve IIR semitone passbands must lie below Nyquist".into(),
        ));
    }
    let mut result = Array2::zeros((BANDS, frames));
    for band in 0..BANDS {
        let center = 32.7 * 2.0_f64.powf(band as f64 / 12.0);
        let sections = butterworth_bandpass(
            center / edge_ratio,
            center * edge_ratio,
            f64::from(sr),
            Some(4),
        )?;
        let filtered = filter(y, &sections);
        for t in 0..frames {
            let start = t * hop;
            let energy: f64 = filtered[start..start + win_length]
                .iter()
                .map(|v| v * v)
                .sum();
            let rms = (energy / win_length as f64).sqrt() as f32;
            if !rms.is_finite() {
                return Err(AudioError::ComputationFailed(
                    "IIR frame RMS exceeds finite f32 range".into(),
                ));
            }
            result[[band, t]] = rms;
        }
    }
    Ok(result)
}

/// Causal direct-form-II-transposed SOS filtering from zero initial state.
fn filter(input: &[f32], sections: &[[f64; 6]]) -> Vec<f64> {
    let mut output: Vec<f64> = input.iter().copied().map(f64::from).collect();
    for s in sections {
        let (mut delay1, mut delay2) = (0.0, 0.0);
        for sample in &mut output {
            let value = s[0] * *sample + delay1;
            delay1 = s[1] * *sample - s[4] * value + delay2;
            delay2 = s[2] * *sample - s[5] * value;
            *sample = value;
        }
    }
    output
}

// ─── Phasor ───────────────────────────────────────────────────────────────────

/// Constructs a complex spectrogram from polar-form phase angles and optional magnitudes.
///
/// Each element `[i, j]` of the returned array equals `magnitude * e^{i·angle}`.
/// When `mag` is `None` all magnitudes are `1.0` (unit phasors).
///
/// Commonly used to reconstruct a complex STFT from a phase matrix returned by
/// [`magphase`] and a new target magnitude spectrogram.
///
/// # Arguments
/// * `angles` — Phase angle matrix in radians, shape `(n_bins, n_frames)`.
/// * `mag` — Optional magnitude matrix with the same shape as `angles`.
///
/// # Examples
/// ```
/// use dasp_rs::proc::phasor;
/// use ndarray::Array2;
/// use std::f32::consts::PI;
/// let angles = Array2::from_elem((5, 10), PI / 2.0);
/// let c = phasor(&angles, None);
/// // Each element ≈ e^{iπ/2} = i
/// assert!(c[[0, 0]].re.abs() < 1e-5);
/// assert!((c[[0, 0]].im - 1.0).abs() < 1e-5);
/// ```
pub fn phasor(
    angles: &Array2<f32>,
    mag: Option<&Array2<f32>>,
) -> Array2<num_complex::Complex<f32>> {
    Array2::from_shape_fn(angles.raw_dim(), |(i, j)| {
        let a = angles[[i, j]];
        let m = mag.map_or(1.0, |m| m[[i, j]]);
        num_complex::Complex::from_polar(m, a)
    })
}

#[cfg(test)]
#[path = "../../tests/common/reference.rs"]
mod iir_reference;

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
        let recon = istft(&spec)
            .hop_length(64)
            .length(y.len())
            .compute()
            .unwrap();
        assert_eq!(recon.len(), y.len());

        let (mut err, mut cnt) = (0.0_f32, 0usize);
        for i in 256..(y.len() - 256) {
            err += (recon[i] - y[i]).abs();
            cnt += 1;
        }
        assert!(
            (err / cnt as f32) < 1e-3,
            "mean abs reconstruction error too high"
        );
    }

    #[test]
    fn stft_locates_sinusoid_peak_bin() {
        // A pure tone at bin k should produce its maximum magnitude at row k.
        let n_fft = 256;
        let k = 10usize;
        let y: Vec<f32> = (0..4096)
            .map(|n| (2.0 * PI * k as f32 * n as f32 / n_fft as f32).sin())
            .collect();
        let spec = stft(&y)
            .n_fft(n_fft)
            .hop_length(64)
            .compute()
            .expect("stft");
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
    fn iir_design_and_impulse_match_independent_scipy_sos() {
        use super::iir_reference::{check_complex, check_real, load, signal};
        let data = load(include_str!("../../tests/fixtures/iir_reference.csv"));
        let indices: Vec<_> = signal(&data, "impulse_indices")
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let mut impulse = vec![0.0; 8192];
        impulse[0] = 1.0;
        for name in ["narrow", "wide", "nyquist", "odd_order"] {
            let config = &data[&format!("{name}_config")];
            let sr = config[[0, 0]].re;
            let sections = butterworth_bandpass(
                config[[0, 1]].re,
                config[[0, 2]].re,
                sr,
                Some(config[[0, 3]].re as usize),
            )
            .unwrap();
            let responses: Vec<_> = data[&format!("{name}_frequencies")]
                .iter()
                .map(|f| {
                    let z = evaluate_filter(&sections, 2.0 * std::f64::consts::PI * f.re / sr);
                    Complex::new(z.re as f32, z.im as f32)
                })
                .collect();
            check_complex(
                &format!("{name}_response"),
                &Array2::from_shape_vec((1, responses.len()), responses).unwrap(),
                &data,
                2e-7,
                2e-6,
            );
            let filtered = filter(&impulse, &sections);
            assert!(filtered.iter().all(|v| v.is_finite()));
            let selected: Vec<_> = indices.iter().map(|&i| filtered[i] as f32).collect();
            check_real(
                &format!("{name}_impulse"),
                &Array2::from_shape_vec((1, selected.len()), selected).unwrap(),
                &data,
                1e-12,
                2e-6,
            );
        }
    }

    #[test]
    fn iir_design_rejects_invalid_parameters() {
        for (low, high, sr, order) in [
            (0.0, 100.0, 8000.0, 4),
            (100.0, 100.0, 8000.0, 4),
            (100.0, 4000.0, 8000.0, 4),
            (f64::NAN, 100.0, 8000.0, 4),
            (1.0, 100.0, f64::INFINITY, 4),
            (1.0, 100.0, 8000.0, 0),
        ] {
            assert!(butterworth_bandpass(low, high, sr, Some(order)).is_err());
        }
    }
}
