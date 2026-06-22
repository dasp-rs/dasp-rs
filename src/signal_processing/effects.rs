//! Audio effects: time stretching, pitch shifting, silence trimming and splitting.

use crate::features::harmonics::phase_vocoder;
use crate::signal_processing::resampling::resample;
use crate::signal_processing::time_frequency::{istft, stft};
use thiserror::Error;

/// Error conditions for audio effects operations.
#[derive(Error, Debug)]
pub enum EffectsError {
    /// Invalid parameter value (e.g., non-positive rate, empty signal).
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// An underlying DSP operation (STFT, phase vocoder, resampler) failed.
    #[error("DSP error: {0}")]
    Dsp(String),
}

// ─── Time stretch ─────────────────────────────────────────────────────────────

/// Builder for [`time_stretch`].
#[derive(Debug, Clone)]
pub struct TimeStretchBuilder<'a> {
    y: &'a [f32],
    rate: f32,
    n_fft: usize,
    hop_length: usize,
}

impl TimeStretchBuilder<'_> {
    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, v: usize) -> Self {
        self.n_fft = v;
        self
    }

    /// Set the hop length between frames (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Compute the time-stretched signal.
    ///
    /// # Errors
    /// Returns an error if the signal is empty, the rate is non-positive, or a
    /// time-frequency operation fails.
    pub fn compute(self) -> Result<Vec<f32>, EffectsError> {
        time_stretch_impl(self.y, self.rate, self.n_fft, self.hop_length)
    }
}

/// Stretches or compresses a signal in time without altering pitch.
///
/// Uses the phase vocoder. `rate > 1` speeds up playback (shorter output),
/// `rate < 1` slows down (longer output).
///
/// # Examples
/// ```no_run
/// use dasp_rs::proc::time_stretch;
/// let y = vec![0.0_f32; 44100];
/// let slow = time_stretch(&y, 0.5).compute()?; // 2× slower
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn time_stretch(y: &[f32], rate: f32) -> TimeStretchBuilder<'_> {
    TimeStretchBuilder { y, rate, n_fft: 2048, hop_length: 512 }
}

fn time_stretch_impl(
    y: &[f32],
    rate: f32,
    n_fft: usize,
    hop_length: usize,
) -> Result<Vec<f32>, EffectsError> {
    if y.is_empty() {
        return Err(EffectsError::InvalidParameter("Signal is empty".into()));
    }
    if !rate.is_finite() || rate <= 0.0 {
        return Err(EffectsError::InvalidParameter(
            "rate must be a positive finite number".into(),
        ));
    }
    if n_fft == 0 || hop_length == 0 {
        return Err(EffectsError::InvalidParameter(
            "n_fft and hop_length must be positive".into(),
        ));
    }
    let spec = stft(y)
        .n_fft(n_fft)
        .hop_length(hop_length)
        .compute()
        .map_err(|e| EffectsError::Dsp(e.to_string()))?;
    let stretched = phase_vocoder(&spec, rate, Some(hop_length), Some(n_fft))
        .map_err(|e| EffectsError::Dsp(e.to_string()))?;
    Ok(istft(&stretched).hop_length(hop_length).compute())
}

// ─── Pitch shift ──────────────────────────────────────────────────────────────

/// Builder for [`pitch_shift`].
#[derive(Debug, Clone)]
pub struct PitchShiftBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    n_steps: f32,
    n_fft: usize,
    hop_length: usize,
}

impl PitchShiftBuilder<'_> {
    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, v: usize) -> Self {
        self.n_fft = v;
        self
    }

    /// Set the hop length between frames (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Compute the pitch-shifted signal.
    ///
    /// # Errors
    /// Returns an error if the signal is empty or a DSP operation fails.
    pub fn compute(self) -> Result<Vec<f32>, EffectsError> {
        pitch_shift_impl(self.y, self.sr, self.n_steps, self.n_fft, self.hop_length)
    }
}

/// Shifts the pitch of a signal by `n_steps` semitones.
///
/// Positive values raise the pitch; negative values lower it.
/// The output has the same length and sample rate as the input.
///
/// # Examples
/// ```no_run
/// use dasp_rs::proc::pitch_shift;
/// let y = vec![0.0_f32; 44100];
/// let up_octave = pitch_shift(&y, 44100, 12.0).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn pitch_shift(y: &[f32], sr: u32, n_steps: f32) -> PitchShiftBuilder<'_> {
    PitchShiftBuilder { y, sr, n_steps, n_fft: 2048, hop_length: 512 }
}

fn pitch_shift_impl(
    y: &[f32],
    sr: u32,
    n_steps: f32,
    n_fft: usize,
    hop_length: usize,
) -> Result<Vec<f32>, EffectsError> {
    if !n_steps.is_finite() {
        return Err(EffectsError::InvalidParameter("n_steps must be finite".into()));
    }
    if sr == 0 {
        return Err(EffectsError::InvalidParameter("sr must be positive".into()));
    }
    // rate < 1 → expand time; we'll restore duration via resampling
    let rate = 2.0_f32.powf(-n_steps / 12.0);
    let stretched = time_stretch_impl(y, rate, n_fft, hop_length)?;

    // Interpreting the stretched signal as recorded at fake_sr, then resampling
    // to sr is mathematically equivalent to shifting all frequencies by 2^(n_steps/12).
    let fake_sr = ((sr as f32) / rate).round() as u32;
    let fake_sr = fake_sr.max(1);
    let mut shifted = resample(&stretched, fake_sr, sr)
        .map_err(|e| EffectsError::Dsp(e.to_string()))?;

    shifted.truncate(y.len());
    shifted.resize(y.len(), 0.0);
    Ok(shifted)
}

// ─── Trim ─────────────────────────────────────────────────────────────────────

/// Builder for [`trim`].
#[derive(Debug, Clone)]
pub struct TrimBuilder<'a> {
    y: &'a [f32],
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
}

impl TrimBuilder<'_> {
    /// Set the silence threshold in dB below the peak RMS (default: 60.0).
    #[must_use]
    pub fn top_db(mut self, v: f32) -> Self {
        self.top_db = v;
        self
    }

    /// Set the RMS frame length in samples (default: 2048).
    #[must_use]
    pub fn frame_length(mut self, v: usize) -> Self {
        self.frame_length = v;
        self
    }

    /// Set the hop length between RMS frames (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Trim silence. Returns `(trimmed_signal, (start_sample, end_sample))`.
    pub fn compute(self) -> (Vec<f32>, (usize, usize)) {
        trim_impl(self.y, self.top_db, self.frame_length, self.hop_length)
    }
}

/// Trims leading and trailing silence from a signal.
///
/// Silence is defined as frames whose RMS is more than `top_db` dB below the
/// peak RMS. Returns `(trimmed, (start, end))` where `start` and `end` are
/// sample indices into the original signal.
///
/// # Examples
/// ```no_run
/// use dasp_rs::proc::trim;
/// let mut y = vec![0.0_f32; 100];
/// y[40] = 0.9; y[41] = 0.9;
/// let (trimmed, (start, end)) = trim(&y).top_db(60.0).compute();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn trim(y: &[f32]) -> TrimBuilder<'_> {
    TrimBuilder { y, top_db: 60.0, frame_length: 2048, hop_length: 512 }
}

fn rms_frames(y: &[f32], frame_length: usize, hop_length: usize) -> Vec<f32> {
    if y.is_empty() || frame_length == 0 || hop_length == 0 {
        return Vec::new();
    }
    let n_frames = y.len().saturating_sub(frame_length) / hop_length + 1;
    (0..n_frames)
        .map(|i| {
            let start = i * hop_length;
            let end = (start + frame_length).min(y.len());
            let sl = &y[start..end];
            let mean_sq = sl.iter().map(|&v| v * v).sum::<f32>() / sl.len() as f32;
            mean_sq.sqrt()
        })
        .collect()
}

fn trim_impl(
    y: &[f32],
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
) -> (Vec<f32>, (usize, usize)) {
    let rms = rms_frames(y, frame_length, hop_length);
    if rms.is_empty() {
        return (y.to_vec(), (0, y.len()));
    }
    let peak = rms.iter().copied().fold(0.0_f32, f32::max).max(1e-10);
    let threshold = peak * 10.0_f32.powf(-top_db / 20.0);

    let first_frame = rms.iter().position(|r| *r >= threshold).unwrap_or(0);
    let last_frame = (0..rms.len()).rfind(|&i| rms[i] >= threshold)
        .unwrap_or_else(|| rms.len().saturating_sub(1));

    let start = (first_frame * hop_length).min(y.len());
    let end = ((last_frame + 1) * hop_length + frame_length).min(y.len());
    (y[start..end].to_vec(), (start, end))
}

// ─── Split ────────────────────────────────────────────────────────────────────

/// Builder for [`split`].
#[derive(Debug, Clone)]
pub struct SplitBuilder<'a> {
    y: &'a [f32],
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
}

impl SplitBuilder<'_> {
    /// Set the silence threshold in dB below the peak RMS (default: 60.0).
    #[must_use]
    pub fn top_db(mut self, v: f32) -> Self {
        self.top_db = v;
        self
    }

    /// Set the RMS frame length in samples (default: 2048).
    #[must_use]
    pub fn frame_length(mut self, v: usize) -> Self {
        self.frame_length = v;
        self
    }

    /// Set the hop length between RMS frames (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Find non-silent intervals. Returns `(start, end)` sample index pairs.
    pub fn compute(self) -> Vec<(usize, usize)> {
        split_impl(self.y, self.top_db, self.frame_length, self.hop_length)
    }
}

/// Splits a signal into non-silent intervals.
///
/// Returns a vector of `(start, end)` sample index pairs — one entry per
/// contiguous non-silent segment. Silence is defined the same way as [`trim`].
///
/// # Examples
/// ```no_run
/// use dasp_rs::proc::split;
/// let mut y = vec![0.0_f32; 44100];
/// y[1000] = 0.9;
/// y[20000] = 0.9;
/// let intervals = split(&y).top_db(60.0).compute();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn split(y: &[f32]) -> SplitBuilder<'_> {
    SplitBuilder { y, top_db: 60.0, frame_length: 2048, hop_length: 512 }
}

fn split_impl(
    y: &[f32],
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
) -> Vec<(usize, usize)> {
    let rms = rms_frames(y, frame_length, hop_length);
    if rms.is_empty() {
        return Vec::new();
    }
    let peak = rms.iter().copied().fold(0.0_f32, f32::max).max(1e-10);
    let threshold = peak * 10.0_f32.powf(-top_db / 20.0);

    let mut intervals: Vec<(usize, usize)> = Vec::new();
    let mut in_segment = false;
    let mut seg_start = 0usize;

    for (i, &r) in rms.iter().enumerate() {
        if r >= threshold && !in_segment {
            seg_start = i * hop_length;
            in_segment = true;
        } else if r < threshold && in_segment {
            let end = (i * hop_length + frame_length).min(y.len());
            intervals.push((seg_start, end));
            in_segment = false;
        }
    }
    if in_segment {
        intervals.push((seg_start, y.len()));
    }
    intervals
}

// ─── Preemphasis / Deemphasis ─────────────────────────────────────────────────

/// Builder for [`preemphasis`].
#[derive(Debug, Clone)]
pub struct PreemphasisBuilder<'a> {
    y: &'a [f32],
    coef: f32,
}

impl PreemphasisBuilder<'_> {
    /// Set the filter coefficient (default: 0.97).
    #[must_use]
    pub fn coef(mut self, v: f32) -> Self {
        self.coef = v;
        self
    }

    /// Apply the filter.
    pub fn compute(self) -> Vec<f32> {
        preemphasis_impl(self.y, self.coef)
    }
}

/// Applies a first-order high-pass pre-emphasis filter: `y[n] = y[n] - coef * y[n-1]`.
///
/// Used to spectrally flatten a signal before analysis (common in speech processing).
/// Inverse operation: [`deemphasis`].
///
/// # Examples
/// ```
/// use dasp_rs::proc::preemphasis;
/// let y = vec![1.0_f32, 2.0, 3.0, 4.0];
/// let out = preemphasis(&y).coef(0.97).compute();
/// assert!((out[0] - 1.0).abs() < 1e-6);
/// assert!((out[1] - (2.0 - 0.97)).abs() < 1e-5);
/// ```
pub fn preemphasis(y: &[f32]) -> PreemphasisBuilder<'_> {
    PreemphasisBuilder { y, coef: 0.97 }
}

fn preemphasis_impl(y: &[f32], coef: f32) -> Vec<f32> {
    if y.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(y.len());
    out.push(y[0]);
    for i in 1..y.len() {
        out.push(y[i] - coef * y[i - 1]);
    }
    out
}

/// Builder for [`deemphasis`].
#[derive(Debug, Clone)]
pub struct DeemphasisBuilder<'a> {
    y: &'a [f32],
    coef: f32,
}

impl DeemphasisBuilder<'_> {
    /// Set the filter coefficient (default: 0.97).
    #[must_use]
    pub fn coef(mut self, v: f32) -> Self {
        self.coef = v;
        self
    }

    /// Apply the filter.
    pub fn compute(self) -> Vec<f32> {
        deemphasis_impl(self.y, self.coef)
    }
}

/// Applies the inverse of [`preemphasis`]: `y[n] = y[n] + coef * out[n-1]`.
///
/// Restores a signal that has been high-pass filtered by `preemphasis`.
///
/// # Examples
/// ```
/// use dasp_rs::proc::{preemphasis, deemphasis};
/// let y = vec![1.0_f32, 2.0, 3.0, 4.0];
/// let pre = preemphasis(&y).compute();
/// let rec = deemphasis(&pre).compute();
/// for (a, b) in y.iter().zip(rec.iter()) {
///     assert!((a - b).abs() < 1e-5, "{a} != {b}");
/// }
/// ```
pub fn deemphasis(y: &[f32]) -> DeemphasisBuilder<'_> {
    DeemphasisBuilder { y, coef: 0.97 }
}

fn deemphasis_impl(y: &[f32], coef: f32) -> Vec<f32> {
    if y.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(y.len());
    out.push(y[0]);
    for i in 1..y.len() {
        out.push(y[i] + coef * out[i - 1]);
    }
    out
}

// ─── Remix ────────────────────────────────────────────────────────────────────

/// Builder for [`remix`].
#[derive(Debug, Clone)]
pub struct RemixBuilder<'a> {
    y: &'a [f32],
    intervals: &'a [(usize, usize)],
    align_zeros: bool,
}

impl RemixBuilder<'_> {
    /// Snap interval boundaries to the nearest zero-crossing to reduce clicks
    /// (default: `false`).
    #[must_use]
    pub fn align_zeros(mut self, v: bool) -> Self {
        self.align_zeros = v;
        self
    }

    /// Concatenate the specified intervals and return the reordered audio.
    pub fn compute(self) -> Vec<f32> {
        remix_impl(self.y, self.intervals, self.align_zeros)
    }
}

/// Reorders and concatenates segments of an audio signal.
///
/// Extracts the sample ranges given by `intervals` and concatenates them in
/// order. Intervals that extend beyond the signal are silently clamped. Empty
/// or inverted intervals are skipped.
///
/// # Arguments
/// * `y` — Input audio samples.
/// * `intervals` — Slice of `(start, end)` sample-index pairs (end exclusive).
///
/// # Examples
/// ```
/// use dasp_rs::proc::remix;
/// let y: Vec<f32> = (0..100).map(|i| i as f32).collect();
/// // Reverse two segments
/// let out = remix(&y, &[(50, 100), (0, 50)]).compute();
/// assert_eq!(out.len(), 100);
/// assert_eq!(out[0], 50.0);
/// ```
pub fn remix<'a>(y: &'a [f32], intervals: &'a [(usize, usize)]) -> RemixBuilder<'a> {
    RemixBuilder { y, intervals, align_zeros: false }
}

fn remix_impl(y: &[f32], intervals: &[(usize, usize)], align_zeros: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for &(start, end) in intervals {
        let s = start.min(y.len());
        let e = end.min(y.len());
        if s >= e {
            continue;
        }
        let (sa, ea) = if align_zeros {
            let sa = nearest_zero_crossing(y, s);
            let ea = nearest_zero_crossing(y, e);
            (sa.min(e), ea.min(y.len()))
        } else {
            (s, e)
        };
        if sa < ea {
            out.extend_from_slice(&y[sa..ea]);
        }
    }
    out
}

fn nearest_zero_crossing(y: &[f32], idx: usize) -> usize {
    let n = y.len();
    if idx == 0 || idx >= n {
        return idx;
    }
    let window = 256;
    let lo = idx.saturating_sub(window);
    let hi = (idx + window).min(n.saturating_sub(1));
    let mut best = idx;
    let mut best_dist = usize::MAX;
    for i in lo..hi {
        if (y[i] >= 0.0) != (y[i + 1] >= 0.0) {
            let dist = i.abs_diff(idx);
            if dist < best_dist {
                best_dist = dist;
                best = i;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_stretch_empty() {
        assert!(time_stretch_impl(&[], 0.5, 2048, 512).is_err());
    }

    #[test]
    fn test_time_stretch_invalid_rate() {
        let y = vec![0.0_f32; 4096];
        assert!(time_stretch_impl(&y, 0.0, 2048, 512).is_err());
        assert!(time_stretch_impl(&y, -1.0, 2048, 512).is_err());
        assert!(time_stretch_impl(&y, f32::NAN, 2048, 512).is_err());
    }

    #[test]
    fn test_time_stretch_silence() {
        let y = vec![0.0_f32; 8192];
        let out = time_stretch_impl(&y, 2.0, 2048, 512).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|&v| v.abs() < 1e-4));
    }

    #[test]
    fn test_pitch_shift_preserves_length() {
        let y = vec![0.0_f32; 8192];
        let out = pitch_shift_impl(&y, 44100, 2.0, 2048, 512).unwrap();
        assert_eq!(out.len(), y.len());
    }

    #[test]
    fn test_trim_all_silence() {
        let y = vec![0.0_f32; 8192];
        let (trimmed, (start, end)) = trim_impl(&y, 60.0, 2048, 512);
        // All silence: trim should return the whole signal (no non-silent content found)
        let _ = (trimmed, start, end);
    }

    #[test]
    fn test_trim_with_signal() {
        let mut y = vec![0.0_f32; 10000];
        y[3000..4000].fill(0.5);
        let (trimmed, (start, end)) = trim_impl(&y, 60.0, 512, 128);
        assert!(start < 3000 + 512, "trim start should be near signal onset");
        assert!(end > 4000 - 512, "trim end should be near signal offset");
        assert!(!trimmed.is_empty());
    }

    #[test]
    fn test_split_two_bursts() {
        let mut y = vec![0.0_f32; 20000];
        y[1000..2000].fill(0.8);
        y[10000..11000].fill(0.8);
        let intervals = split_impl(&y, 60.0, 512, 128);
        assert_eq!(intervals.len(), 2, "should find exactly two non-silent bursts");
    }

    #[test]
    fn test_preemphasis_empty() {
        assert!(preemphasis_impl(&[], 0.97).is_empty());
    }

    #[test]
    fn test_preemphasis_first_sample_unchanged() {
        let y = vec![1.0_f32, 2.0, 3.0];
        let out = preemphasis_impl(&y, 0.97);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (2.0 - 0.97 * 1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_deemphasis_inverts_preemphasis() {
        let y = vec![0.5_f32, -0.3, 0.8, 0.1, -0.6];
        let pre = preemphasis_impl(&y, 0.97);
        let rec = deemphasis_impl(&pre, 0.97);
        for (a, b) in y.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-4, "round-trip mismatch: {a} != {b}");
        }
    }
}
