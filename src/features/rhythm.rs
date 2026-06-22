use ndarray::{s, Array1, Array2, Axis};
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

impl TempoBuilder<'_> {
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

    /// Compute tempo.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
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
/// use dasp_rs::feat::*;
/// use dasp_rs::types::*;
/// let y = vec![1.0, 2.0, 3.0, 4.0];
/// let bpm = tempo(&y, 44100)
///     .hop_length(512)
///     .compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tempo(y: &[f32], sr: u32) -> TempoBuilder<'_> {
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
/// use dasp_rs::feat::*;
/// use dasp_rs::types::*;
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// // Clean, ergonomic API
/// let bpm = tempo(&y, 44100)
///     .hop_length(512)
///     .compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub(crate) fn tempo_impl(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
) -> Result<f32, RhythmError> {
    let sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let onset_owned = if let Some(envelope) = onset_envelope {
        envelope.to_owned()
    } else {
        let y = y.ok_or_else(|| {
            RhythmError::InvalidInput(
                "Audio signal required when onset_envelope is None".to_string(),
            )
        })?;
        let s = stft(y)
            .hop_length(hop)
            .compute()
            .map_err(|e| RhythmError::ComputationFailed(format!("STFT computation failed: {e}")))?
            .mapv(num_complex::Complex::norm);
        s.map_axis(Axis(0), |row| row.iter().map(|&x| x.max(0.0)).sum::<f32>())
    };
    let onset = &onset_owned;
    let tempogram = tempogram(None, Some(sr), Some(onset), hop_length, None)?;
    let freqs = crate::utils::frequency::tempo_frequencies_impl(tempogram.shape()[0], hop, sr);
    Ok(tempogram
        .axis_iter(Axis(1))
        .map(|col| {
            let max_idx = col
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map_or(0, |(i, _)| i);
            freqs[max_idx]
        })
        .sum::<f32>()
        / tempogram.shape()[1] as f32)
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
/// # Errors
/// Returns [`RhythmError::InvalidInput`] if both `y` and `onset_envelope` are `None`,
/// or [`RhythmError::ComputationFailed`] if the STFT cannot be computed.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::tempogram;
/// let y = vec![0.0; 2048];
/// let tgram = tempogram(Some(&y), Some(44100), None, Some(512), Some(384))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tempogram(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
) -> Result<Array2<f32>, RhythmError> {
    let _sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let win = win_length.unwrap_or(384);
    let onset_owned = if let Some(envelope) = onset_envelope {
        envelope.to_owned()
    } else {
        let y = y.ok_or_else(|| {
            RhythmError::InvalidInput(
                "Audio signal required when onset_envelope is None".to_string(),
            )
        })?;
        let s = stft(y)
            .hop_length(hop)
            .compute()
            .map_err(|e| RhythmError::ComputationFailed(format!("STFT computation failed: {e}")))?
            .mapv(num_complex::Complex::norm);
        s.map_axis(Axis(0), |row| row.iter().map(|&x| x.max(0.0)).sum::<f32>())
    };
    let onset = &onset_owned;
    let mut tempogram = Array2::zeros((win / 2 + 1, onset.len()));
    for t in 0..onset.len() {
        for lag in 0..=(win / 2) {
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
/// # Errors
/// Returns [`RhythmError::InvalidInput`] if both `y` and `onset_envelope` are `None`,
/// or [`RhythmError::ComputationFailed`] if the STFT cannot be computed.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::tempogram_ratio;
/// let y = vec![0.0; 2048];
/// let ratio_tgram = tempogram_ratio(Some(&y), Some(44100), None, Some(512), Some(&[2.0, 3.0, 4.0]))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tempogram_ratio(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
    ratios: Option<&[f32]>,
) -> Result<Array2<f32>, RhythmError> {
    let tempogram = tempogram(y, sr, onset_envelope, hop_length, None)?;
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

// ─── Onset strength ───────────────────────────────────────────────────────────

/// Builder for [`onset_strength`].
#[derive(Debug, Clone)]
pub struct OnsetStrengthBuilder<'a> {
    y: &'a [f32],
    hop_length: usize,
    n_fft: usize,
}

impl OnsetStrengthBuilder<'_> {
    /// Set the hop length between STFT frames (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, v: usize) -> Self {
        self.n_fft = v;
        self
    }

    /// Compute the onset strength envelope.
    ///
    /// # Errors
    /// Returns an error if the signal is empty or STFT computation fails.
    pub fn compute(self) -> Result<Array1<f32>, RhythmError> {
        onset_strength_impl(self.y, self.hop_length, self.n_fft)
    }
}

/// Computes the onset strength envelope (positive spectral flux in log space).
///
/// Each value in the returned array represents the amount of spectral change
/// at that frame. High values indicate likely onset positions.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::onset_strength;
/// let y = vec![0.0_f32; 44100];
/// let odf = onset_strength(&y, 44100).hop_length(512).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn onset_strength(y: &[f32], _sr: u32) -> OnsetStrengthBuilder<'_> {
    OnsetStrengthBuilder { y, hop_length: 512, n_fft: 2048 }
}

fn onset_strength_impl(
    y: &[f32],
    hop_length: usize,
    n_fft: usize,
) -> Result<Array1<f32>, RhythmError> {
    if y.is_empty() {
        return Err(RhythmError::InvalidInput("Signal is empty".into()));
    }
    let spec = stft(y)
        .n_fft(n_fft)
        .hop_length(hop_length)
        .compute()
        .map_err(|e| RhythmError::ComputationFailed(format!("STFT failed: {e}")))?;

    let n_frames = spec.shape()[1];
    // Log-power spectrogram
    let log_s = spec.mapv(|x| x.norm().powi(2).max(1e-10_f32).log10());

    let mut odf = Array1::zeros(n_frames);
    for t in 1..n_frames {
        let diff = &log_s.slice(s![.., t]) - &log_s.slice(s![.., t - 1]);
        odf[t] = diff.iter().map(|&v| v.max(0.0)).sum::<f32>();
    }
    Ok(odf)
}

// ─── Onset detect ─────────────────────────────────────────────────────────────

/// Builder for [`onset_detect`].
#[derive(Debug, Clone)]
pub struct OnsetDetectBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    n_fft: usize,
    delta: f32,
    wait: usize,
}

impl OnsetDetectBuilder<'_> {
    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Set the FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, v: usize) -> Self {
        self.n_fft = v;
        self
    }

    /// Set the threshold above the local mean required for a peak (default: 0.07).
    #[must_use]
    pub fn delta(mut self, v: f32) -> Self {
        self.delta = v;
        self
    }

    /// Set the minimum number of frames between consecutive onsets (default: 1).
    #[must_use]
    pub fn wait(mut self, v: usize) -> Self {
        self.wait = v;
        self
    }

    /// Detect onset frame indices.
    ///
    /// # Errors
    /// Returns an error if the signal is empty or STFT computation fails.
    pub fn compute(self) -> Result<Vec<usize>, RhythmError> {
        let odf = onset_strength_impl(self.y, self.hop_length, self.n_fft)?;
        Ok(onset_detect_impl(odf.as_slice().unwrap_or(&[]), self.delta, self.wait))
    }

    /// Detect onset times in seconds.
    ///
    /// # Errors
    /// Returns an error if the signal is empty or STFT computation fails.
    pub fn compute_times(self) -> Result<Vec<f32>, RhythmError> {
        let hop = self.hop_length;
        let sr = self.sr;
        let frames = self.compute()?;
        Ok(frames.into_iter().map(|f| f as f32 * hop as f32 / sr as f32).collect())
    }
}

/// Detects onset frame indices from an audio signal.
///
/// Uses the onset strength envelope with local peak-picking above a mean-based
/// threshold. Convert frames to seconds: `frame * hop_length / sr`.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::onset_detect;
/// let y = vec![0.0_f32; 44100];
/// let frames = onset_detect(&y, 44100).delta(0.07).wait(4).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn onset_detect(y: &[f32], sr: u32) -> OnsetDetectBuilder<'_> {
    OnsetDetectBuilder { y, sr, hop_length: 512, n_fft: 2048, delta: 0.07, wait: 1 }
}

fn onset_detect_impl(odf: &[f32], delta: f32, wait: usize) -> Vec<usize> {
    const PRE_MAX: usize = 3;
    const POST_MAX: usize = 3;
    const PRE_AVG: usize = 3;
    const POST_AVG: usize = 3;

    let n = odf.len();
    if n == 0 {
        return Vec::new();
    }

    let mut peaks: Vec<usize> = Vec::new();
    for t in 0..n {
        let max_left = t.saturating_sub(PRE_MAX);
        let max_right = (t + POST_MAX + 1).min(n);
        let local_max = odf[max_left..max_right]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        // Skip if this frame is not (approximately) the local maximum.
        if odf[t] < local_max - 1e-6 {
            continue;
        }
        let avg_left = t.saturating_sub(PRE_AVG);
        let avg_right = (t + POST_AVG + 1).min(n);
        let window = &odf[avg_left..avg_right];
        let mean = window.iter().sum::<f32>() / window.len() as f32;
        if odf[t] >= mean + delta {
            peaks.push(t);
        }
    }
    if wait <= 1 {
        return peaks;
    }
    let mut filtered: Vec<usize> = Vec::new();
    let mut last = 0usize;
    for &p in &peaks {
        if filtered.is_empty() || p.saturating_sub(last) >= wait {
            filtered.push(p);
            last = p;
        }
    }
    filtered
}

// ─── Beat track ───────────────────────────────────────────────────────────────

/// Builder for [`beat_track`].
#[derive(Debug, Clone)]
pub struct BeatTrackBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    start_bpm: f32,
    tightness: f32,
}

impl BeatTrackBuilder<'_> {
    /// Set the hop length (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Set the initial BPM estimate used when tempo estimation fails (default: 120.0).
    #[must_use]
    pub fn start_bpm(mut self, v: f32) -> Self {
        self.start_bpm = v;
        self
    }

    /// Set the DP tracking tightness (default: 100.0; higher values enforce stricter
    /// adherence to the estimated period).
    #[must_use]
    pub fn tightness(mut self, v: f32) -> Self {
        self.tightness = v;
        self
    }

    /// Track beats and return `(tempo_bpm, beat_frame_indices)`.
    ///
    /// # Errors
    /// Returns an error if the signal is empty or STFT computation fails.
    pub fn compute(self) -> Result<(f32, Vec<usize>), RhythmError> {
        beat_track_impl(self.y, self.sr, self.hop_length, self.start_bpm, self.tightness)
    }
}

/// Tracks beats in an audio signal via dynamic programming on the onset envelope.
///
/// Returns `(tempo_bpm, beat_frame_indices)`. Convert frames to seconds with
/// `frame * hop_length / sr`.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::beat_track;
/// let y = vec![0.0_f32; 44100 * 5];
/// let (bpm, beats) = beat_track(&y, 44100).hop_length(512).compute()?;
/// println!("Tempo: {bpm:.1} BPM, {} beats found", beats.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn beat_track(y: &[f32], sr: u32) -> BeatTrackBuilder<'_> {
    BeatTrackBuilder { y, sr, hop_length: 512, start_bpm: 120.0, tightness: 100.0 }
}

fn beat_track_impl(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    start_bpm: f32,
    tightness: f32,
) -> Result<(f32, Vec<usize>), RhythmError> {
    let odf = onset_strength_impl(y, hop_length, 2048)?;
    let estimated_bpm = tempo_impl(Some(y), Some(sr), None, Some(hop_length))
        .unwrap_or(start_bpm)
        .max(1.0);
    let period = 60.0 * sr as f32 / (hop_length as f32 * estimated_bpm);
    let beat_frames = dp_beat_track(odf.as_slice().unwrap_or(&[]), period, tightness);

    let final_bpm = if beat_frames.len() >= 2 {
        let mut ibis: Vec<f32> = beat_frames.windows(2)
            .map(|w| (w[1] - w[0]) as f32)
            .collect();
        let median_ibi = median_f32(&mut ibis);
        if median_ibi > 0.0 {
            60.0 * sr as f32 / (hop_length as f32 * median_ibi)
        } else {
            estimated_bpm
        }
    } else {
        estimated_bpm
    };

    Ok((final_bpm, beat_frames))
}

fn dp_beat_track(odf: &[f32], period: f32, tightness: f32) -> Vec<usize> {
    let n = odf.len();
    if n == 0 || period <= 0.0 {
        return Vec::new();
    }
    let mut score: Vec<f32> = odf.to_vec();
    let mut backlink: Vec<usize> = (0..n).collect();
    let log_period = period.ln();
    let lag_min = (period * 0.5).max(1.0) as usize;
    let lag_max = (period * 2.0).ceil() as usize + 1;

    for t in 1..n {
        let t_end = t.saturating_sub(lag_min);
        let t_start = t.saturating_sub(lag_max);
        if t_start >= t_end {
            continue;
        }
        let mut best_s = f32::NEG_INFINITY;
        let mut best_prev = t_start;
        for (offset, &s_prev) in score[t_start..t_end].iter().enumerate() {
            let t_prev = t_start + offset;
            let lag = (t - t_prev) as f32;
            let log_ratio = (lag.ln() - log_period) / std::f32::consts::LN_2;
            let transition = -0.5 * tightness * log_ratio * log_ratio;
            let cand = s_prev + transition;
            if cand > best_s {
                best_s = cand;
                best_prev = t_prev;
            }
        }
        if best_s.is_finite() {
            score[t] = odf[t] + best_s;
            backlink[t] = best_prev;
        }
    }

    let last = score.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or_else(|| n.saturating_sub(1), |(i, _)| i);

    let mut beats: Vec<usize> = Vec::new();
    let mut t = last;
    loop {
        beats.push(t);
        if t == 0 {
            break;
        }
        let prev = backlink[t];
        if prev >= t {
            break;
        }
        t = prev;
    }
    beats.reverse();
    beats
}

fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        f32::midpoint(values[mid - 1], values[mid])
    } else {
        values[mid]
    }
}

// ─── Beat sync ────────────────────────────────────────────────────────────────

/// Aggregation strategy for [`beat_sync`].
#[derive(Debug, Clone, Copy, Default)]
pub enum Aggregate {
    /// Arithmetic mean of the segment (default).
    #[default]
    Mean,
    /// Median of the segment.
    Median,
    /// Maximum value in the segment.
    Max,
    /// Minimum value in the segment.
    Min,
}

/// Builder for [`beat_sync`].
#[derive(Debug, Clone)]
pub struct BeatSyncBuilder<'a> {
    data: &'a Array2<f32>,
    beat_frames: &'a [usize],
    aggregate: Aggregate,
    pad: bool,
}

impl BeatSyncBuilder<'_> {
    /// Set the aggregation function (default: [`Aggregate::Mean`]).
    #[must_use]
    pub fn aggregate(mut self, v: Aggregate) -> Self {
        self.aggregate = v;
        self
    }

    /// If `true` (default), add boundary segments at frame 0 and the last frame.
    #[must_use]
    pub fn pad(mut self, v: bool) -> Self {
        self.pad = v;
        self
    }

    /// Aggregate the feature matrix at beat boundaries.
    pub fn compute(self) -> Array2<f32> {
        beat_sync_impl(self.data, self.beat_frames, self.aggregate, self.pad)
    }
}

/// Aggregates a feature matrix at beat frame boundaries.
///
/// Given a feature matrix of shape `(n_features, n_frames)` and a list of beat
/// frame indices, each inter-beat interval is collapsed into a single column
/// using the chosen aggregation strategy.
///
/// When `pad` is `true` (default), boundary segments `[0, beats[0])` and
/// `[beats[-1], n_frames)` are included, giving shape `(n_features, n_beats + 1)`.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::{beat_track, beat_sync};
/// use ndarray::Array2;
/// let y = vec![0.0_f32; 44100 * 4];
/// let (_, beats) = beat_track(&y, 44100).compute()?;
/// let chroma: Array2<f32> = Array2::zeros((12, 345));
/// let synced = beat_sync(&chroma, &beats).compute();
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn beat_sync<'a>(data: &'a Array2<f32>, beat_frames: &'a [usize]) -> BeatSyncBuilder<'a> {
    BeatSyncBuilder { data, beat_frames, aggregate: Aggregate::Mean, pad: true }
}

fn beat_sync_impl(
    data: &Array2<f32>,
    beat_frames: &[usize],
    aggregate: Aggregate,
    pad: bool,
) -> Array2<f32> {
    let n_features = data.shape()[0];
    let n_total = data.shape()[1];

    // Build segment boundaries
    let mut bounds: Vec<usize> = Vec::new();
    if pad {
        bounds.push(0);
    }
    for &f in beat_frames {
        bounds.push(f.min(n_total));
    }
    if pad {
        bounds.push(n_total);
    }
    bounds.dedup();

    let n_segs = bounds.len().saturating_sub(1);
    if n_segs == 0 {
        return Array2::zeros((n_features, 0));
    }

    let mut out = Array2::zeros((n_features, n_segs));
    for seg in 0..n_segs {
        let start = bounds[seg].min(n_total);
        let end = bounds[seg + 1].min(n_total);
        if start >= end {
            continue;
        }
        for feat in 0..n_features {
            let slice: Vec<f32> = data.slice(s![feat, start..end]).iter().copied().collect();
            out[[feat, seg]] = aggregate_slice(&slice, aggregate);
        }
    }
    out
}

fn aggregate_slice(v: &[f32], mode: Aggregate) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    match mode {
        Aggregate::Mean => v.iter().sum::<f32>() / v.len() as f32,
        Aggregate::Max => v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        Aggregate::Min => v.iter().copied().fold(f32::INFINITY, f32::min),
        Aggregate::Median => {
            let mut sorted = v.to_vec();
            sorted.sort_by(f32::total_cmp);
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                f32::midpoint(sorted[mid - 1], sorted[mid])
            } else {
                sorted[mid]
            }
        }
    }
}

// ─── PLP (Predominant Local Pulse) ────────────────────────────────────────────

/// Builder for [`plp`].
#[derive(Debug, Clone)]
pub struct PlpBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    win_length: usize,
    min_tempo: f32,
    max_tempo: f32,
}

impl PlpBuilder<'_> {
    /// Set the hop length between frames (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self {
        self.hop_length = v;
        self
    }

    /// Set the autocorrelation window length in frames (default: 384).
    #[must_use]
    pub fn win_length(mut self, v: usize) -> Self {
        self.win_length = v;
        self
    }

    /// Set the minimum valid tempo in BPM (default: 30.0).
    #[must_use]
    pub fn min_tempo(mut self, v: f32) -> Self {
        self.min_tempo = v;
        self
    }

    /// Set the maximum valid tempo in BPM (default: 300.0).
    #[must_use]
    pub fn max_tempo(mut self, v: f32) -> Self {
        self.max_tempo = v;
        self
    }

    /// Compute the pulse curve and extract beat frames.
    ///
    /// Returns `(pulse_curve, beat_frame_indices)`. `pulse_curve` values are in
    /// `[0, 1]`; peaks correspond to likely beat positions.
    ///
    /// # Errors
    /// Returns an error if the signal is empty or too short.
    pub fn compute(self) -> Result<(Array1<f32>, Vec<usize>), RhythmError> {
        plp_impl(
            self.y,
            self.sr,
            self.hop_length,
            self.win_length,
            self.min_tempo,
            self.max_tempo,
        )
    }
}

/// Computes the Predominant Local Pulse (PLP) from an audio signal.
///
/// Synthesizes a sinusoidal pulse at the locally-dominant tempo for each frame
/// and accumulates these pulses into a single beat-probability curve.  Unlike
/// [`beat_track`], PLP adapts frame-by-frame and is more robust to tempo changes.
///
/// Returns `(pulse_curve, beat_frame_indices)`.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::plp;
/// let y = vec![0.0_f32; 44100 * 5];
/// let (pulse, beats) = plp(&y, 44100).hop_length(512).compute()?;
/// println!("{} beats found", beats.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn plp(y: &[f32], sr: u32) -> PlpBuilder<'_> {
    PlpBuilder { y, sr, hop_length: 512, win_length: 384, min_tempo: 30.0, max_tempo: 300.0 }
}

fn plp_impl(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    win_length: usize,
    min_tempo: f32,
    max_tempo: f32,
) -> Result<(Array1<f32>, Vec<usize>), RhythmError> {
    let odf = onset_strength_impl(y, hop_length, 2048)?;
    let n_frames = odf.len();
    if n_frames < 2 {
        return Err(RhythmError::InvalidInput("Signal too short for PLP".into()));
    }

    // Autocorrelation tempogram
    let tgram = tempogram(None, Some(sr), Some(&odf), Some(hop_length), Some(win_length))?;
    let n_bins = tgram.shape()[0];

    let freqs = crate::utils::frequency::tempo_frequencies_impl(n_bins, hop_length, sr);

    // Valid tempo bin indices
    let valid: Vec<usize> = (0..n_bins)
        .filter(|&i| freqs[i] >= min_tempo && freqs[i] <= max_tempo)
        .collect();

    if valid.is_empty() {
        return Err(RhythmError::InvalidInput(
            "No valid tempo bins in [min_tempo, max_tempo] range".into(),
        ));
    }

    let mut pulse = vec![0.0f32; n_frames];

    for t in 0..n_frames {
        let best = valid
            .iter()
            .copied()
            .max_by(|&a, &b| tgram[[a, t]].total_cmp(&tgram[[b, t]]))
            .unwrap_or(valid[0]);

        let bpm = freqs[best].max(1.0);
        let period = 60.0 * sr as f32 / (hop_length as f32 * bpm);
        let weight = tgram[[best, t]].max(0.0);

        // Raised-cosine lobe of width `period` centered at frame t
        let half = (period / 2.0).ceil() as usize;
        for d in 0..=half {
            let lobe = weight * (std::f32::consts::PI * d as f32 / period).cos().max(0.0);
            if t + d < n_frames {
                pulse[t + d] += lobe;
            }
            if d > 0 && t >= d {
                pulse[t - d] += lobe;
            }
        }
    }

    // Normalize to [0, 1]
    let peak = pulse.iter().copied().fold(0.0_f32, f32::max).max(1e-10);
    for v in &mut pulse {
        *v /= peak;
    }

    let beats = onset_detect_impl(&pulse, 0.1, 1);
    Ok((Array1::from(pulse), beats))
}

// ─── Fourier tempogram ────────────────────────────────────────────────────────

/// Builder for [`fourier_tempogram`].
#[derive(Debug, Clone)]
pub struct FourierTempogramBuilder<'a> {
    y: &'a [f32],
    sr: u32,
    hop_length: usize,
    win_length: usize,
}

impl FourierTempogramBuilder<'_> {
    /// Hop length in samples (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self { self.hop_length = v; self }
    /// Analysis window length (default: 384 frames ≈ 8.7 s at 512-sample hop).
    #[must_use]
    pub fn win_length(mut self, v: usize) -> Self { self.win_length = v; self }

    /// Compute the Fourier tempogram.
    ///
    /// # Errors
    /// Returns an error if the signal is empty or `win_length` is zero.
    pub fn compute(self) -> Result<Array2<f32>, RhythmError> {
        fourier_tempogram_impl(self.y, self.sr, self.hop_length, self.win_length)
    }
}

/// Computes the Fourier-domain tempogram from an audio signal.
///
/// The Fourier tempogram captures periodic rhythmic structure at arbitrary
/// (including non-integer) tempo multiples, unlike the autocorrelation
/// [`tempogram`] which is restricted to the analysis window.
///
/// Algorithm: compute the onset strength envelope, then take its short-time
/// Fourier transform with `n_fft = win_length` and `hop = 1`.
///
/// # Arguments
/// * `y` — Audio samples.
/// * `sr` — Sample rate in Hz.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::fourier_tempogram;
/// let y = vec![0.0_f32; 44100];
/// let tg = fourier_tempogram(&y, 44100).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn fourier_tempogram(y: &[f32], sr: u32) -> FourierTempogramBuilder<'_> {
    FourierTempogramBuilder { y, sr, hop_length: 512, win_length: 384 }
}

fn fourier_tempogram_impl(
    y: &[f32],
    _sr: u32,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>, RhythmError> {
    if win_length == 0 {
        return Err(RhythmError::InvalidInput("win_length must be > 0".into()));
    }

    let oenv = onset_strength_impl(y, hop_length, 2048)?;
    let oenv_vec: Vec<f32> = oenv.into_iter().collect();

    // STFT of onset envelope with n_fft = win_length, hop = 1
    let spec = stft(&oenv_vec)
        .n_fft(win_length)
        .hop_length(1)
        .compute()
        .map_err(|e| RhythmError::InvalidInput(e.to_string()))?;

    // Return magnitude
    Ok(spec.mapv(num_complex::Complex::norm))
}

// ─── Onset strength multi ─────────────────────────────────────────────────────

/// Builder for [`onset_strength_multi`].
#[derive(Debug, Clone)]
pub struct OnsetStrengthMultiBuilder<'a> {
    y: &'a [f32],
    n_bands: usize,
    hop_length: usize,
    n_fft: usize,
}

impl OnsetStrengthMultiBuilder<'_> {
    /// Number of frequency subbands (default: 6).
    #[must_use]
    pub fn n_bands(mut self, v: usize) -> Self { self.n_bands = v; self }
    /// Hop length in samples (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self { self.hop_length = v; self }
    /// FFT size (default: 2048).
    #[must_use]
    pub fn n_fft(mut self, v: usize) -> Self { self.n_fft = v; self }

    /// Compute the multi-band onset strength matrix of shape `(n_bands, n_frames)`.
    ///
    /// # Errors
    /// Returns an error if the signal is empty or any parameter is zero.
    pub fn compute(self) -> Result<Array2<f32>, RhythmError> {
        onset_strength_multi_impl(self.y, self.n_bands, self.hop_length, self.n_fft)
    }
}

/// Computes onset strength independently for multiple frequency subbands.
///
/// Divides the STFT power spectrogram into `n_bands` linearly spaced frequency
/// bands and computes positive log-spectral flux for each, returning a matrix
/// of shape `(n_bands, n_frames)`.
///
/// Useful for rhythm analysis that distinguishes low-frequency (kick/bass) from
/// high-frequency (hi-hat/snare) transients.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::onset_strength_multi;
/// let y = vec![0.0_f32; 44100];
/// let odf = onset_strength_multi(&y, 44100).n_bands(6).compute()?;
/// assert_eq!(odf.shape()[0], 6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn onset_strength_multi(y: &[f32], _sr: u32) -> OnsetStrengthMultiBuilder<'_> {
    OnsetStrengthMultiBuilder { y, n_bands: 6, hop_length: 512, n_fft: 2048 }
}

fn onset_strength_multi_impl(
    y: &[f32],
    n_bands: usize,
    hop_length: usize,
    n_fft: usize,
) -> Result<Array2<f32>, RhythmError> {
    if y.is_empty() {
        return Err(RhythmError::InvalidInput("Signal is empty".into()));
    }
    if n_bands == 0 {
        return Err(RhythmError::InvalidInput("n_bands must be > 0".into()));
    }

    let spec = stft(y)
        .n_fft(n_fft)
        .hop_length(hop_length)
        .compute()
        .map_err(|e| RhythmError::InvalidInput(e.to_string()))?;

    let n_freqs = spec.shape()[0];
    let n_frames = spec.shape()[1];

    if n_frames == 0 {
        return Ok(Array2::zeros((n_bands, 0)));
    }

    // Log power spectrogram
    let log_power: Array2<f32> = spec.mapv(|c| {
        let p = c.norm_sqr();
        if p > 1e-10 { p.log10() * 10.0 } else { -100.0 }
    });

    let band_size = n_freqs.max(n_bands) / n_bands;
    let mut out = Array2::zeros((n_bands, n_frames.saturating_sub(1)));

    for band in 0..n_bands {
        let f_lo = band * band_size;
        let f_hi = ((band + 1) * band_size).min(n_freqs);
        if f_lo >= f_hi {
            continue;
        }
        for t in 1..n_frames {
            let mut flux = 0.0_f32;
            for f in f_lo..f_hi {
                let diff = log_power[[f, t]] - log_power[[f, t - 1]];
                flux += diff.max(0.0);
            }
            out[[band, t - 1]] = flux / (f_hi - f_lo) as f32;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onset_strength_empty() {
        let result = onset_strength_impl(&[], 512, 2048);
        assert!(result.is_err());
    }

    #[test]
    fn test_onset_strength_silence() {
        let y = vec![0.0_f32; 4096];
        let odf = onset_strength_impl(&y, 512, 2048).unwrap();
        assert!(!odf.is_empty());
        // Silence → all frames should be near zero
        assert!(odf.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn test_onset_detect_silence() {
        let y = vec![0.0_f32; 8192];
        let frames = onset_detect(&y, 44100).compute().unwrap();
        assert!(frames.is_empty(), "silence should produce no onsets");
    }

    #[test]
    fn test_beat_track_empty() {
        let result = beat_track_impl(&[], 44100, 512, 120.0, 100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_median_f32() {
        let mut v = vec![3.0, 1.0, 2.0];
        assert!((median_f32(&mut v) - 2.0).abs() < 1e-6);
        let mut v2 = vec![4.0, 1.0, 3.0, 2.0];
        assert!((median_f32(&mut v2) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_beat_sync_mean() {
        // 2 features, 10 frames: values are row index
        let data = ndarray::Array2::from_shape_fn((2, 10), |(r, _)| r as f32);
        let beats = vec![3usize, 7];
        let synced = beat_sync_impl(&data, &beats, Aggregate::Mean, true);
        // With pad=true: segments [0,3), [3,7), [7,10) → 3 columns
        assert_eq!(synced.shape(), [2, 3]);
        // Row 0 is all 0.0, row 1 is all 1.0
        assert!((synced[[0, 0]]).abs() < 1e-6);
        assert!((synced[[1, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_beat_sync_no_pad() {
        let data = ndarray::Array2::from_shape_fn((1, 10), |(_, c)| c as f32);
        let beats = vec![2usize, 5, 8];
        let synced = beat_sync_impl(&data, &beats, Aggregate::Max, false);
        // Without pad: segments [2,5), [5,8) → 2 columns (first beat is boundary only)
        assert_eq!(synced.shape()[1], 2);
    }

    #[test]
    fn test_aggregate_slice_median() {
        let v = vec![1.0_f32, 3.0, 5.0, 7.0];
        assert!((aggregate_slice(&v, Aggregate::Median) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_plp_empty() {
        assert!(plp_impl(&[], 44100, 512, 384, 30.0, 300.0).is_err());
    }

    #[test]
    fn test_plp_returns_normalized_curve() {
        let y = vec![0.0_f32; 8192];
        let result = plp_impl(&y, 44100, 512, 384, 30.0, 300.0);
        // Even for silence the function should succeed (all-zero odf still gives a tempogram)
        if let Ok((curve, _)) = result {
            assert!(!curve.is_empty());
            assert!(curve.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-6));
        }
    }
}
