//! Array and signal utility functions: framing, padding, peak-picking, sync, interval matching.

use ndarray::{Array1, Array2, Axis};

// ─── Local extrema ────────────────────────────────────────────────────────────

/// Returns a boolean mask marking local maxima of `x`.
///
/// Position `i` is a local maximum when `x[i] > x[i-1]` (strictly) and
/// `x[i] >= x[i+1]` (non-strictly).
/// Boundary positions are compared against only one neighbour.
///
/// # Examples
/// ```
/// use dasp_rs::util::localmax;
/// let x = vec![0.0_f32, 1.0, 0.5, 2.0, 0.0];
/// let m = localmax(&x);
/// assert!(m[1]);   // 1.0 > 0.0 and 1.0 >= 0.5
/// assert!(m[3]);   // 2.0 > 0.5 and 2.0 >= 0.0
/// assert!(!m[0]);
/// assert!(!m[4]);
/// ```
pub fn localmax(x: &[f32]) -> Vec<bool> {
    let n = x.len();
    if n == 0 { return vec![]; }
    let mut out = vec![false; n];
    for i in 0..n {
        let prev = if i > 0 { x[i] > x[i - 1] } else { true };
        let next = if i + 1 < n { x[i] >= x[i + 1] } else { true };
        out[i] = prev && next;
    }
    out
}

/// Returns a boolean mask marking local minima of `x`.
///
/// Position `i` is a local minimum when `x[i] < x[i-1]` (strictly) and
/// `x[i] <= x[i+1]` (non-strictly).
///
/// # Examples
/// ```
/// use dasp_rs::util::localmin;
/// let x = vec![1.0_f32, 0.0, 0.5, -1.0, 0.0];
/// let m = localmin(&x);
/// assert!(m[1]);   // 0.0 < 1.0 and 0.0 <= 0.5
/// assert!(m[3]);   // -1.0 < 0.5 and -1.0 <= 0.0
/// ```
pub fn localmin(x: &[f32]) -> Vec<bool> {
    let n = x.len();
    if n == 0 { return vec![]; }
    let mut out = vec![false; n];
    for i in 0..n {
        let prev = if i > 0 { x[i] < x[i - 1] } else { true };
        let next = if i + 1 < n { x[i] <= x[i + 1] } else { true };
        out[i] = prev && next;
    }
    out
}

// ─── Peak pick ────────────────────────────────────────────────────────────────

/// Builder for [`peak_pick`].
#[derive(Debug, Clone)]
pub struct PeakPickBuilder<'a> {
    x: &'a [f32],
    pre_max: usize,
    post_max: usize,
    pre_avg: usize,
    post_avg: usize,
    delta: f32,
    wait: usize,
}

impl PeakPickBuilder<'_> {
    /// Samples before peak that must all be ≤ peak (default: 3).
    #[must_use]
    pub fn pre_max(mut self, v: usize) -> Self { self.pre_max = v; self }
    /// Samples after peak that must all be ≤ peak (default: 3).
    #[must_use]
    pub fn post_max(mut self, v: usize) -> Self { self.post_max = v; self }
    /// Samples before peak used to compute the mean threshold (default: 3).
    #[must_use]
    pub fn pre_avg(mut self, v: usize) -> Self { self.pre_avg = v; self }
    /// Samples after peak used to compute the mean threshold (default: 3).
    #[must_use]
    pub fn post_avg(mut self, v: usize) -> Self { self.post_avg = v; self }
    /// Minimum amount by which peak must exceed the local mean (default: 0.07).
    #[must_use]
    pub fn delta(mut self, v: f32) -> Self { self.delta = v; self }
    /// Minimum number of samples between consecutive peaks (default: 30).
    #[must_use]
    pub fn wait(mut self, v: usize) -> Self { self.wait = v; self }

    /// Compute peak indices.
    pub fn compute(self) -> Vec<usize> {
        peak_pick_impl(
            self.x,
            self.pre_max,
            self.post_max,
            self.pre_avg,
            self.post_avg,
            self.delta,
            self.wait,
        )
    }
}

/// Picks peaks from a 1-D array using a moving-maximum + moving-mean threshold.
///
/// A sample `x[i]` is a peak when:
/// 1. It is a local maximum over `[i - pre_max, i + post_max]`.
/// 2. It exceeds the local mean over `[i - pre_avg, i + post_avg]` by at least `delta`.
/// 3. It is at least `wait` samples away from any previously accepted peak.
///
/// Used internally by onset detection and pitch tracking; also useful standalone.
///
/// # Examples
/// ```
/// use dasp_rs::util::peak_pick;
/// let x = vec![0.0_f32, 0.5, 1.0, 0.5, 0.0, 0.0, 0.8, 0.0];
/// let peaks = peak_pick(&x).pre_max(1).post_max(1).pre_avg(2).post_avg(2).delta(0.1).wait(2).compute();
/// assert!(peaks.contains(&2));
/// ```
pub fn peak_pick(x: &[f32]) -> PeakPickBuilder<'_> {
    PeakPickBuilder { x, pre_max: 3, post_max: 3, pre_avg: 3, post_avg: 3, delta: 0.07, wait: 30 }
}

fn peak_pick_impl(
    x: &[f32],
    pre_max: usize,
    post_max: usize,
    pre_avg: usize,
    post_avg: usize,
    delta: f32,
    wait: usize,
) -> Vec<usize> {
    let n = x.len();
    if n == 0 { return vec![]; }

    let mut peaks = Vec::new();
    let mut last_peak: Option<usize> = None;

    for i in 0..n {
        let lo_max = i.saturating_sub(pre_max);
        let hi_max = (i + post_max + 1).min(n);
        let lo_avg = i.saturating_sub(pre_avg);
        let hi_avg = (i + post_avg + 1).min(n);

        // Local max check
        let local_max = x[lo_max..hi_max].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if x[i] < local_max {
            continue;
        }

        // Must be strictly larger than at least one neighbour
        let is_local_max = (i == 0 || x[i] > x[i - 1]) || (i + 1 == n || x[i] >= x[i + 1]);
        if !is_local_max {
            continue;
        }

        // Mean threshold check
        let window = &x[lo_avg..hi_avg];
        let mean = window.iter().sum::<f32>() / window.len() as f32;
        if x[i] < mean + delta {
            continue;
        }

        // Wait constraint
        if let Some(last) = last_peak {
            if i - last < wait {
                continue;
            }
        }

        peaks.push(i);
        last_peak = Some(i);
    }
    peaks
}

// ─── Frame ────────────────────────────────────────────────────────────────────

/// Builder for [`frame`].
#[derive(Debug, Clone)]
pub struct FrameBuilder<'a> {
    y: &'a [f32],
    frame_length: usize,
    hop_length: usize,
}

impl FrameBuilder<'_> {
    /// Set the hop length in samples (default: 512).
    #[must_use]
    pub fn hop_length(mut self, v: usize) -> Self { self.hop_length = v; self }

    /// Compute the framed matrix of shape `(frame_length, n_frames)`.
    pub fn compute(self) -> Array2<f32> {
        frame_impl(self.y, self.frame_length, self.hop_length)
    }
}

/// Slices a 1-D signal into overlapping frames.
///
/// Returns a matrix of shape `(frame_length, n_frames)` where each column is
/// one frame. The number of frames is `(len(y) - frame_length) / hop_length + 1`.
/// Frames that would extend past the end of `y` are not included.
///
/// # Examples
/// ```
/// use dasp_rs::util::frame;
/// let y: Vec<f32> = (0..20).map(|i| i as f32).collect();
/// let f = frame(&y, 4).hop_length(2).compute();
/// assert_eq!(f.shape(), [4, 9]); // (20-4)/2 + 1 = 9 frames
/// assert_eq!(f[[0, 0]], 0.0);
/// assert_eq!(f[[0, 1]], 2.0);
/// ```
pub fn frame(y: &[f32], frame_length: usize) -> FrameBuilder<'_> {
    FrameBuilder { y, frame_length, hop_length: 512 }
}

fn frame_impl(y: &[f32], frame_length: usize, hop_length: usize) -> Array2<f32> {
    let n = y.len();
    if frame_length == 0 || hop_length == 0 || n < frame_length {
        return Array2::zeros((frame_length.max(1), 0));
    }
    let n_frames = (n - frame_length) / hop_length + 1;
    Array2::from_shape_fn((frame_length, n_frames), |(row, col)| {
        y[col * hop_length + row]
    })
}

// ─── Pad center ───────────────────────────────────────────────────────────────

/// Zero-pads `data` symmetrically so that it is centered in a buffer of length `size`.
///
/// If `size < data.len()` the data is returned truncated to `size`. If `size == data.len()`
/// the data is returned unchanged.
///
/// # Examples
/// ```
/// use dasp_rs::util::pad_center;
/// let x = vec![1.0_f32, 2.0, 3.0];
/// let p = pad_center(&x, 7);
/// assert_eq!(p.len(), 7);
/// assert_eq!(p[2], 1.0);
/// assert_eq!(p[4], 3.0);
/// ```
pub fn pad_center(data: &[f32], size: usize) -> Vec<f32> {
    if size <= data.len() {
        return data[..size].to_vec();
    }
    let total_pad = size - data.len();
    let left = total_pad / 2;
    let mut out = vec![0.0_f32; size];
    out[left..left + data.len()].copy_from_slice(data);
    out
}

// ─── Fix length ───────────────────────────────────────────────────────────────

/// Pads or truncates `data` to exactly `size` samples.
///
/// - If `data.len() < size`, the tail is zero-padded.
/// - If `data.len() > size`, the tail is truncated.
///
/// # Examples
/// ```
/// use dasp_rs::util::fix_length;
/// assert_eq!(fix_length(&[1.0_f32, 2.0], 4), vec![1.0, 2.0, 0.0, 0.0]);
/// assert_eq!(fix_length(&[1.0_f32, 2.0, 3.0, 4.0], 2), vec![1.0, 2.0]);
/// ```
pub fn fix_length(data: &[f32], size: usize) -> Vec<f32> {
    if data.len() >= size {
        data[..size].to_vec()
    } else {
        let mut out = data.to_vec();
        out.resize(size, 0.0);
        out
    }
}

// ─── Sync ─────────────────────────────────────────────────────────────────────

/// Aggregation mode for [`sync`].
#[derive(Debug, Clone, Copy)]
pub enum SyncAggregate {
    /// Arithmetic mean (default).
    Mean,
    /// Median.
    Median,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

/// Builder for [`sync`].
#[derive(Debug, Clone)]
pub struct SyncBuilder<'a> {
    data: &'a Array2<f32>,
    frames: &'a [usize],
    aggregate: SyncAggregate,
    pad: bool,
}

impl SyncBuilder<'_> {
    /// Set the aggregation function (default: [`SyncAggregate::Mean`]).
    #[must_use]
    pub fn aggregate(mut self, v: SyncAggregate) -> Self { self.aggregate = v; self }

    /// If `true`, prepend a segment `[0, frames[0])` and append `[frames[-1], n_frames)`
    /// so the entire time axis is covered (default: `true`).
    #[must_use]
    pub fn pad(mut self, v: bool) -> Self { self.pad = v; self }

    /// Compute the synchronized feature matrix.
    pub fn compute(self) -> Array2<f32> {
        sync_impl(self.data, self.frames, self.aggregate, self.pad)
    }
}

/// Synchronizes a feature matrix to a set of frame boundaries.
///
/// Aggregates columns of `data` within each segment defined by consecutive
/// entries in `frames`, returning a `(n_features, n_segments)` matrix.
/// This is a general-purpose version of beat-synchronous feature extraction.
///
/// # Arguments
/// * `data` — Feature matrix of shape `(n_features, n_frames)`.
/// * `frames` — Sorted boundary frame indices (exclusive start of each segment).
///
/// # Examples
/// ```
/// use dasp_rs::util::sync;
/// use ndarray::Array2;
/// let data = Array2::from_shape_fn((3, 10), |(i, j)| (i + j) as f32);
/// let frames = vec![3, 6];
/// let s = sync(&data, &frames).compute();
/// assert_eq!(s.shape()[1], 3); // 3 segments: [0,3), [3,6), [6,10)
/// ```
pub fn sync<'a>(data: &'a Array2<f32>, frames: &'a [usize]) -> SyncBuilder<'a> {
    SyncBuilder { data, frames, aggregate: SyncAggregate::Mean, pad: true }
}

fn sync_impl(
    data: &Array2<f32>,
    frames: &[usize],
    aggregate: SyncAggregate,
    pad: bool,
) -> Array2<f32> {
    let n_frames = data.shape()[1];
    let n_features = data.shape()[0];

    let mut bounds: Vec<usize> = frames.iter().map(|&f| f.min(n_frames)).collect();
    if pad {
        if bounds.first() != Some(&0) { bounds.insert(0, 0); }
        if bounds.last() != Some(&n_frames) { bounds.push(n_frames); }
    }
    bounds.dedup();

    let n_segs = bounds.len().saturating_sub(1);
    if n_segs == 0 { return Array2::zeros((n_features, 0)); }

    let mut out = Array2::zeros((n_features, n_segs));
    for s in 0..n_segs {
        let lo = bounds[s];
        let hi = bounds[s + 1].min(n_frames);
        if lo >= hi { continue; }
        for f in 0..n_features {
            let vals: Vec<f32> = (lo..hi).map(|t| data[[f, t]]).collect();
            out[[f, s]] = aggregate_vals(&vals, aggregate);
        }
    }
    out
}

fn aggregate_vals(vals: &[f32], mode: SyncAggregate) -> f32 {
    if vals.is_empty() { return 0.0; }
    match mode {
        SyncAggregate::Mean => vals.iter().sum::<f32>() / vals.len() as f32,
        SyncAggregate::Max => vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        SyncAggregate::Min => vals.iter().copied().fold(f32::INFINITY, f32::min),
        SyncAggregate::Median => {
            let mut v = vals.to_vec();
            v.sort_by(f32::total_cmp);
            let mid = v.len() / 2;
            if v.len() % 2 == 0 { f32::midpoint(v[mid - 1], v[mid]) } else { v[mid] }
        }
    }
}

// ─── Match intervals ──────────────────────────────────────────────────────────

/// For each interval in `intervals_from`, finds the index of the best-matching
/// interval in `intervals_to` by maximizing overlap duration.
///
/// # Arguments
/// * `intervals_from` — Slice of `(start, end)` pairs (the query set).
/// * `intervals_to` — Slice of `(start, end)` pairs (the reference set).
///
/// # Returns
/// A `Vec<usize>` of length `intervals_from.len()` where each entry is the index
/// into `intervals_to` of the best match. Returns an empty Vec if `intervals_to`
/// is empty.
///
/// # Examples
/// ```
/// use dasp_rs::util::match_intervals;
/// let from = vec![(0.0_f32, 1.0), (2.0, 3.0)];
/// let to   = vec![(0.0_f32, 0.8), (1.9, 3.1)];
/// let idx = match_intervals(&from, &to);
/// assert_eq!(idx[0], 0); // (0,1) best matches (0,0.8)
/// assert_eq!(idx[1], 1); // (2,3) best matches (1.9,3.1)
/// ```
pub fn match_intervals(
    intervals_from: &[(f32, f32)],
    intervals_to: &[(f32, f32)],
) -> Vec<usize> {
    if intervals_to.is_empty() { return vec![]; }
    intervals_from
        .iter()
        .map(|&(a_start, a_end)| {
            intervals_to
                .iter()
                .enumerate()
                .max_by(|(_, b), (_, c)| {
                    let ov_b = (a_end.min(b.1) - a_start.max(b.0)).max(0.0);
                    let ov_c = (a_end.min(c.1) - a_start.max(c.0)).max(0.0);
                    ov_b.total_cmp(&ov_c)
                })
                .map_or(0, |(i, _)| i)
        })
        .collect()
}

// ─── Expand to ────────────────────────────────────────────────────────────────

/// Expands a 1-D array to a 2-D matrix by inserting a singleton axis.
///
/// * `axis = 0` — data varies along rows: result shape `(n, 1)`.
/// * `axis = 1` — data varies along columns: result shape `(1, n)`.
///
/// Useful for broadcasting 1-D frequency weights or time envelopes against 2-D
/// spectrograms.
///
/// # Examples
/// ```
/// use dasp_rs::util::expand_to;
/// use ndarray::Array1;
/// let x = Array1::from(vec![1.0_f32, 2.0, 3.0]);
/// let col = expand_to(&x, 0); // (3, 1)
/// assert_eq!(col.shape(), [3, 1]);
/// let row = expand_to(&x, 1); // (1, 3)
/// assert_eq!(row.shape(), [1, 3]);
/// ```
pub fn expand_to(x: &Array1<f32>, axis: usize) -> Array2<f32> {
    if axis == 0 {
        x.view().insert_axis(Axis(1)).to_owned().into_dimensionality().unwrap_or_else(|_| {
            Array2::from_shape_fn((x.len(), 1), |(i, _)| x[i])
        })
    } else {
        x.view().insert_axis(Axis(0)).to_owned().into_dimensionality().unwrap_or_else(|_| {
            Array2::from_shape_fn((1, x.len()), |(_, j)| x[j])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_localmax_basic() {
        let x = vec![0.0_f32, 1.0, 0.5, 2.0, 0.0];
        let m = localmax(&x);
        assert!(m[1], "1.0 is local max");
        assert!(m[3], "2.0 is local max");
        assert!(!m[2]);
        assert!(!m[4]);
    }

    #[test]
    fn test_localmin_basic() {
        let x = vec![1.0_f32, 0.0, 0.5, -1.0, 0.0];
        let m = localmin(&x);
        assert!(m[1]);
        assert!(m[3]);
    }

    #[test]
    fn test_peak_pick_finds_obvious_peak() {
        let mut x = vec![0.0_f32; 30];
        x[10] = 1.0;
        let peaks = peak_pick_impl(&x, 3, 3, 3, 3, 0.07, 5);
        assert!(peaks.contains(&10));
    }

    #[test]
    fn test_peak_pick_empty() {
        assert!(peak_pick_impl(&[], 3, 3, 3, 3, 0.07, 5).is_empty());
    }

    #[test]
    fn test_frame_shape() {
        let y: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let f = frame_impl(&y, 4, 2);
        assert_eq!(f.shape(), [4, 9]);
    }

    #[test]
    fn test_frame_content() {
        let y: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let f = frame_impl(&y, 3, 1);
        assert_eq!(f[[0, 0]], 0.0);
        assert_eq!(f[[1, 0]], 1.0);
        assert_eq!(f[[0, 1]], 1.0);
    }

    #[test]
    fn test_pad_center_symmetric() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let p = pad_center(&x, 7);
        assert_eq!(p.len(), 7);
        assert_eq!(p[2], 1.0);
        assert_eq!(p[4], 3.0);
    }

    #[test]
    fn test_fix_length_pad() {
        let v = fix_length(&[1.0_f32, 2.0], 5);
        assert_eq!(v, vec![1.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fix_length_truncate() {
        let v = fix_length(&[1.0_f32, 2.0, 3.0, 4.0], 2);
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sync_segments() {
        let data = Array2::from_shape_fn((2, 6), |(i, j)| (i + j) as f32);
        let frames = vec![3];
        let s = sync_impl(&data, &frames, SyncAggregate::Mean, true);
        assert_eq!(s.shape()[1], 2); // [0,3) and [3,6)
    }

    #[test]
    fn test_match_intervals_basic() {
        let from = vec![(0.0_f32, 1.0), (2.0, 3.0)];
        let to   = vec![(0.0_f32, 0.8), (1.9, 3.1)];
        let idx = match_intervals(&from, &to);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[1], 1);
    }

    #[test]
    fn test_expand_to_column() {
        let x = Array1::from(vec![1.0_f32, 2.0, 3.0]);
        let m = expand_to(&x, 0);
        assert_eq!(m.shape(), [3, 1]);
        assert_eq!(m[[1, 0]], 2.0);
    }

    #[test]
    fn test_expand_to_row() {
        let x = Array1::from(vec![1.0_f32, 2.0, 3.0]);
        let m = expand_to(&x, 1);
        assert_eq!(m.shape(), [1, 3]);
        assert_eq!(m[[0, 2]], 3.0);
    }
}
