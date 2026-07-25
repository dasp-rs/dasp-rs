//! Music structure analysis: self-similarity and cross-similarity matrices.

use ndarray::{Array2, ArrayView1};

/// Similarity metric for recurrence and cross-similarity computation.
#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    /// Cosine similarity (default).
    Cosine,
    /// Euclidean distance (converted to affinity via Gaussian kernel).
    Euclidean,
    /// Manhattan (L1) distance (converted to affinity via Gaussian kernel).
    Manhattan,
}

/// Output mode for recurrence/cross-similarity matrices.
#[derive(Debug, Clone, Copy)]
pub enum RecurrenceMode {
    /// Continuous affinity values in `[0, 1]` (default).
    Affinity,
    /// Binary: 1 where similarity exceeds the median, 0 elsewhere.
    Binary,
}

// ─── Recurrence matrix ────────────────────────────────────────────────────────

/// Builder for [`recurrence_matrix`].
#[derive(Debug, Clone)]
pub struct RecurrenceMatrixBuilder<'a> {
    data: &'a Array2<f32>,
    metric: SimilarityMetric,
    mode: RecurrenceMode,
    sym: bool,
    bandwidth: Option<f32>,
}

impl RecurrenceMatrixBuilder<'_> {
    /// Set the distance/similarity metric (default: [`SimilarityMetric::Cosine`]).
    #[must_use]
    pub fn metric(mut self, v: SimilarityMetric) -> Self {
        self.metric = v;
        self
    }

    /// Set the output mode (default: [`RecurrenceMode::Affinity`]).
    #[must_use]
    pub fn mode(mut self, v: RecurrenceMode) -> Self {
        self.mode = v;
        self
    }

    /// Symmetrize the output matrix: `R = (R + R^T) / 2` (default: `true`).
    #[must_use]
    pub fn sym(mut self, v: bool) -> Self {
        self.sym = v;
        self
    }

    /// Set the Gaussian bandwidth for distance-based affinity (default: median pairwise distance).
    ///
    /// Only used with [`SimilarityMetric::Euclidean`] or [`SimilarityMetric::Manhattan`] in
    /// [`RecurrenceMode::Affinity`] mode.
    #[must_use]
    pub fn bandwidth(mut self, v: f32) -> Self {
        self.bandwidth = Some(v);
        self
    }

    /// Compute the self-similarity matrix of shape `(n_frames, n_frames)`.
    pub fn compute(self) -> Array2<f32> {
        recurrence_matrix_impl(self.data, self.metric, self.mode, self.sym, self.bandwidth)
    }
}

/// Computes a self-similarity (recurrence) matrix from a feature sequence.
///
/// Each element `R[i, j]` represents the similarity between frames `i` and `j`
/// of the input feature matrix. Used for music structure analysis (verse/chorus
/// boundary detection, structural segmentation).
///
/// # Arguments
/// * `data` — Feature matrix of shape `(n_features, n_frames)`
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::recurrence_matrix;
/// use ndarray::Array2;
/// let chroma: Array2<f32> = Array2::zeros((12, 200));
/// let R = recurrence_matrix(&chroma).sym(true).compute();
/// assert_eq!(R.shape(), [200, 200]);
/// ```
pub fn recurrence_matrix(data: &Array2<f32>) -> RecurrenceMatrixBuilder<'_> {
    RecurrenceMatrixBuilder {
        data,
        metric: SimilarityMetric::Cosine,
        mode: RecurrenceMode::Affinity,
        sym: true,
        bandwidth: None,
    }
}

fn recurrence_matrix_impl(
    data: &Array2<f32>,
    metric: SimilarityMetric,
    mode: RecurrenceMode,
    sym: bool,
    bandwidth: Option<f32>,
) -> Array2<f32> {
    let n = data.shape()[1];
    if n == 0 {
        return Array2::zeros((0, 0));
    }

    let mut r = Array2::zeros((n, n));

    // Build raw similarity/distance values
    for i in 0..n {
        for j in i..n {
            let val = col_measure(data.column(i), data.column(j), metric);
            r[[i, j]] = val;
            r[[j, i]] = val;
        }
    }

    // Convert distances to affinities when needed
    match (metric, mode) {
        (SimilarityMetric::Cosine, RecurrenceMode::Affinity) => {
            // Cosine similarity is already in [-1, 1]; clamp to [0, 1]
            r.mapv_inplace(|x| x.max(0.0));
        }
        (SimilarityMetric::Cosine, RecurrenceMode::Binary) => {
            let median = array_median(r.as_slice().unwrap_or(&[]));
            r.mapv_inplace(|x| if x >= median { 1.0 } else { 0.0 });
        }
        (_, RecurrenceMode::Affinity) => {
            // Distance → Gaussian affinity
            let bw = bandwidth.unwrap_or_else(|| {
                let mut vals: Vec<f32> = r.iter().filter(|&&v| v > 0.0).copied().collect();
                if vals.is_empty() { return 1.0; }
                vals.sort_by(f32::total_cmp);
                vals[vals.len() / 2]
            });
            let bw2 = (bw * bw).max(1e-10);
            r.mapv_inplace(|d| (-d * d / bw2).exp());
        }
        (_, RecurrenceMode::Binary) => {
            let median = array_median(r.as_slice().unwrap_or(&[]));
            r.mapv_inplace(|d| if d <= median { 1.0 } else { 0.0 });
        }
    }

    if sym {
        let rt = r.t().to_owned();
        r = (&r + &rt).mapv(|x| x * 0.5);
    }

    r
}

// ─── Cross-similarity ─────────────────────────────────────────────────────────

/// Builder for [`cross_similarity`].
#[derive(Debug, Clone)]
pub struct CrossSimilarityBuilder<'a> {
    data: &'a Array2<f32>,
    query: &'a Array2<f32>,
    metric: SimilarityMetric,
    mode: RecurrenceMode,
    bandwidth: Option<f32>,
}

impl CrossSimilarityBuilder<'_> {
    /// Set the metric (default: [`SimilarityMetric::Cosine`]).
    #[must_use]
    pub fn metric(mut self, v: SimilarityMetric) -> Self {
        self.metric = v;
        self
    }

    /// Set the output mode (default: [`RecurrenceMode::Affinity`]).
    #[must_use]
    pub fn mode(mut self, v: RecurrenceMode) -> Self {
        self.mode = v;
        self
    }

    /// Set the Gaussian bandwidth for distance-based affinity.
    #[must_use]
    pub fn bandwidth(mut self, v: f32) -> Self {
        self.bandwidth = Some(v);
        self
    }

    /// Compute the cross-similarity matrix of shape `(n_frames_data, n_frames_query)`.
    pub fn compute(self) -> Array2<f32> {
        cross_similarity_impl(self.data, self.query, self.metric, self.mode, self.bandwidth)
    }
}

/// Computes a cross-similarity matrix between two feature sequences.
///
/// Element `C[i, j]` is the similarity between frame `i` of `data` and frame `j`
/// of `query`. Useful for cover-song detection and sequence alignment.
///
/// # Arguments
/// * `data` — Reference feature matrix of shape `(n_features, n_frames_data)`
/// * `query` — Query feature matrix of shape `(n_features, n_frames_query)`
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::cross_similarity;
/// use ndarray::Array2;
/// let a: Array2<f32> = Array2::zeros((12, 100));
/// let b: Array2<f32> = Array2::zeros((12, 80));
/// let C = cross_similarity(&a, &b).compute();
/// assert_eq!(C.shape(), [100, 80]);
/// ```
pub fn cross_similarity<'a>(
    data: &'a Array2<f32>,
    query: &'a Array2<f32>,
) -> CrossSimilarityBuilder<'a> {
    CrossSimilarityBuilder {
        data,
        query,
        metric: SimilarityMetric::Cosine,
        mode: RecurrenceMode::Affinity,
        bandwidth: None,
    }
}

fn cross_similarity_impl(
    data: &Array2<f32>,
    query: &Array2<f32>,
    metric: SimilarityMetric,
    mode: RecurrenceMode,
    bandwidth: Option<f32>,
) -> Array2<f32> {
    let nd = data.shape()[1];
    let nq = query.shape()[1];
    if nd == 0 || nq == 0 {
        return Array2::zeros((nd, nq));
    }

    let mut c = Array2::zeros((nd, nq));
    for i in 0..nd {
        for j in 0..nq {
            c[[i, j]] = col_measure(data.column(i), query.column(j), metric);
        }
    }

    match (metric, mode) {
        (SimilarityMetric::Cosine, RecurrenceMode::Affinity) => {
            c.mapv_inplace(|x| x.max(0.0));
        }
        (SimilarityMetric::Cosine, RecurrenceMode::Binary) => {
            let med = array_median(c.as_slice().unwrap_or(&[]));
            c.mapv_inplace(|x| if x >= med { 1.0 } else { 0.0 });
        }
        (_, RecurrenceMode::Affinity) => {
            // Matches `recurrence_matrix`'s default-bandwidth computation: zero
            // distances (e.g. an identical query/reference pair) are excluded so
            // the two functions agree when called on equivalent data.
            let bw = bandwidth.unwrap_or_else(|| {
                let mut vals: Vec<f32> = c.iter().filter(|&&v| v > 0.0).copied().collect();
                if vals.is_empty() { return 1.0; }
                vals.sort_by(f32::total_cmp);
                vals[vals.len() / 2]
            });
            let bw2 = (bw * bw).max(1e-10);
            c.mapv_inplace(|d| (-d * d / bw2).exp());
        }
        (_, RecurrenceMode::Binary) => {
            let med = array_median(c.as_slice().unwrap_or(&[]));
            c.mapv_inplace(|d| if d <= med { 1.0 } else { 0.0 });
        }
    }

    c
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn col_measure(a: ArrayView1<f32>, b: ArrayView1<f32>, metric: SimilarityMetric) -> f32 {
    match metric {
        SimilarityMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            let na = a.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
            let nb = b.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
            dot / (na * nb)
        }
        SimilarityMetric::Euclidean => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt(),
        SimilarityMetric::Manhattan => {
            a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
        }
    }
}

fn array_median(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        f32::midpoint(sorted[mid - 1], sorted[mid])
    } else {
        sorted[mid]
    }
}

// ─── Agglomerative segmentation ───────────────────────────────────────────────

/// Builder for [`agglomerative`].
#[derive(Debug, Clone)]
pub struct AgglomerativeBuilder<'a> {
    data: &'a Array2<f32>,
    k: usize,
    metric: SimilarityMetric,
}

impl AgglomerativeBuilder<'_> {
    /// Set the similarity metric for inter-segment comparison (default: [`SimilarityMetric::Cosine`]).
    #[must_use]
    pub fn metric(mut self, v: SimilarityMetric) -> Self {
        self.metric = v;
        self
    }

    /// Compute segment labels.
    ///
    /// Returns a `Vec<usize>` of length `n_frames` where `labels[t]` is the
    /// segment index (0-based) that frame `t` belongs to.
    pub fn compute(self) -> Vec<usize> {
        agglomerative_impl(self.data, self.k, self.metric)
    }
}

/// Segments a feature sequence into `k` contiguous groups via greedy agglomerative clustering.
///
/// Starts with each frame as its own cluster, then repeatedly merges the two
/// most similar adjacent segments until exactly `k` segments remain. Similarity
/// is computed between segment centroids.
///
/// # Arguments
/// * `data` — Feature matrix of shape `(n_features, n_frames)`.
/// * `k` — Target number of segments.
///
/// # Examples
/// ```
/// use dasp_rs::feat::agglomerative;
/// use ndarray::Array2;
/// let chroma: Array2<f32> = Array2::from_shape_fn((12, 100), |(i, j)| (i * j) as f32 * 0.01);
/// let labels = agglomerative(&chroma, 4).compute();
/// assert_eq!(labels.len(), 100);
/// assert!(*labels.iter().max().unwrap() < 4);
/// ```
pub fn agglomerative(data: &Array2<f32>, k: usize) -> AgglomerativeBuilder<'_> {
    AgglomerativeBuilder { data, k, metric: SimilarityMetric::Cosine }
}

fn agglomerative_impl(data: &Array2<f32>, k: usize, metric: SimilarityMetric) -> Vec<usize> {
    let n = data.shape()[1];
    if n == 0 { return vec![]; }
    let k = k.clamp(1, n);

    // Each frame starts as its own segment (start, exclusive_end)
    let mut segs: Vec<(usize, usize)> = (0..n).map(|i| (i, i + 1)).collect();

    while segs.len() > k {
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_i = 0;
        for i in 0..segs.len() - 1 {
            let sim = seg_centroid_sim(data, segs[i], segs[i + 1], metric);
            if sim > best_sim {
                best_sim = sim;
                best_i = i;
            }
        }
        // Merge segs[best_i] and segs[best_i+1]
        let merged = (segs[best_i].0, segs[best_i + 1].1);
        segs[best_i] = merged;
        segs.remove(best_i + 1);
    }

    let mut labels = vec![0usize; n];
    for (seg_idx, (start, end)) in segs.iter().enumerate() {
        for label in labels.iter_mut().take(*end).skip(*start) {
            *label = seg_idx;
        }
    }
    labels
}

fn seg_centroid_sim(
    data: &Array2<f32>,
    seg_a: (usize, usize),
    seg_b: (usize, usize),
    metric: SimilarityMetric,
) -> f32 {
    let nf = data.shape()[0];
    let centroid = |seg: (usize, usize)| -> Vec<f32> {
        let len = (seg.1 - seg.0) as f32;
        (0..nf)
            .map(|f| (seg.0..seg.1).map(|t| data[[f, t]]).sum::<f32>() / len)
            .collect()
    };
    let ca = centroid(seg_a);
    let cb = centroid(seg_b);

    // For distance metrics, negate so "max sim" = "min dist"
    match metric {
        SimilarityMetric::Cosine => {
            let dot: f32 = ca.iter().zip(&cb).map(|(&a, &b)| a * b).sum();
            let na = ca.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
            let nb = cb.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
            dot / (na * nb)
        }
        SimilarityMetric::Euclidean => {
            -ca.iter()
                .zip(&cb)
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        }
        SimilarityMetric::Manhattan => {
            -ca.iter().zip(&cb).map(|(&a, &b)| (a - b).abs()).sum::<f32>()
        }
    }
}

// ─── Subsegment ───────────────────────────────────────────────────────────────

/// Builder for [`subsegment`].
#[derive(Debug, Clone)]
pub struct SubsegmentBuilder<'a> {
    data: &'a Array2<f32>,
    frames: &'a [usize],
    k: usize,
    metric: SimilarityMetric,
}

impl SubsegmentBuilder<'_> {
    /// Set the similarity metric (default: [`SimilarityMetric::Cosine`]).
    #[must_use]
    pub fn metric(mut self, v: SimilarityMetric) -> Self {
        self.metric = v;
        self
    }

    /// Compute refined segment boundaries.
    ///
    /// Returns a sorted `Vec<usize>` containing all boundary frame indices
    /// (from both the original `frames` and the newly discovered sub-boundaries).
    /// Frame 0 and `n_frames` are excluded.
    pub fn compute(self) -> Vec<usize> {
        subsegment_impl(self.data, self.frames, self.k, self.metric)
    }
}

/// Refines existing segment boundaries by subdividing each segment into `k` parts.
///
/// For each interval between adjacent boundary frames, runs agglomerative
/// clustering with `k` target sub-segments and collects the resulting boundaries.
/// The output includes all original boundaries plus the new sub-boundaries.
///
/// # Arguments
/// * `data` — Feature matrix of shape `(n_features, n_frames)`.
/// * `frames` — Existing boundary frame indices (e.g., from beat tracking).
/// * `k` — Number of sub-segments per interval.
///
/// # Examples
/// ```
/// use dasp_rs::feat::subsegment;
/// use ndarray::Array2;
/// let data: Array2<f32> = Array2::from_shape_fn((12, 100), |(i, j)| (i + j) as f32 * 0.1);
/// let boundaries = vec![25, 50, 75];
/// let refined = subsegment(&data, &boundaries, 2).compute();
/// assert!(refined.len() >= boundaries.len());
/// ```
pub fn subsegment<'a>(
    data: &'a Array2<f32>,
    frames: &'a [usize],
    k: usize,
) -> SubsegmentBuilder<'a> {
    SubsegmentBuilder { data, frames, k, metric: SimilarityMetric::Cosine }
}

fn subsegment_impl(
    data: &Array2<f32>,
    frames: &[usize],
    k: usize,
    metric: SimilarityMetric,
) -> Vec<usize> {
    let n = data.shape()[1];
    if n == 0 || k == 0 { return vec![]; }

    // Build a sorted, deduped list of interval endpoints including 0 and n.
    let mut boundaries: Vec<usize> = std::iter::once(0)
        .chain(frames.iter().map(|&f| f.min(n)))
        .chain(std::iter::once(n))
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();

    let n_segs = boundaries.len().saturating_sub(1);
    if n_segs == 0 { return vec![]; }

    let mut result: Vec<usize> = Vec::new();

    for seg_idx in 0..n_segs {
        let seg_start = boundaries[seg_idx];
        let seg_end = boundaries[seg_idx + 1];
        let seg_len = seg_end - seg_start;

        // Always keep the original boundary (except frame 0)
        if seg_start > 0 {
            result.push(seg_start);
        }

        // Only subdivide if there's room for at least 2 sub-segments
        if seg_len < 2 || k <= 1 {
            continue;
        }

        let nf = data.shape()[0];
        let slice = Array2::from_shape_fn((nf, seg_len), |(f, t)| data[[f, seg_start + t]]);
        let sub_labels = agglomerative_impl(&slice, k.min(seg_len), metric);

        // Turn label changes into boundary indices in the global frame space
        let mut prev = sub_labels[0];
        for (i, &label) in sub_labels[1..].iter().enumerate() {
            if label != prev {
                result.push(seg_start + i + 1);
                prev = label;
            }
        }
    }

    result.sort_unstable();
    result.dedup();
    result
}

// ─── Path enhance ─────────────────────────────────────────────────────────────

/// Builder for [`path_enhance`].
#[derive(Debug, Clone)]
pub struct PathEnhanceBuilder<'a> {
    r: &'a Array2<f32>,
    n: usize,
}

impl PathEnhanceBuilder<'_> {
    /// Set the diagonal blur kernel width (default: 11).
    #[must_use]
    pub fn n(mut self, v: usize) -> Self { self.n = v; self }

    /// Compute the enhanced recurrence matrix.
    pub fn compute(self) -> Array2<f32> {
        path_enhance_impl(self.r, self.n)
    }
}

/// Enhances diagonal paths in a recurrence (self-similarity) matrix.
///
/// Applies a 1-D moving-average blur along the main-diagonal direction (i.e.,
/// each element is replaced by the mean of its `n` diagonal neighbours). This
/// suppresses isolated high-similarity cells and reinforces repeating structural
/// segments that appear as extended diagonal bands.
///
/// # Arguments
/// * `r` — Self-similarity matrix of shape `(n_frames, n_frames)`.
/// * `n` — Diagonal blur kernel width (must be odd; padded to nearest odd if even).
///
/// # Examples
/// ```
/// use dasp_rs::feat::path_enhance;
/// use ndarray::Array2;
/// let r = Array2::eye(10);
/// let enhanced = path_enhance(&r, 3).compute();
/// assert_eq!(enhanced.shape(), [10, 10]);
/// ```
pub fn path_enhance(r: &Array2<f32>, n: usize) -> PathEnhanceBuilder<'_> {
    PathEnhanceBuilder { r, n: n.max(1) }
}

fn path_enhance_impl(r: &Array2<f32>, n: usize) -> Array2<f32> {
    let (rows, cols) = r.dim();
    if rows == 0 || cols == 0 || n == 0 {
        return r.to_owned();
    }
    let half = (n / 2) as isize;
    let mut out = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0_f32;
            let mut cnt = 0_u32;
            for k in -half..=half {
                let ri = i as isize + k;
                let ci = j as isize + k;
                if ri >= 0 && ri < rows as isize && ci >= 0 && ci < cols as isize {
                    sum += r[[ri as usize, ci as usize]];
                    cnt += 1;
                }
            }
            out[[i, j]] = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
        }
    }
    out
}

// ─── Time-lag filter ──────────────────────────────────────────────────────────

/// Builder for [`timelag_filter`].
#[derive(Debug, Clone)]
pub struct TimeLagFilterBuilder<'a> {
    r: &'a Array2<f32>,
    n_window: usize,
}

impl TimeLagFilterBuilder<'_> {
    /// Median filter window size along the lag axis (default: 11).
    #[must_use]
    pub fn n_window(mut self, v: usize) -> Self { self.n_window = v.max(1); self }

    /// Compute the time-lag filtered matrix.
    pub fn compute(self) -> Array2<f32> {
        timelag_filter_impl(self.r, self.n_window)
    }
}

/// Applies a median filter in the time-lag domain to a recurrence matrix.
///
/// Converts the (time × time) recurrence matrix to a (lag × time) representation
/// by extracting each off-diagonal, applies a 1-D median filter along time for
/// each constant lag, then reconstructs a symmetric (time × time) matrix.
///
/// This suppresses spurious repeated elements while preserving genuine
/// periodic structure at each tempo lag.
///
/// # Arguments
/// * `r` — Self-similarity matrix, shape `(n_frames, n_frames)`.
///
/// # Examples
/// ```
/// use dasp_rs::feat::timelag_filter;
/// use ndarray::Array2;
/// let r = Array2::from_shape_fn((8, 8), |(i, j)| if i == j { 1.0_f32 } else { 0.0 });
/// let filtered = timelag_filter(&r).n_window(3).compute();
/// assert_eq!(filtered.shape(), [8, 8]);
/// ```
pub fn timelag_filter(r: &Array2<f32>) -> TimeLagFilterBuilder<'_> {
    TimeLagFilterBuilder { r, n_window: 11 }
}

fn timelag_filter_impl(r: &Array2<f32>, n_window: usize) -> Array2<f32> {
    let n = r.shape()[0].min(r.shape()[1]);
    if n == 0 { return r.to_owned(); }
    let half = (n_window / 2) as isize;
    let mut out = Array2::zeros((n, n));

    // For each lag d (0..n), extract diagonal, median-filter, write back
    for d in 0..n {
        let diag_len = n - d;
        // Raw diagonal values: r[t, t-d] and r[t-d, t] for t = d..n
        let raw: Vec<f32> = (d..n).map(|t| r[[t, t - d]]).collect();

        // Median filter along this diagonal
        let filtered: Vec<f32> = (0..diag_len)
            .map(|idx| {
                let lo = (idx as isize - half).max(0) as usize;
                let hi = (idx as isize + half + 1).min(diag_len as isize) as usize;
                let mut window = raw[lo..hi].to_vec();
                window.sort_by(f32::total_cmp);
                let mid = window.len() / 2;
                if window.len() % 2 == 0 {
                    f32::midpoint(window[mid - 1], window[mid])
                } else {
                    window[mid]
                }
            })
            .collect();

        // Write symmetric
        for (idx, &val) in filtered.iter().enumerate() {
            let t = idx + d;
            out[[t, t - d]] = val;
            if d > 0 {
                out[[t - d, t]] = val;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_recurrence_matrix_identity() {
        // Identical frames → cosine similarity = 1 everywhere
        let data = arr2(&[[1.0_f32, 1.0, 1.0], [0.0, 0.0, 0.0]]);
        let r = recurrence_matrix_impl(&data, SimilarityMetric::Cosine, RecurrenceMode::Affinity, false, None);
        assert_eq!(r.shape(), [3, 3]);
        for i in 0..3 {
            assert!((r[[i, i]] - 1.0).abs() < 1e-5, "diagonal should be 1");
        }
    }

    #[test]
    fn test_recurrence_matrix_orthogonal() {
        // Orthogonal frames → cosine similarity = 0 (clamped from 0)
        let data = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let r = recurrence_matrix_impl(&data, SimilarityMetric::Cosine, RecurrenceMode::Affinity, true, None);
        assert!((r[[0, 1]]).abs() < 1e-5, "orthogonal vectors → similarity 0");
    }

    #[test]
    fn test_cross_similarity_shape() {
        let a = Array2::from_elem((4, 10), 0.5_f32);
        let b = Array2::from_elem((4, 7), 0.5_f32);
        let c = cross_similarity_impl(&a, &b, SimilarityMetric::Cosine, RecurrenceMode::Affinity, None);
        assert_eq!(c.shape(), [10, 7]);
    }

    #[test]
    fn test_cross_similarity_self_equals_recurrence() {
        let data = Array2::from_shape_fn((3, 5), |(i, j)| (i + j) as f32 + 0.1);
        let r = recurrence_matrix_impl(&data, SimilarityMetric::Cosine, RecurrenceMode::Affinity, false, None);
        let c = cross_similarity_impl(&data, &data, SimilarityMetric::Cosine, RecurrenceMode::Affinity, None);
        for i in 0..5 {
            for j in 0..5 {
                assert!((r[[i, j]] - c[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_binary_mode() {
        let data = Array2::from_shape_fn((2, 4), |(_, j)| j as f32);
        let r = recurrence_matrix_impl(&data, SimilarityMetric::Cosine, RecurrenceMode::Binary, false, None);
        assert!(r.iter().all(|&v| v == 0.0 || v == 1.0));
    }
}
