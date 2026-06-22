//! Feature-level matrix utilities: masking, normalization, sparsification.

use ndarray::Array2;

// ─── Soft-mask ────────────────────────────────────────────────────────────────

/// Builder for [`softmask`].
#[derive(Debug, Clone)]
pub struct SoftmaskBuilder<'a> {
    x: &'a Array2<f32>,
    x_ref: &'a Array2<f32>,
    power: f32,
    split_zeros: bool,
}

impl SoftmaskBuilder<'_> {
    /// Exponent applied to both numerator and denominator (default: `1.0`).
    ///
    /// Higher values produce sharper (harder) masks. Use `f32::INFINITY` for a
    /// hard binary mask.
    #[must_use]
    pub fn power(mut self, v: f32) -> Self {
        self.power = v;
        self
    }

    /// When `true`, cells where both `x` and `x_ref` are zero receive a mask
    /// value of `0.5` instead of `0.0` (default: `false`).
    #[must_use]
    pub fn split_zeros(mut self, v: bool) -> Self {
        self.split_zeros = v;
        self
    }

    /// Compute the soft mask.
    pub fn compute(self) -> Array2<f32> {
        softmask_impl(self.x, self.x_ref, self.power, self.split_zeros)
    }
}

/// Computes a soft Wiener-filter mask from two non-negative spectrograms.
///
/// Returns a mask `M` in `[0, 1]` such that `M[i,j] = x[i,j]^p / (x[i,j]^p + x_ref[i,j]^p)`.
/// The complement `1 − M` can be applied to `x_ref` so both masks sum to 1.
///
/// Typical use: estimate a harmonic/noise mask from two HPSS spectrograms, then
/// apply to the original complex STFT for source separation.
///
/// # Arguments
/// * `x` — Non-negative spectrogram (numerator component), shape `(n_bins, n_frames)`.
/// * `x_ref` — Non-negative reference spectrogram (denominator component), same shape.
///
/// # Examples
/// ```
/// use dasp_rs::feat::softmask;
/// use ndarray::Array2;
/// let h = Array2::from_elem((128, 50), 3.0_f32);
/// let p = Array2::from_elem((128, 50), 1.0_f32);
/// let mask = softmask(&h, &p).compute();
/// // Each element ≈ 3/(3+1) = 0.75
/// assert!((mask[[0, 0]] - 0.75).abs() < 1e-5);
/// ```
pub fn softmask<'a>(x: &'a Array2<f32>, x_ref: &'a Array2<f32>) -> SoftmaskBuilder<'a> {
    SoftmaskBuilder { x, x_ref, power: 1.0, split_zeros: false }
}

fn softmask_impl(
    x: &Array2<f32>,
    x_ref: &Array2<f32>,
    power: f32,
    split_zeros: bool,
) -> Array2<f32> {
    const EPS: f32 = 1e-16;
    Array2::from_shape_fn(x.raw_dim(), |(i, j)| {
        let a = x[[i, j]];
        let b = x_ref[[i, j]];
        if power == f32::INFINITY {
            if a > b { 1.0 } else if a < b { 0.0 } else { 0.5 }
        } else {
            let ap = a.powf(power);
            let bp = b.powf(power);
            let total = ap + bp;
            if total < EPS {
                if split_zeros { 0.5 } else { 0.0 }
            } else {
                ap / total
            }
        }
    })
}

// ─── Feature normalization ────────────────────────────────────────────────────

/// Norm type for [`normalize`].
#[derive(Debug, Clone, Copy)]
pub enum FeatureNorm {
    /// L1 norm (sum of absolute values).
    L1,
    /// L2 norm (Euclidean length; default).
    L2,
    /// L∞ norm (maximum absolute value).
    LInf,
}

/// Builder for [`normalize`].
#[derive(Debug, Clone)]
pub struct NormalizeBuilder<'a> {
    x: &'a Array2<f32>,
    norm: FeatureNorm,
    axis: usize,
    threshold: Option<f32>,
}

impl NormalizeBuilder<'_> {
    /// Set the norm type (default: [`FeatureNorm::L2`]).
    #[must_use]
    pub fn norm(mut self, v: FeatureNorm) -> Self {
        self.norm = v;
        self
    }

    /// Axis along which to normalize (default: `0` — normalize each column/frame).
    #[must_use]
    pub fn axis(mut self, v: usize) -> Self {
        self.axis = v;
        self
    }

    /// Minimum norm value below which a slice is left unchanged (default: `1e-10`).
    #[must_use]
    pub fn threshold(mut self, v: f32) -> Self {
        self.threshold = Some(v);
        self
    }

    /// Compute the normalized array.
    pub fn compute(self) -> Array2<f32> {
        normalize_impl(self.x, self.norm, self.axis, self.threshold.unwrap_or(1e-10))
    }
}

/// Normalizes a feature matrix along an axis.
///
/// Divides each vector (column or row) by its norm so that the result has unit
/// norm along the chosen axis. Vectors whose norm is below `threshold` are
/// returned unchanged.
///
/// # Arguments
/// * `x` — Feature matrix of shape `(n_features, n_frames)`.
///
/// # Examples
/// ```
/// use dasp_rs::feat::{normalize, FeatureNorm};
/// use ndarray::arr2;
/// let x = arr2(&[[3.0_f32, 0.0], [4.0, 0.0]]);
/// let n = normalize(&x).norm(FeatureNorm::L2).axis(0).compute();
/// assert!((n[[0, 0]] - 0.6).abs() < 1e-5);
/// assert!((n[[1, 0]] - 0.8).abs() < 1e-5);
/// ```
pub fn normalize(x: &Array2<f32>) -> NormalizeBuilder<'_> {
    NormalizeBuilder { x, norm: FeatureNorm::L2, axis: 0, threshold: None }
}

fn normalize_impl(x: &Array2<f32>, norm: FeatureNorm, axis: usize, threshold: f32) -> Array2<f32> {
    let mut out = x.to_owned();
    match axis {
        0 => {
            for j in 0..x.ncols() {
                let col: Vec<f32> = x.column(j).iter().copied().collect();
                let n = norm_value(&col, norm);
                if n > threshold {
                    for i in 0..x.nrows() {
                        out[[i, j]] /= n;
                    }
                }
            }
        }
        _ => {
            for i in 0..x.nrows() {
                let row: Vec<f32> = x.row(i).iter().copied().collect();
                let n = norm_value(&row, norm);
                if n > threshold {
                    for j in 0..x.ncols() {
                        out[[i, j]] /= n;
                    }
                }
            }
        }
    }
    out
}

fn norm_value(v: &[f32], norm: FeatureNorm) -> f32 {
    match norm {
        FeatureNorm::L1 => v.iter().map(|&x| x.abs()).sum(),
        FeatureNorm::L2 => v.iter().map(|&x| x * x).sum::<f32>().sqrt(),
        FeatureNorm::LInf => v.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max),
    }
}

// ─── Sparsify rows ────────────────────────────────────────────────────────────

/// Builder for [`sparsify_rows`].
#[derive(Debug, Clone)]
pub struct SparsifyRowsBuilder<'a> {
    x: &'a Array2<f32>,
    quantile: f32,
}

impl SparsifyRowsBuilder<'_> {
    /// Fraction of the absolute-value distribution to zero out per row (default: `0.01`).
    ///
    /// A value of `0.01` zeros all entries whose absolute value falls below the
    /// 1st percentile of their row.
    #[must_use]
    pub fn quantile(mut self, v: f32) -> Self {
        self.quantile = v;
        self
    }

    /// Compute the sparsified matrix.
    pub fn compute(self) -> Array2<f32> {
        sparsify_rows_impl(self.x, self.quantile)
    }
}

/// Zeros out small entries in each row to produce a sparse representation.
///
/// For each row, computes the `quantile`-th percentile of absolute values and
/// sets all entries below that threshold to zero.
///
/// # Arguments
/// * `x` — Input feature matrix of shape `(n_features, n_frames)`.
///
/// # Examples
/// ```
/// use dasp_rs::feat::sparsify_rows;
/// use ndarray::arr2;
/// let x = arr2(&[[0.001_f32, 1.0, 2.0, 3.0]]);
/// let s = sparsify_rows(&x).quantile(0.25).compute();
/// assert_eq!(s[[0, 0]], 0.0, "smallest value should be zeroed");
/// ```
pub fn sparsify_rows(x: &Array2<f32>) -> SparsifyRowsBuilder<'_> {
    SparsifyRowsBuilder { x, quantile: 0.01 }
}

fn sparsify_rows_impl(x: &Array2<f32>, quantile: f32) -> Array2<f32> {
    let mut out = x.to_owned();
    for i in 0..x.nrows() {
        let mut vals: Vec<f32> = x.row(i).iter().map(|&v| v.abs()).collect();
        if vals.is_empty() {
            continue;
        }
        vals.sort_by(f32::total_cmp);
        let idx = ((vals.len() as f32 * quantile.clamp(0.0, 1.0)) as usize)
            .min(vals.len().saturating_sub(1));
        let threshold = vals[idx];
        for j in 0..x.ncols() {
            if out[[i, j]].abs() < threshold {
                out[[i, j]] = 0.0;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};

    // softmask

    #[test]
    fn test_softmask_ratio() {
        let x = Array2::from_elem((2, 3), 3.0_f32);
        let r = Array2::from_elem((2, 3), 1.0_f32);
        let m = softmask_impl(&x, &r, 1.0, false);
        assert!((m[[0, 0]] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_softmask_split_zeros() {
        let x: Array2<f32> = Array2::zeros((2, 2));
        let r: Array2<f32> = Array2::zeros((2, 2));
        let m_no_split = softmask_impl(&x, &r, 1.0, false);
        let m_split = softmask_impl(&x, &r, 1.0, true);
        assert_eq!(m_no_split[[0, 0]], 0.0);
        assert_eq!(m_split[[0, 0]], 0.5);
    }

    #[test]
    fn test_softmask_hard_mask() {
        let x = arr2(&[[2.0_f32, 0.5], [1.0, 3.0]]);
        let r = arr2(&[[1.0_f32, 1.5], [1.0, 1.0]]);
        let m = softmask_impl(&x, &r, f32::INFINITY, false);
        assert_eq!(m[[0, 0]], 1.0); // 2 > 1
        assert_eq!(m[[0, 1]], 0.0); // 0.5 < 1.5
    }

    // normalize

    #[test]
    fn test_normalize_l2_columns() {
        let x = arr2(&[[3.0_f32, 0.0], [4.0, 0.0]]);
        let n = normalize_impl(&x, FeatureNorm::L2, 0, 1e-10);
        assert!((n[[0, 0]] - 0.6).abs() < 1e-5);
        assert!((n[[1, 0]] - 0.8).abs() < 1e-5);
        // Zero column should be unchanged
        assert_eq!(n[[0, 1]], 0.0);
    }

    #[test]
    fn test_normalize_l1_rows() {
        let x = arr2(&[[1.0_f32, 2.0, 3.0]]);
        let n = normalize_impl(&x, FeatureNorm::L1, 1, 1e-10);
        // Row sum of abs = 6, so each divided by 6
        assert!((n[[0, 0]] - 1.0 / 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_linf() {
        let x = arr2(&[[1.0_f32, -4.0, 2.0]]);
        let n = normalize_impl(&x, FeatureNorm::LInf, 1, 1e-10);
        assert!((n[[0, 1]] + 1.0).abs() < 1e-5); // -4/4 = -1
    }

    // sparsify_rows

    #[test]
    fn test_sparsify_zeros_smallest() {
        let x = arr2(&[[0.001_f32, 1.0, 2.0, 3.0]]);
        let s = sparsify_rows_impl(&x, 0.25);
        assert_eq!(s[[0, 0]], 0.0, "0.001 is below 25th percentile and should be zeroed");
    }

    #[test]
    fn test_sparsify_zero_quantile_preserves_all() {
        let x = Array2::from_shape_fn((3, 5), |(i, j)| (i * j) as f32 + 0.1);
        let s = sparsify_rows_impl(&x, 0.0);
        for ((i, j), &v) in x.indexed_iter() {
            assert_eq!(s[[i, j]], v);
        }
    }
}
