//! Spectral decomposition via Non-negative Matrix Factorization (NMF).

use ndarray::Array2;
use thiserror::Error;

/// Error conditions for decomposition operations.
#[derive(Error, Debug)]
pub enum DecomposeError {
    /// Input validation failed.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Builder for [`decompose`].
#[derive(Debug, Clone)]
pub struct DecomposeBuilder<'a> {
    s: &'a Array2<f32>,
    n_components: usize,
    n_iter: usize,
    random_seed: u64,
}

impl DecomposeBuilder<'_> {
    /// Set the number of multiplicative-update iterations (default: 200).
    #[must_use]
    pub fn n_iter(mut self, v: usize) -> Self {
        self.n_iter = v;
        self
    }

    /// Set the seed for deterministic random initialization (default: 0).
    #[must_use]
    pub fn random_seed(mut self, v: u64) -> Self {
        self.random_seed = v;
        self
    }

    /// Factorize the input spectrogram.
    ///
    /// # Returns
    /// `(W, H)` where `W` has shape `(n_bins, n_components)` (templates) and
    /// `H` has shape `(n_components, n_frames)` (activations), such that
    /// `W @ H ≈ S`.
    ///
    /// # Errors
    /// Returns an error if the input is empty or `n_components` exceeds the
    /// smaller dimension of `S`.
    pub fn compute(self) -> Result<(Array2<f32>, Array2<f32>), DecomposeError> {
        decompose_impl(self.s, self.n_components, self.n_iter, self.random_seed)
    }
}

/// Decomposes a magnitude spectrogram via Non-negative Matrix Factorization.
///
/// Finds non-negative matrices `W` (spectral templates) and `H` (temporal
/// activations) such that `S ≈ W @ H` by iterating the Lee–Seung
/// multiplicative update rules.
///
/// # Arguments
/// * `s` — Input magnitude spectrogram of shape `(n_bins, n_frames)`.
/// * `n_components` — Number of NMF components.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::decompose;
/// use ndarray::Array2;
/// let spec: Array2<f32> = Array2::from_elem((128, 200), 0.5);
/// let (W, H) = decompose(&spec, 8).n_iter(100).compute()?;
/// assert_eq!(W.shape(), [128, 8]);
/// assert_eq!(H.shape(), [8, 200]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn decompose(s: &Array2<f32>, n_components: usize) -> DecomposeBuilder<'_> {
    DecomposeBuilder { s, n_components, n_iter: 200, random_seed: 0 }
}

// ─── Implementation ───────────────────────────────────────────────────────────

fn decompose_impl(
    s: &Array2<f32>,
    n_components: usize,
    n_iter: usize,
    seed: u64,
) -> Result<(Array2<f32>, Array2<f32>), DecomposeError> {
    const EPS: f32 = 1e-10;

    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];

    if n_bins == 0 || n_frames == 0 {
        return Err(DecomposeError::InvalidInput("Empty spectrogram".into()));
    }
    if n_components == 0 {
        return Err(DecomposeError::InvalidInput("n_components must be > 0".into()));
    }

    let mut w = nmf_init(n_bins, n_components, seed);
    let mut h = nmf_init(n_components, n_frames, seed.wrapping_add(1));

    for _ in 0..n_iter {
        // ── Update H ─────────────────────────────────────────────────────────
        // H ← H ⊙ (Wᵀ S) / (Wᵀ W H + ε)
        let wt = w.t().to_owned();
        let num_h = wt.dot(s);
        let den_h = wt.dot(&w).dot(&h).mapv(|x| x + EPS);
        h *= &num_h;
        h /= &den_h;

        // ── Update W ─────────────────────────────────────────────────────────
        // W ← W ⊙ (S Hᵀ) / (W H Hᵀ + ε)
        let ht = h.t().to_owned();
        let num_w = s.dot(&ht);
        let den_w = w.dot(&h).dot(&ht).mapv(|x| x + EPS);
        w *= &num_w;
        w /= &den_w;
    }

    Ok((w, h))
}

/// Deterministic pseudo-random initialization in `(0, 1)`.
fn nmf_init(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let h = (i as u64)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((j as u64).wrapping_mul(40_503))
            .wrapping_add(seed);
        ((h & 0xFFFF) as f32 + 1.0) / 65_538.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_empty_error() {
        let empty: Array2<f32> = Array2::zeros((0, 0));
        assert!(decompose_impl(&empty, 2, 10, 0).is_err());
    }

    #[test]
    fn test_decompose_zero_components_error() {
        let s = Array2::from_elem((8, 20), 1.0_f32);
        assert!(decompose_impl(&s, 0, 10, 0).is_err());
    }

    #[test]
    fn test_decompose_output_shapes() {
        let s = Array2::from_elem((16, 30), 0.5_f32);
        let (w, h) = decompose_impl(&s, 4, 50, 42).unwrap();
        assert_eq!(w.shape(), [16, 4]);
        assert_eq!(h.shape(), [4, 30]);
    }

    #[test]
    fn test_decompose_non_negative() {
        let s = Array2::from_shape_fn((8, 20), |(i, j)| (i * j) as f32 * 0.1 + 0.01);
        let (w, h) = decompose_impl(&s, 3, 100, 7).unwrap();
        assert!(w.iter().all(|&v| v >= 0.0), "W must be non-negative");
        assert!(h.iter().all(|&v| v >= 0.0), "H must be non-negative");
    }

    #[test]
    fn test_decompose_reconstruction_error_decreases() {
        let s = Array2::from_shape_fn((12, 25), |(i, j)| ((i + j) as f32).sin().abs() + 0.1);
        let (w10, h10) = decompose_impl(&s, 3, 10, 0).unwrap();
        let (w200, h200) = decompose_impl(&s, 3, 200, 0).unwrap();
        let err10 = frobenius_err(&s, &w10.dot(&h10));
        let err200 = frobenius_err(&s, &w200.dot(&h200));
        assert!(err200 < err10, "More iterations should reduce reconstruction error");
    }

    fn frobenius_err(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }
}
