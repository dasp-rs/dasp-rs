use ndarray::{Array2, ArrayView1};
use num_complex::Complex;
use thiserror::Error;

/// Error conditions for harmonic feature extraction and processing.
///
/// Enumerates specific failure modes in harmonic analysis and phase vocoding operations,
/// providing detailed diagnostics for DSP pipeline debugging.
#[derive(Error, Debug)]
pub enum HarmonicsError {
    /// Input arrays have mismatched lengths (e.g., amplitudes vs. frequency bins).
    #[error("Length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),

    /// Invalid input parameter (e.g., empty array, negative rate).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Non-finite input values (e.g., NaN, infinite).
    #[error("Non-finite input: {0}")]
    NonFiniteInput(String),

    /// Numerical computation failure (e.g., division by zero, overflow).
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Default FFT size factor: `n_fft = (n_bins - 1) * 2`.
const N_FFT_FACTOR: usize = 2;

/// Default hop length divisor: `hop_length = n_fft / 4`.
const HOP_DIVISOR: usize = 4;

/// Default weights array for salience computation (supports up to 16 harmonics).
const DEFAULT_WEIGHTS_ARRAY: &[f32] = &[1.0; 16];

/// Validates input arrays for length, emptiness, sortedness, and finite values.
///
/// # Parameters
/// - `arrays`: List of (array, name) pairs to validate (slices of `f32`).
/// - `spectrogram`: Optional `Array2<f32>` to validate (for `salience`).
/// - `complex_spectrogram`: Optional `Array2<Complex<f32>>` to validate (for `phase_vocoder`).
/// - `require_sorted`: If true, checks if arrays are sorted in ascending order.
/// - `check_finite`: If true, checks for NaN or infinite values.
/// - `check_lengths`: If true, enforces equal lengths for all arrays; if false, allows different lengths.
///
/// # Returns
/// - `Ok(())` if all checks pass.
/// - `Err(HarmonicsError)` for length mismatch, empty arrays, unsorted arrays, or non-finite values.
fn validate_inputs(
    arrays: &[(&[f32], &str)],
    spectrogram: Option<&Array2<f32>>,
    complex_spectrogram: Option<&Array2<Complex<f32>>>,
    require_sorted: bool,
    check_finite: bool,
    check_lengths: bool,
) -> Result<(), HarmonicsError> {
    // Check emptiness for arrays
    for &(arr, name) in arrays {
        if arr.is_empty() {
            return Err(HarmonicsError::InvalidInput(format!("{} array is empty", name)));
        }
    }

    // Check spectrogram emptiness
    if let Some(s) = spectrogram {
        if s.shape()[0] == 0 || s.shape()[1] == 0 {
            return Err(HarmonicsError::InvalidInput("Spectrogram is empty".to_string()));
        }
    }
    if let Some(d) = complex_spectrogram {
        if d.shape()[0] == 0 || d.shape()[1] == 0 {
            return Err(HarmonicsError::InvalidInput("Complex spectrogram is empty".to_string()));
        }
    }

    // Check length mismatches
    if check_lengths && !arrays.is_empty() {
        let base_len = arrays[0].0.len();
        for &(arr, _name) in arrays.iter().skip(1) {
            if arr.len() != base_len {
                return Err(HarmonicsError::LengthMismatch(base_len, arr.len()));
            }
        }
    }

    // Check finite values and sortedness for arrays
    for &(arr, name) in arrays {
        if check_finite && arr.iter().any(|&v| !v.is_finite()) {
            return Err(HarmonicsError::NonFiniteInput(format!("{} contain non-finite values", name)));
        }
        if require_sorted && !arr.windows(2).all(|w| w[0] <= w[1]) {
            return Err(HarmonicsError::InvalidInput(format!("{} must be sorted", name)));
        }
    }

    // Check finite values for spectrogram
    if check_finite {
        if let Some(s) = spectrogram {
            if s.iter().any(|&v| !v.is_finite()) {
                return Err(HarmonicsError::NonFiniteInput("Spectrogram contains non-finite values".to_string()));
            }
        }
        if let Some(d) = complex_spectrogram {
            if d.iter().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
                return Err(HarmonicsError::NonFiniteInput("Complex spectrogram contains non-finite values".to_string()));
            }
        }
    }

    Ok(())
}

/// Interpolates a value at a target frequency using linear interpolation.
///
/// # Parameters
/// - `x`: Amplitude spectrum.
/// - `freqs`: Frequency bins (assumed sorted and non-empty).
/// - `target_freq`: Frequency to interpolate at.
///
/// # Returns
/// - `Ok(f32)`: Interpolated amplitude, or 0.0 if `target_freq` exceeds `freqs.last()`.
/// - `Err(HarmonicsError)`: If computation fails (e.g., empty frequencies).
fn interpolate_at(x: &[f32], freqs: &[f32], target_freq: f32) -> Result<f32, HarmonicsError> {
    let max_freq = freqs.last().ok_or_else(|| {
        HarmonicsError::InvalidInput("Frequency bins cannot be empty".to_string())
    })?;
    if target_freq > *max_freq {
        return Ok(0.0);
    }

    let left_idx = freqs
        .binary_search_by(|&x| x.partial_cmp(&target_freq).unwrap_or(std::cmp::Ordering::Less))
        .unwrap_or_else(|e| e.saturating_sub(1))
        .min(freqs.len() - 2);
    let right_idx = left_idx + 1;
    let left_freq = freqs[left_idx];
    let right_freq = freqs[right_idx];

    if (right_freq - left_freq).abs() < f32::EPSILON {
        return Ok(x[left_idx]);
    }

    let alpha = (target_freq - left_freq) / (right_freq - left_freq);
    if !alpha.is_finite() {
        return Err(HarmonicsError::ComputationFailed(
            "Invalid interpolation coefficient".to_string(),
        ));
    }

    Ok(x[left_idx] * (1.0 - alpha) + x[right_idx] * alpha)
}

/// Interpolates harmonic amplitudes across frequency bins.
///
/// Performs linear interpolation of amplitude values at harmonic frequencies derived
/// from a fundamental frequency grid.
///
/// # Parameters
/// - `x`: Amplitude spectrum as a slice of `f32` (frequency bin values).
/// - `freqs`: Frequency bins corresponding to `x` (monotonically increasing).
/// - `harmonics`: Harmonic multipliers (e.g., `[1.0, 2.0, 3.0]` for first three harmonics).
///
/// # Returns
/// - `Ok(Array2<f32>)`: Interpolated amplitudes, shape `(n_harmonics, n_bins)`.
/// - `Err(HarmonicsError)`: Failure due to length mismatch, invalid input, or computation error.
///
/// # Constraints
/// - `x.len() == freqs.len()`.
/// - `freqs` must be sorted in ascending order and contain finite values.
/// - Harmonic frequencies exceeding `freqs.last()` are clamped to zero.
///
/// # Examples
/// ```
/// use dasp_rs::feat::{interp_harmonics, HarmonicsError};
/// let x = vec![0.1, 0.2, 0.3, 0.4];
/// let freqs = vec![0.0, 100.0, 200.0, 300.0];
/// let harmonics = vec![1.0, 2.0];
/// let result = interp_harmonics(&x, &freqs, &harmonics)?;
/// assert_eq!(result.shape(), &[2, 4]);
/// assert_eq!(result[[0, 0]], 0.1);
/// assert_eq!(result[[1, 1]], 0.3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn interp_harmonics(x: &[f32], freqs: &[f32], harmonics: &[f32]) -> Result<Array2<f32>, HarmonicsError> {
    validate_inputs(&[(&x, "amplitudes"), (&freqs, "frequencies")], None, None, true, true, true)?;

    let n_bins = freqs.len();
    let n_harmonics = harmonics.len();
    let mut result = Array2::zeros((n_harmonics, n_bins));

    result.axis_iter_mut(ndarray::Axis(0)).enumerate().for_each(|(h_idx, mut row)| {
        let h = harmonics[h_idx];
        for (bin, &f) in freqs.iter().enumerate() {
            if let Ok(value) = interpolate_at(x, freqs, f * h) {
                row[bin] = value;
            }
        }
    });

    Ok(result)
}

/// Computes a salience map from a spectrogram via harmonic summation.
///
/// Sums weighted harmonic contributions across frequency bins and frames, producing
/// a salience map for pitch detection or harmonic analysis.
///
/// # Parameters
/// - `s`: Spectrogram as `Array2<f32>`, shape `(n_bins, n_frames)`.
/// - `freqs`: Frequency bins corresponding to spectrogram rows (monotonically increasing).
/// - `harmonics`: Harmonic multipliers (e.g., `[1.0, 2.0]`).
/// - `weights`: Optional harmonic weights; defaults to uniform `1.0` if `None`.
///
/// # Returns
/// - `Ok(Array2<f32>)`: Salience map, shape `(n_bins, n_frames)`.
/// - `Err(HarmonicsError)`: Failure due to dimension mismatch, invalid input, or computation error.
///
/// # Constraints
/// - `s.shape()[0] == freqs.len()`.
/// - `weights.len() == harmonics.len()` if provided.
/// - `freqs` must be sorted in ascending order and contain finite values.
///
/// # Examples
/// ```
/// use dasp_rs::feat::salience;
/// use ndarray::array;
/// let s = array![[0.1, 0.2], [0.3, 0.4]];
/// let freqs = vec![0.0, 100.0];
/// let harmonics = vec![1.0, 2.0];
/// let weights: Option<&[f32]> = Some(&[1.0, 0.5]);
/// let result = salience(&s, &freqs, &harmonics, weights)?;
/// assert_eq!(result.shape(), &[2, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn salience(s: &Array2<f32>, freqs: &[f32], harmonics: &[f32], weights: Option<&[f32]>) -> Result<Array2<f32>, HarmonicsError> {
    validate_inputs(&[(&freqs, "frequencies")], Some(s), None, true, true, false)?;
    if s.shape()[0] != freqs.len() {
        return Err(HarmonicsError::LengthMismatch(s.shape()[0], freqs.len()));
    }

    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let n_harmonics = harmonics.len();
    let weights = weights.unwrap_or(DEFAULT_WEIGHTS_ARRAY);
    if weights.len() < n_harmonics {
        return Err(HarmonicsError::LengthMismatch(weights.len(), n_harmonics));
    }

    let mut salience_map = Array2::zeros((n_bins, n_frames));
    salience_map.axis_iter_mut(ndarray::Axis(1)).enumerate().for_each(|(frame, mut col)| {
        let column = s.column(frame).to_vec(); // Convert to Vec to ensure reliable access
        for (bin, &f) in freqs.iter().enumerate() {
            let mut total = 0.0;
            for (h_idx, &h) in harmonics.iter().enumerate() {
                if let Ok(interp) = interpolate_at(&column, freqs, f * h) {
                    total += interp * weights[h_idx];
                }
            }
            col[bin] = total;
        }
    });

    Ok(salience_map)
}

/// Extracts harmonic amplitudes from time-varying fundamental frequencies.
///
/// Interpolates amplitudes at harmonic frequencies based on frame-wise `f0` values.
///
/// # Parameters
/// - `x`: Amplitude spectrum as a slice of `f32`.
/// - `f0`: Fundamental frequencies per frame.
/// - `freqs`: Frequency bins corresponding to `x` (monotonically increasing).
/// - `harmonics`: Harmonic multipliers.
///
/// # Returns
/// - `Ok(Array2<f32>)`: Harmonic amplitudes, shape `(n_harmonics, n_frames)`.
/// - `Err(HarmonicsError)`: Failure due to length mismatch, invalid input, or computation error.
///
/// # Constraints
/// - `x.len() == freqs.len()`.
/// - `f0` must be non-empty and contain finite values.
/// - `freqs` must be sorted in ascending order and contain finite values.
///
/// # Examples
/// ```
/// use dasp_rs::feat::{f0_harmonics, HarmonicsError};
/// let x = vec![0.1, 0.2, 0.3, 0.4];
/// let f0 = vec![100.0, 150.0];
/// let freqs = vec![0.0, 100.0, 200.0, 300.0];
/// let harmonics = vec![1.0, 2.0];
/// let result = f0_harmonics(&x, &f0, &freqs, &harmonics)?;
/// assert_eq!(result.shape(), &[2, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn f0_harmonics(x: &[f32], f0: &[f32], freqs: &[f32], harmonics: &[f32]) -> Result<Array2<f32>, HarmonicsError> {
    validate_inputs(&[(&x, "amplitudes"), (&f0, "f0"), (&freqs, "frequencies")], None, None, true, true, false)?;
    if x.len() != freqs.len() {
        return Err(HarmonicsError::LengthMismatch(x.len(), freqs.len()));
    }

    let n_frames = f0.len();
    let n_harmonics = harmonics.len();
    let mut result = Array2::zeros((n_harmonics, n_frames));

    result.axis_iter_mut(ndarray::Axis(1)).enumerate().for_each(|(frame, mut col)| {
        let fund = f0[frame];
        for (h_idx, &h) in harmonics.iter().enumerate() {
            if let Ok(value) = interpolate_at(x, freqs, fund * h) {
                col[h_idx] = value;
            }
        }
    });

    Ok(result)
}

/// Computes angular frequencies for phase vocoding.
fn compute_omega(n_bins: usize, n_fft: usize) -> Vec<f32> {
    (0..n_bins)
        .map(|k| 2.0 * std::f32::consts::PI * k as f32 / n_fft as f32)
        .collect()
}

/// Unwraps phase differences for phase vocoding.
fn unwrap_phase(delta_phase: ArrayView1<f32>) -> Vec<f32> {
    delta_phase
        .iter()
        .map(|&dp| dp - 2.0 * std::f32::consts::PI * (dp / (2.0 * std::f32::consts::PI)).round())
        .collect()
}

/// Advances phase accumulation for phase vocoding.
fn advance_phase(delta_phase_unwrapped: &[f32], hop: usize, rate: f32) -> Array2<f32> {
    ArrayView1::from(delta_phase_unwrapped)
        .mapv(|x| x / hop as f32 * (hop as f32 * rate))
        .to_shape((delta_phase_unwrapped.len(), 1))
        .unwrap()
        .into_owned()
}

/// Performs phase vocoding for time-scale modification of a complex spectrogram.
///
/// Adjusts the temporal resolution of a spectrogram while preserving frequency content,
/// implementing phase unwrapping and accumulation for coherent resynthesis.
///
/// # Parameters
/// - `d`: Complex spectrogram as `Array2<Complex<f32>>`, shape `(n_bins, n_frames)`.
/// - `rate`: Time stretching factor (>1 stretches, <1 compresses).
/// - `hop_length`: Optional hop length between frames; defaults to `n_fft / 4`.
/// - `n_fft`: Optional FFT size; defaults to `(n_bins - 1) * 2`.
///
/// # Returns
/// - `Ok(Array2<Complex<f32>>)`: Time-scaled spectrogram with adjusted frame count.
/// - `Err(HarmonicsError)`: Failure due to invalid rate, dimensions, or non-finite input.
///
/// # Constraints
/// - `rate > 0.0`.
/// - `hop_length > 0` if provided.
/// - `d` must be non-empty and contain finite values.
///
/// # Examples
/// ```no_run
/// use dasp_rs::feat::phase_vocoder;
/// use ndarray::array;
/// use num_complex::Complex;
/// let d = array![[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]];
/// let result = phase_vocoder(&d, 0.5, None, None)?;
/// assert_eq!(result.shape()[0], 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn phase_vocoder(
    d: &Array2<Complex<f32>>,
    rate: f32,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
) -> Result<Array2<Complex<f32>>, HarmonicsError> {
    if rate <= 0.0 {
        return Err(HarmonicsError::InvalidInput("Rate must be positive".to_string()));
    }
    let n_bins = d.shape()[0];
    let orig_frames = d.shape()[1];
    validate_inputs(&[], None, Some(d), false, true, false)?;

    let n_fft = n_fft.unwrap_or((n_bins - 1) * N_FFT_FACTOR);
    if n_fft == 0 {
        return Err(HarmonicsError::InvalidInput("FFT size must be positive".to_string()));
    }
    let hop = hop_length.unwrap_or(n_fft / HOP_DIVISOR);
    if hop == 0 {
        return Err(HarmonicsError::InvalidInput("Hop length must be positive".to_string()));
    }

    let new_frames = ((orig_frames as f32 * hop as f32) / rate / hop as f32).ceil() as usize;
    let mut output = Array2::zeros((n_bins, new_frames));
    let mut phase_acc = Array2::zeros((n_bins, 1));

    let omega = compute_omega(n_bins, n_fft);

    for t in 0..new_frames {
        let orig_t = (t as f32 * rate * hop as f32 / hop as f32) as usize;
        let orig_t_next = ((t + 1) as f32 * rate * hop as f32 / hop as f32) as usize;

        if orig_t >= orig_frames || orig_t_next >= orig_frames {
            continue;
        }

        let mag = d.column(orig_t).mapv(|c| c.norm());
        let phase = d.column(orig_t).mapv(|c| c.arg());
        let phase_next = d.column(orig_t_next).mapv(|c| c.arg());
        let delta_phase = phase_next - phase - &ArrayView1::from(&omega) * hop as f32;

        let delta_phase_unwrapped = unwrap_phase(delta_phase.view());
        let phase_advance = advance_phase(&delta_phase_unwrapped, hop, rate);
        phase_acc = phase_acc + phase_advance;

        for (i, &m) in mag.iter().enumerate() {
            output[[i, t]] = Complex::from_polar(m, phase_acc[[i, 0]]);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_interpolate_at() {
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let freqs = vec![0.0, 100.0, 200.0, 300.0];
        assert_abs_diff_eq!(interpolate_at(&x, &freqs, 100.0).unwrap(), 0.2, epsilon = EPSILON);
        assert_abs_diff_eq!(interpolate_at(&x, &freqs, 150.0).unwrap(), 0.25, epsilon = EPSILON);
        assert_abs_diff_eq!(interpolate_at(&x, &freqs, 400.0).unwrap(), 0.0, epsilon = EPSILON);
        assert_abs_diff_eq!(
            interpolate_at(&x, &vec![100.0, 100.0], 100.0).unwrap(),
            0.1,
            epsilon = EPSILON
        );
        assert!(matches!(
            interpolate_at(&x, &[], 100.0),
            Err(HarmonicsError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_interp_harmonics() {
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let freqs = vec![0.0, 100.0, 200.0, 300.0];
        let harmonics = vec![1.0, 2.0];
        let result = interp_harmonics(&x, &freqs, &harmonics).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        assert_abs_diff_eq!(result[[0, 0]], 0.1, epsilon = EPSILON);
        assert_abs_diff_eq!(result[[1, 0]], 0.1, epsilon = EPSILON);
        assert_abs_diff_eq!(result[[0, 1]], 0.2, epsilon = EPSILON);
        assert_abs_diff_eq!(result[[1, 1]], 0.3, epsilon = EPSILON);
    }

    #[test]
    fn test_interp_harmonics_mismatch() {
        let x = vec![0.1, 0.2];
        let freqs = vec![0.0, 100.0, 200.0];
        let harmonics = vec![1.0];
        let result = interp_harmonics(&x, &freqs, &harmonics);
        assert!(matches!(result, Err(HarmonicsError::LengthMismatch(2, 3))));
    }

    #[test]
    fn test_interp_harmonics_non_finite() {
        let x = vec![0.1, f32::NAN];
        let freqs = vec![0.0, 100.0];
        let harmonics = vec![1.0];
        let result = interp_harmonics(&x, &freqs, &harmonics);
        if !matches!(result, Err(HarmonicsError::NonFiniteInput(_))) {
            panic!("Expected NonFiniteInput error, got {:?}", result);
        }
    }

    #[test]
    fn test_salience() {
        let s = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]];
        let freqs = vec![0.0, 100.0, 200.0, 300.0];
        let harmonics = vec![1.0, 2.0];
        let weights: Option<&[f32]> = Some(&[1.0, 0.5]);
        let result = salience(&s, &freqs, &harmonics, weights).unwrap();
        assert_eq!(result.shape(), &[4, 3]);
        assert_abs_diff_eq!(result[[0, 0]], 0.1 + 0.1 * 0.5, epsilon = EPSILON);
        assert_abs_diff_eq!(result[[1, 0]], 0.4 + 0.7 * 0.5, epsilon = EPSILON);
    }

    #[test]
    fn test_salience_weight_mismatch() {
        let s = array![[0.1, 0.2], [0.3, 0.4]];
        let freqs = vec![0.0, 100.0];
        let harmonics = vec![1.0, 2.0];
        let weights: Option<&[f32]> = Some(&[1.0]);
        let result = salience(&s, &freqs, &harmonics, weights);
        assert!(matches!(result, Err(HarmonicsError::LengthMismatch(1, 2))));
    }

    #[test]
    fn test_salience_identical_freqs() {
        let s = array![[0.1, 0.2], [0.3, 0.4]];
        let freqs = vec![100.0, 100.0];
        let harmonics = vec![1.0];
        let result = salience(&s, &freqs, &harmonics, None).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_abs_diff_eq!(result[[0, 0]], 0.1, epsilon = EPSILON);
    }

    #[test]
    fn test_f0_harmonics() {
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let f0 = vec![100.0, 150.0];
        let freqs = vec![0.0, 100.0, 200.0, 300.0];
        let harmonics = vec![1.0, 2.0];
        let result = f0_harmonics(&x, &f0, &freqs, &harmonics).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_abs_diff_eq!(result[[0, 0]], 0.2, epsilon = EPSILON);
        assert_abs_diff_eq!(result[[1, 0]], 0.3, epsilon = EPSILON);
        assert_abs_diff_eq!(result[[0, 1]], 0.25, epsilon = EPSILON);
    }

    #[test]
    fn test_f0_harmonics_empty() {
        let x = vec![];
        let f0 = vec![100.0];
        let freqs = vec![];
        let harmonics = vec![1.0];
        let result = f0_harmonics(&x, &f0, &freqs, &harmonics);
        assert!(matches!(result, Err(HarmonicsError::InvalidInput(_))));
    }

    #[test]
    fn test_f0_harmonics_non_finite() {
        let x = vec![0.1, 0.2];
        let f0 = vec![f32::INFINITY];
        let freqs = vec![0.0, 100.0];
        let harmonics = vec![1.0];
        let result = f0_harmonics(&x, &f0, &freqs, &harmonics);
        assert!(matches!(result, Err(HarmonicsError::NonFiniteInput(_))));
    }

    #[test]
    fn test_phase_vocoder() {
        let d = array![
            [
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0)
            ],
            [
                Complex::new(5.0, 0.0),
                Complex::new(6.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(8.0, 0.0)
            ],
        ];
        let result = phase_vocoder(&d, 0.5, Some(1), Some(4)).unwrap();
        assert_eq!(result.shape(), &[2, 8]);
        assert_abs_diff_eq!(result[[0, 0]].norm(), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_phase_vocoder_invalid_rate() {
        let d = array![[Complex::new(1.0, 0.0)]];
        let result = phase_vocoder(&d, 0.0, None, None);
        assert!(matches!(result, Err(HarmonicsError::InvalidInput(_))));
    }

    #[test]
    fn test_phase_vocoder_non_finite() {
        let d = array![[Complex::new(f32::NAN, 0.0)]];
        let result = phase_vocoder(&d, 1.0, None, None);
        assert!(matches!(result, Err(HarmonicsError::NonFiniteInput(_))));
    }
}
