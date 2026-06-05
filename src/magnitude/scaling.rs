use ndarray::{Array1, Array2};
use thiserror::Error;

use crate::core::AudioError;

/// Errors specific to spectrogram scaling and weighting operations.
#[derive(Error, Debug)]
pub enum ScalingError {
    /// Insufficient data for the requested operation (e.g., empty spectrogram).
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Invalid input parameters (e.g., negative values, mismatched dimensions).
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl From<ScalingError> for AudioError {
    fn from(err: ScalingError) -> Self {
        match err {
            ScalingError::InsufficientData(msg) => AudioError::InsufficientData(msg),
            ScalingError::InvalidInput(msg) => AudioError::InvalidInput(msg),
        }
    }
}

/// Converts an amplitude spectrogram to decibels (dB).
///
/// Computes the decibel representation of an amplitude spectrogram using the formula:
/// `db = 20 * log10(max(x, amin) / ref_val)`, with clipping at `-top_db` below the maximum.
/// This is useful for audio visualization and perceptual scaling.
///
/// # Arguments
/// * `spectrogram` - Amplitude spectrogram as a 2D array (`Array2<f32>`).
/// * `ref_val` - Reference amplitude for 0 dB (defaults to 1.0 if `None`).
/// * `amin` - Minimum amplitude threshold to avoid log of zero (defaults to 1e-5 if `None`).
/// * `top_db` - Maximum dB below the reference level (defaults to 80.0 if `None`).
///
/// # Returns
/// A `Result` containing the decibel spectrogram as `Array2<f32>`.
/// Values are clipped to ensure they do not fall below `max_db - top_db`.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If the spectrogram is empty.
/// * `ScalingError::InvalidInput` - If `ref_val`, `amin`, or `top_db` is non-positive, or if the spectrogram contains negative values.
///
/// # Example
/// ```
/// use ndarray::arr2;
/// use dasp_rs::mag::amplitude_to_db;
/// let s = arr2(&[[1.0, 2.0], [0.1, 0.01]]);
/// let s_db = amplitude_to_db(&s, None, None, None)?;
/// assert_eq!(s_db[[0, 0]], 0.0); // 20 * log10(1.0 / 1.0)
/// assert!((s_db[[0, 1]] - 6.0206).abs() < 1e-4); // 20 * log10(2.0 / 1.0)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn amplitude_to_db(
    spectrogram: &Array2<f32>,
    ref_val: Option<f32>,
    amin: Option<f32>,
    top_db: Option<f32>,
) -> Result<Array2<f32>, ScalingError> {
    let ref_val = ref_val.unwrap_or(1.0);
    let amin = amin.unwrap_or(1e-5);
    let top_db = top_db.unwrap_or(80.0);

    validate_spectrogram(spectrogram, "amplitude")?;
    validate_positive_params(ref_val, amin, top_db, "Reference value", "Minimum amplitude", "Top dB")?;

    Ok(spectrogram.mapv(|x| {
        let x_clipped = x.max(amin);
        let db = 20.0 * (x_clipped / ref_val).log10();
        let max_db = db.max(-top_db);
        db.max(max_db)
    }))
}

/// Converts a decibel (dB) spectrogram to amplitude.
///
/// Converts a decibel spectrogram back to amplitude using the formula:
/// `amplitude = ref_val * 10^(db / 20)`.
///
/// # Arguments
/// * `spectrogram_db` - Decibel spectrogram as a 2D array (`Array2<f32>`).
/// * `ref_val` - Reference amplitude for 0 dB (defaults to 1.0 if `None`).
///
/// # Returns
/// A `Result` containing the amplitude spectrogram as `Array2<f32>`.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If the spectrogram is empty.
/// * `ScalingError::InvalidInput` - If `ref_val` is non-positive.
///
/// # Example
/// ```
/// use ndarray::arr2;
/// use dasp_rs::mag::db_to_amplitude;
/// let s_db = arr2(&[[0.0, 6.0206], [-20.0, -40.0]]);
/// let s = db_to_amplitude(&s_db, None)?;
/// assert_eq!(s[[0, 0]], 1.0);
/// assert!((s[[0, 1]] - 2.0).abs() < 1e-4); // 10^(6.0206 / 20)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn db_to_amplitude(
    spectrogram_db: &Array2<f32>,
    ref_val: Option<f32>,
) -> Result<Array2<f32>, ScalingError> {
    let ref_val = ref_val.unwrap_or(1.0);

    validate_spectrogram(spectrogram_db, "decibel")?;
    if ref_val <= 0.0 {
        return Err(ScalingError::InvalidInput(
            "Reference value must be positive".to_string(),
        ));
    }

    Ok(spectrogram_db.mapv(|x| ref_val * 10.0f32.powf(x / 20.0)))
}

/// Converts a power spectrogram to decibels (dB).
///
/// Computes the decibel representation of a power spectrogram using the formula:
/// `db = 10 * log10(max(x, amin) / ref_val)`, with clipping at `-top_db` below the maximum.
/// This is suitable for power-based spectrograms (e.g., squared amplitude).
///
/// # Arguments
/// * `spectrogram` - Power spectrogram as a 2D array (`Array2<f32>`).
/// * `ref_val` - Reference power for 0 dB (defaults to 1.0 if `None`).
/// * `amin` - Minimum power threshold to avoid log of zero (defaults to 1e-10 if `None`).
/// * `top_db` - Maximum dB below the reference level (defaults to 80.0 if `None`).
///
/// # Returns
/// A `Result` containing the decibel spectrogram as `Array2<f32>`.
/// Values are clipped to ensure they do not fall below `max_db - top_db`.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If the spectrogram is empty.
/// * `ScalingError::InvalidInput` - If `ref_val`, `amin`, or `top_db` is non-positive, or if the spectrogram contains negative values.
///
/// # Example
/// ```
/// use ndarray::arr2;
/// use dasp_rs::mag::power_to_db;
/// let s = arr2(&[[1.0, 4.0], [0.1, 0.01]]);
/// let s_db = power_to_db(&s, None, None, None)?;
/// assert_eq!(s_db[[0, 0]], 0.0); // 10 * log10(1.0 / 1.0)
/// assert!((s_db[[0, 1]] - 6.0206).abs() < 1e-4); // 10 * log10(4.0 / 1.0)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn power_to_db(
    spectrogram: &Array2<f32>,
    ref_val: Option<f32>,
    amin: Option<f32>,
    top_db: Option<f32>,
) -> Result<Array2<f32>, ScalingError> {
    let ref_val = ref_val.unwrap_or(1.0);
    let amin = amin.unwrap_or(1e-10);
    let top_db = top_db.unwrap_or(80.0);

    validate_spectrogram(spectrogram, "power")?;
    validate_positive_params(ref_val, amin, top_db, "Reference value", "Minimum power", "Top dB")?;

    Ok(spectrogram.mapv(|x| {
        let x_clipped = x.max(amin);
        let db = 10.0 * (x_clipped / ref_val).log10();
        let max_db = db.max(-top_db);
        db.max(max_db)
    }))
}

/// Converts a decibel (dB) spectrogram to power.
///
/// Converts a decibel spectrogram back to power using the formula:
/// `power = ref_val * 10^(db / 10)`.
///
/// # Arguments
/// * `spectrogram_db` - Decibel spectrogram as a 2D array (`Array2<f32>`).
/// * `ref_val` - Reference power for 0 dB (defaults to 1.0 if `None`).
///
/// # Returns
/// A `Result` containing the power spectrogram as `Array2<f32>`.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If the spectrogram is empty.
/// * `ScalingError::InvalidInput` - If `ref_val` is non-positive.
///
/// # Example
/// ```
/// use ndarray::arr2;
/// use dasp_rs::mag::db_to_power;
/// let s_db = arr2(&[[0.0, 6.0206], [-10.0, -20.0]]);
/// let s = db_to_power(&s_db, None)?;
/// assert_eq!(s[[0, 0]], 1.0);
/// assert!((s[[0, 1]] - 4.0).abs() < 1e-4); // 10^(6.0206 / 10)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn db_to_power(
    spectrogram_db: &Array2<f32>,
    ref_val: Option<f32>,
) -> Result<Array2<f32>, ScalingError> {
    let ref_val = ref_val.unwrap_or(1.0);

    validate_spectrogram(spectrogram_db, "decibel")?;
    if ref_val <= 0.0 {
        return Err(ScalingError::InvalidInput(
            "Reference value must be positive".to_string(),
        ));
    }

    Ok(spectrogram_db.mapv(|x| ref_val * 10.0f32.powf(x / 10.0)))
}

/// Applies perceptual frequency weighting to a spectrogram.
///
/// Applies frequency-dependent weighting (e.g., A, B, C, or D) to a spectrogram to emphasize
/// perceptually relevant frequencies. The spectrogram is multiplied by weights computed for each frequency bin.
///
/// # Arguments
/// * `spectrogram` - Spectrogram as a 2D array (`Array2<f32>`, frequencies Ãƒâ€” time).
/// * `frequencies` - Slice of frequencies (Hz) corresponding to spectrogram rows.
/// * `kind` - Weighting type ("A", "B", "C", or "D"; defaults to "A" if `None`).
///
/// # Returns
/// A `Result` containing the weighted spectrogram as `Array2<f32>`.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If the spectrogram is empty.
/// * `ScalingError::InvalidInput` - If frequencies length mismatches spectrogram rows, spectrogram contains negative values, or `kind` is invalid.
///
/// # Example
/// ```
/// use ndarray::arr2;
/// use dasp_rs::mag::perceptual_weighting;
/// let s = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
/// let freqs = vec![1000.0, 2000.0];
/// let s_weighted = perceptual_weighting(&s, &freqs, None)?;
/// assert_eq!(s_weighted.shape(), s.shape());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn perceptual_weighting(
    spectrogram: &Array2<f32>,
    frequencies: &[f32],
    kind: Option<&str>,
) -> Result<Array2<f32>, ScalingError> {
    validate_spectrogram(spectrogram, "spectrogram")?;
    validate_frequencies(frequencies, spectrogram.shape()[0])?;

    let weights = frequency_weighting(frequencies, kind)?;
    let weights_array = Array1::from_vec(weights);
    let weights_2d = weights_array
            .clone()
            .into_shape_with_order((weights_array.len(), 1))
            .map_err(|e| ScalingError::InvalidInput(format!("Failed to reshape weights: {}", e)))?;

    // Broadcasting weights across time dimension
    let s_weighted = spectrogram * &weights_2d;

    Ok(s_weighted)
}

/// Computes frequency weighting coefficients for a given type.
///
/// Supports A, B, C, or D weightings, which adjust frequency amplitudes based on human auditory perception.
/// Returns weights as amplitude multipliers (not dB).
///
/// # Arguments
/// * `frequencies` - Slice of frequencies in Hz.
/// * `kind` - Weighting type ("A", "B", "C", or "D"; defaults to "A" if `None`).
///
/// # Returns
/// A `Result` containing a `Vec<f32>` of weighting coefficients.
///
/// # Errors
/// * `ScalingError::InvalidInput` - If `kind` is not "A", "B", "C", or "D".
///
/// # pÃƒÂ©lda
/// ```
/// use dasp_rs::mag::frequency_weighting;
/// let freqs = vec![1000.0, 2000.0];
/// let weights = frequency_weighting(&freqs, Some("A"))?;
/// assert_eq!(weights.len(), 2);
/// assert!(weights[0] > 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn frequency_weighting(
    frequencies: &[f32],
    kind: Option<&str>,
) -> Result<Vec<f32>, ScalingError> {
    match kind.unwrap_or("A") {
        "A" => a_weighting(frequencies, None),
        "B" => b_weighting(frequencies, None),
        "C" => c_weighting(frequencies, None),
        "D" => d_weighting(frequencies, None),
        k => Err(ScalingError::InvalidInput(format!("Unknown weighting kind: {}", k))),
    }
}

/// Computes multiple frequency weighting coefficients for various types.
///
/// Generates weighting coefficients for multiple weighting types, useful for comparing different perceptual models.
///
/// # Arguments
/// * `frequencies` - Slice of frequencies in Hz.
/// * `kinds` - Slice of weighting types (e.g., ["A", "C"]).
///
/// # Returns
/// A `Result` containing a `Vec<Vec<f32>>`, where each inner vector corresponds to the weights for one kind.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If `frequencies` or `kinds` is empty.
/// * `ScalingError::InvalidInput` - If any `kind` is not "A", "B", "C", or "D".
///
/// # Example
/// ```
/// use dasp_rs::mag::multi_frequency_weighting;
/// let freqs = vec![1000.0, 2000.0];
/// let weights = multi_frequency_weighting(&freqs, &["A", "C"])?;
/// assert_eq!(weights.len(), 2);
/// assert_eq!(weights[0].len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn multi_frequency_weighting(
    frequencies: &[f32],
    kinds: &[&str],
) -> Result<Vec<Vec<f32>>, ScalingError> {
    if frequencies.is_empty() {
        return Err(ScalingError::InsufficientData(
            "Frequency array is empty".to_string(),
        ));
    }
    if kinds.is_empty() {
        return Err(ScalingError::InvalidInput(
            "No weighting kinds provided".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(kinds.len());
    for &kind in kinds {
        results.push(frequency_weighting(frequencies, Some(kind))?);
    }
    Ok(results)
}

/// Computes A-weighting coefficients for given frequencies.
///
/// A-weighting approximates human ear sensitivity, emphasizing frequencies around 1-6 kHz.
/// The formula is based on IEC 61672-1, adjusted to return amplitude weights.
///
/// # Arguments
/// * `frequencies` - Slice of frequencies in Hz.
/// * `min_db` - Minimum dB threshold for weights (defaults to -80.0 if `None`).
///
/// # Returns
/// A `Result` containing a `Vec<f32>` of A-weighting coefficients as amplitude multipliers.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If `frequencies` is empty.
/// * `ScalingError::InvalidInput` - If `frequencies` contains negative values.
///
/// # Example
/// ```no_run
/// use dasp_rs::mag::a_weighting;
/// let freqs = vec![1000.0];
/// let weights = a_weighting(&freqs, None)?;
/// assert!((weights[0] - 1.2589).abs() < 1e-4); // A-weighting at 1 kHz
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn a_weighting(
    frequencies: &[f32],
    min_db: Option<f32>,
) -> Result<Vec<f32>, ScalingError> {
    compute_weighting(frequencies, min_db, |f| {
        let f2 = f * f;
        let f4 = f2 * f2;
        let num = 12194.0_f32.powi(2) * f4;
        let den = (f2 + 20.6_f32.powi(2))
            * (f2 + 12194.0_f32.powi(2))
            * ((f2 + 107.7_f32.powi(2)) * (f2 + 737.9_f32.powi(2))).sqrt();
        20.0 * (num / den).log10() + 2.0
    })
}

/// Computes B-weighting coefficients for given frequencies.
///
/// B-weighting is less common but used for medium sound levels, with less attenuation at low frequencies than A-weighting.
///
/// # Arguments
/// * `frequencies` - Slice of frequencies in Hz.
/// * `min_db` - Minimum dB threshold for weights (defaults to -80.0 if `None`).
///
/// # Returns
/// A `Result` containing a `Vec<f32>` of B-weighting coefficients as amplitude multipliers.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If `frequencies` is empty.
/// * `ScalingError::InvalidInput` - If `frequencies` contains negative values.
///
/// # Example
/// ```
/// use dasp_rs::mag::b_weighting;
/// let freqs = vec![1000.0];
/// let weights = b_weighting(&freqs, None)?;
/// assert!(weights[0] > 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn b_weighting(
    frequencies: &[f32],
    min_db: Option<f32>,
) -> Result<Vec<f32>, ScalingError> {
    compute_weighting(frequencies, min_db, |f| {
        let f2 = f * f;
        let num = 12194.0_f32.powi(2) * f2;
        let den = (f2 + 20.6_f32.powi(2)) * (f2 + 12194.0_f32.powi(2));
        10.0 * (num / den + 1.0).log10()
    })
}

/// Computes C-weighting coefficients for given frequencies.
///
/// C-weighting is flatter than A-weighting, used for high sound levels, with minimal attenuation at low and high frequencies.
///
/// # Arguments
/// * `frequencies` - Slice of frequencies in Hz.
/// * `min_db` - Minimum dB threshold for weights (defaults to -80.0 if `None`).
///
/// # Returns
/// A `Result` containing a `Vec<f32>` of C-weighting coefficients as amplitude multipliers.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If `frequencies` is empty.
/// * `ScalingError::InvalidInput` - If `frequencies` contains negative values.
///
/// # Example
/// ```
/// use dasp_rs::mag::c_weighting;
/// let freqs = vec![1000.0];
/// let weights = c_weighting(&freqs, None)?;
/// assert!(weights[0] > 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn c_weighting(
    frequencies: &[f32],
    min_db: Option<f32>,
) -> Result<Vec<f32>, ScalingError> {
    compute_weighting(frequencies, min_db, |f| {
        let f2 = f * f;
        let num = 12194.0_f32.powi(2) * f2;
        let den = (f2 + 20.6_f32.powi(2)) * (f2 + 12194.0_f32.powi(2));
        10.0 * (num / den).log10() + 0.06
    })
}

/// Computes D-weighting coefficients for given frequencies.
///
/// D-weighting is used for aircraft noise, emphasizing mid-frequencies more than A-weighting.
///
/// # Arguments
/// * `frequencies` - Slice of frequencies in Hz.
/// * `min_db` - Minimum dB threshold for weights (defaults to -80.0 if `None`).
///
/// # Returns
/// A `Result` containing a `Vec<f32>` of D-weighting coefficients as amplitude multipliers.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If `frequencies` is empty.
/// * `ScalingError::InvalidInput` - If `frequencies` contains negative values.
///
/// # Example
/// ```
/// use dasp_rs::mag::d_weighting;
/// let freqs = vec![1000.0];
/// let weights = d_weighting(&freqs, None)?;
/// assert!(weights[0] > 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn d_weighting(
    frequencies: &[f32],
    min_db: Option<f32>,
) -> Result<Vec<f32>, ScalingError> {
    compute_weighting(frequencies, min_db, |f| {
        let f2 = f * f;
        let f4 = f2 * f2;
        let num = 6532.0_f32.powi(2) * f4;
        let den = (f2 + 148.0_f32.powi(2))
            * (f2 + 6532.0_f32.powi(2))
            * (f + 1087.0).powi(2);
        10.0 * (num / den).log10()
    })
}

/// Applies Per-Channel Energy Normalization (PCEN) to a spectrogram.
///
/// PCEN normalizes a spectrogram to reduce background noise and enhance foreground signals.
/// The formula is: `P[f, t] = (S[f, t] / (eps + M[f, t]))^gain + bias - bias`, where
/// `M[f, t]` is an exponentially smoothed version of the spectrogram.
///
/// # Arguments
/// * `spectrogram` - Spectrogram as a 2D array (`Array2<f32>`, frequencies Ãƒâ€” time).
/// * `sample_rate` - Sample rate in Hz (defaults to 44100 if `None`).
/// * `hop_length` - Hop length in samples (defaults to 512 if `None`).
/// * `gain` - Gain exponent for normalization (defaults to 0.8 if `None`).
/// * `bias` - Bias term to stabilize output (defaults to 10.0 if `None`).
///
/// # Returns
/// A `Result` containing the normalized spectrogram as `Array2<f32>`.
///
/// # Errors
/// * `ScalingError::InsufficientData` - If the spectrogram is empty.
/// * `ScalingError::InvalidInput` - If the spectrogram contains negative values, or if `gain` or `bias` is negative.
///
/// # Example
/// ```
/// use ndarray::arr2;
/// use dasp_rs::mag::pcen;
/// let s = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
/// let p = pcen(&s, None, None, None, None)?;
/// assert_eq!(p.shape(), s.shape());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn pcen(
    spectrogram: &Array2<f32>,
    sample_rate: Option<u32>,
    hop_length: Option<usize>,
    gain: Option<f32>,
    bias: Option<f32>,
) -> Result<Array2<f32>, ScalingError> {
    const EPS: f32 = 1e-6;
    const SMOOTH_COEF: f32 = 0.025;

    let sr = sample_rate.unwrap_or(44_100);
    let hop_length = hop_length.unwrap_or(512);
    let gain = gain.unwrap_or(0.8);
    let bias = bias.unwrap_or(10.0);

    validate_spectrogram(spectrogram, "spectrogram")?;
    if gain < 0.0 || bias < 0.0 {
        return Err(ScalingError::InvalidInput(
            "Gain and bias must be non-negative".to_string(),
        ));
    }

    let n_freqs = spectrogram.shape()[0];
    let n_frames = spectrogram.shape()[1];
    let alpha = (-SMOOTH_COEF * sr as f32 / hop_length as f32).exp();
    let one_minus_alpha = 1.0 - alpha;

    let mut m = Array2::zeros((n_freqs, n_frames));
    for f in 0..n_freqs {
        m[[f, 0]] = spectrogram[[f, 0]];
        for t in 1..n_frames {
            m[[f, t]] = alpha * m[[f, t - 1]] + one_minus_alpha * spectrogram[[f, t]];
        }
    }

    let mut p = Array2::zeros((n_freqs, n_frames));
    for f in 0..n_freqs {
        for t in 0..n_frames {
            let m_val = m[[f, t]] + EPS;
            p[[f, t]] = (spectrogram[[f, t]] / m_val).powf(gain) + bias - bias;
        }
    }

    Ok(p)
}

// Helper functions to reduce code duplication and improve maintainability.

fn validate_spectrogram(spectrogram: &Array2<f32>, context: &str) -> Result<(), ScalingError> {
    if spectrogram.is_empty() {
        return Err(ScalingError::InsufficientData(format!(
            "{} spectrogram is empty",
            context
        )));
    }
    if context != "decibel" && spectrogram.iter().any(|&x| x < 0.0) {
        return Err(ScalingError::InvalidInput(format!(
            "{} spectrogram contains negative values",
            context
        )));
    }
    Ok(())
}

fn validate_positive_params(
    ref_val: f32,
    amin: f32,
    top_db: f32,
    ref_name: &str,
    amin_name: &str,
    top_db_name: &str,
) -> Result<(), ScalingError> {
    if ref_val <= 0.0 {
        return Err(ScalingError::InvalidInput(format!(
            "{} must be positive",
            ref_name
        )));
    }
    if amin <= 0.0 {
        return Err(ScalingError::InvalidInput(format!(
            "{} must be positive",
            amin_name
        )));
    }
    if top_db <= 0.0 {
        return Err(ScalingError::InvalidInput(format!(
            "{} must be positive",
            top_db_name
        )));
    }
    Ok(())
}

fn validate_frequencies(frequencies: &[f32], n_rows: usize) -> Result<(), ScalingError> {
    if frequencies.len() != n_rows {
        return Err(ScalingError::InvalidInput(format!(
            "Frequency length {} does not match spectrogram rows {}",
            frequencies.len(),
            n_rows
        )));
    }
    if frequencies.iter().any(|&f| f < 0.0) {
        return Err(ScalingError::InvalidInput(
            "Frequencies must be non-negative".to_string(),
        ));
    }
    Ok(())
}

fn compute_weighting<F>(
    frequencies: &[f32],
    min_db: Option<f32>,
    gain_fn: F,
) -> Result<Vec<f32>, ScalingError>
where
    F: Fn(f32) -> f32,
{
    const EPS: f32 = 1e-6;
    let min_db = min_db.unwrap_or(-80.0);

    if frequencies.is_empty() {
        return Err(ScalingError::InsufficientData(
            "Frequency array is empty".to_string(),
        ));
    }
    if frequencies.iter().any(|&f| f < 0.0) {
        return Err(ScalingError::InvalidInput(
            "Frequencies must be non-negative".to_string(),
        ));
    }

    let weights = frequencies
        .iter()
        .map(|&f| {
            if f < EPS {
                0.0
            } else {
                let gain_db = gain_fn(f);
                if gain_db < min_db {
                    0.0
                } else {
                    10.0_f32.powf(gain_db / 20.0)
                }
            }
        })
        .collect::<Vec<f32>>();

    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_amplitude_to_db_invalid_inputs() {
        let s = arr2(&[[1.0, 2.0]]);
        assert!(amplitude_to_db(&s, Some(0.0), None, None).is_err());
        assert!(amplitude_to_db(&s, None, Some(-1e-5), None).is_err());
        assert!(amplitude_to_db(&s, None, None, Some(0.0)).is_err());
        let s_neg = arr2(&[[-1.0, 2.0]]);
        assert!(amplitude_to_db(&s_neg, None, None, None).is_err());
        let s_empty = Array2::zeros((0, 0));
        assert!(amplitude_to_db(&s_empty, None, None, None).is_err());
    }

    #[test]
    fn test_db_to_amplitude_empty() {
        let s_empty = Array2::zeros((0, 0));
        assert!(db_to_amplitude(&s_empty, None).is_err());
    }

    #[test]
    fn test_power_to_db_accuracy() {
        let s = arr2(&[[1.0, 4.0], [0.1, 0.01]]);
        let s_db = power_to_db(&s, None, None, None).unwrap();
        assert_eq!(s_db[[0, 0]], 0.0);
        assert!((s_db[[0, 1]] - 6.0206).abs() < 1e-4);
        assert!((s_db[[1, 0]] - (-10.0)).abs() < 1e-4);
    }

    #[test]
    fn test_perceptual_weighting_mismatch() {
        let s = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        let freqs = vec![1000.0]; // Wrong length
        assert!(perceptual_weighting(&s, &freqs, None).is_err());
    }

    #[test]
    fn test_frequency_weighting_invalid_kind() {
        let freqs = vec![1000.0];
        assert!(frequency_weighting(&freqs, Some("X")).is_err());
    }

    #[test]
    fn test_multi_frequency_weighting_empty() {
        let freqs: Vec<f32> = vec![];
        let kinds = ["A", "C"];
        assert!(multi_frequency_weighting(&freqs, &kinds).is_err());
        let freqs = vec![1000.0];
        let kinds: [&str; 0] = [];
        assert!(multi_frequency_weighting(&freqs, &kinds).is_err());
    }

    #[test]
    fn test_a_weighting_zero_freq() {
        let freqs = vec![0.0];
        let weights = a_weighting(&freqs, None).unwrap();
        assert_eq!(weights, vec![0.0]);
    }

    #[test]
    fn test_pcen_negative_gain() {
        let s = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(pcen(&s, None, None, Some(-0.8), None).is_err());
    }
}
