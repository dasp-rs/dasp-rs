use thiserror::Error;

/// Custom error types for signal generation operations.
#[derive(Error, Debug)]
pub enum GeneratorError {
    /// Invalid input parameters or data.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Builder for a click track.
#[derive(Debug, Clone, Default)]
pub struct ClicksBuilder<'a> {
    times: Option<&'a [f32]>,
    frames: Option<&'a [usize]>,
    sr: Option<u32>,
    hop_length: Option<usize>,
}

impl<'a> ClicksBuilder<'a> {
    /// Place clicks at the given times in seconds (takes precedence over frames).
    #[must_use]
    pub fn times(mut self, times: &'a [f32]) -> Self {
        self.times = Some(times);
        self
    }

    /// Place clicks at the given frame indices.
    #[must_use]
    pub fn frames(mut self, frames: &'a [usize]) -> Self {
        self.frames = Some(frames);
        self
    }

    /// Set the sample rate in Hz (default: 44100).
    #[must_use]
    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sr = Some(sr);
        self
    }

    /// Set the hop length in samples, used with `frames` (default: 512).
    #[must_use]
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = Some(hop_length);
        self
    }

    /// Render the click track.
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
    pub fn compute(self) -> Result<Vec<f32>, GeneratorError> {
        clicks_impl(self.times, self.frames, self.sr, self.hop_length)
    }
}

/// Generates a click signal at specified times or frame indices.
///
/// Returns a builder. Configure with [`ClicksBuilder::times`] or
/// [`ClicksBuilder::frames`], then call `.compute()`.
///
/// # Examples
/// ```
/// use dasp_rs::generate::clicks;
/// let times = [0.1, 0.2, 0.3];
/// let signal = clicks().times(&times).compute()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn clicks<'a>() -> ClicksBuilder<'a> {
    ClicksBuilder::default()
}

fn clicks_impl(
    times: Option<&[f32]>,
    frames: Option<&[usize]>,
    sr: Option<u32>,
    hop_length: Option<usize>,
) -> Result<Vec<f32>, GeneratorError> {
    let sample_rate = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let max_samples = if let Some(t) = times {
        let max_time = t.iter().copied().max_by(f32::total_cmp).ok_or_else(|| {
            GeneratorError::InvalidInput("Times array cannot be empty".to_string())
        })?;
        (max_time * sample_rate as f32) as usize + 1
    } else if let Some(f) = frames {
        let max_frame = f.iter().copied().max().ok_or_else(|| {
            GeneratorError::InvalidInput("Frames array cannot be empty".to_string())
        })?;
        max_frame * hop + 1
    } else {
        44100
    };
    let mut signal = vec![0.0; max_samples];

    if let Some(ts) = times {
        for &t in ts {
            let idx = (t * sample_rate as f32) as usize;
            if idx < signal.len() {
                signal[idx] = 1.0;
            }
        }
    } else if let Some(fs) = frames {
        for &f in fs {
            let idx = f * hop;
            if idx < signal.len() {
                signal[idx] = 1.0;
            }
        }
    }
    Ok(signal)
}

/// Generates a pure tone (sine wave) at a specified frequency.
///
/// # Arguments
/// * `frequency` - Frequency of the tone in Hz
/// * `sr` - Sample rate in Hz
///
/// # Returns
/// Returns a builder that can be configured with method chaining.
///
/// # Examples
/// ```
/// use dasp_rs::generate::*;
/// use dasp_rs::types::*;
/// let tone_signal = tone(440.0, 44100)
///     .duration(0.5)
///     .phase(0.0)
///     .compute();
/// ```
pub fn tone(frequency: f32, sr: u32) -> ToneBuilder {
    ToneBuilder {
        frequency,
        sr,
        duration: 1.0,
        phase: 0.0,
    }
}

/// Tone builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct ToneBuilder {
    frequency: f32,
    sr: u32,
    duration: f32,
    phase: f32,
}

impl ToneBuilder {
    /// Set the duration in seconds (default: 1.0).
    #[must_use]
    pub fn duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }

    /// Set the initial phase in radians (default: 0.0).
    #[must_use]
    pub fn phase(mut self, phase: f32) -> Self {
        self.phase = phase;
        self
    }

    /// Compute the tone signal with the configured parameters.
    pub fn compute(self) -> Vec<f32> {
        tone_impl(self.frequency, self.sr, self.duration, self.phase)
    }
}

/// Internal tone implementation.
fn tone_impl(frequency: f32, sr: u32, duration: f32, phase: f32) -> Vec<f32> {
    let len = (duration * sr as f32) as usize;
    (0..len)
        .map(|n| {
            (2.0 * std::f32::consts::PI * frequency * n as f32 / sr as f32 + phase).cos()
        })
        .collect()
}

/// Generates a linear chirp signal with frequency sweeping from fmin to fmax.
///
/// # Arguments
/// * `fmin` - Starting frequency in Hz
/// * `fmax` - Ending frequency in Hz
/// * `sr` - Sample rate in Hz
///
/// # Returns
/// Returns a builder that can be configured with method chaining.
///
/// # Examples
/// ```
/// use dasp_rs::generate::*;
/// use dasp_rs::types::*;
/// let chirp_signal = chirp(200.0, 800.0, 44100)
///     .duration(2.0)
///     .compute();
/// ```
pub fn chirp(fmin: f32, fmax: f32, sr: u32) -> ChirpBuilder {
    ChirpBuilder {
        fmin,
        fmax,
        sr,
        duration: 1.0,
    }
}

/// Chirp builder for method chaining (internal use only).
#[derive(Debug, Clone)]
pub struct ChirpBuilder {
    fmin: f32,
    fmax: f32,
    sr: u32,
    duration: f32,
}

impl ChirpBuilder {
    /// Set the duration in seconds (default: 1.0).
    #[must_use]
    pub fn duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }

    /// Compute the chirp signal with the configured parameters.
    pub fn compute(self) -> Vec<f32> {
        chirp_impl(self.fmin, self.fmax, self.sr, self.duration)
    }
}

/// Internal chirp implementation.
fn chirp_impl(fmin: f32, fmax: f32, sr: u32, duration: f32) -> Vec<f32> {
    let len = (duration * sr as f32) as usize;
    let sr_f = sr as f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    // For linear chirp: f(t) = f0 + (f1 - f0) * t / T
    // Phase is integral of frequency: φ(t) = 2π * ∫[f0 + (f1-f0)*τ/T]dτ
    // φ(t) = 2π * [f0*t + (f1-f0)*t²/(2*T)]
    (0..len)
        .map(|n| {
            let t = n as f32 / sr_f; // Time in seconds
            let phase = two_pi * (fmin * t + (fmax - fmin) * t * t / (2.0 * duration));
            phase.cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn clicks_from_times_and_frames() {
        let signal = clicks().times(&[0.0, 0.001]).sample_rate(1000).compute().unwrap();
        assert_eq!(signal.len(), 2);
        assert_eq!(signal, vec![1.0, 1.0]);

        let frames = clicks().frames(&[0, 2]).sample_rate(8000).hop_length(2).compute().unwrap();
        assert_eq!(frames.len(), 5);
        assert!(approx_eq(frames[0], 1.0, f32::EPSILON));
        assert!(approx_eq(frames[4], 1.0, f32::EPSILON));
        assert!(frames[1].abs() < f32::EPSILON);
        assert!(frames[2].abs() < f32::EPSILON);
        assert!(frames[3].abs() < f32::EPSILON);
    }

    #[test]
    fn clicks_rejects_empty_inputs() {
        assert!(clicks().times(&[]).compute().is_err());
        assert!(clicks().frames(&[]).compute().is_err());
    }

    #[test]
    fn tone_builder_respects_duration_and_phase() {
        let samples = tone(440.0, 44_100).duration(0.001).phase(std::f32::consts::FRAC_PI_2).compute();
        assert_eq!(samples.len(), 44); // 44.1 samples -> 44 truncation
        // First sample uses phase only.
        assert!(approx_eq(samples[0], 0.0, 1e-6));
        assert!(samples.iter().all(|s| s.abs() <= 1.0));
    }

    #[test]
    fn chirp_builder_generates_expected_length() {
        let samples = chirp(100.0, 200.0, 10_000).duration(0.01).compute();
        assert_eq!(samples.len(), 100); // 0.01 * 10k
        assert!(samples.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    }
}
