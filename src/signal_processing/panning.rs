use crate::core::io::AudioData;
use thiserror::Error;

/// Custom error types for panning operations.
///
/// This enum defines errors specific to distributing a mono signal across stereo or
/// multi-channel audio fields.
#[derive(Error, Debug)]
pub enum PanningError {
    /// Error when the input signal is not mono.
    #[error("Signal must be mono: {0} channels found")]
    NotMono(u16),

    /// Error when panning parameters are invalid (e.g., pan value out of range).
    #[error("Invalid panning parameter: {0}")]
    InvalidParameter(String),

    /// Error when the target channel count is unsupported.
    #[error("Unsupported channel count: {0}")]
    UnsupportedChannels(u16),
}

/// Pans a mono signal across a stereo field.
///
/// This function distributes a mono signal between left and right channels based on a
/// pan value, using an equal-power (constant loudness) pan law: gains follow
/// `cos`/`sin` of the pan angle so `left² + right² == 1` everywhere, avoiding the
/// "hole in the middle" dip a linear (amplitude-summing) pan law produces at
/// center. A pan of -1.0 is fully left, 0.0 is center, and 1.0 is fully right.
/// The signal must be mono (1 channel).
///
/// # Arguments
/// * `signal` - The mono audio signal to pan.
/// * `pan` - The panning value (-1.0 to 1.0, where -1.0 is left, 1.0 is right).
///
/// # Returns
/// Returns `Result<AudioData, PanningError>` containing the stereo signal or an error.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let signal = AudioData { samples: vec![1.0, 1.0], sample_rate: 44100, channels: 1 };
/// let panned = stereo_pan(&signal, 0.0)?; // Center: equal power, ~0.707 each side
/// assert!((panned.samples[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
/// assert!((panned.samples[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
/// assert_eq!(panned.channels, 2);
///
/// let panned_left = stereo_pan(&signal, -1.0)?; // Fully left
/// assert!((panned_left.samples[0] - 1.0).abs() < 1e-5);
/// assert!(panned_left.samples[1].abs() < 1e-5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn stereo_pan(signal: &AudioData, pan: f32) -> Result<AudioData, PanningError> {
    if signal.channels != 1 {
        return Err(PanningError::NotMono(signal.channels));
    }
    if !(-1.0..=1.0).contains(&pan) {
        return Err(PanningError::InvalidParameter(
            "Pan value must be between -1.0 and 1.0".to_string(),
        ));
    }

    let theta = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
    let left_gain = theta.cos();
    let right_gain = theta.sin();

    let mut samples = Vec::with_capacity(signal.samples.len() * 2);
    for &sample in &signal.samples {
        samples.push(sample * left_gain);  
        samples.push(sample * right_gain); 
    }

    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: 2,
    })
}

/// Pans a mono signal across a multi-channel sound field.
///
/// This function distributes a mono signal across a specified number of channels
/// (e.g., 5.1 surround) based on an azimuth angle (in degrees). The signal must be mono.
/// Supported channel layouts: 2 (stereo), 4 (quad), 6 (5.1).
///
/// # Arguments
/// * `signal` - The mono audio signal to pan.
/// * `azimuth` - The angle in degrees (0° front, 90° right, 180° rear, 270° left).
/// * `channels` - The target number of channels (2, 4, or 6).
///
/// # Returns
/// Returns `Result<AudioData, PanningError>` containing the multi-channel signal or an error.
///
/// # Examples
/// ```
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let signal = AudioData { samples: vec![1.0, 1.0], sample_rate: 44100, channels: 1 };
/// let panned = multi_channel_pan(&signal, 0.0, 6)?; // Front center for 5.1
/// assert_eq!(panned.channels, 6);
/// // Samples: [FL, FR, C, LFE, BL, BR], center emphasized
/// assert_eq!(panned.samples[2], 1.0); // Center channel full
/// assert_eq!(panned.samples[3], 0.0); // LFE off
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// # Errors
/// Returns an error if the input is invalid (e.g., empty signal or
/// out-of-range parameters) or if the computation cannot be completed.
pub fn multi_channel_pan(
    signal: &AudioData,
    azimuth: f32,
    channels: u16,
) -> Result<AudioData, PanningError> {
    if signal.channels != 1 {
        return Err(PanningError::NotMono(signal.channels));
    }
    if !matches!(channels, 2 | 4 | 6) {
        return Err(PanningError::UnsupportedChannels(channels));
    }

    let azimuth = (azimuth % 360.0 + 360.0) % 360.0;
    let mut gains = vec![0.0; channels as usize];

    match channels {
        2 => {
            let pan = if azimuth <= 90.0 {
                azimuth / 90.0
            } else if azimuth <= 180.0 {
                1.0 - (azimuth - 90.0) / 90.0
            } else if azimuth <= 270.0 {
                -(azimuth - 180.0) / 90.0
            } else {
                -1.0 + (azimuth - 270.0) / 90.0
            };
            gains[0] = (1.0 - pan) / 2.0;
            gains[1] = f32::midpoint(pan, 1.0);
        }
        4 => {
            if azimuth <= 90.0 {
                gains[0] = 1.0 - azimuth / 90.0;
                gains[1] = azimuth / 90.0;
            } else if azimuth <= 180.0 {
                gains[1] = 1.0 - (azimuth - 90.0) / 90.0;
                gains[3] = (azimuth - 90.0) / 90.0;
            } else if azimuth <= 270.0 {
                gains[3] = 1.0 - (azimuth - 180.0) / 90.0;
                gains[2] = (azimuth - 180.0) / 90.0;
            } else {
                gains[2] = 1.0 - (azimuth - 270.0) / 90.0;
                gains[0] = (azimuth - 270.0) / 90.0;
            }
        }
        6 => {
            // 5.1 (ITU speaker layout): pan between the two adjacent speakers that
            // bracket `azimuth`, crossfading linearly. Channel order is
            // [FL, FR, C, LFE, BL, BR]; LFE carries no directional signal.
            const SPEAKERS: [(usize, f32); 5] =
                [(2, 0.0), (1, 30.0), (5, 110.0), (4, 250.0), (0, 330.0)];
            let n = SPEAKERS.len();
            let mut lo = n - 1;
            for (i, &(_, angle)) in SPEAKERS.iter().enumerate() {
                if azimuth < angle {
                    lo = if i == 0 { n - 1 } else { i - 1 };
                    break;
                }
            }
            let hi = (lo + 1) % n;
            let (lo_ch, lo_angle) = SPEAKERS[lo];
            let (hi_ch, hi_angle) = SPEAKERS[hi];
            let span = if hi_angle > lo_angle { hi_angle - lo_angle } else { 360.0 - lo_angle + hi_angle };
            let offset = if azimuth >= lo_angle { azimuth - lo_angle } else { 360.0 - lo_angle + azimuth };
            let pan = (offset / span).clamp(0.0, 1.0);
            gains[lo_ch] = 1.0 - pan;
            gains[hi_ch] = pan;
            gains[3] = 0.0; // LFE always off
        }
        _ => return Err(PanningError::UnsupportedChannels(channels)),
    }

    let mut samples = Vec::with_capacity(signal.samples.len() * channels as usize);
    for &sample in &signal.samples {
        for gain in &gains {
            samples.push(sample * gain);
        }
    }

    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels,
    })
}
