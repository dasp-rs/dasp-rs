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
/// pan value. A pan of -1.0 is fully left, 0.0 is center, and 1.0 is fully right.
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
/// ```no_run
/// use dasp_rs::proc::*;
/// use dasp_rs::types::*;
/// let signal = AudioData { samples: vec![1.0, 1.0], sample_rate: 44100, channels: 1 };
/// let panned = stereo_pan(&signal, 0.0)?; // Center
/// assert_eq!(panned.samples, vec![1.0, 1.0, 1.0, 1.0]); // Left, Right, Left, Right
/// assert_eq!(panned.channels, 2);
///
/// let panned_left = stereo_pan(&signal, -1.0)?; // Fully left
/// assert_eq!(panned_left.samples, vec![1.0, 0.0, 1.0, 0.0]);
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

    let left_gain = (1.0 - pan) / 2.0;
    let right_gain = f32::midpoint(pan, 1.0);

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
            // 5.1: Front Left, Front Right, Center, LFE, Back Left, Back Right
            if azimuth <= 45.0 {
                gains[2] = 1.0 - azimuth / 45.0; // Center
                gains[1] = azimuth / 45.0;       // FR
            } else if azimuth <= 135.0 {
                gains[1] = 1.0;                  // FR full from 45° to 135°
            } else if azimuth <= 225.0 {
                gains[1] = 1.0 - (azimuth - 135.0) / 90.0; // FR fades
                gains[5] = (azimuth - 135.0) / 90.0;       // BR rises
            } else if azimuth <= 315.0 {
                gains[5] = 1.0 - (azimuth - 225.0) / 90.0; // BR fades
                gains[4] = (azimuth - 225.0) / 90.0;       // BL rises
            } else {
                gains[4] = 1.0 - (azimuth - 315.0) / 45.0; // BL fades
                gains[0] = (azimuth - 315.0) / 45.0;       // FL rises
            }
            // Handle left side (270° to 360°)
            if azimuth >= 315.0 {
                gains[2] = (azimuth - 315.0) / 45.0; // Center fades in
            } else if azimuth >= 225.0 {
                gains[0] = (azimuth - 225.0) / 90.0; // FL
            } else if (45.0..=135.0).contains(&azimuth) {
                gains[0] = 0.0; // No FL contribution in this range
            } else if azimuth <= 45.0 {
                gains[0] = 0.0; // No FL at front center
            }
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
