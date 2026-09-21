//! # DASP-RS: Digital Audio Signal Processing in Rust
//!
//! DASP-RS provides a collection of tools and utilities for audio signal processing,
//! analysis, and generation. It includes functionality for handling audio input/output,
//! performing signal transformations, generating synthetic signals, extracting audio features,
//! working with magnitude spectra, and pitch-related operations. The library is designed
//! to be modular and extensible, leveraging Rust's performance and safety features.
//!
//! ## Key Features
//! - Audio I/O: Loading and saving audio files with flexible options.
//! - Signal Processing: Time-frequency transforms (e.g., STFT, CQT) and filtering.
//! - Signal Generation: Creating synthetic waveforms and noise.
//! - Feature Extraction: Computing audio features like tempo, pitch, and spectral properties.
//! - Magnitude Operations: Manipulating and analyzing magnitude spectra.
//! - Pitch Utilities: Converting between frequency, MIDI, and musical notations.
//! - Utilities: General-purpose functions for audio analysis and conversion.
//!
//! ## Usage
//! To use this library, add it to your `Cargo.toml` and import the desired modules:
//!
//! ```toml
//! [dependencies]
//! dasp-rs = "0.4"
//! ```
//!
//! ```no_run
//! // Option 1: Use prelude for convenience
//! use dasp_rs::prelude::*;
//!
//! let audio = Decoder::new("example.wav")
//!     .sample_rate(22050)
//!     .mono()
//!     .load()?;
//!
//! let duration = get_duration(&audio);
//! println!("Duration: {} seconds", duration);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ```no_run
//! // Option 2: Explicit imports for clarity
//! use dasp_rs::{types::AudioData, io::{Decoder, export}, util::get_duration};
//!
//! let audio = Decoder::new("example.wav")
//!     .sample_rate(22050)
//!     .mono()
//!     .load()?;
//!
//! let duration = get_duration(&audio);
//! println!("Duration: {} seconds", duration);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## API Structure
//! The API is organized by concern for clarity and discoverability:
//! - `prelude` - Convenient imports for common use cases
//! - `types` - Core audio data types (`AudioData`, `AudioError`)
//! - `io` - Audio input/output operations
//! - `proc` - Signal processing algorithms
//! - `feat` - Audio feature extraction
//! - `util` - Utility functions
//! - `pitch` - Pitch detection and conversion
//! - `mag` - Magnitude spectrum operations
//! - `generate` - Signal generation

// Internal modules
mod core;
mod features;
mod magnitude;
mod pitch_core;
mod signal_generation;
mod signal_processing;
mod utils;

/// Core audio data types
pub mod types {
    pub use crate::core::{AudioData, AudioError};
}

/// Audio input/output operations
pub mod io {
    pub use crate::core::io::{Decoder, export, load, stream, stream_lazy};
}

/// Sample-wise signal operations
pub mod ops {
    pub use crate::core::ops::*;
}

/// Signal processing algorithms
pub mod proc {
    pub use crate::signal_processing::{
        amplitude::*, effects::*, mixing::*, mono::*, panning::*, resampling::*, time_domain::*,
        time_frequency::*,
    };
}

/// Audio feature extraction
pub mod feat {
    pub use crate::features::spectral::*;
    pub use crate::features::{
        decompose::*, feat_util::*, harmonics::*, inverse::*, manipulation::*, phase_recovery::*,
        rhythm::*, segment::*, sequence::*,
    };
}

/// Magnitude spectrum operations
pub mod mag {
    pub use crate::magnitude::scaling::*;
}

/// Pitch detection and conversion
pub mod pitch {
    pub use crate::pitch_core::*;
}

/// Utility functions
pub mod util {
    pub use crate::utils::{array::*, frequency::*, notation::*, time::*};
}

/// Signal generation
pub mod generate {
    pub use crate::signal_generation::*;
}

/// Prelude module for convenient imports.
///
/// This module re-exports the most commonly used types and functions
/// to make it easier to use the library without verbose imports.
///
/// # Example
/// ```no_run
/// use dasp_rs::prelude::*;
///
/// // Now you can use common items directly
/// let audio = Decoder::new("file.wav").mono().load()?;
/// let duration = get_duration(&audio);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub mod prelude {
    // Core types
    pub use crate::core::{AudioData, AudioError};

    // I/O operations
    pub use crate::core::io::{Decoder, export};

    // Utility functions
    pub use crate::utils::frequency::{hz_to_midi, midi_to_hz};
    pub use crate::utils::time::get_duration;

    // Common signal processing
    pub use crate::signal_processing::{mono::to_mono, resampling::resample, time_frequency::stft};

    // Common features
    pub use crate::features::spectral::spectral;
    pub use crate::features::{harmonics::salience, rhythm::tempo};

    // Pitch operations
    pub use crate::pitch_core::*;
}
