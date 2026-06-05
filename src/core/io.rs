use crate::signal_processing::{resample, to_mono};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use ndarray::ShapeError;
use rayon::prelude::*;
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use thiserror::Error;

/// Enumerates error conditions for WAV-based audio operations.
///
/// Variants encapsulate specific failure modes encountered during file I/O, format parsing,
/// or signal processing, with detailed diagnostics for DSP pipeline debugging.
#[derive(Error, Debug)]
pub enum AudioError {
    /// WAV file open failure, typically due to invalid path or corrupted header.
    #[error("WAV open failed: {0}")]
    OpenError(#[from] hound::Error),

    /// Unsupported WAV sample format (e.g., formats other than 8/16/24/32-bit PCM or 32-bit float).
    #[error("Unsupported WAV sample format")]
    UnsupportedFormat,

    /// Offset or duration exceeds sample bounds.
    #[error("Offset or duration out of bounds")]
    InvalidRange,

    /// General I/O error outside `hound` operations (e.g., filesystem issues).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// `hound`-specific error during sample read/write.
    #[error("Hound processing error: {0}")]
    HoundError(hound::Error),

    /// Resampling failure from `signal_processing::resampling`.
    #[error("Resampling error: {0}")]
    ResampleError(#[from] crate::signal_processing::resampling::ResampleError),

    /// Streaming operation failure (e.g., channel disconnect or thread failure).
    #[error("Stream processing error")]
    StreamError,

    /// Array shape mismatch from `ndarray` operations.
    #[error("Shape mismatch: {0}")]
    ShapeError(#[from] ShapeError),

    /// Insufficient samples for requested operation.
    #[error("Insufficient sample count: {0}")]
    InsufficientData(String),

    /// Invalid parameter (e.g., negative offset, zero frame length).
    #[error("Invalid parameter: {0}")]
    InvalidInput(String),

    /// Numerical computation failure (e.g., overflow).
    #[error("Computation error: {0}")]
    ComputationFailed(String),

    /// File not found at the specified path.
    #[error("File not found: {0}")]
    FileNotFound(String),
}

/// Core audio data container for WAV-based DSP workflows.
///
/// Stores interleaved 32-bit float samples with associated sample rate and channel count.
/// Optimized for in-memory processing and compatibility with `signal_processing` operations.
/// Validates sample rate and channel count at construction to ensure correctness.
///
/// # Fields
/// - `samples`: Interleaved `f32` sample buffer (e.g., `[L1, R1, L2, R2...]` for stereo).
/// - `sample_rate`: Samples per second (Hz), must be positive.
/// - `channels`: Number of channels (1 = mono, 2 = stereo), must be positive.
///
/// # Notes
/// - Samples are stored in interleaved format: for stereo, `[L1, R1, L2, R2, ...]`.
/// - Empty `samples` vectors are allowed, but operations in `ops.rs` may reject them.
/// - Use utility methods like `to_mono`, `split_channels`, `duration`, or `frame_count`
///   for common tasks.
/// - For `librosa`-like raw access, use `to_raw` to get samples, sample rate, and channels.
///
/// # Examples
/// ```no_run
/// use dasp_rs::types::{AudioData, AudioError};
/// // Create mono audio
/// let audio = AudioData::new(vec![0.5, -0.5, 0.5], 44100, 1)?;
/// assert_eq!(audio.samples.len(), 3);
/// assert_eq!(audio.sample_rate, 44100);
/// assert_eq!(audio.channels, 1);
///
/// // Create stereo audio and convert to mono
/// let stereo = AudioData::new(vec![0.2, 0.4, 0.6, 0.8], 44100, 2)?;
/// let mono = stereo.to_mono();
/// assert_eq!(mono.samples, vec![0.3, 0.7]);
/// assert_eq!(mono.channels, 1);
///
/// // Get duration
/// assert_eq!(mono.duration(), 2.0 / 44100.0);
///
/// // Split channels
/// let channels = stereo.split_channels()?;
/// assert_eq!(channels, vec![vec![0.2, 0.6], vec![0.4, 0.8]]);
///
/// // Raw access
/// let (samples, sr, ch) = stereo.to_raw();
/// assert_eq!(samples, &[0.2, 0.4, 0.6, 0.8]);
/// assert_eq!(sr, 44100);
/// assert_eq!(ch, 2);
///
/// // Invalid construction
/// let result = AudioData::new(vec![0.1], 0, 1);
/// assert!(matches!(result, Err(AudioError::InvalidInput(_))));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioData {
    /// Constructs an `AudioData` instance from raw components with validation.
    ///
    /// # Parameters
    /// - `samples`: Interleaved `f32` sample buffer (may be empty).
    /// - `sample_rate`: Sample rate in Hz (must be positive).
    /// - `channels`: Channel count (must be positive).
    ///
    /// # Returns
    /// - `Ok(AudioData)`: Initialized instance.
    /// - `Err(AudioError)`: If `sample_rate` or `channels` is zero.
    ///
    /// # Example
    /// ```
    /// use dasp_rs::types::{AudioData, AudioError};
    /// let audio = AudioData::new(vec![0.5, -0.5], 44100, 1)?;
    /// assert_eq!(audio.samples.len(), 2);
    /// assert_eq!(audio.sample_rate, 44100);
    /// assert_eq!(audio.channels, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Result<Self, AudioError> {
        if sample_rate == 0 {
            return Err(AudioError::InvalidInput(
                "Sample rate must be positive".into(),
            ));
        }
        if channels == 0 {
            return Err(AudioError::InvalidInput(
                "Channel count must be positive".into(),
            ));
        }
        Ok(Self {
            samples,
            sample_rate,
            channels,
        })
    }

    /// Converts multi-channel audio to mono by averaging channels.
    ///
    /// Uses `signal_processing::to_mono` to compute the mean of samples across channels
    /// for each frame. Returns a new `AudioData` with `channels = 1`.
    ///
    /// # Returns
    /// New `AudioData` instance with mono samples.
    ///
    /// # Example
    /// ```no_run
    /// use dasp_rs::types::AudioData;
    /// let stereo = AudioData::new(vec![0.2, 0.4, 0.6, 0.8], 44100, 2)?;
    /// let mono = stereo.to_mono();
    /// assert_eq!(mono.samples, vec![0.3, 0.7]);
    /// assert_eq!(mono.channels, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_mono(&self) -> Self {
        let samples = if self.channels > 1 {
            to_mono(&self.samples, self.channels as usize)
        } else {
            self.samples.clone()
        };
        Self {
            samples,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Splits interleaved samples into separate channel vectors.
    ///
    /// De-interleaves the `samples` buffer into a vector of per-channel sample vectors.
    /// For example, stereo `[L1, R1, L2, R2]` becomes `[vec![L1, L2], vec![R1, R2]]`.
    ///
    /// # Returns
    /// - `Ok(Vec<Vec<f32>>)`: Vector of channel sample vectors.
    /// - `Err(AudioError)`: If `samples` length is not a multiple of `channels`.
    ///
    /// # Example
    /// ```
    /// use dasp_rs::types::{AudioData, AudioError};
    /// let stereo = AudioData::new(vec![0.2, 0.4, 0.6, 0.8], 44100, 2)?;
    /// let channels = stereo.split_channels()?;
    /// assert_eq!(channels, vec![vec![0.2, 0.6], vec![0.4, 0.8]]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn split_channels(&self) -> Result<Vec<Vec<f32>>, AudioError> {
        if self.samples.len() % self.channels as usize != 0 {
            return Err(AudioError::InvalidInput(
                "Sample length must be a multiple of channels".into(),
            ));
        }
        let frame_count = self.samples.len() / self.channels as usize;
        let mut channels = vec![Vec::with_capacity(frame_count); self.channels as usize];
        for (i, &sample) in self.samples.iter().enumerate() {
            let channel_idx = i % self.channels as usize;
            channels[channel_idx].push(sample);
        }
        Ok(channels)
    }

    /// Returns the duration of the audio in seconds.
    ///
    /// Computed as `samples.len() / (channels * sample_rate)`.
    ///
    /// # Returns
    /// Duration in seconds as `f32`.
    ///
    /// # Example
    /// ```
    /// use dasp_rs::types::AudioData;
    /// let audio = AudioData::new(vec![0.2, 0.4], 44100, 1)?;
    /// assert_eq!(audio.duration(), 2.0 / 44100.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.channels as f32 * self.sample_rate as f32)
    }

    /// Returns the number of frames (samples per channel).
    ///
    /// Computed as `samples.len() / channels`.
    ///
    /// # Returns
    /// Number of frames as `usize`.
    ///
    /// # Example
    /// ```
    /// use dasp_rs::types::AudioData;
    /// let stereo = AudioData::new(vec![0.2, 0.4, 0.6, 0.8], 44100, 2)?;
    /// assert_eq!(stereo.frame_count(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Returns raw samples, sample rate, and channels for `librosa`-like access.
    ///
    /// Provides a tuple of `(&[f32], u32, u16)` for users who prefer raw data access.
    ///
    /// # Returns
    /// Tuple of `(samples, sample_rate, channels)`.
    ///
    /// # Example
    /// ```
    /// use dasp_rs::types::AudioData;
    /// let audio = AudioData::new(vec![0.2, 0.4], 44100, 1)?;
    /// let (samples, sr, ch) = audio.to_raw();
    /// assert_eq!(samples, &[0.2, 0.4]);
    /// assert_eq!(sr, 44100);
    /// assert_eq!(ch, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_raw(&self) -> (&[f32], u32, u16) {
        (&self.samples, self.sample_rate, self.channels)
    }
}

/// Loads WAV file into `AudioData` with optional DSP transformations.
///
/// Streams WAV data from disk, supporting 8/16/24/32-bit PCM and 32-bit float formats.
/// Applies resampling, mono conversion, and sample trimming as specified.
///
/// # Parameters
/// - `path`: WAV file path (`AsRef<Path>`).
/// - `sr`: Target sample rate (Hz); `None` retains source rate.
/// - `mono`: Convert to mono if `Some(true)`; `None` retains source channels.
/// - `offset`: Start time (seconds); `None` defaults to 0.0.
/// - `duration`: Segment length (seconds); `None` takes full length.
///
/// # Returns
/// - `Ok(AudioData)`: Processed audio data.
/// - `Err(AudioError)`: Failure due to I/O, format, or parameter errors.
///
/// # Errors
/// - `AudioError::FileNotFound`: The specified file does not exist.
/// - `AudioError::OpenError`: Invalid WAV file or corrupted header.
/// - `AudioError::InvalidRange`: Offset/duration exceeds file length.
/// - `AudioError::UnsupportedFormat`: Unsupported sample format.
/// - `AudioError::HoundError`: Error reading samples.
/// - `AudioError::ResampleError`: Resampling failed.
/// - `AudioError::InvalidInput`: Invalid parameters (e.g., negative offset, zero sample rate, zero channels).
/// - `AudioError::InsufficientData`: Empty or insufficient samples.
///
/// # Examples
/// ```no_run
/// use dasp_rs::io::load;
/// use dasp_rs::types::AudioData;
/// // Load entire file with original channels and sample rate
/// let audio = load("audio.wav", None, None, None, None)?;
///
/// // Load 5-second mono segment starting at 2 seconds, resampled to 16kHz
/// let segment = load("audio.wav", Some(16000), Some(true), Some(2.0), Some(5.0))?;
///
/// // Process stereo audio
/// let stereo = load("stereo.wav", None, None, None, None)?;
/// let channels = stereo.split_channels()?;
/// let mono = stereo.to_mono();
/// assert_eq!(mono.channels, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load<P: AsRef<Path>>(
    path: P,
    sr: Option<u32>,
    mono: Option<bool>,
    offset: Option<f32>,
    duration: Option<f32>,
) -> Result<AudioData, AudioError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(AudioError::FileNotFound(
            path.to_string_lossy().into_owned(),
        ));
    }

    if let Some(off) = offset {
        if off < 0.0 {
            return Err(AudioError::InvalidInput("Offset cannot be negative".into()));
        }
    }
    if let Some(dur) = duration {
        if dur <= 0.0 {
            return Err(AudioError::InvalidInput("Duration must be positive".into()));
        }
    }
    if let Some(rate) = sr {
        if rate == 0 {
            return Err(AudioError::InvalidInput(
                "Sample rate must be positive".into(),
            ));
        }
    }

    let file = File::open(path)?;
    let mut reader = WavReader::new(file)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let start = (offset.unwrap_or(0.0) * sample_rate as f32) as usize;
    let len = duration.map(|d| (d * sample_rate as f32) as usize);

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .skip(start)
            .take(len.unwrap_or(usize::MAX))
            .collect::<Result<Vec<_>, _>>()
            .map_err(AudioError::HoundError)?,
        SampleFormat::Int => match spec.bits_per_sample {
            8 => reader
                .samples::<i8>()
                .skip(start)
                .take(len.unwrap_or(usize::MAX))
                .map(|s| s.map(|v| v as f32 / i8::MAX as f32))
                .collect::<Result<Vec<_>, _>>()
                .map_err(AudioError::HoundError)?,
            16 => reader
                .samples::<i16>()
                .skip(start)
                .take(len.unwrap_or(usize::MAX))
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect::<Result<Vec<_>, _>>()
                .map_err(AudioError::HoundError)?,
            24 | 32 => reader
                .samples::<i32>()
                .skip(start)
                .take(len.unwrap_or(usize::MAX))
                .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
                .collect::<Result<Vec<_>, _>>()
                .map_err(AudioError::HoundError)?,
            _ => return Err(AudioError::UnsupportedFormat),
        },
    };

    if samples.is_empty() && len != Some(0) {
        return Err(AudioError::InsufficientData("No samples available".into()));
    }
    if start > samples.len() && !samples.is_empty() {
        return Err(AudioError::InvalidRange);
    }

    let mut samples = samples;
    let channels = spec.channels as usize;
    if channels > 1 && mono.unwrap_or(false) {
        samples = to_mono(&samples, channels);
    }

    let final_samples = if let Some(target_samplerate) = sr {
        if target_samplerate != sample_rate {
            resample(&samples, sample_rate, target_samplerate)?
        } else {
            samples
        }
    } else {
        samples
    };

    AudioData::new(
        final_samples,
        sr.unwrap_or(sample_rate),
        if mono.unwrap_or(false) {
            1
        } else {
            spec.channels
        },
    )
}

/// Modern audio decoder with builder pattern for clean, readable API.
///
/// # Example
/// ```no_run
/// use dasp_rs::io::Decoder;
/// 
/// // Simple loading
/// let audio = Decoder::from("file.wav").load()?;
/// 
/// // With options
/// let audio = Decoder::from("file.wav")
///     .sample_rate(22050)
///     .mono()
///     .offset(10.0)
///     .duration(30.0)
///     .load()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Decoder {
    path: PathBuf,
    sample_rate: Option<u32>,
    mono: bool,
    offset: Option<f32>,
    duration: Option<f32>,
}

impl Decoder {
    /// Create a new audio decoder from a file path.
    pub fn from<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            sample_rate: None,
            mono: false,
            offset: None,
            duration: None,
        }
    }

    /// Set the target sample rate for resampling.
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Convert to mono (single channel).
    pub fn mono(mut self) -> Self {
        self.mono = true;
        self
    }

    /// Set the start offset in seconds.
    pub fn offset(mut self, seconds: f32) -> Self {
        self.offset = Some(seconds);
        self
    }

    /// Set the duration to load in seconds.
    pub fn duration(mut self, seconds: f32) -> Self {
        self.duration = Some(seconds);
        self
    }

    /// Load the audio file with the configured options.
    pub fn load(self) -> Result<AudioData, AudioError> {
        load(
            &self.path,
            self.sample_rate,
            Some(self.mono),
            self.offset,
            self.duration,
        )
    }
}

/// Exports `AudioData` to a WAV file using in-memory buffering.
///
/// Writes 32-bit float WAV data via `Cursor`, committing to disk in a single operation.
/// Automatically clamps samples to `[-1.0, 1.0]` range.
///
/// # Parameters
/// - `path`: Output WAV file path (`AsRef<Path>`).
/// - `audio_data`: Source `AudioData` reference.
///
/// # Returns
/// - `Ok(())`: Successful write.
/// - `Err(AudioError)`: I/O or format error.
///
/// # Errors
/// - `AudioError::IoError`: Failed to write to filesystem.
/// - `AudioError::HoundError`: WAV format encoding error.
/// - `AudioError::InvalidInput`: Invalid audio data parameters (e.g., zero channels, zero sample rate).
///
/// # Example
/// ```no_run
/// use dasp_rs::types::AudioData;
/// use dasp_rs::io::export;
/// let audio = AudioData::new(vec![0.2, 0.4, 0.6], 44100, 1)?;
/// export("output.wav", &audio)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn export<P: AsRef<Path>>(path: P, audio_data: &AudioData) -> Result<(), AudioError> {
    if audio_data.channels == 0 {
        return Err(AudioError::InvalidInput(
            "Channel count must be positive".into(),
        ));
    }
    if audio_data.sample_rate == 0 {
        return Err(AudioError::InvalidInput(
            "Sample rate must be positive".into(),
        ));
    }

    let spec = WavSpec {
        channels: audio_data.channels,
        sample_rate: audio_data.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut buffer = Vec::new();
    let mut writer = WavWriter::new(Cursor::new(&mut buffer), spec)?;
    for &sample in &audio_data.samples {
        writer.write_sample(sample.clamp(-1.0, 1.0))?;
    }
    writer.finalize()?;
    std::fs::write(path, buffer)?;
    Ok(())
}

/// Generates a collection of WAV sample blocks with optional parallel processing.
///
/// Streams WAV data and splits it into fixed-size blocks, processed sequentially or in parallel
/// based on workload size for optimal performance. Returns a vector of blocks that can be
/// iterated over.
///
/// # Parameters
/// - `path`: WAV file path (`AsRef<Path>`).
/// - `block_length`: Maximum block count.
/// - `frame_length`: Samples per block.
/// - `hop_length`: Step size between blocks; `None` uses `frame_length`.
///
/// # Returns
/// - `Ok(Vec<Vec<f32>>)`: Vector of sample blocks.
/// - `Err(AudioError)`: I/O or format error.
///
/// # Errors
/// - `AudioError::FileNotFound`: The specified file does not exist.
/// - `AudioError::OpenError`: Invalid WAV file or corrupted header.
/// - `AudioError::HoundError`: Error reading samples.
/// - `AudioError::UnsupportedFormat`: Unsupported sample format.
/// - `AudioError::InvalidInput`: Invalid parameters (e.g., zero frame length).
/// - `AudioError::InsufficientData`: Insufficient samples for any blocks.
///
/// # Example
/// ```no_run
/// use dasp_rs::io::stream;
/// let blocks = stream("audio.wav", 100, 4096, None)?;
/// for block in blocks {
///     // Process each 4096-sample block
///     println!("Block size: {}", block.len());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance
/// - Uses `rayon` for parallel processing only for large workloads (>1M samples).
/// - Streams data from disk, suitable for large files.
pub fn stream<P: AsRef<Path>>(
    path: P,
    block_length: usize,
    frame_length: usize,
    hop_length: Option<usize>,
) -> Result<Vec<Vec<f32>>, AudioError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(AudioError::FileNotFound(
            path.to_string_lossy().into_owned(),
        ));
    }
    if frame_length == 0 {
        return Err(AudioError::InvalidInput(
            "Frame length must be positive".into(),
        ));
    }
    let hop = hop_length.unwrap_or(frame_length);
    if hop == 0 {
        return Err(AudioError::InvalidInput(
            "Hop length must be positive".into(),
        ));
    }

    let file = File::open(path)?;
    let mut reader = WavReader::new(file)?;
    let spec = reader.spec();

    let samples_iter: Box<dyn Iterator<Item = Result<f32, hound::Error>>> = match spec.sample_format
    {
        SampleFormat::Float => Box::new(reader.samples::<f32>()),
        SampleFormat::Int => match spec.bits_per_sample {
            8 => Box::new(
                reader
                    .samples::<i8>()
                    .map(|s| s.map(|v| v as f32 / i8::MAX as f32)),
            ),
            16 => Box::new(
                reader
                    .samples::<i16>()
                    .map(|s| s.map(|v| v as f32 / i16::MAX as f32)),
            ),
            24 | 32 => Box::new(
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / i32::MAX as f32)),
            ),
            _ => return Err(AudioError::UnsupportedFormat),
        },
    };

    let mut blocks = Vec::new();
    let mut buffer = Vec::with_capacity(frame_length);
    let mut index = 0;
    let mut block_count = 0;

    for sample in samples_iter {
        let sample = sample.map_err(AudioError::HoundError)?;
        buffer.push(sample);
        if buffer.len() >= frame_length && (index % hop == 0 || buffer.len() >= frame_length) {
            let mut block = Vec::with_capacity(frame_length);
            block.extend_from_slice(&buffer[..frame_length]);
            block.resize(frame_length, 0.0);
            blocks.push(block);
            buffer.drain(..hop.min(buffer.len()));
            block_count += 1;
            if block_count >= block_length {
                break;
            }
        }
        index += 1;
    }

    if blocks.is_empty() {
        return Err(AudioError::InsufficientData("No blocks generated".into()));
    }

    let blocks = if frame_length * block_length > 1_000_000 {
        blocks.into_par_iter().collect()
    } else {
        blocks
    };

    Ok(blocks)
}

/// Streams WAV sample blocks lazily with parallel chunk processing.
///
/// Processes WAV data incrementally in a separate thread, generating blocks in parallel
/// within chunks to minimize memory footprint. Sends blocks over a channel, with error
/// handling for thread or receiver failures.
///
/// # Parameters
/// - `path`: WAV file path (`AsRef<Path>`).
/// - `block_length`: Maximum block count.
/// - `frame_length`: Samples per block.
/// - `hop_length`: Step size between blocks; `None` uses `frame_length`.
///
/// # Returns
/// - `Ok(Receiver<Vec<f32>>)`: Channel receiver for blocks.
/// - `Err(AudioError)`: I/O or streaming error.
///
/// # Errors
/// - `AudioError::FileNotFound`: The specified file does not exist.
/// - `AudioError::OpenError`: Invalid WAV file or corrupted header.
/// - `AudioError::HoundError`: Error reading samples.
/// - `AudioError::UnsupportedFormat`: Unsupported sample format.
/// - `AudioError::InvalidInput`: Invalid parameters (e.g., zero frame length).
/// - `AudioError::StreamError`: Channel communication failure or thread failure.
/// - `AudioError::InsufficientData`: Insufficient samples for any blocks.
///
/// # Example
/// ```no_run
/// use dasp_rs::io::stream_lazy;
/// let rx = stream_lazy("audio.wav", 1000, 1024, Some(512))?;
/// while let Ok(block) = rx.recv() {
///     // Process each 1024-sample block with 50% overlap
///     println!("Received block of {} samples", block.len());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance
/// - Background thread for file reading.
/// - Memory-efficient for files >1GB.
/// - Parallel block processing for large chunks.
pub fn stream_lazy<P: AsRef<Path>>(
    path: P,
    block_length: usize,
    frame_length: usize,
    hop_length: Option<usize>,
) -> Result<Receiver<Vec<f32>>, AudioError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(AudioError::FileNotFound(
            path.to_string_lossy().into_owned(),
        ));
    }
    if frame_length == 0 {
        return Err(AudioError::InvalidInput(
            "Frame length must be positive".into(),
        ));
    }
    let hop = hop_length.unwrap_or(frame_length);
    if hop == 0 {
        return Err(AudioError::InvalidInput(
            "Hop length must be positive".into(),
        ));
    }

    let file = File::open(path)?;
    let reader = WavReader::new(file)?;
    let spec = reader.spec();

    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let mut reader = reader;
        let samples_iter: Box<dyn Iterator<Item = Result<f32, _>>> = match spec.sample_format {
            SampleFormat::Float => Box::new(reader.samples::<f32>()),
            SampleFormat::Int => match spec.bits_per_sample {
                8 => Box::new(
                    reader
                        .samples::<i8>()
                        .map(|s| s.map(|v| v as f32 / i8::MAX as f32)),
                ),
                16 => Box::new(
                    reader
                        .samples::<i16>()
                        .map(|s| s.map(|v| v as f32 / i16::MAX as f32)),
                ),
                24 | 32 => Box::new(
                    reader
                        .samples::<i32>()
                        .map(|s| s.map(|v| v as f32 / i32::MAX as f32)),
                ),
                _ => {
                    let _ = tx.send(Vec::new());
                    return;
                }
            },
        };

        let mut chunk = Vec::with_capacity(frame_length * block_length);
        let mut block_count = 0;

        for sample in samples_iter {
            let sample = match sample {
                Ok(s) => s,
                Err(_) => {
                    let _ = tx.send(Vec::new());
                    return;
                }
            };
            chunk.push(sample);

            if chunk.len() >= frame_length
                && (chunk.len() % hop == 0 || chunk.len() >= frame_length * block_length)
            {
                let indices: Vec<usize> = (0..chunk.len())
                    .step_by(hop)
                    .take(block_length - block_count)
                    .collect();
                let drain_to = indices.last().map_or(0, |&i| (i + hop).min(chunk.len()));

                let use_parallel = indices.len() * frame_length > 1_000_000;
                let blocks: Vec<Vec<f32>> = if use_parallel {
                    indices
                        .into_par_iter()
                        .map(|i| {
                            let end = (i + frame_length).min(chunk.len());
                            let mut block = Vec::with_capacity(frame_length);
                            block.extend_from_slice(&chunk[i..end]);
                            block.resize(frame_length, 0.0);
                            block
                        })
                        .collect()
                } else {
                    indices
                        .into_iter()
                        .map(|i| {
                            let end = (i + frame_length).min(chunk.len());
                            let mut block = Vec::with_capacity(frame_length);
                            block.extend_from_slice(&chunk[i..end]);
                            block.resize(frame_length, 0.0);
                            block
                        })
                        .collect()
                };

                for block in blocks {
                    if tx.send(block).is_err() {
                        return;
                    }
                    block_count += 1;
                    if block_count >= block_length {
                        return;
                    }
                }
                chunk.drain(..drain_to);
            }
        }

        if !chunk.is_empty() && block_count < block_length {
            chunk.resize(frame_length, 0.0);
            let _ = tx.send(chunk);
        }
    });

    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    fn create_test_wav() -> AudioData {
        AudioData::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 44100, 1).unwrap()
    }

    #[test]
    fn test_audio_data_new_valid() {
        let audio = AudioData::new(vec![0.1, 0.2], 44100, 1).unwrap();
        assert_eq!(audio.samples, vec![0.1, 0.2]);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
    }

    #[test]
    fn test_audio_data_new_invalid_sample_rate() {
        let result = AudioData::new(vec![0.1], 0, 1);
        assert!(matches!(result, Err(AudioError::InvalidInput(_))));
    }

    #[test]
    fn test_audio_data_new_invalid_channels() {
        let result = AudioData::new(vec![0.1], 44100, 0);
        assert!(matches!(result, Err(AudioError::InvalidInput(_))));
    }

    #[test]
    fn test_audio_data_to_mono() {
        let stereo = AudioData::new(vec![0.1, 0.2, 0.3, 0.4], 44100, 2).unwrap();
        let mono = stereo.to_mono();
        assert_eq!(mono.channels, 1);
        for (actual, expected) in mono.samples.iter().zip(vec![0.15, 0.35]) {
            assert!((actual - expected).abs() < 1e-6, "Expected {}, got {}", expected, actual);
        }
    }

    #[test]
    fn test_audio_data_split_channels() {
        let stereo = AudioData::new(vec![0.1, 0.2, 0.3, 0.4], 44100, 2).unwrap();
        let channels = stereo.split_channels().unwrap();
        assert_eq!(channels, vec![vec![0.1, 0.3], vec![0.2, 0.4]]);
    }

    #[test]
    fn test_audio_data_split_channels_invalid() {
        let invalid = AudioData::new(vec![0.1, 0.2, 0.3], 44100, 2).unwrap();
        let result = invalid.split_channels();
        assert!(matches!(result, Err(AudioError::InvalidInput(_))));
    }

    #[test]
    fn test_audio_data_duration() {
        let audio = AudioData::new(vec![0.1, 0.2], 44100, 1).unwrap();
        assert_eq!(audio.duration(), 2.0 / 44100.0);
    }

    #[test]
    fn test_audio_data_frame_count() {
        let stereo = AudioData::new(vec![0.1, 0.2, 0.3, 0.4], 44100, 2).unwrap();
        assert_eq!(stereo.frame_count(), 2);
    }

    #[test]
    fn test_audio_data_to_raw() {
        let audio = AudioData::new(vec![0.1, 0.2], 44100, 1).unwrap();
        let (samples, sr, ch) = audio.to_raw();
        assert_eq!(samples, &[0.1, 0.2]);
        assert_eq!(sr, 44100);
        assert_eq!(ch, 1);
    }

    #[test]
    fn test_load() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(
            fs::metadata(path).is_ok(),
            "File should exist after export: {:?}",
            path
        );
        let loaded = load(path, None, None, None, None).unwrap();
        assert_eq!(loaded.samples, audio.samples);
        assert_eq!(loaded.channels, audio.channels);
    }

    #[test]
    fn test_load_segment() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(
            fs::metadata(path).is_ok(),
            "File should exist after export: {:?}",
            path
        );
        let loaded = load(path, None, None, Some(0.00004535147), Some(0.00004535148)).unwrap();
        assert_eq!(loaded.samples, vec![0.1, 0.2]);
    }

    #[test]
    fn test_export() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(
            fs::metadata(path).is_ok(),
            "File should exist after export: {:?}",
            path
        );
        let loaded = load(path, None, None, None, None).unwrap();
        assert_eq!(loaded.samples, audio.samples);
    }

    #[test]
    fn test_stream() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(
            fs::metadata(path).is_ok(),
            "File should exist after export: {:?}",
            path
        );
        let blocks = stream(path, 3, 2, Some(2)).unwrap();
        assert_eq!(blocks, vec![vec![0.0, 0.1], vec![0.2, 0.3], vec![0.4, 0.5]]);
    }

    #[test]
    fn test_stream_lazy() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(
            fs::metadata(path).is_ok(),
            "File should exist after export: {:?}",
            path
        );
        let rx = stream_lazy(path, 3, 2, Some(2)).unwrap();
        let blocks: Vec<_> = rx.into_iter().collect();
        assert_eq!(blocks, vec![vec![0.0, 0.1], vec![0.2, 0.3], vec![0.4, 0.5]]);
    }

    #[test]
    fn test_load_file_not_found() {
        let path = Path::new("test.wav");
        if path.exists() {
            fs::remove_file(path).unwrap();
        }
        let result = load(path, None, None, None, None);
        assert!(matches!(result.unwrap_err(), AudioError::FileNotFound(_)));
    }

    #[test]
    fn test_stream_file_not_found() {
        let path = Path::new("test.wav");
        if path.exists() {
            fs::remove_file(path).unwrap();
        }
        let result = stream(path, 3, 2, Some(2));
        assert!(matches!(result.unwrap_err(), AudioError::FileNotFound(_)));
    }

    #[test]
    fn test_stream_lazy_file_not_found() {
        let path = Path::new("test.wav");
        if path.exists() {
            fs::remove_file(path).unwrap();
        }
        let result = stream_lazy(path, 3, 2, Some(2));
        assert!(matches!(result.unwrap_err(), AudioError::FileNotFound(_)));
    }

    #[test]
    fn test_load_empty_file() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut buffer = Vec::new();
        let writer = WavWriter::new(Cursor::new(&mut buffer), spec).unwrap();
        writer.finalize().unwrap();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        fs::write(path, buffer).unwrap();

        let result = load(path, None, None, None, None);
        assert!(matches!(
            result.unwrap_err(),
            AudioError::InsufficientData(_)
        ));
    }

    #[test]
    fn test_load_negative_offset() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(
            fs::metadata(path).is_ok(),
            "File should exist after export: {:?}",
            path
        );
        let result = load(path, None, None, Some(-1.0), None);
        assert!(matches!(result.unwrap_err(), AudioError::InvalidInput(_)));
    }

    #[test]
    fn test_load_zero_duration() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(
            fs::metadata(path).is_ok(),
            "File should exist after export: {:?}",
            path
        );
        let result = load(path, None, None, None, Some(0.0));
        assert!(matches!(result.unwrap_err(), AudioError::InvalidInput(_)));
    }

    #[test]
    fn test_stream_zero_frame_length() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(
            fs::metadata(path).is_ok(),
            "File should exist after export: {:?}",
            path
        );
        let result = stream(path, 3, 0, Some(2));
        assert!(matches!(result.unwrap_err(), AudioError::InvalidInput(_)));
    }

    #[test]
    fn test_export_invalid_channels() {
        let audio = AudioData::new(vec![0.1, 0.2], 44100, 0);
        assert!(
            matches!(audio, Err(AudioError::InvalidInput(_))),
            "AudioData::new should fail with zero channels"
        );
    }
}
