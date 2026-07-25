use crate::signal_processing::{resample, to_mono};
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::ShapeError;
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use thiserror::Error;

/// Enumerates error conditions for audio I/O operations.
///
/// Variants encapsulate specific failure modes encountered during file I/O, format parsing,
/// or signal processing, with detailed diagnostics for DSP pipeline debugging.
#[derive(Error, Debug)]
pub enum AudioError {
    /// Audio file open or format-probe failure (invalid path, unsupported container, etc.).
    #[error("Audio open/probe failed: {0}")]
    OpenError(String),

    /// Unsupported codec or sample format inside an otherwise valid container.
    #[error("Unsupported audio format")]
    UnsupportedFormat,

    /// Offset or duration exceeds sample bounds.
    #[error("Offset or duration out of bounds")]
    InvalidRange,

    /// General I/O error (e.g., filesystem issues).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// `hound` error during WAV write (`export`).
    #[error("WAV write error: {0}")]
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

impl From<hound::Error> for AudioError {
    fn from(e: hound::Error) -> Self {
        Self::HoundError(e)
    }
}

/// Core audio data container for DSP workflows.
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
/// - For raw access, use `to_raw` to get samples, sample rate, and channels.
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
    /// Interleaved `f32` sample buffer (e.g., `[L1, R1, L2, R2, …]` for stereo).
    pub samples: Vec<f32>,
    /// Sample rate in Hz (must be positive).
    pub sample_rate: u32,
    /// Number of interleaved channels (1 = mono, 2 = stereo; must be positive).
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
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
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
    #[must_use]
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
    /// # Errors
    /// Returns an error if the input is invalid (e.g., empty signal or
    /// out-of-range parameters) or if the computation cannot be completed.
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
        self.samples.len() as f32 / (f32::from(self.channels) * self.sample_rate as f32)
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

    /// Returns raw samples, sample rate, and channels for direct access.
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

/// Decodes an audio file into interleaved `f32` samples using Symphonia.
///
/// Supports WAV, MP3, FLAC, OGG Vorbis, AIFF, AAC/M4A, and any other format
/// supported by the enabled Symphonia codec/format features.
///
/// Returns `(samples, sample_rate, channels)`.
fn decode_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32, u16), AudioError> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AudioError::OpenError(e.to_string()))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(AudioError::UnsupportedFormat)?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track
        .codec_params
        .channels
        .map_or(1, |c| c.count() as u16);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::OpenError(e.to_string()))?;

    let mut samples: Vec<f32> = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A single malformed packet is recoverable; skip it and keep decoding.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            // Anything else (I/O failure, reset required, etc.) means the stream
            // is no longer trustworthy — surface it instead of silently returning
            // a shorter-than-expected `AudioData`.
            Err(e) => return Err(AudioError::OpenError(format!("Decode failed: {e}"))),
        };
        let spec = *decoded.spec();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buf.samples());
    }

    Ok((samples, sample_rate, channels))
}

/// Loads an audio file into `AudioData` with optional DSP transformations.
///
/// Decodes any format supported by Symphonia (WAV, MP3, FLAC, OGG, AIFF, AAC/M4A, …).
/// Applies resampling, mono conversion, and sample trimming as specified.
///
/// # Parameters
/// - `path`: Audio file path (`AsRef<Path>`).
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
/// - `AudioError::OpenError`: Unrecognised or corrupted audio file.
/// - `AudioError::UnsupportedFormat`: File container / codec not supported.
/// - `AudioError::InvalidRange`: Offset/duration exceeds file length.
/// - `AudioError::ResampleError`: Resampling failed.
/// - `AudioError::InvalidInput`: Invalid parameters (e.g., negative offset, zero sample rate).
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
/// let segment = load("audio.mp3", Some(16000), Some(true), Some(2.0), Some(5.0))?;
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

    let (raw_samples, sample_rate, channels) = decode_audio(path)?;

    let start = (offset.unwrap_or(0.0) * sample_rate as f32) as usize;
    let len = duration.map(|d| (d * sample_rate as f32) as usize);

    let end = len.map_or(raw_samples.len(), |l| (start + l).min(raw_samples.len()));
    let start = start.min(raw_samples.len());

    let mut samples = raw_samples[start..end].to_vec();

    // `duration` is validated to be strictly positive above, so a `len` that
    // rounds down to 0 samples (a tiny duration at a low sample rate) is still
    // a real request that couldn't be satisfied, not an intentional empty read.
    if samples.is_empty() {
        return Err(AudioError::InsufficientData("No samples available".into()));
    }

    if channels > 1 && mono.unwrap_or(false) {
        samples = to_mono(&samples, channels as usize);
    }

    let final_samples = if let Some(target_sr) = sr {
        if target_sr == sample_rate {
            samples
        } else {
            resample(&samples, sample_rate, target_sr)?
        }
    } else {
        samples
    };

    AudioData::new(
        final_samples,
        sr.unwrap_or(sample_rate),
        if mono.unwrap_or(false) { 1 } else { channels },
    )
}

/// Modern audio decoder with builder pattern for clean, readable API.
///
/// # Example
/// ```no_run
/// use dasp_rs::io::Decoder;
///
/// // Simple loading (WAV, MP3, FLAC, OGG, …)
/// let audio = Decoder::new("file.flac").load()?;
///
/// // With options
/// let audio = Decoder::new("file.mp3")
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
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            sample_rate: None,
            mono: false,
            offset: None,
            duration: None,
        }
    }

    /// Create a new audio decoder from a file path.
    #[deprecated(
        since = "0.4.0",
        note = "use `Decoder::new` instead; `from` is reserved for `From` conversions"
    )]
    pub fn from<P: AsRef<Path>>(path: P) -> Self {
        Self::new(path)
    }

    /// Set the target sample rate for resampling.
    #[must_use]
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Convert to mono (single channel).
    #[must_use]
    pub fn mono(mut self) -> Self {
        self.mono = true;
        self
    }

    /// Set the start offset in seconds.
    #[must_use]
    pub fn offset(mut self, seconds: f32) -> Self {
        self.offset = Some(seconds);
        self
    }

    /// Set the duration to load in seconds.
    #[must_use]
    pub fn duration(mut self, seconds: f32) -> Self {
        self.duration = Some(seconds);
        self
    }

    /// Load the audio file with the configured options.
    ///
    /// # Errors
    /// Returns any error produced by [`load`]: file-not-found, unsupported
    /// format, invalid offset/duration, or resampling failure.
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

/// Exports `AudioData` to a WAV file.
///
/// Writes 32-bit float WAV data via an in-memory buffer, committing to disk in a single
/// operation. Automatically clamps samples to `[-1.0, 1.0]`.
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

/// Collects fixed-size blocks of decoded audio samples from a file.
///
/// Decodes any Symphonia-supported format and splits the samples into
/// fixed-size blocks with configurable overlap.
///
/// # Parameters
/// - `path`: Audio file path (`AsRef<Path>`).
/// - `block_length`: Maximum number of blocks to return.
/// - `frame_length`: Samples per block.
/// - `hop_length`: Step size between blocks; `None` uses `frame_length` (no overlap).
///
/// # Returns
/// - `Ok(Vec<Vec<f32>>)`: Vector of sample blocks.
/// - `Err(AudioError)`: I/O or format error.
///
/// # Errors
/// - `AudioError::FileNotFound`: The specified file does not exist.
/// - `AudioError::OpenError`: Unrecognised or corrupted audio file.
/// - `AudioError::InvalidInput`: Invalid parameters (e.g., zero frame length).
/// - `AudioError::InsufficientData`: File contains no samples.
///
/// # Example
/// ```no_run
/// use dasp_rs::io::stream;
/// let blocks = stream("audio.flac", 100, 4096, None)?;
/// for block in blocks {
///     println!("Block size: {}", block.len());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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

    let (samples, _, _) = decode_audio(path)?;

    if samples.is_empty() {
        return Err(AudioError::InsufficientData("No blocks generated".into()));
    }

    let mut blocks = Vec::new();
    let mut pos = 0usize;

    while blocks.len() < block_length {
        if pos + frame_length <= samples.len() {
            blocks.push(samples[pos..pos + frame_length].to_vec());
        } else if pos < samples.len() {
            let mut block = samples[pos..].to_vec();
            block.resize(frame_length, 0.0);
            blocks.push(block);
            break;
        } else {
            break;
        }
        pos += hop;
    }

    if blocks.is_empty() {
        return Err(AudioError::InsufficientData("No blocks generated".into()));
    }

    Ok(blocks)
}

/// Streams fixed-size blocks of decoded audio samples lazily via a channel.
///
/// Decodes the file in a background thread, emitting blocks through an `mpsc` channel.
/// Each received item is `Ok(block)` or `Err(AudioError)` on mid-stream failure.
///
/// # Parameters
/// - `path`: Audio file path (`AsRef<Path>`).
/// - `block_length`: Maximum number of blocks to emit.
/// - `frame_length`: Samples per block.
/// - `hop_length`: Step size between blocks; `None` uses `frame_length`.
///
/// # Returns
/// - `Ok(Receiver<Result<Vec<f32>, AudioError>>)`: Channel receiver for blocks.
/// - `Err(AudioError)`: I/O or parameter error detected before thread spawn.
///
/// # Errors
/// - `AudioError::FileNotFound`: The specified file does not exist.
/// - `AudioError::IoError`: File cannot be opened.
/// - `AudioError::InvalidInput`: Invalid parameters (e.g., zero frame length).
///
/// Errors that occur after streaming starts are delivered through the channel.
///
/// # Example
/// ```no_run
/// use dasp_rs::io::stream_lazy;
/// let rx = stream_lazy("audio.ogg", 1000, 1024, Some(512))?;
/// for block in rx {
///     println!("Received block of {} samples", block?.len());
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn stream_lazy<P: AsRef<Path>>(
    path: P,
    block_length: usize,
    frame_length: usize,
    hop_length: Option<usize>,
) -> Result<Receiver<Result<Vec<f32>, AudioError>>, AudioError> {
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
    // Validate file accessibility before spawning the thread.
    File::open(path)?;

    let path_buf = path.to_path_buf();
    let (tx, rx) = channel();

    std::thread::spawn(move || {
        let (samples, _, _) = match decode_audio(&path_buf) {
            Ok(data) => data,
            Err(e) => {
                let _ = tx.send(Err(e));
                return;
            }
        };

        let mut block_count = 0usize;
        let mut pos = 0usize;

        while block_count < block_length {
            let block = if pos + frame_length <= samples.len() {
                samples[pos..pos + frame_length].to_vec()
            } else if pos < samples.len() {
                let mut b = samples[pos..].to_vec();
                b.resize(frame_length, 0.0);
                let _ = tx.send(Ok(b));
                return;
            } else {
                break;
            };
            if tx.send(Ok(block)).is_err() {
                return;
            }
            block_count += 1;
            pos += hop;
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
            assert!((actual - expected).abs() < 1e-6, "Expected {expected}, got {actual}");
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
        assert!((audio.duration() - 2.0 / 44100.0).abs() < f32::EPSILON);
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
        assert!(fs::metadata(path).is_ok());
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
        let loaded = load(path, None, None, Some(0.000_045_351_47), Some(0.000_045_351_48)).unwrap();
        assert_eq!(loaded.samples, vec![0.1, 0.2]);
    }

    #[test]
    fn test_export() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        assert!(fs::metadata(path).is_ok());
        let loaded = load(path, None, None, None, None).unwrap();
        assert_eq!(loaded.samples, audio.samples);
    }

    #[test]
    fn test_stream() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        let blocks = stream(path, 3, 2, Some(2)).unwrap();
        assert_eq!(blocks, vec![vec![0.0, 0.1], vec![0.2, 0.3], vec![0.4, 0.5]]);
    }

    #[test]
    fn test_stream_lazy() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        let rx = stream_lazy(path, 3, 2, Some(2)).unwrap();
        let blocks: Vec<Vec<f32>> = rx.into_iter().collect::<Result<_, _>>().unwrap();
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
        let result = load(path, None, None, Some(-1.0), None);
        assert!(matches!(result.unwrap_err(), AudioError::InvalidInput(_)));
    }

    #[test]
    fn test_load_zero_duration() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
        let result = load(path, None, None, None, Some(0.0));
        assert!(matches!(result.unwrap_err(), AudioError::InvalidInput(_)));
    }

    #[test]
    fn test_stream_zero_frame_length() {
        let audio = create_test_wav();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        export(path, &audio).unwrap();
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

    fn write_int_wav<S: hound::Sample + Copy>(path: &std::path::Path, bits: u16, samples: &[S]) {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: bits,
            sample_format: SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn test_load_16bit_int() {
        let temp = NamedTempFile::new().unwrap();
        write_int_wav(temp.path(), 16, &[0i16, 16384, -16384]);
        let loaded = load(temp.path(), None, None, None, None).unwrap();
        // Symphonia normalises signed 16-bit by dividing by 32768 (2^15).
        let expected = [0.0f32, 0.5, -0.5];
        assert_eq!(loaded.samples.len(), 3);
        for (a, e) in loaded.samples.iter().zip(expected) {
            assert!((a - e).abs() < 1e-4, "16-bit: expected {e}, got {a}");
        }
    }

    #[test]
    fn test_load_24bit_int() {
        let temp = NamedTempFile::new().unwrap();
        write_int_wav(temp.path(), 24, &[0i32, 4_194_304, -4_194_304]);
        let loaded = load(temp.path(), None, None, None, None).unwrap();
        let expected = [0.0, 0.5, -0.5];
        for (a, e) in loaded.samples.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "24-bit: expected {e}, got {a}");
        }
    }

    #[test]
    fn test_load_32bit_int() {
        let temp = NamedTempFile::new().unwrap();
        write_int_wav(temp.path(), 32, &[0i32, 1_073_741_824, -1_073_741_824]);
        let loaded = load(temp.path(), None, None, None, None).unwrap();
        let expected = [0.0, 0.5, -0.5];
        for (a, e) in loaded.samples.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "32-bit int: expected {e}, got {a}");
        }
    }

    #[test]
    fn test_load_32bit_float() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let temp = NamedTempFile::new().unwrap();
        {
            let mut writer = hound::WavWriter::create(temp.path(), spec).unwrap();
            for &s in &[0.0f32, 0.5, -0.5] {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        let loaded = load(temp.path(), None, None, None, None).unwrap();
        for (a, e) in loaded.samples.iter().zip([0.0, 0.5, -0.5]) {
            assert!((a - e).abs() < 1e-6, "32-bit float: expected {e}, got {a}");
        }
    }

    #[test]
    fn test_stream_24bit_int() {
        let temp = NamedTempFile::new().unwrap();
        write_int_wav(temp.path(), 24, &[0i32, 4_194_304, -4_194_304, 4_194_304, -4_194_304, 0]);
        let blocks = stream(temp.path(), 3, 2, Some(2)).unwrap();
        assert_eq!(blocks, vec![vec![0.0, 0.5], vec![-0.5, 0.5], vec![-0.5, 0.0]]);
    }

    #[test]
    fn test_stream_lazy_24bit_int() {
        let temp = NamedTempFile::new().unwrap();
        write_int_wav(temp.path(), 24, &[0i32, 4_194_304, -4_194_304, 4_194_304, -4_194_304, 0]);
        let rx = stream_lazy(temp.path(), 3, 2, Some(2)).unwrap();
        let blocks: Vec<Vec<f32>> = rx.into_iter().collect::<Result<_, _>>().unwrap();
        assert_eq!(blocks, vec![vec![0.0, 0.5], vec![-0.5, 0.5], vec![-0.5, 0.0]]);
    }
}
