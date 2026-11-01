use crate::signal_processing::{resample, to_mono};
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::ShapeError;
use std::collections::VecDeque;
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder as CodecDecoder, DecoderOptions};
use symphonia::core::formats::{FormatOptions, FormatReader};
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
    let mut packets = AudioPackets::open(path)?;
    let mut samples = Vec::new();
    for chunk in &mut packets {
        samples.extend(chunk?);
    }
    Ok((samples, packets.sample_rate, packets.channels))
}

// Both batch and streaming entry points consume the same fallible packet decoder.
struct AudioPackets {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn CodecDecoder>,
    track_id: u32,
    sample_rate: u32,
    channels: u16,
    expected_frames: Option<u64>,
    decoded_frames: u64,
    finished: bool,
}

impl AudioPackets {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self, AudioError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| AudioError::OpenError(e.to_string()))?;

        let format = probed.format;

        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or(AudioError::UnsupportedFormat)?;

        let track_id = track.id;
        let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
        let channels = track.codec_params.channels.map_or(1, |c| c.count() as u16);
        // PCM-like streams (one frame per coded block) can declare an exact
        // sample-frame length. Do not compare against estimated compressed-stream
        // durations or timebases whose ticks are not sample frames.
        let expected_frames = if track.codec_params.frames_per_block == Some(1)
            && track
                .codec_params
                .time_base
                .is_some_and(|base| base.numer == 1 && base.denom == sample_rate)
        {
            track.codec_params.n_frames
        } else {
            None
        };

        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| AudioError::OpenError(e.to_string()))?;

        Ok(Self {
            format,
            decoder,
            track_id,
            sample_rate,
            channels,
            expected_frames,
            decoded_frames: 0,
            finished: false,
        })
    }
}

// Symphonia signals normal end-of-stream with UnexpectedEof. Other packet
// errors must not be mistaken for successful completion of a shorter file.
fn packet_or_eof<T>(
    packet: Result<T, symphonia::core::errors::Error>,
) -> Result<Option<T>, AudioError> {
    match packet {
        Ok(packet) => Ok(Some(packet)),
        Err(symphonia::core::errors::Error::IoError(e))
            if e.kind() == std::io::ErrorKind::UnexpectedEof =>
        {
            Ok(None)
        }
        Err(e) => Err(AudioError::OpenError(format!("Packet read failed: {e}"))),
    }
}

impl Iterator for AudioPackets {
    type Item = Result<Vec<f32>, AudioError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            let packet = match packet_or_eof(self.format.next_packet()) {
                Ok(Some(packet)) => packet,
                Ok(None) => {
                    self.finished = true;
                    if let Some(expected) = self.expected_frames {
                        if self.decoded_frames < expected {
                            return Some(Err(AudioError::IoError(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                format!(
                                    "Truncated audio: decoded {} of {expected} declared frames",
                                    self.decoded_frames
                                ),
                            ))));
                        }
                    }
                    return None;
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            };
            if packet.track_id() != self.track_id {
                continue;
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(AudioError::OpenError(format!("Decode failed: {e}"))));
                }
            };
            self.decoded_frames = self.decoded_frames.saturating_add(decoded.frames() as u64);
            let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
            buffer.copy_interleaved_ref(decoded);
            return Some(Ok(buffer.samples().to_vec()));
        }
    }
}

// Retain only the current analysis window plus at most one decoded packet.
// `skip` is applied before filling the next window, including when hop > frame.
struct SampleBlocks<I> {
    packets: I,
    buffer: VecDeque<f32>,
    frame_length: usize,
    hop: usize,
    remaining: usize,
    skip: usize,
    finished: bool,
}

impl<I> SampleBlocks<I> {
    fn new(packets: I, limit: usize, frame_length: usize, hop: usize) -> Self {
        Self {
            packets,
            buffer: VecDeque::new(),
            frame_length,
            hop,
            remaining: limit,
            skip: 0,
            finished: false,
        }
    }
}

impl<I: Iterator<Item = Result<Vec<f32>, AudioError>>> Iterator for SampleBlocks<I> {
    type Item = Result<Vec<f32>, AudioError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.remaining == 0 {
            return None;
        }
        let discard = self.skip.min(self.buffer.len());
        self.buffer.drain(..discard);
        self.skip -= discard;
        while self.buffer.len() < self.frame_length {
            match self.packets.next() {
                Some(Ok(packet)) => {
                    let discard = self.skip.min(packet.len());
                    self.skip -= discard;
                    self.buffer.extend(packet.into_iter().skip(discard));
                }
                Some(Err(e)) => {
                    self.finished = true;
                    return Some(Err(e));
                }
                None => {
                    self.finished = true;
                    break;
                }
            }
        }
        if self.buffer.is_empty() {
            return None;
        }
        let mut block: Vec<f32> = self
            .buffer
            .iter()
            .take(self.frame_length)
            .copied()
            .collect();
        block.resize(self.frame_length, 0.0);
        self.skip = self.hop;
        self.remaining -= 1;
        Some(Ok(block))
    }
}

fn send_blocks(
    blocks: impl Iterator<Item = Result<Vec<f32>, AudioError>>,
    tx: &SyncSender<Result<Vec<f32>, AudioError>>,
) {
    for block in blocks {
        if tx.send(block).is_err() {
            break;
        }
    }
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
        if !off.is_finite() || off < 0.0 {
            return Err(AudioError::InvalidInput(
                "Offset must be finite and non-negative".into(),
            ));
        }
    }
    if let Some(dur) = duration {
        if !dur.is_finite() || dur <= 0.0 {
            return Err(AudioError::InvalidInput(
                "Duration must be finite and positive".into(),
            ));
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

    let channel_count = usize::from(channels);
    let total_frames = raw_samples.len() / channel_count;
    let start_frame =
        ((f64::from(offset.unwrap_or(0.0)) * f64::from(sample_rate)) as usize).min(total_frames);
    let end_frame = duration.map_or(total_frames, |d| {
        start_frame
            .saturating_add((f64::from(d) * f64::from(sample_rate)) as usize)
            .min(total_frames)
    });
    let mut samples = raw_samples[start_frame * channel_count..end_frame * channel_count].to_vec();

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
        } else if channels == 1 || mono.unwrap_or(false) {
            resample(&samples, sample_rate, target_sr)?
        } else {
            // A filter must never treat adjacent channels as adjacent times.
            let resampled: Vec<Vec<f32>> = (0..channel_count)
                .map(|channel| {
                    let input: Vec<f32> = samples
                        .iter()
                        .skip(channel)
                        .step_by(channel_count)
                        .copied()
                        .collect();
                    resample(&input, sample_rate, target_sr)
                })
                .collect::<Result<_, _>>()?;
            let mut interleaved = Vec::with_capacity(resampled[0].len() * channel_count);
            for frame in 0..resampled[0].len() {
                for channel in &resampled {
                    interleaved.push(channel[frame]);
                }
            }
            interleaved
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
/// operation. Float samples are preserved without clipping, including values
/// outside `[-1.0, 1.0]`. Apply explicit limiting before export if desired.
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
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    std::fs::write(path, buffer)?;
    Ok(())
}

/// Collects fixed-size blocks of decoded audio samples from a file.
///
/// Decodes any Symphonia-supported format and splits the samples into
/// fixed-size blocks with configurable overlap. Decoding stops after the requested
/// blocks; the returned collection still uses memory proportional to their size.
/// Block and hop lengths count interleaved scalar samples; use multiples of the
/// channel count when blocks must align with complete channel frames.
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

    let packets = AudioPackets::open(path)?;
    let blocks = SampleBlocks::new(packets, block_length, frame_length, hop)
        .collect::<Result<Vec<_>, _>>()?;
    if blocks.is_empty() {
        return Err(AudioError::InsufficientData("No blocks generated".into()));
    }
    Ok(blocks)
}

/// Streams fixed-size blocks of decoded audio samples lazily via a channel.
///
/// Decodes the file in a background thread, emitting blocks through an `mpsc` channel.
/// Each received item is `Ok(block)` or `Err(AudioError)` on mid-stream failure.
/// A bounded two-block queue applies backpressure. Memory depends on block and
/// codec packet sizes, not file duration. Dropping the receiver stops the worker
/// at its next send; an in-progress packet read/decode is not interrupted.
/// Block and hop lengths count interleaved scalar samples, as in [`stream`].
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
    let (tx, rx) = sync_channel(2);

    std::thread::spawn(move || {
        if block_length == 0 {
            return;
        }
        let packets = match AudioPackets::open(&path_buf) {
            Ok(packets) => packets,
            Err(e) => {
                let _ = tx.send(Err(e));
                return;
            }
        };
        send_blocks(
            SampleBlocks::new(packets, block_length, frame_length, hop),
            &tx,
        );
    });

    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn incremental_blocks_match_reference_across_packet_boundaries() {
        for len in [0, 1, 3, 4, 7, 15] {
            let samples: Vec<f32> = (0..len).map(|i| i as f32).collect();
            for frame in [1, 2, 4, 5] {
                for hop in [1, 2, 4, 7, usize::MAX] {
                    for limit in [0, 1, 10] {
                        let packets = samples.chunks(3).map(|chunk| Ok(chunk.to_vec()));
                        let actual = SampleBlocks::new(packets, limit, frame, hop)
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap();
                        let mut expected = Vec::new();
                        let mut pos = 0usize;
                        while pos < samples.len() && expected.len() < limit {
                            let end = (pos + frame).min(samples.len());
                            let mut block = samples[pos..end].to_vec();
                            block.resize(frame, 0.0);
                            expected.push(block);
                            if end - pos < frame {
                                break;
                            }
                            pos = pos.saturating_add(hop);
                        }
                        assert_eq!(
                            actual, expected,
                            "len={len}, frame={frame}, hop={hop}, limit={limit}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn block_limit_stops_packet_consumption() {
        let mut reads = 0;
        let packets = std::iter::from_fn(|| {
            reads += 1;
            assert_eq!(reads, 1, "read past requested block limit");
            Some(Ok(vec![1.0, 2.0, 3.0]))
        });
        let blocks = SampleBlocks::new(packets, 1, 2, 2)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(blocks, [vec![1.0, 2.0]]);
        assert_eq!(reads, 1);
    }

    #[test]
    fn packet_errors_are_not_successful_eof() {
        use symphonia::core::errors::Error;
        assert_eq!(packet_or_eof(Ok(42)).unwrap(), Some(42));
        assert!(
            packet_or_eof::<()>(Err(Error::IoError(std::io::Error::from(
                std::io::ErrorKind::UnexpectedEof
            ))))
            .unwrap()
            .is_none()
        );
        for error in [
            Error::IoError(std::io::Error::other("read failure")),
            Error::DecodeError("bad packet"),
            Error::ResetRequired,
        ] {
            assert!(packet_or_eof::<()>(Err(error)).is_err());
        }
    }

    #[test]
    fn late_packet_error_reaches_receiver_and_stops_iteration() {
        let packets = [
            Ok(vec![1.0, 2.0, 3.0]),
            Err(AudioError::StreamError),
            Ok(vec![4.0, 5.0]),
        ];
        let blocks = SampleBlocks::new(packets.into_iter(), 10, 2, 2);
        let (tx, rx) = sync_channel(2);
        send_blocks(blocks, &tx);
        drop(tx);
        assert_eq!(rx.recv().unwrap().unwrap(), [1.0, 2.0]);
        assert!(matches!(rx.recv().unwrap(), Err(AudioError::StreamError)));
        assert!(rx.recv().is_err());
    }

    #[test]
    fn bounded_sender_applies_backpressure_and_stops_on_receiver_drop() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        let (tx, rx) = sync_channel(2);
        let (progress_tx, progress_rx) = channel();
        let worker = std::thread::spawn(move || {
            let blocks = (0..100).map(|i| {
                progress_tx.send(i).unwrap();
                Ok(vec![i as f32])
            });
            send_blocks(blocks, &tx);
        });
        // Two queued blocks and one pending send; a fourth block cannot be produced.
        for expected in 0..3 {
            assert_eq!(
                progress_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
                expected
            );
        }
        assert!(progress_rx.try_recv().is_err());
        drop(rx);
        worker.join().unwrap();
        assert!(progress_rx.try_recv().is_err());
    }

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
            assert!(
                (actual - expected).abs() < 1e-6,
                "Expected {expected}, got {actual}"
            );
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
        let loaded = load(
            path,
            None,
            None,
            Some(0.000_045_351_47),
            Some(0.000_045_351_48),
        )
        .unwrap();
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
        write_int_wav(
            temp.path(),
            24,
            &[0i32, 4_194_304, -4_194_304, 4_194_304, -4_194_304, 0],
        );
        let blocks = stream(temp.path(), 3, 2, Some(2)).unwrap();
        assert_eq!(
            blocks,
            vec![vec![0.0, 0.5], vec![-0.5, 0.5], vec![-0.5, 0.0]]
        );
    }

    #[test]
    fn test_stream_lazy_24bit_int() {
        let temp = NamedTempFile::new().unwrap();
        write_int_wav(
            temp.path(),
            24,
            &[0i32, 4_194_304, -4_194_304, 4_194_304, -4_194_304, 0],
        );
        let rx = stream_lazy(temp.path(), 3, 2, Some(2)).unwrap();
        let blocks: Vec<Vec<f32>> = rx.into_iter().collect::<Result<_, _>>().unwrap();
        assert_eq!(
            blocks,
            vec![vec![0.0, 0.5], vec![-0.5, 0.5], vec![-0.5, 0.0]]
        );
    }
}
