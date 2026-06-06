# dasp-rs

[![Crates.io](https://img.shields.io/crates/v/dasp-rs.svg)](https://crates.io/crates/dasp-rs)
[![Documentation](https://docs.rs/dasp-rs/badge.svg)](https://docs.rs/dasp-rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`dasp-rs` is a pure-Rust library for digital audio signal processing, analysis, and
synthesis. It covers the same ground as Python's `librosa` — STFT/CQT transforms,
spectral and MIR features, pitch tracking, and music/phonetics notation — as a fast,
dependency-light Rust crate that builds with no system libraries.

It is aimed at developers, audio/ML researchers, phoneticians, and music-information-
retrieval work.

## Highlights

- **Pure Rust, no system dependencies.** No OpenBLAS, `pkg-config`, or C toolchain
  required (unlike earlier versions). Builds out of the box on Linux, macOS, and Windows.
- **Ergonomic builder APIs** for the common entry points (`Decoder`, `stft`, `cqt`,
  `tempo`, `griffinlim`, …), with positional functions underneath for full control.
- **WAV I/O** for 8/16/24/32-bit PCM and 32-bit float, with segment loading, on-load
  resampling / mono conversion, and streaming readers for large files.
- **Broad feature set**: time–frequency transforms, spectral/MIR features, pitch &
  tuning, magnitude scaling & loudness weighting, signal generation, and extensive
  music/phonetics notation utilities.
- **Parallelized** with `rayon` where it helps.

## Installation

```toml
[dependencies]
dasp-rs = "0.3.1"
```

No additional system packages are required.

## Quick start

```rust
use dasp_rs::io::Decoder;
use dasp_rs::proc::stft;
use dasp_rs::feat::mfcc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a WAV file as mono at 22.05 kHz (builder API)
    let audio = Decoder::from("input.wav")
        .sample_rate(22_050)
        .mono()
        .load()?;

    // Short-Time Fourier Transform (builder API)
    let spec = stft(&audio.samples).n_fft(2048).hop_length(512).compute()?;
    println!("STFT: {} bins x {} frames", spec.nrows(), spec.ncols());

    // MFCCs
    let cc = mfcc(&audio, None, Some(20), None, None)?;
    println!("MFCC: {:?}", cc.shape());

    Ok(())
}
```

The crate root re-exports the API by concern: `types`, `io`, `ops`, `proc`, `feat`,
`pitch`, `mag`, `generate`, `util`, and a `prelude`.

## Features

### Audio I/O — `io` / `types`
- `AudioData` — `f32` sample container (`samples`, `sample_rate`,
  `channels`) with `to_mono`, `split_channels`, `duration`, `frame_count`, `to_raw`.
- `Decoder` — builder for loading: `.sample_rate()`, `.mono()`, `.offset()`,
  `.duration()`, `.load()`.
- `load` — load a WAV (with optional resample, mono, offset, duration).
- `export` — write a 32-bit float WAV.
- `stream` / `stream_lazy` — block/streaming readers for large files.
- Reads 8/16/24/32-bit PCM and 32-bit float WAV.

### Sample-wise operations — `ops`
- `mix_signals`, `subtract_signals`, `multiply_signals`, `divide_signals`,
  `scalar_operation`.

### Signal processing — `proc`
- **Channels / amplitude**: `to_mono`, `amplify`, `attenuate`, `normalize`.
- **Mixing**: `stereo_mix`, `multi_channel_mix`, `dry_wet_mix`.
- **Panning**: `stereo_pan`, `multi_channel_pan`.
- **Resampling**: `resample` (sinc-based, via `rubato`).
- **Time domain**: `delay`, `time_reversal`, `time_crop`, `zero_padding`,
  `autocorrelate`, `lpc`, `zero_crossings`, `mu_compress`, `mu_expand`, `log_energy`.
- **Time–frequency**: `stft` / `istft`, `cqt` / `icqt`, `vqt`, `pseudo_cqt`,
  `hybrid_cqt`, `fmt`, `iirt`, `reassigned_spectrogram`, `magphase`.

### Feature extraction — `feat`
- **Spectral / MIR**: `melspectrogram`, `mfcc`, `rms`, `chroma_stft`, `chroma_cqt`,
  `chroma_cens`, `spectral_centroid`, `spectral_bandwidth`, `spectral_contrast`,
  `spectral_flatness`, `spectral_rolloff`, `spectral_flux`, `spectral_entropy`,
  `poly_features`, `tonnetz`, `pitch_chroma`, `cmvn`, `hpss`, `pitch_autocorr`,
  `vad_features`, `spectral_subband_centroids`, `formant_frequencies`.
- **Harmonics**: `interp_harmonics`, `salience`, `f0_harmonics`, `phase_vocoder`.
- **Rhythm**: `tempo`, `tempogram`, ratio tempogram.
- **Manipulation**: `stack_memory`, `temporal_kurtosis`, `zero_crossing_rate`.
- **Phase reconstruction**: `griffinlim`.
- **Inverse transforms**: `compute_delta`, `mel_to_stft`, `mel_to_audio`,
  `mfcc_to_mel`, `mfcc_to_audio`.

### Pitch & tuning — `pitch`
- `yin`, `pyin` — fundamental-frequency estimation (with independent
  `frame_length` / `hop_length`).
- `piptrack` — spectral peak pitch tracking.
- `estimate_tuning`, `pitch_tuning` — tuning-deviation estimation.

### Magnitude & loudness — `mag`
- `amplitude_to_db`, `db_to_amplitude`, `power_to_db`, `db_to_power`.
- `perceptual_weighting`, `frequency_weighting`, `multi_frequency_weighting`.
- `a_weighting`, `b_weighting`, `c_weighting`, `d_weighting`.
- `pcen` — per-channel energy normalization.

### Signal generation — `generate`
- `tone` (sine), `chirp` (linear sweep), `clicks`.

### Utilities — `util`
- **Time / framing**: `get_duration`, `get_duration_from_path`, `get_samplerate`,
  `frames_to_samples`, `frames_to_time`, `samples_to_frames`, `samples_to_time`,
  `time_to_frames`, `time_to_samples`, `blocks_to_frames`, `blocks_to_samples`,
  `blocks_to_time`, `samples_like`, `times_like`.
- **Frequency / scales**: `hz_to_midi`, `midi_to_hz`, `hz_to_note`, `note_to_hz`,
  `note_to_midi`, `midi_to_note`, `hz_to_mel`, `mel_to_hz`, `hz_to_octs`, `octs_to_hz`,
  `a4_to_tuning`, `tuning_to_a4`, `fft_frequencies`, `cqt_frequencies`,
  `mel_frequencies`, `tempo_frequencies`, `fourier_tempo_frequencies`.
- **Notation**:
  - Western: `key_to_notes`, `key_to_degrees`, `fifths_to_note`.
  - Carnatic: `mela_to_svara`, `mela_to_degrees`, `list_mela`, `hz_to_svara_c`,
    `midi_to_svara_c`, `note_to_svara_c`.
  - Hindustani: `thaat_to_degrees`, `list_thaat`, `hz_to_svara_h`, `midi_to_svara_h`,
    `note_to_svara_h`.
  - Just intonation: `interval_to_fjs`, `hz_to_fjs`, `interval_frequencies`,
    `pythagorean_intervals`, `plimit_intervals`.

Full item-level documentation, with signatures and examples, is on
[docs.rs](https://docs.rs/dasp-rs).

## Audio format support

`dasp-rs` reads and writes WAV via [`hound`]. Internally all audio is `f32`.

| Format | Read | Write |
|--------|:----:|:-----:|
| 8-bit PCM  | ✅ | – |
| 16-bit PCM | ✅ | – |
| 24-bit PCM | ✅ | – |
| 32-bit PCM | ✅ | – |
| 32-bit float | ✅ | ✅ |

`export` writes 32-bit float WAV (lossless for the internal representation).

## Performance

- `rayon`-based parallelism for large workloads.
- Streaming readers (`stream`, `stream_lazy`) keep memory bounded for large files.
- FFTs via [`rustfft`]; resampling via [`rubato`]; linear algebra via pure-Rust
  [`nalgebra`].

## Contributing

Issues and pull requests are welcome on [GitHub]. Please run `cargo test` before
submitting.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

Thanks to [@Levitanus](https://github.com/Levitanus) for reporting the OpenBLAS build
failure and 24-bit decoding issues, and for the diagnosis in
[#4](https://github.com/dasp-rs/dasp-rs/pull/4).

[`hound`]: https://crates.io/crates/hound
[`rustfft`]: https://crates.io/crates/rustfft
[`rubato`]: https://crates.io/crates/rubato
[`nalgebra`]: https://crates.io/crates/nalgebra
[GitHub]: https://github.com/dasp-rs/dasp-rs
