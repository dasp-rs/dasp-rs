# dasp-rs

[![Crates.io](https://img.shields.io/crates/v/dasp-rs.svg)](https://crates.io/crates/dasp-rs)
[![Documentation](https://docs.rs/dasp-rs/badge.svg)](https://docs.rs/dasp-rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`dasp-rs` is a pure-Rust library for digital audio signal processing, analysis, and
synthesis. It targets the same ground as Python's `librosa` — STFT/CQT transforms,
spectral and MIR features, pitch tracking, and music/phonetics notation — but as a
fast, dependency-light Rust crate that builds with no system libraries.

It is aimed at developers, audio/ML researchers, phoneticians, and music-information-
retrieval work.

## Highlights

- **Pure Rust, no system dependencies.** No OpenBLAS, no `pkg-config`, no C toolchain that previously supported.
- **WAV I/O** with 8/16/24/32-bit PCM and 32-bit float, including segment loading,
  on-load resampling/mono conversion, and streaming readers for large files.
- **Time–frequency transforms**: STFT/iSTFT, CQT/iCQT, VQT, pseudo/hybrid CQT, FMT,
  IIRT, reassigned spectrograms, and Griffin–Lim phase reconstruction.
- **Spectral & MIR features**: mel spectrogram, MFCC, chroma (STFT/CQT/CENS), spectral
  centroid/bandwidth/contrast/flatness/rolloff/flux/entropy, tonnetz, RMS, HPSS, and more.
- **Pitch & tuning**: YIN, pYIN, `piptrack`, tuning estimation.
- **Music & phonetics notation**: Hz/MIDI/note conversion, mel scale, Western keys,
  Carnatic melakarta and Hindustani thaat svaras, Functional Just System, Pythagorean
  and p-limit intervals.
- **Parallelized** with `rayon` where it helps.

## Installation

Add it with Cargo:

```toml
[dependencies]
dasp-rs = "0.3.0"
```

No additional system packages are required.

## Quick start

```rust
use dasp_rs::io::load;
use dasp_rs::proc::stft;
use dasp_rs::feat::mfcc;
use dasp_rs::types::AudioData;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a WAV file as mono at 22.05 kHz
    let audio: AudioData = load("input.wav", Some(22_050), Some(true), None, None)?;

    // Short-Time Fourier Transform (builder API)
    let spec = stft(&audio.samples).n_fft(2048).hop_length(512).compute()?;
    println!("STFT: {} bins x {} frames", spec.nrows(), spec.ncols());

    // MFCCs
    let cc = mfcc(&audio, None, Some(20), None, None)?;
    println!("MFCC: {:?}", cc.shape());

    Ok(())
}
```

## API overview

The public API is organized by concern, re-exported from the crate root:

| Module | Contents |
|--------|----------|
| `types` | `AudioData`, `AudioError` |
| `io` | `load`, `export`, `stream`, `stream_lazy`, `Decoder` (builder) |
| `ops` | sample-wise `mix_signals`, `subtract_signals`, `multiply_signals`, `divide_signals`, `scalar_operation` |
| `proc` | mono/amplitude/mixing/panning/resampling; time-domain (`delay`, `lpc`, `autocorrelate`, `zero_crossings`, μ-law, …); time–frequency (`stft`/`istft`, `cqt`/`icqt`, `vqt`, `pseudo_cqt`, `hybrid_cqt`, `fmt`, `iirt`, `reassigned_spectrogram`, `magphase`) |
| `feat` | spectral features (`melspectrogram`, `mfcc`, `chroma_*`, `spectral_*`, `tonnetz`, `hpss`, `rms`, `formant_frequencies`, …), harmonics, rhythm (`tempo`, `tempogram`), `griffinlim`, inverse transforms (`mel_to_audio`, `mfcc_to_audio`, …) |
| `pitch` | `yin`, `pyin`, `piptrack`, `estimate_tuning`, `pitch_tuning` |
| `mag` | dB/amplitude/power scaling, A/B/C/D weighting, perceptual weighting, `pcen` |
| `generate` | `tone`, `chirp`, `clicks` |
| `util` | time/frame/sample conversions, frequency & note conversions, music/phonetics notation |
| `prelude` | the most common imports |

Full item-level documentation is on [docs.rs](https://docs.rs/dasp-rs).

## Audio format support

`dasp-rs` reads and writes WAV via [`hound`]. Supported sample formats:

| Format | Read | Write |
|--------|:----:|:-----:|
| 8-bit PCM  | ✅ | – |
| 16-bit PCM | ✅ | – |
| 24-bit PCM | ✅ | – |
| 32-bit PCM | ✅ | – |
| 32-bit float | ✅ | ✅ |

Internally all audio is `f32`. `export` writes 32-bit float WAV.

## Performance

- `rayon`-based parallelism for large workloads.
- Streaming readers (`stream`, `stream_lazy`) keep memory bounded for large files.
- FFTs via [`rustfft`]; resampling via [`rubato`]; linear algebra via pure-Rust
  [`nalgebra`].

## Contributing

Issues and pull requests are welcome on [GitHub]. Please run `cargo test` and
`cargo clippy` before submitting.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

Thanks to [@Levitanus](https://github.com/Levitanus) for reporting the OpenBLAS build
failure and 24-bit decoding issues and contributing the diagnosis in
[#4](https://github.com/amirhosseinghanipour/dasp-rs/pull/4).

[`hound`]: https://crates.io/crates/hound
[`rustfft`]: https://crates.io/crates/rustfft
[`rubato`]: https://crates.io/crates/rubato
[`nalgebra`]: https://crates.io/crates/nalgebra
