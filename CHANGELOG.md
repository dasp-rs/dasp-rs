# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [0.3.0]

### Changed
- **Removed the OpenBLAS / `ndarray-linalg` dependency.** Linear algebra now uses
  pure-Rust `nalgebra`, so the crate builds with no system libraries (no OpenBLAS,
  no `pkg-config`) on Linux, macOS, and Windows. Fixes the build failure reported in
  [#1](https://github.com/dasp-rs/dasp-rs/issues/1).
- License is now consistently **MIT** across `Cargo.toml`, `LICENSE`, and the README
  (previously inconsistent).

### Added
- `hop_length` parameter on `pitch::yin`, `pitch::pyin`, and `pitch::estimate_tuning`,
  defaulting to `frame_length / 4`, so frame and hop length can be set independently
  and consistently with `piptrack`. Resolves
  [#3](https://github.com/dasp-rs/dasp-rs/issues/3).
- Tests covering WAV decoding for 8/16/24/32-bit PCM and 32-bit float, including
  24-bit coverage for `load`, `stream`, and `stream_lazy`.

### Fixed
- **24-bit WAV decoding.** 24-bit samples were normalized by `i32::MAX` instead of
  `2^23`, loading them roughly 48 dB too quiet. `load`, `stream`, and `stream_lazy`
  now normalize each bit depth correctly. Fixes
  [#2](https://github.com/dasp-rs/dasp-rs/issues/2).
- Several signal-processing and notation bugs surfaced by the test suite
  (mono averaging of trailing partial frames, IIR filter index underflow, off-by-one
  in `clicks`, and the Carnatic melakarta-to-svara mapping).
- Documentation examples now compile and match the public API.

### Breaking
- `pitch::yin`, `pitch::pyin`, and `pitch::estimate_tuning` take one additional
  trailing `hop_length: Option<usize>` argument.

### Acknowledgements
- Thanks to [@Levitanus](https://github.com/Levitanus) for reporting issues
  [#1](https://github.com/dasp-rs/dasp-rs/issues/1),
  [#2](https://github.com/dasp-rs/dasp-rs/issues/2), and
  [#3](https://github.com/dasp-rs/dasp-rs/issues/3), and for the diagnosis
  in [#4](https://github.com/dasp-rs/dasp-rs/pull/4).

## [0.2.0]

Initial documented release on crates.io.
