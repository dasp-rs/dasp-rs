# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [0.4.0]

### Changed
- **Idiomatic-Rust overhaul.** The crate now compiles warning-free under
  `clippy::pedantic` + `clippy::nursery` (curated allows are documented in
  `Cargo.toml` `[lints]`), denies `unsafe_code`, and warns on `missing_docs`.
- **Panic-free library paths.** All `partial_cmp().unwrap()` float comparisons were
  replaced with `f32::total_cmp`, the `unreachable!()` in `multi_channel_pan` is now
  an error return, and framing helpers (`temporal_kurtosis`, `zero_crossing_rate`,
  reassignment STFT) validate signal/frame lengths instead of panicking on underflow.
- `stream_lazy` no longer swallows decode errors by sending an empty block: the
  receiver now yields `Result<Vec<f32>, AudioError>` items, closing after the first
  error.
- `Decoder::new` replaces `Decoder::from` (deprecated alias retained), following the
  convention that `from` is reserved for `From` conversions.
- Internal `*_impl` helpers (`chroma_stft_impl`, `tempo_impl`, `fft_frequencies_impl`,
  `mel_frequencies_impl`, `tempo_frequencies_impl`) are no longer exported; use the
  corresponding builders.
- Builder methods carry `#[must_use]`; fallible public functions document their
  errors under `# Errors`.
- Repaired double-encoded UTF-8 throughout the source: Unicode note names (`C♯`),
  IAST svara transliterations (`ṣaḍjam`, …), and degree symbols now render correctly
  in docs and runtime output (previously mojibake like `Câ™¯`).
- Dependency hygiene: `thiserror` upgraded to 2.x, `approx` moved to
  `[dev-dependencies]` (it was only used in tests), redundant internal re-exports
  removed, and the non-functional `.cargoignore` file deleted.

- **STFT reworked to standard conventions.** `stft` uses a periodic **Hann** window
  by default (was Hamming), centers the signal via reflect padding (`center = true`),
  and centers a `win_length` window inside `n_fft`. Output is `(n_fft/2 + 1, n_frames)`
  with `n_frames = 1 + len / hop_length`. This corrects every downstream spectral
  feature, which all build on the STFT.
- `istft` reimplemented as a consistent inverse (same window, COLA overlap-add with
  window-squared normalization, center trimming).

### Added
- **Builder APIs everywhere.** Every function that
  previously took multiple optional/`bool` arguments is now a builder with named,
  defaulted setters and a terminal `.compute()`:
  - Transforms: `stft`/`istft` (`.window(Window)`, `.center(bool)`), `cqt`, `icqt`,
    `vqt`, `pseudo_cqt`, `hybrid_cqt`, `iirt`, `fmt`, `reassigned_spectrogram`.
  - Spectral features: `spectral(&y, sr)` with terminal methods for the whole family
    (`.mfcc()`, `.melspectrogram()`, `.chroma_cqt()`, `.chroma_cens()`, `.hpss()`,
    `.spectral_contrast()`, `.spectral_rolloff()`, `.tonnetz()`, `.formant_frequencies()`,
    `.rms()`, `.vad_features()`, …) plus `cmvn(&feats)`.
  - Pitch: `yin`, `pyin`, `estimate_tuning`, `piptrack`.
  - Magnitude: `amplitude_to_db`, `power_to_db`, `pcen`.
  - Inverse: `compute_delta`, `mel_to_stft`, `mel_to_audio`, `mfcc_to_mel`,
    `mfcc_to_audio`.
  - Time domain: `delay`, `zero_crossings`, `mu_compress`, `mu_expand`, `log_energy`.
  - Generation: `clicks` (`.times()`/`.frames()`/…).
  - `proc::normalize(&signal, target)` builder with a `NormalizeMode` enum
    (`Peak`/`Rms`) replacing the stringly-typed `mode: &str`.
  - `util` 2+ option helpers are builders: `fft_frequencies`, `mel_frequencies`,
    `tempo_frequencies`, `frames_to_time`, `time_to_frames`, `blocks_to_time`,
    `times_like`, `hz_to_fjs`, `midi_to_svara_h`, `midi_to_svara_c`, `mela_to_svara`.
  - Dead/unused `Option` parameters removed from the remaining `util` helpers; the
    rest take at most one `Option`. No public function exposes option/boolean soup.
  - Single low-level positional functions remain as escape hatches where one already
    has a builder (`load` behind `Decoder`); trivial numeric conversion helpers in
    `util` keep their one or two `Option` arguments.
  - Public `Window` enum (`Hann`, `Hamming`).
- A correctness test suite (`tests/correctness.rs`) asserting DSP invariants
  (centered frame count, COLA reconstruction, sinusoid peak bin, window selection).

### Fixed
- `estimate_tuning` was unusable with a real signal: it passed both the signal and a
  computed spectrogram to `piptrack`, which rejects that combination. It now works.

### Breaking
- `stft` default window changed Hamming → Hann and is now centered by default;
  numerical output and frame counts differ from 0.3.x.
- Many functions are now builders (call sites must use the chained `.compute()` form):
  `istft`, `icqt`, `vqt`, `pseudo_cqt`, `hybrid_cqt`, `iirt`, `fmt`, `yin`, `pyin`,
  `estimate_tuning`, `piptrack`, `clicks`, `amplitude_to_db`, `power_to_db`, `pcen`,
  `cmvn`, `compute_delta`, `mel_to_stft`, `mel_to_audio`, `mfcc_to_mel`,
  `mfcc_to_audio`, `delay`, `zero_crossings`, `mu_compress`, `mu_expand`, `log_energy`.
- `tempogram_impl` / `tempogram_ratio_impl` renamed to `tempogram` / `tempogram_ratio`.
- The positional spectral feature functions (`mfcc`, `melspectrogram`, `chroma_cqt`,
  `hpss`, `formant_frequencies`, `spectral_*`, `rms`, …) are no longer part of the
  public API; use the `spectral(&y, sr)` builder and its terminal methods instead.
- `stream_lazy` returns `Receiver<Result<Vec<f32>, AudioError>>` (was
  `Receiver<Vec<f32>>`).
- The `*_impl` positional helpers are no longer exported from `feat`/`util`.
- `Decoder::from` is deprecated in favor of `Decoder::new`.

## [0.3.1]

### Changed
- Crate metadata updates only.

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
