# dasp-rs

[![Crates.io](https://img.shields.io/crates/v/dasp-rs.svg)](https://crates.io/crates/dasp-rs)
[![Documentation](https://docs.rs/dasp-rs/badge.svg)](https://docs.rs/dasp-rs)
[![CI](https://github.com/dasp-rs/dasp-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dasp-rs/dasp-rs/actions/workflows/ci.yml)
[![MSRV: 1.85](https://img.shields.io/badge/rustc-1.85%2B-blue.svg)](Cargo.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22871121.svg)](https://doi.org/10.5281/zenodo.22871121)

**Audio analysis, processing, and synthesis in Rust.** `dasp-rs` provides audio
file I/O, time–frequency transforms, spectral features, pitch tracking, rhythm
analysis, and music-notation utilities for speech, music, and general DSP workflows.
Its decoding, FFT, resampling, and linear-algebra backends are implemented in
Rust; building the crate requires no external BLAS or audio libraries.

Use it to turn recordings into feature matrices, experiment with signal
transformations, or assemble analysis pipelines. Common operations expose builder
APIs, while numerical conventions and independent-reference coverage are documented
below. The library allocates during processing; suitability for a real-time audio
callback must be evaluated for your application.

[Quick start](#quick-start) · [API guide](#api-guide) · [Examples](#examples) ·
[Numerical conventions](#numerical-conventions) · [Migration](#migrating-from-05) ·
[Validation](#validation) · [Performance](#performance) · [Contributing](#contributing)

## Installation

```toml
[dependencies]
dasp-rs = "0.6"
```

Or run `cargo add dasp-rs@0.6`. Import the crate as `dasp_rs` in Rust code.
The minimum supported Rust version is **1.85**; the crate uses Rust edition 2024.
CI checks Linux, macOS, and Windows. No Python environment is needed to use the
library or run its Rust tests; Python is used by the optional validation and
performance-reporting tools.

See the [API documentation](https://docs.rs/dasp-rs),
[changelog](CHANGELOG.md), and [GitHub releases](https://github.com/dasp-rs/dasp-rs/releases).

## Quick start

This complete example generates its own input, extracts features, and reconstructs
the waveform. Save it as `src/main.rs` in a project with the dependency above and
run `cargo run`.

```rust
use dasp_rs::{feat, generate, proc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 16_000;
    let y = generate::tone(440.0, sr).duration(1.0).compute();

    let spectrum = proc::stft(&y)
        .n_fft(1024)
        .hop_length(256)
        .compute()?;

    let mfcc = feat::spectral(&y, sr)
        .n_fft(1024)
        .hop_length(256)
        .n_mels(40)
        .mfcc()?;

    let reconstructed = proc::istft(&spectrum)
        .n_fft(1024)
        .hop_length(256)
        .length(y.len())
        .compute()?;

    println!("STFT: {:?} (frequency bins, frames)", spectrum.dim());
    println!("MFCC: {:?} (coefficients, frames)", mfcc.dim());
    println!("Reconstructed {} samples", reconstructed.len());
    Ok(())
}
```

### Analyze an audio file

Load a supported recording, explicitly convert it to mono, and estimate pitch.
This example expects an `input.flac` file in the working directory.

```rust
use dasp_rs::{io::Decoder, pitch};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio = Decoder::new("input.flac")
        .sample_rate(16_000)
        .mono()
        .load()?;

    let tracked = pitch::pyin(&audio.samples, 65.0, 1_200.0)
        .sample_rate(audio.sample_rate)
        .frame_length(1024)
        .hop_length(256)
        .compute_with_voicing()?;

    let voiced = tracked.voiced_flag.iter().filter(|&&v| v).count();
    println!("{voiced} voiced frames out of {}", tracked.f0.len());
    Ok(())
}
```

`tracked.f0` contains frequency estimates in Hz, with zero for unvoiced frames.
`voiced_prob` contains frame-local model scores, not calibrated confidence in the
final pitch. Supply at least one complete analysis frame and choose a frame long
enough for two periods at the requested minimum frequency.

## API guide

The public API is grouped by concern. Transforms and feature matrices use
`ndarray`; complex spectra use `num_complex::Complex<f32>`. Import modules
explicitly, as above, or use `dasp_rs::prelude::*` for common entry points.

| Public module | What it provides | Representative entry points |
| --- | --- | --- |
| [`types`](https://docs.rs/dasp-rs/latest/dasp_rs/types/index.html) | Audio containers and shared errors | `AudioData`, `AudioError` |
| [`io`](https://docs.rs/dasp-rs/latest/dasp_rs/io/index.html) | Decode files, select segments, stream blocks, export WAV | `Decoder`, `load`, `stream_lazy`, `export` |
| [`proc`](https://docs.rs/dasp-rs/latest/dasp_rs/proc/index.html) | Transforms, resampling, filtering, channels, effects | `stft`, `istft`, `cqt`, `icqt`, `resample`, `iirt`, `time_stretch` |
| [`feat`](https://docs.rs/dasp-rs/latest/dasp_rs/feat/index.html) | Spectral/MIR features, rhythm, separation, reconstruction, sequences | `spectral`, `onset_detect`, `tempo`, `hpss`, `griffinlim`, `dtw`, `viterbi` |
| [`pitch`](https://docs.rs/dasp-rs/latest/dasp_rs/pitch/index.html) | Fundamental frequency, voicing, spectral pitch, tuning | `yin`, `pyin`, `piptrack`, `estimate_tuning` |
| [`mag`](https://docs.rs/dasp-rs/latest/dasp_rs/mag/index.html) | Decibel conversion, perceptual weighting, normalization | `power_to_db`, `amplitude_to_db`, `a_weighting`, `pcen` |
| [`generate`](https://docs.rs/dasp-rs/latest/dasp_rs/generate/index.html) | Synthetic signals | `tone`, `chirp`, `clicks` |
| [`ops`](https://docs.rs/dasp-rs/latest/dasp_rs/ops/index.html) | Sample-wise arithmetic | `mix_signals`, `multiply_signals`, `scalar_operation` |
| [`util`](https://docs.rs/dasp-rs/latest/dasp_rs/util/index.html) | Framing, time/frequency conversion, array operations, notation | `frame`, `frames_to_time`, `hz_to_midi`, `note_to_hz`, `key_to_notes` |
| [`prelude`](https://docs.rs/dasp-rs/latest/dasp_rs/prelude/index.html) | Common imports | `Decoder`, `AudioData`, `stft`, `spectral`, pitch utilities |

<details>
<summary><strong>Browse the detailed feature catalog</strong></summary>


### Audio I/O — `io` / `types`

- `AudioData` — `f32` sample container (`samples`, `sample_rate`,
  `channels`) with `to_mono`, `split_channels`, `duration`, `frame_count`, `to_raw`.
- `Decoder` — builder for loading: `.sample_rate()`, `.mono()`, `.offset()`,
  `.duration()`, `.load()`.
- `load` — decode audio through Symphonia (with optional resample, mono, offset, duration).
- `export` — write a 32-bit float WAV.
- `stream` — incrementally decode and collect a requested number of blocks.
- `stream_lazy` — incrementally decode through a bounded two-block queue, delivering
  `Result` items so mid-stream decode errors surface. Drop the receiver to stop delivery.
- Reads 8/16/24/32-bit PCM and 32-bit float WAV.

### Sample-wise operations — `ops`

- `mix_signals`, `subtract_signals`, `multiply_signals`, `divide_signals`,
  `scalar_operation`.

### Signal processing — `proc`

- **Channels / amplitude**: `to_mono`, `amplify`, `attenuate`, `normalize`.
- **Mixing**: `stereo_mix`, `multi_channel_mix`, `dry_wet_mix`.
- **Panning**: `stereo_pan`, `multi_channel_pan`.
- **Resampling**: `resample` (FFT-based band-limited conversion, via `rubato`).
- **Time domain**: `delay`, `time_reversal`, `time_crop`, `zero_padding`,
  `autocorrelate`, `lpc`, `zero_crossings`, `mu_compress`, `mu_expand`, `log_energy`.
- **Time–frequency**: `stft` / `istft`, `cqt` / `icqt`, `vqt`, `pseudo_cqt`,
  `hybrid_cqt`, `fmt`, `iirt`, `reassigned_spectrogram`, `magphase`, `phasor`.
- **Effects**: `trim`, `split`, `time_stretch`, `pitch_shift`, `remix`.

### Feature extraction — `feat`

- **Spectral / MIR** via the `spectral(&y, sr)` builder with terminal methods:
  `.melspectrogram()`, `.mfcc()`, `.rms()`, `.chroma_cqt()`, `.chroma_cens()`,
  `.spectral_centroid()`, `.spectral_bandwidth()`, `.spectral_contrast()`,
  `.spectral_flatness()`, `.spectral_rolloff()`, `.spectral_flux()`,
  `.spectral_entropy()`, `.poly_features()`, `.tonnetz()`, `.pitch_chroma()`,
  `.hpss()`, `.pitch_autocorr()`, `.vad_features()`, `.spectral_subband_centroids()`,
  `.formant_frequencies()`. Also `chroma_stft`, `chroma_vqt`, and `cmvn(&feats)` builders.
- **Harmonics**: `interp_harmonics`, `salience`, `f0_harmonics`, `phase_vocoder`.
- **Rhythm**: `tempo`, `tempogram`, `tempogram_ratio`, `fourier_tempogram`,
  `onset_strength`, `onset_detect`, `beat_track`, `beat_sync`, `plp`, and `onset_strength_multi`.
- **Manipulation**: `stack_memory`, `temporal_kurtosis`, `zero_crossing_rate`.
- **Masks & normalisation**: `softmask`, `normalize` (L1/L2/L∞), `sparsify_rows`.
- **Phase reconstruction**: `griffinlim`, `griffinlim_cqt`.
- **Inverse transforms**: `compute_delta`, `mel_to_stft`, `mel_to_audio`,
  `mfcc_to_mel`, `mfcc_to_audio`.
- **Harmonic/percussive separation**: `harmonic`, `percussive` convenience builders.
- **Segmentation & structure**: `recurrence_matrix`, `cross_similarity`,
  `agglomerative`, `subsegment`, `path_enhance`, `timelag_filter`.
- **Sequence / HMM**: `dtw`, `viterbi`, `viterbi_discriminative`, `viterbi_binary`;
  transition-matrix builders `transition_loop`, `transition_local`,
  `transition_uniform`, `transition_cycle`, `transition_acyclic`.
- **NMF decomposition**: `decompose` (Lee–Seung multiplicative updates).

### Pitch & tuning — `pitch`

- `yin(&y, fmin, fmax)`, `pyin(&y, fmin, fmax)` — fundamental-frequency estimation
  (builders: `.sample_rate()`, `.frame_length()`, `.hop_length()`, `.compute()`).
- YIN selects the first acceptable trough with parabolic interpolation;
  `.trough_threshold()` controls its acceptance threshold (default 0.1).
- pYIN combines probabilistic threshold candidates with voiced/unvoiced Viterbi
  decoding on a 10-cent pitch grid. `.compute_with_voicing()` returns `PyinResult`
  (`f0`, `voiced_flag`, and frame-local `voiced_prob`). `.compute()` still returns
  only frequencies, using zero for unvoiced frames. Transition rate and voicing
  switch probability are configurable.
- Both trackers use complete, uncentered frames and require enough samples per
  frame to cover two periods at `fmin` plus an interpolation neighbour. Nonfinite
  samples, invalid frequency ranges, and zero hop lengths are rejected.
- `piptrack` — spectral peak pitch tracking.
- `estimate_tuning(&y)` (builder), `pitch_tuning` — tuning-deviation estimation.

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
- **Array / signal utilities**: `localmax`, `localmin`, `peak_pick`, `frame`,
  `pad_center`, `fix_length`, `sync` (Mean/Median/Max/Min), `match_intervals`,
  `expand_to`.
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

</details>

## Audio format support

`dasp-rs` reads audio via [`symphonia`] and writes WAV via [`hound`]. Public audio samples
are `f32`; algorithms may use higher-precision intermediates. Enabled decoders include WAV, MP3, FLAC, Ogg/Vorbis, and AAC/M4A.
The following table describes supported WAV sample encodings:

| Format | Read | Write |
|--------|:----:|:-----:|
| 8-bit PCM  | ✅ | – |
| 16-bit PCM | ✅ | – |
| 24-bit PCM | ✅ | – |
| 32-bit PCM | ✅ | – |
| 32-bit float | ✅ | ✅ |

`export` writes 32-bit float WAV without clipping, preserving samples outside
`[-1.0, 1.0]`. Apply limiting explicitly if you need it.

For multichannel streams, block and hop lengths count interleaved scalar samples,
not per-channel frames. `stream` collects a requested number of blocks;
`stream_lazy` delivers `Result` items through a bounded queue and surfaces
mid-stream decode errors. See [the streaming API](https://docs.rs/dasp-rs/latest/dasp_rs/io/fn.stream_lazy.html)
for shutdown and buffering behavior.

## Examples

The [examples directory](examples) contains complete programs. From a repository
checkout, run an example with `cargo run --example <name>`:

| Workflow | Examples |
| --- | --- |
| First pipeline and audio files | [`getting_started`](examples/getting_started.rs), [`audio_io`](examples/audio_io.rs) |
| Transforms and effects | [`time_frequency`](examples/time_frequency.rs), [`effects`](examples/effects.rs) |
| Spectral features and separation | [`spectral_features`](examples/spectral_features.rs), [`harmonic_percussive`](examples/harmonic_percussive.rs), [`decomposition`](examples/decomposition.rs) |
| Pitch and rhythm | [`pitch_detection`](examples/pitch_detection.rs), [`rhythm_analysis`](examples/rhythm_analysis.rs) |
| Signals and magnitude | [`signal_generation`](examples/signal_generation.rs), [`magnitude_scaling`](examples/magnitude_scaling.rs) |
| Sequences and segmentation | [`sequence_hmm`](examples/sequence_hmm.rs), [`segmentation`](examples/segmentation.rs) |
| Arrays, features, and notation | [`array_utilities`](examples/array_utilities.rs), [`feature_utilities`](examples/feature_utilities.rs), [`utilities`](examples/utilities.rs) |

For signatures, defaults, and function-specific errors, consult
[docs.rs](https://docs.rs/dasp-rs). Examples illustrate usage; they are not substitutes
for the numerical-reference tests described below.

## Numerical conventions

These choices affect feature values, alignment, and compatibility with other libraries.

| Area | Convention |
| --- | --- |
| Audio and channels | Public samples are `f32`; `AudioData` stores interleaved channels. Slice-based analysis generally operates on one channel: use `Decoder::mono()` or `AudioData::split_channels()` explicitly. |
| STFT | Frequency-bin × frame matrices; periodic Hann and reflect-centered frames by default. Frame centers are `frame_index * hop_length`. |
| ISTFT | Fallible overlap-add reconstruction. Supply the matching `n_fft` for odd FFT sizes and `.length(original_len)` when the original sample count matters. |
| Mel and MFCC | Slaney mel spacing and area normalization. MFCCs use an orthonormal DCT-II of **natural-log mel power**, rather than a default decibel transform. |
| Pitch | YIN/pYIN use complete, uncentered frames; silence/unvoiced output is zero. Pitch-frame counts can therefore differ from centered spectral features. |
| HPSS | Harmonic and percussive outputs are power spectra. Clipped median neighborhoods and exponent-one masks preserve their summed input power. |
| Resampling | Band-limited FFT conversion with startup delay removed and the filter flushed; output length is `ceil(input_len * target_sr / original_sr)`. |
| CQT/VQT | Direct, full-rate wavelet analysis with 12 bins per octave; VQT uses a fixed 24 Hz bandwidth offset. These are not drop-in replicas of librosa's recursive transforms. |
| Reassignment | A nearest-grid **magnitude raster**, not arrays of reassigned frequencies and times. Quiet coefficients are retained; the API documents its tie and boundary rules. |

Algorithms expose different parameter constraints and error types; follow the
individual API documentation rather than assuming every function accepts arbitrary
geometry or scale. Inverse CQT is iterative and can remain ill-conditioned. Several
higher-level features have less independent-reference coverage than the core
transforms. The [validation section](#validation) records tested conventions,
known differences, and remaining gaps without claiming universal accuracy or
complete librosa compatibility.

## Migrating from 0.5

Update your dependency to `dasp-rs = "0.6"`. Inverse STFT now returns a
`Result`; propagate or handle reconstruction errors:

```rust
let reconstructed = dasp_rs::proc::istft(&spectrum).compute()?;
```

Numerical outputs also change for CQT/VQT, IIRT, resampling, phase recovery,
pitch tracking, and several features. Recompute stored feature baselines and
review the [0.6.0 changelog](CHANGELOG.md) for input-validation and channel
conventions. The minimum supported Rust version remains 1.85.

## Validation

The repository combines analytic regressions, independent numerical references,
fixed-seed property tests, and attributed recordings. Each provides different
evidence; the commands and limitations below make those distinctions explicit.

### Numerical reference benchmarks

Run the independent accuracy comparisons and produce a JSON report with Python
3 and Cargo (no Python packages required for this command):

```sh
python3 scripts/reference_benchmarks.py
python3 scripts/reference_benchmarks.py --release --output target/reference-release.json
python3 scripts/reference_benchmarks.py --self-test
```

The default report is `target/reference-report.json`, with the full Rust test log
beside it. It records maximum absolute error, RMSE, declared error budgets, Rust
version, source/test/generator/fixture hashes, and the tested revision. These are
numerical accuracy measurements; runtime performance is not measured. Reports
identify comparisons between different algorithms separately from parity tests.
The input manifest audits fixture inputs, exact metadata, expected test counts,
and metric counts. Missing observations, nonfinite metrics, or source changes
during a run fail the report. The stdlib-only self-test checks these failure paths.

The `tests/reference_*.rs` suites use NumPy 2.3.5, SciPy 1.18.1, and librosa
0.11.0 fixtures. Generators never invoke the Rust implementation. Inputs are
quantized to `f32` before the independent calculation; most reference arithmetic
uses `f64`. Fixtures cover:

| Area | Independent reference and checks |
| --- | --- |
| STFT, ISTFT, phase vocoder, time stretch | librosa; odd/even FFT sizes, window/padding conventions, endpoints, multiple rates |
| Resampling | SciPy polyphase FIR; separate interior/boundary errors, impulse delay, DC gain, passband and stopband behavior |
| Mel, MFCC, inverse DCT, spectral features | librosa and SciPy; individual mel weights, DCT types I–IV, deltas, centroid, bandwidth, rolloff, flatness, RMS, contrast, HPSS |
| CQT/VQT and inverse | librosa wavelets with SciPy correlation; calibrated tones/impulses, independent NumPy SVD inverse, conditioning and reconstruction checks |
| LPC, autocorrelation, magnitude, sequences | SciPy Toeplitz solves/correlation and librosa; dB, weighting, PCEN, mu-law, DTW costs/paths, Viterbi scores/paths |
| IIR, reassignment, rhythm, recovery | SciPy SOS response/filtering; explicit reassignment raster; flux/tempograms/PLP; analytic pulse grids; matched Griffin–Lim iterations and NMF reconstruction objectives |
| Pitch and recorded audio | Librosa YIN/pYIN with a declared 15-cent allowance and voicing checks; independent mel/MFCC summaries for the attributed corpus |

To verify fixture regeneration without changing checked-in data, create a Python
3.14 environment with the pinned dependencies:

```sh
python3.14 -m venv target/reference-python
target/reference-python/bin/python -m pip install -r tests/fixtures/requirements-reference.txt
python3 scripts/reference_benchmarks.py --check-fixtures --python target/reference-python/bin/python
```

Use the corresponding `Scripts/python.exe` path on Windows. `--regenerate`
replaces fixtures intentionally; review the generator and resulting numerical
changes together. CI regenerates fixtures in a temporary directory, rejects
unexpected drift, runs the comparisons, and uploads the report on pushes and
pull requests. The checked-in references also run in the ordinary Rust test
matrix, including release builds and Rust 1.85.

Reference settings match this crate's documented conventions: natural-log MFCCs,
nearest-edge deltas, PCEN initialized from its first frame, and Viterbi scores
without an initial uniform-state log prior. Waveform RMS uses complete,
uncentered rectangular frames; spectrum RMS measures the supplied windowed
signal. Contrast retains inclusive octave bands with an additive floor. HPSS
uses clipped median neighborhoods and exponent-one masks over power spectra.
The generator headers describe these choices and link their sources.

Some differences remain intentional or limited by conditioning:

- CQT/VQT use full-rate wavelet correlation. The calibrated tone fixtures differ
  from librosa's recursive implementations by about 2.54%/4.72% relative L2,
  respectively; tight parity assertions use the independent full-rate reference.
  `hybrid_cqt` currently aliases full CQT; `pseudo_cqt` samples STFT bins.
- Inverse CQT uses at most 32 conjugate-gradient iterations. It cannot recover
  frequencies or time detail absent from its input. Ill-conditioned inverse
  fixtures assert coefficient consistency and report waveform differences.
  The inverse retains only candidates that improve the coefficient residual
  after conversion to `f32`. This guard is not regularization: ill-conditioned
  inputs can still produce large waveform magnitudes. The broader sweep includes
  a 72-sample system where the iteration budget leaves a relative coefficient
  residual around `3.58e-4`, versus an independent SVD optimum around `2.30e-8`.
  Direct CQT and iterative inversion cost more than recursive implementations.
- Rubato and SciPy use different resampling filters, so boundary and impulse
  differences have separate budgets. Nearly zero Hann endpoints amplify
  finite-precision spectrum errors; tests report these separately from
  well-conditioned samples.
- Time stretching rounds output length using Rust's halfway-away-from-zero rule;
  NumPy's halfway-to-even rule can differ by one sample. D-weighting retains its
  published coefficient convention, with a 0.03% relative allowance against
  librosa's differently rounded coefficients.

Passing these fixtures establishes accuracy for the tested signals and settings;
it does not establish complete librosa API compatibility or cover every DSP
algorithm in the crate.

The extended CQT sweeps cover sample rates from 8 to 48 kHz, varied frequency
grids/hops, one-sample and longer inputs, amplitudes from subnormal values to
`1e20`, and tones, chirps, noise, and transients. Analytic tone checks measure
gain, phase, and localization. Comparisons with recursive librosa also distinguish
frames whose complete wavelet support fits inside the signal from boundary
frames; measured gaps reach about 11.3% relative L2 for a boundary VQT chirp case.
These measurements describe individual fixtures, not universal error bounds.
To run just this area, pass `--group cqt --group cqt_sweep --group cqt_inverse`
to the reference runner. Report observations have unique target/test identifiers
so checks through different public entry points remain distinguishable.

### Randomized and recorded-audio validation

`tests/randomized_dsp.rs` and `tests/randomized_features.rs` exercise fixed-seed
geometry, scale, channel, and sequence cases, including exhaustive small DTW and
Viterbi oracles. Failure messages retain the seed and case settings. These are
bounded property tests, not an exhaustive fuzzing campaign.

`tests/real_audio_workflows.rs` uses three checked-in, four-second mono FLAC
excerpts: speech, trumpet, and instrumental music with pitched percussion.
Their approximately 190 KB of audio requires no network access in Rust tests.
Workflows check decoding, resampling duration, STFT reconstruction, independent
mel/MFCC summaries, HPSS conservation, and pitch/rhythm output validity.
The corpus has no annotated pitch, beat, or separation ground truth.

```sh
python3 scripts/prepare_audio_corpus.py --verify
# With the pinned reference environment, recompute the independent summaries:
target/reference-python/bin/python scripts/prepare_audio_corpus.py --check-references
```

Source hashes, conversion settings, full license texts, attribution, and explicit
corpus limits are retained in [the audio manifest](tests/fixtures/audio/manifest.json)
and [attribution](tests/fixtures/audio/ATTRIBUTION.md). Regeneration requires an
explicit `--prepare`; downloads additionally require `--download` and are checked
against pinned source hashes.

### Release validation

[validation/release.json](validation/release.json) inventories public free
functions, maps evidence by area, lists remaining independent-reference gaps,
and defines local release gates. New public functions require an inventory review.
This catalog is not a complete builder-method inventory or branch-coverage report.

```sh
python3 scripts/validate_release.py --check
python3 scripts/validate_release.py --run --reference-python target/reference-python/bin/python
```

The full run checks the reference runner's failure paths, formatting, strict
Clippy, debug/release tests, doctests,
documentation warnings, Rust 1.85, corpus hashes, regenerated references, and
performance workloads. It saves per-gate logs and `target/release-validation/report.json`.
Use an idle machine for the final performance measurements. CI also checks
Linux/macOS/Windows and uploads numerical/performance reports; a local run does
not verify remote CI results or publish a release.

Remaining independent-reference work includes FMT, chroma/CENS/tonnetz, formants,
some higher-level spectral features, complete mel/MFCC-to-audio pipelines,
end-to-end pitch shifting, and segmentation/harmonic utilities. Existing tests
for these areas remain useful, but they do not establish external reference
parity. Twenty existing ignored documentation examples also remain outside
executed doctest coverage. Numerical/API changes require changelog and version
review before publication.

## Performance

- Selected operations use `rayon`; benchmark the actual workload and thread configuration.
- `stream_lazy` retains only a bounded queue, the current block, and codec packet
  buffers; storage does not grow with file duration. `stream` additionally retains
  the requested output blocks in its returned collection.
- FFTs via [`rustfft`]; resampling via [`rubato`]; linear algebra via pure-Rust
  [`nalgebra`].

Run the release performance workloads on an idle machine:

```sh
python3 scripts/performance_benchmarks.py run --rss --output target/performance-baseline.json
# After a change, repeat with --output target/performance-candidate.json, then:
python3 scripts/performance_benchmarks.py compare target/performance-baseline.json target/performance-candidate.json
```

The 13 deterministic cases cover short/long STFTs, resampling, CQT/iCQT, and
YIN/pYIN. Reports include warmed median/p95 latency, throughput, separate
allocation counts/bytes, and optional whole-process peak RSS on Linux. Timing
uses an uninstrumented executable; allocations use a separate executable with
[`stats_alloc`](https://docs.rs/stats_alloc/0.1.10/stats_alloc/). Input preparation
is excluded, while per-call planning/setup and output destruction are included.
Allocated bytes measure cumulative traffic, not peak live memory.

Comparisons require matching environments and workload definitions. Reports
record source/binary hashes and reject source changes during measurement.
Wall-clock results describe that machine and workload; they are not promises of
real-time callback suitability or noisy shared-runner pass/fail thresholds.
`cargo test --all-targets` compiles the benchmark targets without running them.

CQT wavelets store only taps that can overlap the analyzed signal. Full-kernel
phase and normalization are preserved, while stored support per filter is bounded
by twice the signal length. This avoids allocating enormous full filters for
very short inputs with low minimum frequencies.

## Contributing

Issues and pull requests are welcome on [GitHub]. Before submitting, run:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --doc
cargo test --release --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

CI also checks the minimum supported Rust version (1.85) and runs the test suite
on Linux, macOS, and Windows. The integration suites in `tests/` use synthetic
signals, analytic expectations, independent linear solves, exhaustive small
sequence models, and checked-in numerical references; Python is not required to
run the Rust tests. For a bug fix, add a regression that reproduces the failure
before changing the implementation. Cover boundary conditions and numerical
behavior, not just output dimensions.

## License

The software is licensed under the MIT License. See [LICENSE](LICENSE).
The recorded-audio test fixtures retain their individual Creative Commons
licenses; see [audio attribution](tests/fixtures/audio/ATTRIBUTION.md).

## Citation

If you use `dasp-rs` in research, cite the software using [CITATION.cff](CITATION.cff)
or GitHub’s **Cite this repository** action. The
[Zenodo concept DOI](https://doi.org/10.5281/zenodo.22871121) identifies the project
across versions; include the exact crate version used in your experiments.

## Acknowledgements

Thanks to [@Levitanus](https://github.com/Levitanus) for reporting the OpenBLAS build
failure and 24-bit decoding issues, and for the diagnosis in
[#4](https://github.com/dasp-rs/dasp-rs/pull/4).

[`hound`]: https://crates.io/crates/hound
[`symphonia`]: https://crates.io/crates/symphonia
[`rustfft`]: https://crates.io/crates/rustfft
[`rubato`]: https://crates.io/crates/rubato
[`nalgebra`]: https://crates.io/crates/nalgebra
[GitHub]: https://github.com/dasp-rs/dasp-rs
