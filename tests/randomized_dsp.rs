//! Bounded deterministic randomized DSP checks. Seeds and geometry accompany
//! failures; no arbitrary rates, allocations, or unbounded filter supports.
use dasp_rs::proc::{
    Window, cqt, delay, dry_wet_mix, icqt, iirt, istft, multi_channel_mix, resample, stereo_mix,
    stft, time_crop, time_reversal, to_mono, vqt, zero_padding,
};
use dasp_rs::types::AudioData;
use dasp_rs::util::samples_to_time;
use ndarray::Array2;
use num_complex::Complex;

#[test]
fn fft_overflow_is_reported() {
    let forward = stft(&[f32::MAX; 16]).n_fft(16).center(false).compute();
    assert!(
        forward.is_err(),
        "Unrepresentable forward FFT returned success: {forward:?}"
    );
}

#[test]
fn representable_inverse_survives_large_intermediate_values() {
    // Alternating real spectrum is a centered unit impulse, scaled to MAX.
    // The unnormalized inverse exceeds f32, but the normalized output fits.
    let spectrum = Array2::from_shape_fn((9, 1), |(bin, _)| {
        Complex::new(if bin % 2 == 0 { f32::MAX } else { -f32::MAX }, 0.0)
    });
    let result = istft(&spectrum)
        .n_fft(16)
        .hop_length(4)
        .length(1)
        .compute()
        .unwrap();
    assert!(
        result[0].is_finite(),
        "Representable inverse returned {result:?}"
    );
    assert!((f64::from(result[0]) / f64::from(f32::MAX) - 1.0).abs() < 1e-6);
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn integer(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.next() as usize % (hi - lo)
    }
    fn sample(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / 16_777_216.0) - 0.5
    }
    fn signal(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.sample()).collect()
    }
}

fn max_difference(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn randomized_stft_roundtrip_linearity_and_parseval() {
    const SEED: u64 = 0x95a7_231d_5fa2_7e18;
    let mut rng = Rng(SEED);
    let mut worst = 0.0_f64;
    for case in 0..64 {
        let nfft = rng.integer(4, 130);
        let win = rng.integer((nfft / 2).max(2), nfft + 1);
        let hop = rng.integer(1, (win / 4).max(1) + 1);
        let length = rng.integer(1, 513);
        let window = if rng.next() & 1 == 0 {
            Window::Hann
        } else {
            Window::Hamming
        };
        let context = format!(
            "seed={SEED:#x} case={case} nfft={nfft} win={win} hop={hop} length={length} window={window:?}"
        );
        let x = rng.signal(length);
        let y = rng.signal(length);
        let blend: Vec<_> = x.iter().zip(&y).map(|(a, b)| 0.25 * a - 0.5 * b).collect();
        let transform = |input: &[f32]| {
            stft(input)
                .n_fft(nfft)
                .win_length(win)
                .hop_length(hop)
                .window(window)
                .compute()
                .expect(&context)
        };
        let sx = transform(&x);
        let sy = transform(&y);
        let combined = transform(&blend);
        let expected_frames = 1 + (length - nfft % 2) / hop;
        assert_eq!(sx.dim(), (nfft / 2 + 1, expected_frames), "{context}");
        for ((a, b), c) in sx.iter().zip(&sy).zip(&combined) {
            assert!(
                (*c - (*a * 0.25 - *b * 0.5)).norm() < 2e-5,
                "{context}: STFT linearity"
            );
        }
        let recovered = istft(&sx)
            .n_fft(nfft)
            .win_length(win)
            .hop_length(hop)
            .window(window)
            .length(length)
            .compute()
            .expect(&context);
        let error = max_difference(&x, &recovered);
        worst = worst.max(error);
        assert!(error < 2e-5, "{context}: roundtrip max error {error}");
        // Independent Parseval identity on a single noncentered full Hann frame,
        // including the distinct odd/even one-sided Nyquist multiplicity.
        let frame = rng.signal(nfft);
        let fft = stft(&frame)
            .n_fft(nfft)
            .center(false)
            .compute()
            .expect(&context);
        let frequency_energy: f64 = fft
            .column(0)
            .iter()
            .enumerate()
            .map(|(bin, z)| {
                let weight = if bin == 0 || (nfft % 2 == 0 && bin == nfft / 2) {
                    1.0
                } else {
                    2.0
                };
                weight * (f64::from(z.re).powi(2) + f64::from(z.im).powi(2)) / nfft as f64
            })
            .sum();
        let time_energy: f64 = frame
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let window =
                    0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / nfft as f64).cos();
                (f64::from(value) * window).powi(2)
            })
            .sum();
        assert!(
            (frequency_energy - time_energy).abs() < 2e-6 * time_energy + 1e-8,
            "{context}: Parseval frequency={frequency_energy} time={time_energy}"
        );
    }
    eprintln!("RANDOMIZED stft: seed={SEED:#x} cases=64 maximum_roundtrip_error={worst:.6e}");
}

#[test]
fn randomized_wavelet_linearity_geometry_and_well_conditioned_inverse() {
    const SEED: u64 = 0x72e1_a930_83bc_4729;
    let mut rng = Rng(SEED);
    for case in 0..24 {
        let sr = [2048, 4096, 8000][rng.integer(0, 3)];
        let fmin = sr as f32 / 32.0 * (1.0 + rng.sample().abs());
        let bins = rng.integer(1, 25);
        let hop = rng.integer(1, 33);
        let length = rng.integer(1, 130);
        let context = format!(
            "seed={SEED:#x} case={case} sr={sr} fmin={fmin} bins={bins} hop={hop} length={length}"
        );
        let x = rng.signal(length);
        let y = rng.signal(length);
        let blend: Vec<_> = x.iter().zip(&y).map(|(a, b)| 0.25 * a - 0.5 * b).collect();
        for variable in [false, true] {
            let transform = |input: &[f32]| {
                if variable {
                    vqt(input, sr)
                        .fmin(fmin)
                        .n_bins(bins)
                        .hop_length(hop)
                        .compute()
                        .expect(&context)
                } else {
                    cqt(input, sr)
                        .fmin(fmin)
                        .n_bins(bins)
                        .hop_length(hop)
                        .compute()
                        .expect(&context)
                }
            };
            let a = transform(&x);
            let b = transform(&y);
            let actual = transform(&blend);
            assert_eq!(actual.dim(), (bins, 1 + length / hop), "{context}");
            for ((x, y), z) in a.iter().zip(&b).zip(&actual) {
                assert!(z.re.is_finite() && z.im.is_finite(), "{context}");
                assert!(
                    (*z - (*x * 0.25 - *y * 0.5)).norm() < 1e-6,
                    "{context}: variable={variable} linearity"
                );
            }
        }
    }
    let mut worst = 0.0_f64;
    // Wide frequency coverage and dense framing produce a well-conditioned
    // small inverse. An extra terminal zero follows the documented iCQT length.
    for case in 0..20 {
        let sr = [2048, 4096][rng.integer(0, 2)];
        let fmin = sr as f32 / 32.0;
        let length = rng.integer(5, 20);
        let context = format!("seed={SEED:#x} inverse case={case} sr={sr} length={length}");
        let mut x = rng.signal(length);
        let c = cqt(&x, sr)
            .fmin(fmin)
            .n_bins(48)
            .hop_length(1)
            .compute()
            .expect(&context);
        let y = icqt(&c)
            .sample_rate(sr)
            .fmin(fmin)
            .hop_length(1)
            .compute()
            .expect(&context);
        x.push(0.0);
        let error = max_difference(&x, &y);
        worst = worst.max(error);
        assert!(
            error < 3e-5,
            "{context}: well-conditioned reconstruction error {error}"
        );
    }
    eprintln!(
        "RANDOMIZED wavelet: seed={SEED:#x} forward_cases=48 inverse_cases=20 maximum_inverse_error={worst:.6e}"
    );
}

#[test]
fn randomized_iir_gain_and_causal_time_shift() {
    const SEED: u64 = 0x8174_992a_ac58_5d31;
    let mut rng = Rng(SEED);
    for case in 0..24 {
        let sr = [128, 8000, 16000, 44100][rng.integer(0, 4)];
        let length = rng.integer(256, 1025);
        let win = rng.integer(16, 129);
        let hop = rng.integer(1, 65);
        let shifts = rng.integer(1, 4);
        let gain = 0.25 + rng.sample().abs() * 2.0;
        let context = format!(
            "seed={SEED:#x} case={case} sr={sr} length={length} win={win} hop={hop} shifts={shifts} gain={gain}"
        );
        let input = rng.signal(length);
        let transformed = iirt(&input, sr)
            .win_length(win)
            .hop_length(hop)
            .compute()
            .expect(&context);
        let scaled: Vec<_> = input.iter().map(|x| -gain * x).collect();
        let changed = iirt(&scaled, sr)
            .win_length(win)
            .hop_length(hop)
            .compute()
            .expect(&context);
        let mut delayed = vec![0.0; shifts * hop];
        delayed.extend_from_slice(&input);
        let shifted = iirt(&delayed, sr)
            .win_length(win)
            .hop_length(hop)
            .compute()
            .expect(&context);
        assert_eq!(
            transformed.dim(),
            (12, 1 + (length - win) / hop),
            "{context}"
        );
        for ((band, frame), &value) in transformed.indexed_iter() {
            let expected = f64::from(gain) * f64::from(value);
            assert!(
                (f64::from(changed[[band, frame]]) - expected).abs() < 3e-6 * expected + 1e-14,
                "{context}: RMS homogeneity"
            );
            assert!(
                (shifted[[band, frame + shifts]] - value).abs() < f32::EPSILON,
                "{context}: causal shift"
            );
        }
    }
    eprintln!("RANDOMIZED iir: seed={SEED:#x} cases=24");
}

#[test]
fn randomized_resampling_duration_linearity_and_impulse_timing() {
    const SEED: u64 = 0xad15_871a_317c_9273;
    let mut rng = Rng(SEED);
    let rates = [8000, 12000, 16000, 24000, 44100, 48000];
    for case in 0..24 {
        let orig = rates[rng.integer(0, rates.len())];
        let target = rates[rng.integer(0, rates.len())];
        let length = rng.integer(64, 2305);
        let context =
            format!("seed={SEED:#x} case={case} orig={orig} target={target} length={length}");
        let x = rng.signal(length);
        let y = rng.signal(length);
        let blend: Vec<_> = x.iter().zip(&y).map(|(a, b)| 0.25 * a - 0.5 * b).collect();
        let a = resample(&x, orig, target).expect(&context);
        let b = resample(&y, orig, target).expect(&context);
        let combined = resample(&blend, orig, target).expect(&context);
        let expected = (length * target as usize).div_ceil(orig as usize);
        assert_eq!(a.len(), expected, "{context}");
        let expected_blend: Vec<_> = a.iter().zip(&b).map(|(a, b)| 0.25 * a - 0.5 * b).collect();
        assert!(
            max_difference(&combined, &expected_blend) < 1e-5,
            "{context}: resampling linearity"
        );
        let duration = length as f64 / f64::from(orig);
        let new_duration = a.len() as f64 / f64::from(target);
        assert!(
            new_duration + 1e-12 >= duration
                && new_duration - duration < 1.0 / f64::from(target) + 1e-12,
            "{context}: duration"
        );
        let indices: Vec<_> = (0..a.len()).collect();
        let times = samples_to_time(&indices, Some(target));
        assert!(
            times.windows(2).all(|w| w[1] > w[0]),
            "{context}: monotone sample times"
        );
        if orig == target {
            assert_eq!(a, x, "{context}: identity rate");
        }
        let position = rng.integer(256, 768);
        let mut impulse = vec![0.0; 1024];
        impulse[position] = 1.0;
        let response = resample(&impulse, orig, target).expect(&context);
        let peak = response
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        let expected_peak = position as f64 * f64::from(target) / f64::from(orig);
        assert!(
            (peak as f64 - expected_peak).abs() <= 1.0,
            "{context}: impulse peak {peak}, expected {expected_peak}"
        );
    }
    eprintln!("RANDOMIZED resampling: seed={SEED:#x} cases=24");
}

#[test]
fn randomized_channel_and_time_operations_preserve_frames() {
    const SEED: u64 = 0xca93_173b_4b51_836d;
    let mut rng = Rng(SEED);
    for case in 0..64 {
        let channels = rng.integer(1, 7);
        let frames = rng.integer(1, 129);
        let sr = [128, 8000, 44100][rng.integer(0, 3)];
        let context =
            format!("seed={SEED:#x} case={case} channels={channels} frames={frames} sr={sr}");
        let inputs: Vec<_> = (0..channels)
            .map(|_| AudioData {
                samples: rng.signal(frames),
                sample_rate: sr,
                channels: 1,
            })
            .collect();
        let refs: Vec<_> = inputs.iter().collect();
        let combined = multi_channel_mix(&refs, channels as u16).expect(&context);
        let split = combined.split_channels().expect(&context);
        for channel in 0..channels {
            assert_eq!(split[channel], inputs[channel].samples, "{context}");
        }
        if channels == 2 {
            assert_eq!(
                stereo_mix(&inputs[0], &inputs[1]).expect(&context).samples,
                combined.samples,
                "{context}"
            );
        }
        let mono = to_mono(&combined.samples, channels);
        assert_eq!(mono.len(), frames, "{context}: downmix frame count");
        for (frame, &sample) in mono.iter().enumerate() {
            let expected = inputs
                .iter()
                .map(|v| f64::from(v.samples[frame]))
                .sum::<f64>()
                / channels as f64;
            let error = (f64::from(sample) - expected).abs();
            assert!(error < 1e-7, "{context}: downmix error {error}");
        }
        let reversed = time_reversal(&combined).expect(&context);
        assert_eq!(
            time_reversal(&reversed).expect(&context).samples,
            combined.samples,
            "{context}: reversal involution"
        );
        let offset = rng.integer(0, frames);
        let take = rng.integer(0, frames - offset + 1);
        let cropped = time_crop(
            &combined,
            offset as f32 / sr as f32,
            take as f32 / sr as f32,
        )
        .expect(&context);
        assert_eq!(
            cropped.samples,
            combined.samples[offset * channels..(offset + take) * channels],
            "{context}: crop"
        );
        let count = rng.integer(0, frames + 1);
        let seconds = count as f32 / sr as f32;
        let padded = zero_padding(&combined, seconds, seconds).expect(&context);
        assert_eq!(padded.frame_count(), frames + 2 * count, "{context}");
        assert_eq!(
            &padded.samples[count * channels..(count + frames) * channels],
            &combined.samples,
            "{context}: pad placement"
        );
        let delayed = delay(&combined, seconds).compute().expect(&context);
        assert_eq!(
            delayed.frame_count(),
            frames + count,
            "{context}: delay duration"
        );
        assert_eq!(
            &delayed.samples[count * channels..],
            &combined.samples,
            "{context}: delay placement"
        );
        let retained = delay(&combined, seconds)
            .preserve_length(true)
            .compute()
            .expect(&context);
        assert_eq!(
            retained.samples,
            delayed.samples[..frames * channels],
            "{context}: preserved delay"
        );
        let factor = (rng.sample() + 0.5).clamp(0.0, 1.0);
        let wet = dry_wet_mix(&combined, &reversed, factor).expect(&context);
        for (i, &sample) in wet.samples.iter().enumerate() {
            let expected = f64::from(combined.samples[i]) * (1.0 - f64::from(factor))
                + f64::from(reversed.samples[i]) * f64::from(factor);
            assert!(
                (f64::from(sample) - expected).abs() < 1e-7,
                "{context}: mix factor {factor}"
            );
        }
    }
    eprintln!("RANDOMIZED channels_time: seed={SEED:#x} cases=64");
}

fn rejects_without_panic<T, E: std::fmt::Debug>(
    context: &str,
    operation: impl FnOnce() -> Result<T, E> + std::panic::UnwindSafe,
) {
    let result = std::panic::catch_unwind(operation);
    assert!(result.is_ok(), "{context}: panicked");
    assert!(
        result.unwrap().is_err(),
        "{context}: malformed input accepted"
    );
}

#[test]
fn randomized_malformed_parameters_return_errors_without_panics() {
    const SEED: u64 = 0xc47a_3197_95ab_62e1;
    let mut rng = Rng(SEED);
    for case in 0..48 {
        let length = rng.integer(16, 65);
        let signal = rng.signal(length);
        let context = format!("seed={SEED:#x} case={case} length={length}");
        let invalid = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][rng.integer(0, 3)];
        let mut bad = signal.clone();
        bad[rng.integer(0, length)] = invalid;
        rejects_without_panic(&format!("{context} STFT nonfinite"), || {
            stft(&bad).n_fft(16).compute()
        });
        rejects_without_panic(&format!("{context} STFT dimension"), || {
            stft(&signal)
                .n_fft(if case % 2 == 0 { 0 } else { usize::MAX })
                .compute()
        });
        rejects_without_panic(&format!("{context} STFT hop"), || {
            stft(&signal).n_fft(16).hop_length(0).compute()
        });
        let spectrum = Array2::from_elem((9, 2), Complex::new(0.0, 0.0));
        rejects_without_panic(&format!("{context} ISTFT overflow"), || {
            istft(&spectrum).hop_length(usize::MAX).compute()
        });
        rejects_without_panic(&format!("{context} ISTFT length"), || {
            istft(&spectrum).length(usize::MAX).compute()
        });
        let mut bad_spectrum = spectrum.clone();
        bad_spectrum[[rng.integer(0, 9), rng.integer(0, 2)]].re = invalid;
        rejects_without_panic(&format!("{context} ISTFT nonfinite"), || {
            istft(&bad_spectrum).compute()
        });
        rejects_without_panic(&format!("{context} CQT fmin"), || {
            cqt(&signal, 8000).n_bins(12).fmin(invalid).compute()
        });
        rejects_without_panic(&format!("{context} CQT hop"), || {
            cqt(&signal, 8000).n_bins(12).hop_length(0).compute()
        });
        rejects_without_panic(&format!("{context} CQT bins"), || {
            cqt(&signal, 8000).n_bins(usize::MAX).compute()
        });
        rejects_without_panic(&format!("{context} VQT nonfinite"), || {
            vqt(&bad, 8000).n_bins(12).compute()
        });
        rejects_without_panic(&format!("{context} iCQT overflow"), || {
            icqt(&spectrum).hop_length(usize::MAX).compute()
        });
        rejects_without_panic(&format!("{context} IIR nonfinite"), || {
            iirt(&bad, 8000).win_length(16).compute()
        });
        rejects_without_panic(&format!("{context} IIR hop"), || {
            iirt(&signal, 8000).win_length(16).hop_length(0).compute()
        });
        rejects_without_panic(&format!("{context} resample zero rate"), || {
            resample(&signal, 0, 8000)
        });
        rejects_without_panic(&format!("{context} resample nonfinite"), || {
            resample(&bad, 8000, 16000)
        });
        let audio = AudioData {
            samples: signal,
            sample_rate: 8000,
            channels: 1,
        };
        rejects_without_panic(&format!("{context} delay time"), || {
            delay(&audio, invalid).compute()
        });
        rejects_without_panic(&format!("{context} pad huge time"), || {
            zero_padding(&audio, f32::MAX, 0.0)
        });
        rejects_without_panic(&format!("{context} crop time"), || {
            time_crop(&audio, 0.0, invalid)
        });
        rejects_without_panic(&format!("{context} mix factor"), || {
            dry_wet_mix(&audio, &audio, invalid)
        });
        let malformed = AudioData {
            samples: vec![1.0; 3],
            sample_rate: 8000,
            channels: if case % 2 == 0 { 0 } else { 2 },
        };
        rejects_without_panic(&format!("{context} channel metadata"), || {
            time_reversal(&malformed)
        });
    }
    eprintln!("RANDOMIZED invalid: seed={SEED:#x} cases=48 fallible_calls=960");
}

#[test]
fn resampling_extreme_finite_input_returns_finite_output_or_error() {
    let result = resample(&[f32::MAX; 64], 8000, 16000);
    if let Ok(output) = result {
        assert!(
            output.iter().all(|v| v.is_finite()),
            "Resampling returned successful nonfinite output"
        );
    }
}

#[test]
fn inverse_rejects_unrepresentable_window_normalized_output() {
    // This spectrum is an impulse at sample 12 of a 16-point frame. The Hann
    // value there is 1/2, so overlap normalization would require 2*f32::MAX.
    let spectrum = Array2::from_shape_fn((9, 1), |(bin, _)| match bin % 4 {
        0 => Complex::new(f32::MAX, 0.0),
        1 => Complex::new(0.0, f32::MAX),
        2 => Complex::new(-f32::MAX, 0.0),
        _ => Complex::new(0.0, -f32::MAX),
    });
    assert!(
        istft(&spectrum)
            .n_fft(16)
            .hop_length(4)
            .length(5)
            .compute()
            .is_err()
    );
}

#[test]
fn high_amplitude_resampling_matches_normalized_result() {
    const SEED: u64 = 0x2c47_83ab_93fa_8176;
    let mut rng = Rng(SEED);
    // Longer than the resampler block verifies reuse across multiple blocks.
    let input = rng.signal(4097);
    let reference = resample(&input, 44100, 48000).unwrap();
    for scale in [4.0_f32, 1e20_f32, 1e30_f32] {
        let scaled: Vec<_> = input.iter().map(|value| value * scale).collect();
        let output = resample(&scaled, 44100, 48000).unwrap();
        let normalized: Vec<_> = output
            .iter()
            .map(|&value| (f64::from(value) / f64::from(scale)) as f32)
            .collect();
        let error = max_difference(&normalized, &reference);
        assert!(
            error < 2e-6,
            "seed={SEED:#x} scale={scale} normalized max error={error}"
        );
    }
}

#[test]
fn resampling_preserves_distant_quiet_audio_beside_a_large_peak() {
    // A long zero gap places the quiet region well beyond the resampler's FIR
    // support around the isolated peak. Local FFT precision cannot mix them.
    let mut quiet = vec![0.0_f32; 8192];
    quiet[4096..].fill(1e-30);
    let reference = resample(&quiet, 8000, 16000).unwrap();
    let mut mixed = quiet;
    mixed[64] = 1e20;
    let actual = resample(&mixed, 8000, 16000).unwrap();
    let region = 12_000..14_000;
    assert!(reference[region.clone()].iter().all(|&v| v > 0.9e-30));
    let error = max_difference(&actual[region.clone()], &reference[region]);
    assert!(
        error < 1e-35,
        "Distant quiet-region maximum error {error:.6e}; clip-wide f32 normalization must not erase representable samples"
    );
}
