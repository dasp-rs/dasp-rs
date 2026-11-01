//! Independent librosa/SciPy references. Regeneration and numerical conventions
//! are documented in `fixtures/generate_transforms_reference.py`; Python is not
//! required to run these tests. Tolerances reflect f32 FFT/window roundoff rather
//! than bitwise equality with the float64 reference implementation.
#![allow(missing_docs)]
#[path = "common/reference.rs"]
mod reference;

use dasp_rs::feat::phase_vocoder;
use dasp_rs::proc::{Window, istft, resample, stft, time_stretch};
use ndarray::Array2;
use num_complex::Complex;
use reference::{References, check_complex, check_real, load, signal};

fn references() -> References {
    load(include_str!("fixtures/transforms_reference.csv"))
}

fn spectrum(data: &References, name: &str) -> Array2<Complex<f32>> {
    data[name].mapv(|z| Complex::new(z.re as f32, z.im as f32))
}

fn vector(y: Vec<f32>) -> Array2<f32> {
    Array2::from_shape_vec((1, y.len()), y).unwrap()
}

#[test]
fn stft_matches_librosa_periodic_windows_and_frame_conventions() {
    let data = references();
    let y = signal(&data, "input");
    for (name, fft, hop, win, window, center) in [
        ("even", 64, 16, 64, Window::Hann, true),
        ("odd", 63, 13, 63, Window::Hann, true),
        ("short_window", 64, 11, 39, Window::Hamming, true),
        ("noncenter", 64, 16, 64, Window::Hamming, false),
        ("odd_noncenter", 63, 13, 33, Window::Hann, false),
    ] {
        let actual = stft(&y)
            .n_fft(fft)
            .hop_length(hop)
            .win_length(win)
            .window(window)
            .center(center)
            .compute()
            .unwrap();
        check_complex(&format!("stft_{name}"), &actual, &data, 3e-6, 3e-6);
    }
    let short = signal(&data, "input_short");
    let actual = stft(&short).n_fft(64).hop_length(8).compute().unwrap();
    check_complex("stft_short_signal", &actual, &data, 3e-6, 3e-6);
}

#[test]
fn istft_matches_independent_spectra_not_only_roundtrips() {
    let data = references();
    for (name, fft, hop, win, window, center) in [
        ("even", 64, 16, 64, Window::Hann, true),
        ("odd", 63, 13, 63, Window::Hann, true),
        ("short_window", 64, 11, 39, Window::Hamming, true),
        ("noncenter", 64, 16, 64, Window::Hamming, false),
        ("odd_noncenter", 63, 13, 33, Window::Hann, false),
    ] {
        let d = spectrum(&data, &format!("spectrum_{name}"));
        for (suffix, length) in [("natural", None), ("length", Some(257))] {
            let mut builder = istft(&d)
                .n_fft(fft)
                .hop_length(hop)
                .win_length(win)
                .window(window)
                .center(center);
            if let Some(length) = length {
                builder = builder.length(length);
            }
            let actual = vector(builder.compute().unwrap());
            // Noncentered small Hann endpoint weights amplify f32 FFT noise.
            check_real(
                &format!("istft_{name}_{suffix}"),
                &actual,
                &data,
                2e-5,
                2e-5,
            );
        }
    }
}

#[test]
fn istft_normalizes_small_nonzero_window_weights() {
    let data = references();
    let d = spectrum(&data, "spectrum_large");
    let actual = vector(
        istft(&d)
            .n_fft(2048)
            .hop_length(512)
            .center(false)
            .compute()
            .unwrap(),
    );
    // A single Hann frame is ill-conditioned at its ends: 1/w amplification
    // reaches 4.25e5. The 0.03 boundary tolerance follows f32 error amplification;
    // the interior is checked separately at ordinary FFT precision below.
    check_real("istft_large", &actual, &data, 0.03, 1e-5);
    let expected = &data["istft_large"];
    for i in 32..2016 {
        let error = (f64::from(actual[[0, i]]) - expected[[0, i]].re).abs();
        assert!(error < 5e-5, "interior sample {i}: {error}");
    }
}

#[test]
fn phase_vocoder_and_time_stretch_match_librosa_at_multiple_rates() {
    let data = references();
    let y = signal(&data, "input");
    for (name, fft, hop) in [("even", 64, 16), ("odd", 63, 13)] {
        let d = spectrum(&data, &format!("spectrum_{name}"));
        for (tag, rate) in [("0_75", 0.75), ("1_3", 1.3), ("2_0", 2.0)] {
            let actual = phase_vocoder(&d, rate, Some(hop), Some(fft)).unwrap();
            // f32 atan2 and magnitudes, with f64 accumulation, accumulate phase
            // error across output frames; inputs and reference use identical D.
            check_complex(&format!("pv_{name}_{tag}"), &actual, &data, 3e-5, 3e-5);
            let mut stretched = time_stretch(&y, rate)
                .n_fft(fft)
                .hop_length(hop)
                .compute()
                .unwrap();
            // Rust's documented round uses ties away from zero; Python rounds
            // 257/2 to the even 128. Compare all shared samples and explicitly
            // check this intentional one-sample length convention.
            if tag == "2_0" {
                assert_eq!(stretched.len(), 129);
                stretched.truncate(128);
            }
            let actual = vector(stretched);
            check_real(&format!("stretch_{name}_{tag}"), &actual, &data, 3e-5, 3e-5);
        }
    }
}

fn region_error(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    range: impl Iterator<Item = usize>,
    max_bound: f64,
    rms_bound: f64,
) {
    let mut max = 0.0_f64;
    let mut square = 0.0;
    let mut count = 0;
    for i in range {
        assert!(actual[i].is_finite());
        let error = f64::from(actual[i]) - f64::from(expected[i]);
        max = max.max(error.abs());
        square += error * error;
        count += 1;
    }
    let rms = (square / f64::from(count)).sqrt();
    eprintln!(
        "REFERENCE {name}: count={count} max_abs={max:.6e} rmse={rms:.6e} max_bound={max_bound:.3e} rms_bound={rms_bound:.3e}"
    );
    assert!(
        max <= max_bound && rms <= rms_bound,
        "{name}: max={max}, rms={rms}"
    );
}

#[test]
fn resampling_matches_scipy_passband_timing_and_stopband_metrics() {
    let data = references();
    // SciPy's Kaiser FIR and rubato's FFT filter differ in transition width and
    // ringing. Low-frequency passband/interior agreement is required to 0.2%
    // of full scale; a separately reported 3% boundary budget accommodates the
    // differing zero-extension responses. Exclude 128 output samples per end,
    // far beyond SciPy's FIR transient for these ratios and low-frequency inputs.
    for (name, orig, target) in [
        ("tone", 8000, 12000),
        ("chirp", 12000, 8000),
        ("multitone", 44100, 16000),
        ("stopband", 48000, 16000),
    ] {
        let y = signal(&data, &format!("resample_input_{name}"));
        let expected = signal(&data, &format!("resample_{name}"));
        let actual = resample(&y, orig, target).unwrap();
        assert_eq!(actual.len(), expected.len(), "resample {name} length");
        let end = actual.len() - 128;
        region_error(
            &format!("resample_{name}_interior"),
            &actual,
            &expected,
            128..end,
            0.002,
            0.001,
        );
        region_error(
            &format!("resample_{name}_boundary"),
            &actual,
            &expected,
            (0..128).chain(end..actual.len()),
            0.03,
            0.01,
        );
        if name == "stopband" {
            // At 17.76kHz into a 16kHz stream both filters must reject aliasing.
            let rms = (actual[128..end]
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                / (end - 128) as f64)
                .sqrt();
            assert!(rms < 0.001, "stopband RMS {rms}");
        }
    }
    let impulse = signal(&data, "resample_input_impulse");
    let actual = resample(&impulse, 8000, 12000).unwrap();
    let expected = signal(&data, "resample_impulse");
    assert_eq!(actual.len(), expected.len());
    let peak = actual
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap()
        .0;
    assert_eq!(peak, 192, "resampler must remove filter delay");
    let mass: f64 = actual.iter().map(|&v| f64::from(v)).sum();
    assert!((mass - 1.5).abs() < 0.005, "impulse DC gain: {mass}");
    // A broadband impulse excites the differing transition bands. Permit 5%
    // of input energy in the filter difference, and separately require close
    // passband frequency responses so this cannot conceal timing/gain errors.
    let energy: f64 = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    eprintln!("REFERENCE resample_impulse: squared_error={energy:.6e} mass={mass:.6e} peak={peak}");
    assert!(energy < 0.05, "impulse error energy: {energy}");
    for frequency in [0.0, 0.05, 0.1, 0.2] {
        let response = |samples: &[f32]| -> Complex<f64> {
            samples
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    Complex::from_polar(f64::from(v), -std::f64::consts::TAU * frequency * i as f64)
                })
                .sum()
        };
        let error = (response(&actual) - response(&expected)).norm();
        assert!(
            error < 0.003,
            "impulse passband {frequency}: complex response error={error}"
        );
    }
}
