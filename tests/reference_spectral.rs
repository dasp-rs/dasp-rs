//! Independent librosa 0.11.0/SciPy 1.18.1 references; regenerate with the pinned
//! generator in `fixtures/generate_spectral_reference.py` (conventions and URLs there).
//! Mixed tolerances accommodate f32 FFT/filter sums and cancellation near zero;
//! they remain far below a spectral bin, or a meaningful MFCC/flatness change.
//! Mel slope endpoints are computed in f32, warranting 8e-9 absolute weight error
//! and 8e-6 relative output error (65 bins summed). MFCC's 3e-5 absolute bound
//! covers cancellation of twelve equal log-floor values. Inverse DCT uses f64
//! intermediates and allows 1e-7 + 1e-7*|reference|. Centroid/bandwidth tolerate
//! sub-millihertz error including f32 STFT leakage; rolloff permits only rounding
//! of the same frequency bin. Delta permits 1e-7 + 3e-6*|reference| for f32 sums.
//! RMS and HPSS use pure relative bounds (1e-7 to 3e-7), so tiny expected results
//! cannot pass by collapsing to zero. Contrast uses 1e-7 + 1e-7*|reference| dB.
#[path = "common/reference.rs"]
mod reference;

use dasp_rs::{feat, types::AudioData};
use ndarray::{Array1, Array2, Axis};
use reference::{check_real, load, real, signal};

const FIXTURE: &str = include_str!("fixtures/spectral_reference.csv");
const CASES: [(&str, usize, f32, f32); 7] = [
    ("noise_even", 128, 0.0, 4000.0),
    ("noise_odd", 127, 73.0, 3200.0),
    ("tone_even", 128, 150.0, 3500.0),
    ("impulse_odd", 127, 0.0, 4000.0),
    ("quiet_even", 128, 0.0, 4000.0),
    ("scaled_odd", 127, 73.0, 3200.0),
    ("silence_even", 128, 0.0, 4000.0),
];

#[test]
fn mel_filters_and_spectrogram_match_librosa() {
    let data = load(FIXTURE);
    for (name, n_fft, fmin, fmax) in CASES {
        let audio = AudioData::new(signal(&data, &format!("{name}_input")), 8000, 1).unwrap();
        let identity = Array2::eye(n_fft / 2 + 1);
        let power = real(&data, &format!("{name}_power"));
        for (input, suffix, atol, rtol) in [
            (Some(&identity), "filters", 8e-9, 5e-6),
            (Some(&power), "mel", 2e-7, 8e-6),
            (None, "mel_audio", 2e-7, 8e-6),
        ] {
            let actual = feat::melspectrogram(
                &audio,
                input,
                Some(n_fft),
                Some(48),
                Some(12),
                Some(fmin),
                Some(fmax),
            )
            .unwrap();
            // Preserve a meaningful relative check on the 1e-24-power input.
            let atol = if name == "quiet_even" && suffix != "filters" {
                atol * 1e-24
            } else {
                atol
            };
            check_real(&format!("{name}_{suffix}"), &actual, &data, atol, rtol);
        }
    }
}

#[test]
fn mfcc_matches_scipy_natural_log_orthonormal_dct() {
    let data = load(FIXTURE);
    for (name, _, _, _) in CASES {
        let audio = AudioData::new(signal(&data, &format!("{name}_input")), 8000, 1).unwrap();
        let mel = real(&data, &format!("{name}_mel_input"));
        // Both API norm settings intentionally produce the orthonormal DCT.
        for norm in [None, Some("ortho")] {
            let actual = feat::mfcc(&audio, Some(&mel), Some(12), Some(2), norm).unwrap();
            check_real(&format!("{name}_mfcc"), &actual, &data, 3e-5, 2e-6);
        }
    }
}

fn row(array: Array1<f32>) -> Array2<f32> {
    array.insert_axis(Axis(0))
}

#[test]
fn centroid_and_bandwidth_match_librosa_at_all_scales() {
    let data = load(FIXTURE);
    for (name, n_fft, _, _) in CASES {
        let audio = AudioData::new(signal(&data, &format!("{name}_input")), 8000, 1).unwrap();
        let magnitude = real(&data, &format!("{name}_magnitude"));
        for input in [Some(&magnitude), None] {
            let actual = feat::spectral_centroid(&audio, input, Some(n_fft), Some(48)).unwrap();
            check_real(
                &format!("{name}_centroid"),
                &row(actual),
                &data,
                0.0002,
                1e-7,
            );
            for p in [1, 2, 3] {
                let actual =
                    feat::spectral_bandwidth(&audio, input, Some(n_fft), Some(48), Some(p))
                        .unwrap();
                check_real(
                    &format!("{name}_bandwidth{p}"),
                    &row(actual),
                    &data,
                    0.0005,
                    5e-7,
                );
            }
        }
    }
}

#[test]
fn rolloff_matches_librosa_bins() {
    let data = load(FIXTURE);
    for (name, n_fft, _, _) in CASES {
        let audio = AudioData::new(signal(&data, &format!("{name}_input")), 8000, 1).unwrap();
        let magnitude = real(&data, &format!("{name}_magnitude"));
        for input in [Some(&magnitude), None] {
            let actual =
                feat::spectral_rolloff(&audio, input, Some(n_fft), Some(48), Some(0.85)).unwrap();
            check_real(
                &format!("{name}_rolloff"),
                &row(actual),
                &data,
                0.0002,
                1e-7,
            );
        }
    }
}

#[test]
fn magnitude_flatness_matches_librosa_including_zeros() {
    let data = load(FIXTURE);
    for (name, n_fft, _, _) in CASES.into_iter().rev() {
        let audio = AudioData::new(signal(&data, &format!("{name}_input")), 8000, 1).unwrap();
        let magnitude = real(&data, &format!("{name}_magnitude"));
        for input in [Some(&magnitude), None] {
            let actual = feat::spectral_flatness(&audio, input, Some(n_fft), Some(48)).unwrap();
            // Geometric means amplify relative FFT error in a tone's near-zero
            // bins. Only this audio-to-feature path needs a 3e-5 absolute bound.
            let atol = if input.is_none() && name == "tone_even" {
                3e-5
            } else {
                1e-7
            };
            check_real(&format!("{name}_flatness"), &row(actual), &data, atol, 1e-7);
        }
    }
}

#[test]
fn delta_matches_librosa_nearest_edge_on_both_axes() {
    let data = load(FIXTURE);
    let input = real(&data, "delta_input");
    for axis in [0, 1, -1] {
        for width in [3, 5, 9] {
            let actual = feat::compute_delta(&input)
                .width(width)
                .axis(axis)
                .compute()
                .unwrap();
            let name = format!(
                "delta_axis{}_width{width}",
                if axis == -1 { 1 } else { axis }
            );
            check_real(&name, &actual, &data, 1e-7, 3e-6);
        }
    }
}

#[test]
fn inverse_mfcc_matches_scipy_all_dct_types_and_lengths() {
    let data = load(FIXTURE);
    let input = real(&data, "inverse_input");
    for kind in [1, 2, 3, 4] {
        for n_mels in [8, 5, 12] {
            let actual = feat::mfcc_to_mel(&input)
                .n_mels(n_mels)
                .dct_type(kind)
                .compute()
                .unwrap();
            check_real(
                &format!("inverse_type{kind}_mels{n_mels}"),
                &actual,
                &data,
                1e-7,
                1e-7,
            );
        }
    }
}

#[test]
fn invalid_flatness_inputs_and_type_one_singleton_are_rejected() {
    let audio = AudioData::new(vec![0.0; 128], 8000, 1).unwrap();
    for input in [
        Array2::zeros((0, 1)),
        Array2::zeros((65, 0)),
        Array2::ones((64, 2)),
        Array2::from_elem((65, 2), -1.0),
        Array2::from_elem((65, 2), f32::NAN),
        Array2::from_elem((65, 2), f32::INFINITY),
    ] {
        assert!(feat::spectral_flatness(&audio, Some(&input), Some(128), Some(48)).is_err());
    }
    assert!(
        feat::mfcc_to_mel(&Array2::ones((8, 4)))
            .n_mels(1)
            .dct_type(1)
            .compute()
            .is_err()
    );
}

#[test]
fn rms_matches_librosa_waveform_and_parseval_spectrum() {
    let data = load(FIXTURE);
    for (name, n_fft, _, _) in CASES {
        let audio = AudioData::new(signal(&data, &format!("{name}_input")), 8000, 1).unwrap();
        let magnitude = real(&data, &format!("{name}_magnitude"));
        for (input, suffix) in [(None, "audio"), (Some(&magnitude), "spectrum")] {
            let actual = feat::rms(&audio, input, Some(n_fft), Some(48)).unwrap();
            // Pure relative tolerance preserves sensitivity for quiet inputs.
            check_real(
                &format!("{name}_rms_{suffix}"),
                &row(actual),
                &data,
                0.0,
                2e-7,
            );
        }
    }
}

#[test]
fn rms_preserves_extreme_finite_amplitudes() {
    let data = load(FIXTURE);
    for name in ["tiny", "huge"] {
        let audio = AudioData::new(signal(&data, &format!("rms_{name}_input")), 8000, 1).unwrap();
        let actual = feat::rms(&audio, None, Some(128), Some(48)).unwrap();
        check_real(&format!("rms_{name}"), &row(actual), &data, 0.0, 1e-7);
    }
}

#[test]
fn contrast_matches_declared_octave_quantile_definition() {
    let data = load(FIXTURE);
    for (name, n_fft, _, _) in CASES {
        let audio = AudioData::new(signal(&data, &format!("{name}_input")), 8000, 1).unwrap();
        let magnitude = real(&data, &format!("{name}_magnitude"));
        let actual =
            feat::spectral_contrast(&audio, Some(&magnitude), Some(n_fft), Some(48), Some(4))
                .unwrap();
        check_real(&format!("{name}_contrast"), &actual, &data, 1e-7, 1e-7);
    }
}

#[test]
fn contrast_quantile_means_remain_finite_at_large_scale() {
    let data = load(FIXTURE);
    let audio = AudioData::new(vec![0.0; 128], 8000, 1).unwrap();
    let magnitude = real(&data, "contrast_extreme_input");
    let actual =
        feat::spectral_contrast(&audio, Some(&magnitude), Some(1024), Some(48), Some(4)).unwrap();
    check_real("contrast_extreme", &actual, &data, 1e-7, 1e-7);
}

#[test]
fn hpss_matches_scipy_clipped_medians_and_librosa_softmasks() {
    let data = load(FIXTURE);
    let audio = AudioData::new(vec![0.0; 128], 8000, 1).unwrap();
    for name in [
        "sparse", "mixed", "quiet", "loud", "huge", "ratio", "silence",
    ] {
        let spectrum = real(&data, &format!("hpss_{name}_input"));
        for (harm_win, perc_win) in [(3, 5), (31, 31)] {
            let (harmonic, percussive) = feat::hpss(
                &audio,
                Some(&spectrum),
                Some(16),
                Some(4),
                Some(harm_win),
                Some(perc_win),
            )
            .unwrap();
            let prefix = format!("hpss_{name}_{harm_win}_{perc_win}");
            // Scale-free tolerances catch lost energy at 1e-30 and overflow at 3e38.
            check_real(&format!("{prefix}_harmonic"), &harmonic, &data, 0.0, 3e-7);
            check_real(
                &format!("{prefix}_percussive"),
                &percussive,
                &data,
                0.0,
                3e-7,
            );
            check_real(
                &format!("hpss_{name}_input"),
                &(&harmonic + &percussive),
                &data,
                0.0,
                2e-7,
            );
        }
    }
}

#[test]
fn spectrum_rms_weights_dc_nyquist_and_interior_bins() {
    let data = load(FIXTURE);
    let audio = AudioData::new(vec![0.0; 128], 8000, 1).unwrap();
    for n_fft in [127, 128] {
        let magnitude = real(&data, &format!("rms_bins{n_fft}_input"));
        let actual = feat::rms(&audio, Some(&magnitude), Some(n_fft), Some(48)).unwrap();
        check_real(&format!("rms_bins{n_fft}"), &row(actual), &data, 0.0, 1e-7);
    }
}
