//! Tuning is measured in cents relative to the nearest equal-tempered semitone.
use dasp_rs::pitch::{estimate_tuning, piptrack, pitch_tuning};
use ndarray::Array2;

#[test]
fn pitch_tuning_retains_signed_cent_offsets_and_resolution() {
    for cents in [-40.0_f32, -20.0, 0.0, 15.0, 35.0] {
        let frequencies: Vec<f32> = [220.0, 440.0, 880.0]
            .iter()
            .map(|f| f * 2.0_f32.powf(cents / 1200.0))
            .collect();
        assert!((pitch_tuning(&frequencies, None).unwrap() - cents).abs() < 0.01);
    }
    let frequency = 440.0 * 2.0_f32.powf(12.3 / 1200.0);
    assert!((pitch_tuning(&[frequency], Some(5.0)).unwrap() - 10.0).abs() < 0.01);
    for invalid in [f32::NAN, f32::INFINITY] {
        assert!(pitch_tuning(&[440.0], Some(invalid)).is_err());
        assert!(pitch_tuning(&[invalid], None).is_err());
    }
}

#[test]
fn tuning_builder_accepts_spectrogram_and_reports_flat_pitch_as_negative() {
    let n_fft = 1024;
    let sr = 8000;
    let mut s = Array2::zeros((n_fft / 2 + 1, 3));
    s.row_mut(55).fill(1.0); // 429.6875 Hz, about 41 cents flat of A4.
    let actual = estimate_tuning(&[])
        .sample_rate(sr)
        .n_fft(n_fft)
        .spectrogram(&s)
        .compute()
        .unwrap();
    let expected = 1200.0 * (429.6875_f32 / 440.0).log2();
    assert!((actual - expected).abs() < 0.01, "{actual} != {expected}");
}

#[test]
fn peak_tracking_is_scale_invariant_and_does_not_shift_boundary_bins() {
    for scale in [1e-25, 1.0, 1e25] {
        for bin in [0, 1, 4] {
            let mut s = Array2::zeros((5, 1));
            s[[bin, 0]] = scale;
            let (pitch, magnitude) = piptrack(&[])
                .sample_rate(8000)
                .n_fft(8)
                .spectrogram(&s)
                .compute()
                .unwrap();
            assert!((pitch[[bin, 0]] - 1000.0 * bin as f32).abs() < 1e-5);
            assert_eq!(magnitude[[bin, 0]], scale);
        }
    }
}

#[test]
fn peak_tracking_validates_supplied_spectra_and_analysis_parameters() {
    let s = Array2::zeros((5, 1));
    assert!(
        piptrack(&[])
            .n_fft(usize::MAX)
            .spectrogram(&s)
            .compute()
            .is_err()
    );
    assert!(
        piptrack(&[])
            .sample_rate(0)
            .n_fft(8)
            .spectrogram(&s)
            .compute()
            .is_err()
    );
    assert!(
        piptrack(&[])
            .n_fft(8)
            .hop_length(0)
            .spectrogram(&s)
            .compute()
            .is_err()
    );
    assert!(
        piptrack(&[])
            .n_fft(8)
            .spectrogram(&Array2::zeros((5, 0)))
            .compute()
            .is_err()
    );
    for invalid in [f32::NAN, f32::INFINITY, -1.0] {
        assert!(
            piptrack(&[])
                .n_fft(8)
                .spectrogram(&Array2::from_elem((5, 1), invalid))
                .compute()
                .is_err()
        );
    }
}
