//! Public I/O contracts: lossless float output and consistent streamed blocks.
use dasp_rs::{
    io::{Decoder, export, stream, stream_lazy},
    types::AudioData,
};
use tempfile::NamedTempFile;

#[test]
fn float_export_preserves_headroom_and_negative_zero() {
    let samples = vec![1.5, -2.0, 0.0, -0.0, 0.125, f32::MIN_POSITIVE];
    let audio = AudioData::new(samples.clone(), 48_000, 1).unwrap();
    let file = NamedTempFile::new().unwrap();
    export(file.path(), &audio).unwrap();
    // Independent reader checks the file representation, not only our decoder.
    let decoded: Vec<_> = hound::WavReader::open(file.path())
        .unwrap()
        .samples::<f32>()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        samples.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(Decoder::new(file.path()).load().unwrap().samples, samples);
}

#[test]
fn lazy_and_collected_streams_match_across_real_packets() {
    let samples: Vec<_> = (0..20_000).map(|i| (i % 100) as f32 / 100.0).collect();
    let file = NamedTempFile::new().unwrap();
    export(
        file.path(),
        &AudioData::new(samples.clone(), 48_000, 2).unwrap(),
    )
    .unwrap();
    for (frame, hop, limit) in [(1024, 512, 50), (512, 1536, 50), (2048, 2048, 2)] {
        let eager = stream(file.path(), limit, frame, Some(hop)).unwrap();
        let lazy = stream_lazy(file.path(), limit, frame, Some(hop))
            .unwrap()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(eager, lazy);
        for (i, block) in eager.iter().enumerate() {
            let start = i * hop;
            let end = (start + frame).min(samples.len());
            assert_eq!(&block[..end - start], &samples[start..end]);
            assert!(block[end - start..].iter().all(|&v| v == 0.0));
        }
    }
}

#[test]
fn lazy_stream_reports_probe_errors_and_zero_limit_finishes() {
    let file = NamedTempFile::new().unwrap();
    std::fs::write(file.path(), b"not an audio container").unwrap();
    let rx = stream_lazy(file.path(), 2, 16, None).unwrap();
    assert!(rx.recv().unwrap().is_err());
    assert!(rx.recv().is_err());
    let rx = stream_lazy(file.path(), 0, 16, None).unwrap();
    assert!(rx.recv().is_err());
}

#[test]
fn truncated_wav_is_not_reported_as_successful_completion() {
    let file = NamedTempFile::new().unwrap();
    export(
        file.path(),
        &AudioData::new(vec![0.25; 20_000], 48_000, 2).unwrap(),
    )
    .unwrap();
    let original_len = file.as_file().metadata().unwrap().len();
    file.as_file().set_len(original_len - 4000).unwrap();
    assert!(
        Decoder::new(file.path()).load().is_err(),
        "truncated WAV loaded successfully"
    );
    assert!(stream(file.path(), usize::MAX, 512, None).is_err());
    let blocks: Vec<_> = stream_lazy(file.path(), usize::MAX, 512, None)
        .unwrap()
        .into_iter()
        .collect();
    assert!(
        blocks.iter().any(Result::is_ok),
        "expected successful blocks before failure"
    );
    assert!(
        matches!(blocks.last(), Some(Err(_))),
        "truncation must reach the receiver"
    );
    // A bounded prefix read need not inspect a damaged tail it never requests.
    assert_eq!(stream(file.path(), 1, 512, None).unwrap().len(), 1);
}
