//! Structural segmentation: recurrence matrices, agglomerative clustering,
//! path enhancement, and time-lag filtering.
//!
//! Run with: `cargo run --example segmentation`

use dasp_rs::{feat, generate::tone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sr = 22050_u32;
    let y = tone(440.0, sr).duration(4.0).compute();

    // Compute chroma as the feature matrix (12 × n_frames).
    let chroma = feat::spectral(&y, sr).hop_length(512).chroma_cqt()?;
    println!("Feature matrix: {:?}", chroma.shape());

    // ── Recurrence matrix ─────────────────────────────────────────────────────
    // Self-similarity matrix — high values where two frames sound alike.
    // SimilarityMetric: Cosine (default) | Euclidean | Manhattan
    // RecurrenceMode:   Affinity (default) | Binary
    use dasp_rs::feat::{RecurrenceMode, SimilarityMetric};
    let r = feat::recurrence_matrix(&chroma)
        .metric(SimilarityMetric::Cosine)
        .mode(RecurrenceMode::Affinity)
        .compute();
    println!("Recurrence matrix: {:?}", r.shape());

    // ── Cross-similarity ──────────────────────────────────────────────────────
    // Pairwise similarity between two different feature sequences.
    let chroma2 = feat::spectral(&y, sr).hop_length(256).chroma_cqt()?;
    let c = feat::cross_similarity(&chroma, &chroma2)
        .metric(SimilarityMetric::Cosine)
        .compute();
    println!("Cross-similarity:  {:?}", c.shape());

    // ── Path enhancement ──────────────────────────────────────────────────────
    // Blurs along the main diagonal to reinforce repeated segment bands.
    let r_enhanced = feat::path_enhance(&r, 11).compute();
    println!("Path enhanced:     {:?}", r_enhanced.shape());

    // ── Time-lag filter ───────────────────────────────────────────────────────
    // Applies a median filter in the time-lag domain to remove artefacts.
    let r_filtered = feat::timelag_filter(&r).n_window(11).compute();
    println!("Time-lag filtered: {:?}", r_filtered.shape());

    // ── Agglomerative segmentation ────────────────────────────────────────────
    // Merge contiguous frames into k segments by centroid similarity.
    let k = 4_usize;
    let seg_labels = feat::agglomerative(&chroma, k).compute();
    println!("Agglomerative ({k} segs): {} frame labels", seg_labels.len());
    println!("  Unique segments: {:?}", unique(&seg_labels));

    // ── Subsegmentation ───────────────────────────────────────────────────────
    // Given a coarse segmentation boundary list, subdivide each segment.
    // Here we create simple 4-boundary markers and sub-divide each segment.
    // boundaries: frame indices that delimit coarse segments; k: sub-segments per segment.
    let boundaries = vec![chroma.shape()[1] / 4, chroma.shape()[1] / 2];
    let sub_labels = feat::subsegment(&chroma, &boundaries, 2).compute();
    println!("Subsegmented ({} boundaries → {} sub-boundary indices)", boundaries.len(), sub_labels.len());

    // ── Beat synchronisation ──────────────────────────────────────────────────
    // Aggregate feature frames to beat-aligned frames.
    let (_, beat_frames) = feat::beat_track(&y, sr).compute()?;
    let sync_chroma = feat::beat_sync(&chroma, &beat_frames).compute();
    println!("Beat-sync chroma:  {:?} (12 × n_beats)", sync_chroma.shape());

    Ok(())
}

fn unique(v: &[usize]) -> Vec<usize> {
    let mut s: Vec<usize> = v.to_vec();
    s.sort_unstable();
    s.dedup();
    s
}
