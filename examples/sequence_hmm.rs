//! Sequence analysis and HMM decoding: DTW, Viterbi, and transition matrices.
//!
//! Run with: `cargo run --example sequence_hmm`

use dasp_rs::feat;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Dynamic Time Warping ──────────────────────────────────────────────────
    // Align two feature sequences of potentially different lengths.
    use dasp_rs::feat::DtwMetric;
    let x = Array2::from_shape_fn((4, 10), |(i, j)| (i + j) as f32);
    let y = Array2::from_shape_fn((4, 8),  |(i, j)| (i + j) as f32 + 0.5);

    let (cost, path) = feat::dtw(&x, &y)
        .metric(DtwMetric::Euclidean)
        .compute()?;
    println!("DTW cost:    {cost:.3}");
    println!("DTW path:    {} steps (first 5: {:?})", path.len(), &path[..path.len().min(5)]);

    // ── Transition matrices ───────────────────────────────────────────────────
    // All return log-probability matrices ready for viterbi().
    let n = 4_usize;

    // Self-loop: stay in same state with high prob, jump randomly otherwise.
    let t_loop    = feat::transition_loop(n, 0.9);
    // Local: near-diagonal Gaussian bandwidth of ±width states.
    let t_local   = feat::transition_local(n, 2);
    // Uniform: every transition equally likely.
    let t_uniform = feat::transition_uniform(n);
    // Cyclic: advance to next state (wraps n-1 → 0).
    let t_cycle   = feat::transition_cycle(n, 0.8);
    // Acyclic: strictly left-to-right, absorbing at the end.
    let t_acyclic = feat::transition_acyclic(n, 0.3);
    println!("Transition matrices (log-prob) for {n} states:");
    println!("  loop:    {:?}", t_loop.shape());
    println!("  local:   {:?}", t_local.shape());
    println!("  uniform: {:?}", t_uniform.shape());
    println!("  cycle:   {:?}", t_cycle.shape());
    println!("  acyclic: {:?}", t_acyclic.shape());

    // ── Viterbi decoding ──────────────────────────────────────────────────────
    // log_prob shape: (n_states, n_frames) — log P(observation | state)
    let n_frames = 20_usize;
    let log_prob = Array2::from_shape_fn((n, n_frames), |(s, t)| {
        // State 0 active in first half, state 2 active in second half.
        if s == 0 && t < n_frames / 2 { -0.1 }
        else if s == 2 && t >= n_frames / 2 { -0.1 }
        else { -5.0 }
    });

    let (ll, states) = feat::viterbi(&log_prob, &t_loop)?;
    println!("Viterbi (loop):    ll={ll:.2}, states[..5]={:?}", &states[..5]);

    let (ll2, states2) = feat::viterbi(&log_prob, &t_acyclic)?;
    println!("Viterbi (acyclic): ll={ll2:.2}, states[..5]={:?}", &states2[..5]);

    // ── Discriminative Viterbi ────────────────────────────────────────────────
    // Takes POSTERIOR probabilities P(state|obs), not log-likelihoods.
    let posteriors = Array2::from_shape_fn((n, n_frames), |(s, t)| {
        if s == 0 && t < n_frames / 2 { 0.8 }
        else if s == 2 && t >= n_frames / 2 { 0.8 }
        else { 0.05 }
    });
    let (ll3, states3) = feat::viterbi_discriminative(&posteriors, &t_loop, None)?;
    println!("Viterbi disc:      ll={ll3:.2}, states[..5]={:?}", &states3[..5]);

    // ── Binary Viterbi ────────────────────────────────────────────────────────
    // Specialised 2-state decoder that takes P(state=1 | obs) as a plain slice.
    let log_t2 = feat::transition_loop(2, 0.9);
    let prob1: Vec<f32> = (0..n_frames).map(|t| if t < n_frames / 2 { 0.1 } else { 0.9 }).collect();
    let (ll4, states4) = feat::viterbi_binary(&prob1, &log_t2)?;
    println!("Viterbi binary:    ll={ll4:.2}, state[0]={}, state[last]={}", states4[0], states4[n_frames - 1]);

    Ok(())
}
