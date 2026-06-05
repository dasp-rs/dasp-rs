/// Converts multi-channel audio samples to mono by averaging across channels.
///
/// # Arguments
/// * `samples` - Interleaved audio samples (e.g., [L1, R1, L2, R2, ...] for stereo)
/// * `channels` - Number of channels in the input samples
///
/// # Returns
/// Returns a `Vec<f32>` containing the mono audio signal, where each sample is the average
/// of the corresponding samples across all channels.
///
/// # Panics
/// Does not explicitly panic, but if `samples.len()` is not a multiple of `channels`,
/// the last incomplete chunk will be averaged over fewer samples, potentially leading
/// to unexpected results.
///
/// # Examples
/// ```
/// let stereo = vec![0.5, 0.7, 0.3, 0.9]; // [L1, R1, L2, R2]
/// let mono = to_mono(&stereo, 2);
/// assert_eq!(mono, vec![0.6, 0.6]); // [(0.5 + 0.7)/2, (0.3 + 0.9)/2]
/// ```
pub fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    let mut mono = Vec::with_capacity(samples.len() / channels);
    for chunk in samples.chunks(channels) {
        let sum: f32 = chunk.iter().sum();
        mono.push(sum / chunk.len() as f32);
    }
    mono
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_stereo_frames() {
        let stereo = vec![0.5, 0.7, 0.3, 0.9];
        let mono = to_mono(&stereo, 2);
        assert_eq!(mono, vec![0.6, 0.6]);
    }

    #[test]
    fn handles_single_channel_no_copy_needed() {
        let mono_in = vec![0.1, -0.1, 0.2];
        let mono_out = to_mono(&mono_in, 1);
        assert_eq!(mono_out, mono_in);
    }

    #[test]
    fn ignores_incomplete_final_chunk_gracefully() {
        let data = vec![1.0, 3.0, 5.0]; // channels=2 leaves last sample alone in chunk of len 1
        let mono = to_mono(&data, 2);
        // First frame average, second frame uses single value
        assert_eq!(mono, vec![2.0, 5.0]);
    }
}