use ndarray::Array1;
use crate::utils::notation;

/// Converts frequencies in Hz to Western musical note names.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
///
/// # Returns
/// Returns a `Vec<String>` containing note names (e.g., "C4", "G#5").
///
/// # Examples
/// ```
/// let freqs = vec![261.63, 329.63];
/// let notes = hz_to_note(&freqs);
/// assert_eq!(notes, vec!["C4", "E4"]);
/// ```
pub fn hz_to_note(frequencies: &[f32]) -> Vec<String> {
    frequencies.iter().map(|&f| {
        let midi = hz_to_midi(&[f])[0];
        midi_to_note(&[midi], None, None, None)[0].clone()
    }).collect()
}

/// Converts frequencies in Hz to MIDI note numbers.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
///
/// # Returns
/// Returns a `Vec<f32>` containing MIDI note numbers (A4 = 440 Hz = MIDI 69).
///
/// # Examples
/// ```
/// let freqs = vec![440.0];
/// let midi = hz_to_midi(&freqs);
/// assert_eq!(midi, vec![69.0]);
/// ```
pub fn hz_to_midi(frequencies: &[f32]) -> Vec<f32> {
    frequencies.iter().map(|&f| 12.0 * (f / 440.0).log2() + 69.0).collect()
}

/// Converts frequencies in Hz to Hindustani svara notation.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `Sa` - Frequency of the tonic (Sa) in Hz
/// * `abbr` - Optional flag for abbreviated notation (defaults to false)
///
/// # Returns
/// Returns a `Vec<String>` containing Hindustani svara names (e.g., "S", "R1" or "Shadjam", "Shuddha Rishabham").
///
/// # Examples
/// ```
/// let freqs = vec![261.63, 293.66];
/// let svaras = hz_to_svara_h(&freqs, 261.63, Some(true));
/// assert_eq!(svaras, vec!["S", "R2"]);
/// ```
pub fn hz_to_svara_h(frequencies: &[f32], sa: f32, abbr: Option<bool>) -> Vec<String> {
    let abbr = abbr.unwrap_or(false);
    let midi_sa = hz_to_midi(&[sa])[0];
    let midi_notes = hz_to_midi(frequencies);
    let svara_names = if abbr {
        vec!["S", "R1", "R2", "G1", "G2", "M1", "M2", "P", "D1", "D2", "N1", "N2"]
    } else {
        vec!["Shadjam", "Shuddha Rishabham", "Chatushruti Rishabham",
             "Shuddha Gandharam", "Sadharana Gandharam", "Shuddha Madhyamam",
             "Prati Madhyamam", "Panchamam", "Shuddha Dhaivatam", "Chatushruti Dhaivatam",
             "Shuddha Nishadam", "Kaisiki Nishadam"]
    };
    midi_notes.iter().map(|&m| {
        let degree = ((m - midi_sa + 0.5).round() as i32 % 12 + 12) % 12;
        svara_names[degree as usize].to_string()
    }).collect()
}

/// Converts frequencies in Hz to Carnatic svara notation based on a melakarta raga.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `Sa` - Frequency of the tonic (Sa) in Hz
/// * `mela` - Optional melakarta raga index (1-72, defaults to 29, Dheerashankarabharanam)
///
/// # Returns
/// Returns a `Vec<String>` containing Carnatic svara names (e.g., "S", "R2").
///
/// # Examples
/// ```
/// let freqs = vec![261.63, 293.66];
/// let svaras = hz_to_svara_c(&freqs, 261.63, None);
/// assert_eq!(svaras, vec!["S", "R2"]);
/// ```
pub fn hz_to_svara_c(frequencies: &[f32], sa: f32, mela: Option<usize>) -> Vec<String> {
    let mela = mela.unwrap_or(29);
    let degrees = notation::mela_to_degrees(mela);
    let midi_sa = hz_to_midi(&[sa])[0];
    let midi_notes = hz_to_midi(frequencies);
    midi_notes.iter().map(|&m| {
        let semitone = ((m - midi_sa + 0.5).round() as i32 % 12 + 12) % 12;
        let idx = degrees.iter().position(|&d| d == semitone as usize).unwrap_or(0);
        let base = match idx {
            0 => "S", 1..=3 => "R", 4..=6 => "G", 7 => "M", 8 => "P", 9..=11 => "D", 12..=14 => "N", _ => "S",
        };
        let variant = match degrees[idx] % 12 {
            1 => "1", 2 => "2", 3 => "3", 5 => "1", 6 => "2", 7 => "3", 8 => "1", 9 => "2", 10 => "3", _ => "",
        };
        format!("{}{}", base, variant)
    }).collect()
}

/// Converts frequencies in Hz to Functional Just System (FJS) notation.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `fmin` - Optional minimum frequency (defaults to 16.35 Hz, C0)
/// * `unison` - Optional unison interval ratio (defaults to 1.0)
///
/// # Returns
/// Returns a `Vec<String>` containing FJS note names (e.g., "C4 1/1").
///
/// # Examples
/// ```
/// let freqs = vec![261.63];
/// let fjs = hz_to_fjs(&freqs, None, None);
/// assert_eq!(fjs, vec!["C4 1/1"]);
/// ```
pub fn hz_to_fjs(frequencies: &[f32], fmin: Option<f32>, unison: Option<f32>) -> Vec<String> {
    let fmin = fmin.unwrap_or(16.35);
    let unison = unison.unwrap_or(1.0);
    frequencies.iter().map(|&f| {
        let octaves = (f / fmin).log2().floor();
        let interval = f / (fmin * 2.0f32.powf(octaves)) / unison;
        let ratio = notation::interval_to_fjs(interval, Some(1.0));
        format!("C{} {}", octaves as i32, ratio)
    }).collect()
}

/// Converts MIDI note numbers to frequencies in Hz.
///
/// # Arguments
/// * `notes` - Array of MIDI note numbers
///
/// # Returns
/// Returns a `Vec<f32>` containing frequencies in Hz (A4 = MIDI 69 = 440 Hz).
///
/// # Examples
/// ```
/// let midi = vec![69.0];
/// let freqs = midi_to_hz(&midi);
/// assert_eq!(freqs, vec![440.0]);
/// ```
pub fn midi_to_hz(notes: &[f32]) -> Vec<f32> {
    notes.iter().map(|&n| 440.0 * 2.0f32.powf((n - 69.0) / 12.0)).collect()
}

/// Converts MIDI note numbers to Western musical note names.
///
/// # Arguments
/// * `midi` - Array of MIDI note numbers
/// * `octave` - Optional flag to include octave number (defaults to true)
/// * `_cents` - Optional flag for cents (unused, defaults to None)
/// * `_key` - Optional key signature (unused, defaults to None)
///
/// # Returns
/// Returns a `Vec<String>` containing note names (e.g., "C4", "G#").
///
/// # Examples
/// ```
/// let midi = vec![60.0, 61.0];
/// let notes = midi_to_note(&midi, None, None, None);
/// assert_eq!(notes, vec!["C4", "C#4"]);
/// ```
pub fn midi_to_note(midi: &[f32], octave: Option<bool>, _cents: Option<bool>, _key: Option<&str>) -> Vec<String> {
    let note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    midi.iter().map(|&m| {
        let note_idx = (m.round() as usize) % 12;
        let oct = if octave.unwrap_or(true) { format!("{}", (m.round() as i32 - 12) / 12) } else { "".to_string() };
        format!("{}{}", note_names[note_idx], oct)
    }).collect()
}

/// Converts MIDI note numbers to Hindustani svara notation.
///
/// # Arguments
/// * `midi` - Array of MIDI note numbers
/// * `Sa` - Frequency of the tonic (Sa) in Hz
/// * `abbr` - Optional flag for abbreviated notation (defaults to false)
/// * `octave` - Optional flag to include octave number (defaults to false)
///
/// # Returns
/// Returns a `Vec<String>` containing Hindustani svara names with optional octave.
///
/// # Examples
/// ```
/// let midi = vec![60.0, 62.0];
/// let svaras = midi_to_svara_h(&midi, 261.63, Some(true), None);
/// assert_eq!(svaras, vec!["S", "R2"]);
/// ```
pub fn midi_to_svara_h(midi: &[f32], sa: f32, abbr: Option<bool>, octave: Option<bool>) -> Vec<String> {
    let abbr = abbr.unwrap_or(false);
    let octave = octave.unwrap_or(false);
    let midi_sa = hz_to_midi(&[sa])[0];
    let svara_names = if abbr {
        vec!["S", "R1", "R2", "G1", "G2", "M1", "M2", "P", "D1", "D2", "N1", "N2"]
    } else {
        vec!["Shadjam", "Shuddha Rishabham", "Chatushruti Rishabham",
             "Shuddha Gandharam", "Sadharana Gandharam", "Shuddha Madhyamam",
             "Prati Madhyamam", "Panchamam", "Shuddha Dhaivatam", "Chatushruti Dhaivatam",
             "Shuddha Nishadam", "Kaisiki Nishadam"]
    };
    midi.iter().map(|&m| {
        let degree = ((m - midi_sa + 0.5).round() as i32 % 12 + 12) % 12;
        let oct = if octave { format!("{}", (m - midi_sa).round() as i32 / 12) } else { "".to_string() };
        format!("{}{}", svara_names[degree as usize], oct)
    }).collect()
}

/// Converts MIDI note numbers to Carnatic svara notation based on a melakarta raga.
///
/// # Arguments
/// * `midi` - Array of MIDI note numbers
/// * `Sa` - Frequency of the tonic (Sa) in Hz
/// * `mela` - Optional melakarta raga index (1-72, defaults to 29)
/// * `abbr` - Optional flag for abbreviated notation (defaults to false)
///
/// # Returns
/// Returns a `Vec<String>` containing Carnatic svara names.
///
/// # Examples
/// ```
/// let midi = vec![60.0, 62.0];
/// let svaras = midi_to_svara_c(&midi, 261.63, None, Some(true));
/// assert_eq!(svaras, vec!["S", "R2"]);
/// ```
pub fn midi_to_svara_c(midi: &[f32], sa: f32, mela: Option<usize>, abbr: Option<bool>) -> Vec<String> {
    let mela = mela.unwrap_or(29);
    let abbr = abbr.unwrap_or(false);
    let degrees = notation::mela_to_degrees(mela);
    let midi_sa = hz_to_midi(&[sa])[0];
    midi.iter().map(|&m| {
        let semitone = ((m - midi_sa + 0.5).round() as i32 % 12 + 12) % 12;
        let idx = degrees.iter().position(|&d| d == semitone as usize).unwrap_or(0);
        let base = match idx {
            0 => "S", 1..=3 => "R", 4..=6 => "G", 7 => "M", 8 => "P", 9..=11 => "D", 12..=14 => "N", _ => "S",
        };
        let variant = match degrees[idx] % 12 {
            1 => "1", 2 => "2", 3 => "3", 5 => "1", 6 => "2", 7 => "3", 8 => "1", 9 => "2", 10 => "3", _ => "",
        };
        if abbr { format!("{}{}", base, variant) } else { notation::mela_to_svara(mela, Some(false), Some(false))[idx].clone() }
    }).collect()
}

/// Converts note names to frequencies in Hz.
///
/// # Arguments
/// * `note` - Array of note names (e.g., "C4", "G#5")
///
/// # Returns
/// Returns a `Vec<f32>` containing frequencies in Hz.
///
/// # Examples
/// ```
/// let notes = vec!["C4", "E4"];
/// let freqs = note_to_hz(&notes);
/// assert!(freqs[0] > 261.0 && freqs[0] < 262.0);
/// ```
pub fn note_to_hz(note: &[&str]) -> Vec<f32> {
    note.iter().map(|&n| {
        let midi = note_to_midi(&[n], None)[0];
        midi_to_hz(&[midi])[0]
    }).collect()
}

/// Converts note names to MIDI note numbers.
///
/// # Arguments
/// * `note` - Array of note names (e.g., "C4", "G#5")
/// * `round_midi` - Optional flag to round MIDI numbers (defaults to true)
///
/// # Returns
/// Returns a `Vec<f32>` containing MIDI note numbers.
///
/// # Examples
/// ```
/// let notes = vec!["C4", "C#4"];
/// let midi = note_to_midi(&notes, None);
/// assert_eq!(midi, vec![60.0, 61.0]);
/// ```
pub fn note_to_midi(note: &[&str], round_midi: Option<bool>) -> Vec<f32> {
    let note_map = [("C", 0), ("C#", 1), ("Db", 1), ("D", 2), ("D#", 3), ("Eb", 3), ("E", 4), ("F", 5), ("F#", 6), ("Gb", 6), ("G", 7), ("G#", 8), ("Ab", 8), ("A", 9), ("A#", 10), ("Bb", 10), ("B", 11)];
    note.iter().map(|&n| {
        let (note_part, octave_part) = n.split_at(n.find(|c: char| c.is_ascii_digit()).unwrap_or(n.len()));
        let note_val = note_map.iter().find(|&&(name, _)| name == note_part).map(|&(_, val)| val).unwrap_or(0) as f32;
        let octave = octave_part.parse::<i32>().unwrap_or(4);
        let midi = note_val + (octave + 1) as f32 * 12.0;
        if round_midi.unwrap_or(true) { midi.round() } else { midi }
    }).collect()
}

/// Converts note names to Hindustani svara notation.
///
/// # Arguments
/// * `notes` - Array of note names (e.g., "C4", "D4")
/// * `Sa` - Frequency of the tonic (Sa) in Hz
/// * `abbr` - Optional flag for abbreviated notation (defaults to false)
///
/// # Returns
/// Returns a `Vec<String>` containing Hindustani svara names.
///
/// # Examples
/// ```
/// let notes = vec!["C4", "D4"];
/// let svaras = note_to_svara_h(&notes, 261.63, Some(true));
/// assert_eq!(svaras, vec!["S", "R2"]);
/// ```
pub fn note_to_svara_h(notes: &[&str], sa: f32, abbr: Option<bool>) -> Vec<String> {
    let midi = note_to_midi(notes, Some(true));
    hz_to_svara_h(&midi_to_hz(&midi), sa, abbr)
}

/// Converts note names to Carnatic svara notation.
///
/// # Arguments
/// * `notes` - Array of note names (e.g., "C4", "D4")
/// * `Sa` - Frequency of the tonic (Sa) in Hz
/// * `mela` - Optional melakarta raga index (1-72, defaults to 29)
/// * `abbr` - Optional flag for abbreviated notation (defaults to false)
///
/// # Returns
/// Returns a `Vec<String>` containing Carnatic svara names.
///
/// # Examples
/// ```
/// let notes = vec!["C4", "D4"];
/// let svaras = note_to_svara_c(&notes, 261.63, None, Some(true));
/// assert_eq!(svaras, vec!["S", "R2"]);
/// ```
pub fn note_to_svara_c(notes: &[&str], sa: f32, mela: Option<usize>, _abbr: Option<bool>) -> Vec<String> {
    let midi = note_to_midi(notes, Some(true));
    hz_to_svara_c(&midi_to_hz(&midi), sa, mela)
}

/// Converts frequencies in Hz to mel scale.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `htk` - Optional flag for HTK formula (defaults to false)
///
/// # Returns
/// Returns a `Vec<f32>` containing mel values.
///
/// # Examples
/// ```
/// let freqs = vec![440.0];
/// let mels = hz_to_mel(&freqs, None);
/// ```
pub fn hz_to_mel(frequencies: &[f32], htk: Option<bool>) -> Vec<f32> {
    if htk.unwrap_or(false) {
        frequencies.iter().map(|&f| 2595.0 * (1.0 + f / 700.0).log10()).collect()
    } else {
        frequencies.iter().map(|&f| 1125.0 * (1.0 + f / 700.0).ln()).collect()
    }
}

/// Converts frequencies in Hz to octave numbers.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `tuning` - Optional tuning adjustment in semitones (defaults to 0.0)
///
/// # Returns
/// Returns a `Vec<f32>` containing octave numbers (A4 = 440 Hz = octave 4).
///
/// # Examples
/// ```
/// let freqs = vec![440.0];
/// let octs = hz_to_octs(&freqs, None);
/// assert_eq!(octs, vec![4.0]);
/// ```
pub fn hz_to_octs(frequencies: &[f32], tuning: Option<f32>) -> Vec<f32> {
    let tune = tuning.unwrap_or(0.0);
    frequencies.iter().map(|&f| (f / (440.0 * 2.0f32.powf(tune / 12.0))).log2() + 4.0).collect()
}

/// Converts mel values to frequencies in Hz.
///
/// # Arguments
/// * `mels` - Array of mel values
/// * `htk` - Optional flag for HTK formula (defaults to false)
///
/// # Returns
/// Returns a `Vec<f32>` containing frequencies in Hz.
///
/// # Examples
/// ```
/// let mels = vec![1125.0];
/// let freqs = mel_to_hz(&mels, None);
/// ```
pub fn mel_to_hz(mels: &[f32], htk: Option<bool>) -> Vec<f32> {
    if htk.unwrap_or(false) {
        mels.iter().map(|&m| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0)).collect()
    } else {
        mels.iter().map(|&m| 700.0 * (m / 1125.0).exp() - 700.0).collect()
    }
}

/// Converts octave numbers to frequencies in Hz.
///
/// # Arguments
/// * `octs` - Array of octave numbers
/// * `tuning` - Optional tuning adjustment in semitones (defaults to 0.0)
/// * `_bins_per_octave` - Optional bins per octave (unused, defaults to None)
///
/// # Returns
/// Returns a `Vec<f32>` containing frequencies in Hz.
///
/// # Examples
/// ```
/// let octs = vec![4.0];
/// let freqs = octs_to_hz(&octs, None, None);
/// assert_eq!(freqs, vec![440.0]);
/// ```
pub fn octs_to_hz(octs: &[f32], tuning: Option<f32>, _bins_per_octave: Option<usize>) -> Vec<f32> {
    let tune = tuning.unwrap_or(0.0);
    octs.iter().map(|&o| 440.0 * 2.0f32.powf(o - 4.0 + tune / 12.0)).collect()
}

/// Converts an A4 frequency to a tuning offset in semitones.
///
/// # Arguments
/// * `A4` - Frequency of A4 in Hz
/// * `_bins_per_octave` - Optional bins per octave (unused, defaults to None)
///
/// # Returns
/// Returns a `f32` representing the tuning offset from 440 Hz in semitones.
///
/// # Examples
/// ```
/// let tuning = A4_to_tuning(432.0, None);
/// assert!(tuning < 0.0);
/// ```
pub fn a4_to_tuning(a4: f32, _bins_per_octave: Option<usize>) -> f32 {
    12.0 * (a4 / 440.0).log2()
}

/// Converts a tuning offset in semitones to an A4 frequency.
///
/// # Arguments
/// * `tuning` - Tuning offset in semitones
/// * `_bins_per_octave` - Optional bins per octave (unused, defaults to None)
///
/// # Returns
/// Returns a `f32` representing the A4 frequency in Hz.
///
/// # Examples
/// ```
/// let A4 = tuning_to_A4(-0.317667, None);
/// assert!(A4 > 431.0 && A4 < 433.0);
/// ```
pub fn tuning_to_a4(tuning: f32, _bins_per_octave: Option<usize>) -> f32 {
    440.0 * 2.0f32.powf(tuning / 12.0)
}

/// Generates FFT frequency bins.
///
/// # Arguments
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `n_fft` - Optional FFT size (defaults to 2048)
///
/// # Returns
/// Returns a `Vec<f32>` containing frequency bins from 0 to Nyquist (sr/2).
///
/// # Examples
/// ```
/// let freqs = fft_frequencies(None, Some(4));
/// assert_eq!(freqs, vec![0.0, 11025.0, 22050.0]);
/// ```
pub fn fft_frequencies(sr: Option<u32>, n_fft: Option<usize>) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let n = n_fft.unwrap_or(2048);
    Array1::linspace(0.0, sample_rate as f32 / 2.0, n / 2 + 1).to_vec()
}

/// Generates Constant-Q Transform (CQT) frequency bins.
///
/// # Arguments
/// * `n_bins` - Number of frequency bins
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
///
/// # Returns
/// Returns a `Vec<f32>` containing CQT frequency bins.
///
/// # Examples
/// ```
/// let freqs = cqt_frequencies(3, None);
/// ```
pub fn cqt_frequencies(n_bins: usize, fmin: Option<f32>) -> Vec<f32> {
    let fmin = fmin.unwrap_or(32.70);
    let bins_per_octave = 12;
    (0..n_bins).map(|k| fmin * 2.0f32.powf(k as f32 / bins_per_octave as f32)).collect()
}

/// Generates mel-scale frequency bins.
///
/// # Arguments
/// * `n_mels` - Optional number of mel bins (defaults to 128)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 0.0)
/// * `fmax` - Optional maximum frequency in Hz (defaults to 11025.0)
/// * `_htk` - Optional flag for HTK formula (unused, defaults to None)
///
/// # Returns
/// Returns a `Vec<f32>` containing mel-scale frequency bins.
///
/// # Examples
/// ```
/// let freqs = mel_frequencies(Some(3), None, None, None);
/// ```
pub fn mel_frequencies(n_mels: Option<usize>, fmin: Option<f32>, fmax: Option<f32>, _htk: Option<bool>) -> Vec<f32> {
    let n = n_mels.unwrap_or(128);
    let min_freq = fmin.unwrap_or(0.0);
    let max_freq = fmax.unwrap_or(11025.0);
    let min_mel = hz_to_mel(&[min_freq], None)[0];
    let max_mel = hz_to_mel(&[max_freq], None)[0];
    let mel_steps = Array1::linspace(min_mel, max_mel, n);
    mel_to_hz(&mel_steps.to_vec(), None)
}

/// Generates tempo-related frequency bins.
///
/// # Arguments
/// * `n_bins` - Number of frequency bins
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
///
/// # Returns
/// Returns a `Vec<f32>` containing tempo frequencies in beats per minute (BPM).
///
/// # Examples
/// ```
/// let freqs = tempo_frequencies(3, None, None);
/// ```
pub fn tempo_frequencies(n_bins: usize, hop_length: Option<usize>, sr: Option<u32>) -> Vec<f32> {
    let sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let frame_rate = sr as f32 / hop as f32;
    Array1::linspace(0.0, frame_rate / 2.0, n_bins).mapv(|f| f * 60.0).to_vec()
}

/// Generates Fourier tempo frequency bins.
///
/// # Arguments
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
///
/// # Returns
/// Returns a `Vec<f32>` containing tempo frequencies in BPM with fixed parameters.
///
/// # Examples
/// ```
/// let freqs = fourier_tempo_frequencies(None);
/// ```
pub fn fourier_tempo_frequencies(sr: Option<u32>) -> Vec<f32> {
    let sr = sr.unwrap_or(44100);
    let hop_length = 512;
    let n_bins = 256;
    let frame_rate = sr as f32 / hop_length as f32;
    Array1::linspace(0.0, frame_rate / 2.0, n_bins).mapv(|f| f * 60.0).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn midi_and_hz_conversions_round_trip() {
        let notes = vec![60.0, 69.0];
        let freqs = midi_to_hz(&notes);
        assert!(approx_eq(freqs[0], 261.62555));
        assert!(approx_eq(freqs[1], 440.0));
        let back = hz_to_midi(&freqs);
        assert!(approx_eq(back[0], 60.0));
        assert!(approx_eq(back[1], 69.0));
    }

    #[test]
    fn note_name_conversions_match_expectations() {
        let midi = vec![60.0, 61.0, 72.0];
        let names = midi_to_note(&midi, None, None, None);
        assert_eq!(names, vec!["C4", "C#4", "C5"]);

        let back = note_to_midi(&["C4", "C#4", "C5"], None);
        assert_eq!(back, midi);

        let freqs = note_to_hz(&["A4", "C5"]);
        assert!(approx_eq(freqs[0], 440.0));
        assert!(freqs[1] > 520.0 && freqs[1] < 525.0);
    }

    #[test]
    fn mel_and_hz_conversions_are_inverse() {
        let hz = vec![0.0, 440.0, 22050.0];
        let mels = hz_to_mel(&hz, None);
        let back = mel_to_hz(&mels, None);
        for (orig, recon) in hz.iter().zip(back.iter()) {
            let tol = 1e-4 * orig.abs().max(1.0);
            assert!((orig - recon).abs() <= tol);
        }
    }

    #[test]
    fn cqt_and_fft_frequencies_generate_expected_bins() {
        let cqt = cqt_frequencies(3, Some(32.7));
        assert_eq!(cqt.len(), 3);
        assert!(approx_eq(cqt[0], 32.7));
        assert!(cqt[1] > cqt[0]);

        let fft = fft_frequencies(Some(44_100), Some(4));
        assert_eq!(fft, vec![0.0, 11_025.0, 22_050.0]);
    }

    #[test]
    fn tempo_and_fourier_bins_have_expected_scale() {
        let tempo = tempo_frequencies(3, Some(512), Some(44_100));
        assert_eq!(tempo.len(), 3);
        assert!(approx_eq(tempo[0], 0.0));
        assert!(tempo[2] > tempo[1]);

        let fourier = fourier_tempo_frequencies(Some(44_100));
        assert_eq!(fourier.len(), 256);
        assert!(fourier.first().unwrap().abs() < 1e-6);
        // Last bin is the frame-rate Nyquist in BPM: (44100/512)/2 * 60 ≈ 2584.
        assert!(fourier.last().unwrap() > &2500.0);
    }

    #[test]
    fn tuning_and_octave_conversions_match() {
        let tuning = a4_to_tuning(432.0, None);
        assert!(tuning < 0.0);
        let a4 = tuning_to_a4(tuning, None);
        assert!(approx_eq(a4, 432.0));

        let octs = hz_to_octs(&[440.0], None);
        assert!(approx_eq(octs[0], 4.0));
        let freqs = octs_to_hz(&octs, None, None);
        assert!(approx_eq(freqs[0], 440.0));
    }
}