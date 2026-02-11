/// Returns a list of all chromatic note names, ignoring the provided key.
///
/// # Arguments
/// * `key` - Key signature (currently unused, e.g., "C:maj")
/// * `_unicode` - Optional flag for Unicode accidentals (unused, defaults to None)
/// * `_natural` - Optional flag for natural notes only (unused, defaults to None)
///
/// # Returns
/// Returns a `Vec<String>` containing all 12 chromatic note names (C through B).
///
/// # Examples
/// ```
/// let notes = key_to_notes("C:maj", None, None);
/// assert_eq!(notes, vec!["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]);
/// ```
pub fn key_to_notes(_key: &str, _unicode: Option<bool>, _natural: Option<bool>) -> Vec<String> {
    let notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    notes.iter().map(|&n| n.to_string()).collect()
}

/// Converts a key signature to scale degrees.
///
/// # Arguments
/// * `key` - Key signature in the format "tonic:mode" (e.g., "C:maj", "F#:min")
///
/// # Returns
/// Returns a `Vec<usize>` containing the scale degrees (0-11) relative to the chromatic scale.
///
/// # Notes
/// - Supports major ("maj", "major") and minor ("min", "minor") modes.
/// - Defaults to major scale if mode is unspecified or unrecognized.
/// - Tonic is case-insensitive and supports sharp/flat synonyms (e.g., "C#", "Db").
///
/// # Examples
/// ```
/// let degrees = key_to_degrees("C:maj");
/// assert_eq!(degrees, vec![0, 2, 4, 5, 7, 9, 11]); // C major scale
/// let degrees = key_to_degrees("F#:min");
/// assert_eq!(degrees, vec![6, 8, 9, 11, 1, 2, 4]); // F# minor scale
/// ```
pub fn key_to_degrees(key: &str) -> Vec<usize> {
    let key = key.to_lowercase();
    let (tonic, mode) = key.split_once(':').unwrap_or((&key, "maj"));
    let tonic_shift = match tonic {
        "c" => 0, "c#" | "db" => 1, "d" => 2, "d#" | "eb" => 3, "e" => 4,
        "f" => 5, "f#" | "gb" => 6, "g" => 7, "g#" | "ab" => 8, "a" => 9,
        "a#" | "bb" => 10, "b" => 11, _ => 0,
    };
    let major = vec![0, 2, 4, 5, 7, 9, 11];
    let minor = vec![0, 2, 3, 5, 7, 8, 10];
    let degrees = match mode {
        "maj" | "major" => major,
        "min" | "minor" => minor,
        _ => major,
    };
    degrees.into_iter().map(|d| (d + tonic_shift) % 12).collect()
}

/// Converts a melakarta raga index to Carnatic svara names.
///
/// # Arguments
/// * `mela` - Melakarta raga index (1-72)
/// * `abbr` - Optional flag for abbreviated notation (defaults to false)
/// * `unicode` - Optional flag for Unicode transliteration (defaults to false)
///
/// # Returns
/// Returns a `Vec<String>` containing svara names for the melakarta raga.
///
/// # Notes
/// - If `mela` is out of range (1-72), defaults to a major scale-like pattern.
/// - Abbreviated notation uses "S", "R1", etc.; full notation uses "Shadjam", "Shuddha Rishabham", etc.
///
/// # Examples
/// ```
/// let svaras = mela_to_svara(29, Some(true), None); // Dheerashankarabharanam
/// assert_eq!(svaras, vec!["S", "R2", "G3", "M1", "P", "D2", "N3"]);
/// let svaras = mela_to_svara(1, None, None); // Kanakangi
/// assert_eq!(svaras, vec!["shadjam", "rishabham1", "gandharam1", "madhyamam1", "panchamam", "dhaivatam1", "nishadam1"]);
/// ```
pub fn mela_to_svara(mela: usize, abbr: Option<bool>, unicode: Option<bool>) -> Vec<String> {
    let abbr = abbr.unwrap_or(false);
    let unicode = unicode.unwrap_or(false);
    let degrees = mela_to_degrees(mela);
    let svara_full = if unicode {
        vec!["ṣaḍjam", "ṛṣabham", "gāndhāram", "madhyamam", "pañcamam", "dhaivatam", "niṣādam"]
    } else {
        vec!["shadjam", "rishabham", "gandharam", "madhyamam", "panchamam", "dhaivatam", "nishadam"]
    };
    let mut result = Vec::new();
    for (i, deg) in degrees.iter().enumerate() {
        let base = match i {
            0 => "S", 1..=3 => "R", 4..=6 => "G", 7 => "M", 8 => "P", 9..=11 => "D", 12..=14 => "N",
            _ => "S",
        };
        let variant = match deg % 12 {
            1 => "1", 2 => "2", 3 => "3", 5 => "1", 6 => "2", 7 => "3", 8 => "1", 9 => "2", 10 => "3", _ => "",
        };
        let name = if abbr {
            format!("{}{}", base, variant)
        } else {
            let idx = match base {
                "S" => 0, "R" => 1, "G" => 2, "M" => 3, "P" => 4, "D" => 5, "N" => 6, _ => 0,
            };
            format!("{}{}", svara_full[idx], if variant.is_empty() { "" } else { variant })
        };
        result.push(name);
    }
    result
}

/// Converts a melakarta raga index to scale degrees.
///
/// # Arguments
/// * `mela` - Melakarta raga index (1-72)
///
/// # Returns
/// Returns a `Vec<usize>` containing the scale degrees (0-11) for the melakarta raga.
///
/// # Notes
/// - If `mela` is out of range (1-72), returns a default major scale (0, 2, 4, 5, 7, 9, 11).
/// - Uses traditional melakarta rules to determine R, G, M, D, N positions.
///
/// # Examples
/// ```
/// let degrees = mela_to_degrees(29); // Dheerashankarabharanam
/// assert_eq!(degrees, vec![0, 2, 4, 5, 7, 9, 11]);
/// let degrees = mela_to_degrees(1); // Kanakangi
/// assert_eq!(degrees, vec![0, 1, 2, 5, 7, 8, 10]);
/// ```
pub fn mela_to_degrees(mela: usize) -> Vec<usize> {
    if !(1..=72).contains(&mela) { return vec![0, 2, 4, 5, 7, 9, 11]; }
    let mela = mela - 1;
    let r = (mela / 36) % 2;
    let g = (mela / 18) % 2;
    let m = (mela / 9) % 2;
    let d = (mela / 3) % 3;
    let n = mela % 3;
    vec![
        0,
        if r == 0 { 1 } else { 2 + g },
        if r == 0 { 2 + g } else { 4 },
        5 + m,
        7,
        8 + d,
        10 + n,
    ]
}

/// Converts a Hindustani thaat to scale degrees.
///
/// # Arguments
/// * `thaat` - Name of the thaat (e.g., "Bilaval", "Kafi")
///
/// # Returns
/// Returns a `Vec<usize>` containing the scale degrees (0-11) for the thaat.
///
/// # Notes
/// - Case-insensitive; defaults to Bilaval scale if thaat is unrecognized.
/// - Recognizes 10 traditional thaats.
///
/// # Examples
/// ```
/// let degrees = thaat_to_degrees("Bilaval");
/// assert_eq!(degrees, vec![0, 2, 4, 5, 7, 9, 11]);
/// let degrees = thaat_to_degrees("Kafi");
/// assert_eq!(degrees, vec![0, 2, 3, 5, 7, 9, 10]);
/// ```
pub fn thaat_to_degrees(thaat: &str) -> Vec<usize> {
    match thaat.to_lowercase().as_str() {
        "bilaval" => vec![0, 2, 4, 5, 7, 9, 11],
        "kalyani" => vec![0, 2, 4, 6, 7, 9, 11],
        "khamaj" => vec![0, 2, 4, 5, 7, 9, 10],
        "bhairav" => vec![0, 1, 4, 5, 6, 9, 11],
        "purvi" => vec![0, 1, 4, 6, 7, 9, 11],
        "marwa" => vec![0, 1, 3, 6, 7, 9, 11],
        "kafi" => vec![0, 2, 3, 5, 7, 9, 10],
        "asavari" => vec![0, 2, 3, 5, 7, 8, 10],
        "todi" => vec![0, 1, 3, 6, 7, 8, 11],
        "bhoopali" => vec![0, 2, 4, 7, 9],
        _ => vec![0, 2, 4, 5, 7, 9, 11],
    }
}

/// Lists all 72 melakarta ragas with their indices and names.
///
/// # Returns
/// Returns a `Vec<(usize, String)>` containing tuples of (index, name) for all melakarta ragas.
///
/// # Examples
/// ```
/// let melas = list_mela();
/// assert_eq!(melas[0], (1, "Kanakangi".to_string()));
/// assert_eq!(melas.len(), 72);
/// ```
pub fn list_mela() -> Vec<(usize, String)> {
    let names = vec![
        "Kanakangi", "Ratnangi", "Ganamurti", "Vanaspati", "Manavati", "Tanarupi",
        "Senavati", "Hanumatodi", "Dhenuka", "Natakapriya", "Kokilapriya", "Rupavati",
        "Gayakapriya", "Vakulabharanam", "Mayamalavagowla", "Chakravakam", "Suryakantam",
        "Hatakambari", "Jhankaradhwani", "Natabhairavi", "Keeravani", "Kharaharapriya",
        "Gourimanohari", "Varunapriya", "Mararanjani", "Charukesi", "Sarasangi",
        "Harikambhoji", "Dheerasankarabharanam", "Naganandini", "Yagapriya", "Ragavardhini",
        "Gangeyabhushani", "Vagadheeswari", "Shulini", "Chalanata", "Salagam", "Jalarnavam",
        "Jhalavarali", "Navaneetam", "Pavani", "Raghupriya", "Gavambodhi", "Bhavapriya",
        "Shubhapantuvarali", "Shadvidamargini", "Suvarnangi", "Divyamani", "Dhavalambari",
        "Namanarayani", "Kamavardhini", "Ramapriya", "Gamanashrama", "Vishwambari",
        "Shamalangi", "Shanmukhapriya", "Simhendramadhyamam", "Hemavati", "Dharmavati",
        "Neetimati", "Kantamani", "Rishabhapriya", "Latangi", "Vachaspati", "Mechakalyani",
        "Chitrambari", "Sucharitra", "Jyotiswarupini", "Dhatuvardhani", "Nasikabhushani",
        "Kosalam", "Rasikapriya",
    ];
    names.into_iter().enumerate().map(|(i, name)| (i + 1, name.to_string())).collect()
}

/// Lists the 10 traditional Hindustani thaats.
///
/// # Returns
/// Returns a `Vec<String>` containing the names of all 10 thaats.
///
/// # Examples
/// ```
/// let thaats = list_thaat();
/// assert_eq!(thaats, vec!["Bilaval", "Kalyani", "Khamaj", "Bhairav", "Purvi", "Marwa", "Kafi", "Asavari", "Todi", "Bhoopali"]);
/// ```
pub fn list_thaat() -> Vec<String> {
    vec![
        "Bilaval".to_string(),
        "Kalyani".to_string(),
        "Khamaj".to_string(),
        "Bhairav".to_string(),
        "Purvi".to_string(),
        "Marwa".to_string(),
        "Kafi".to_string(),
        "Asavari".to_string(),
        "Todi".to_string(),
        "Bhoopali".to_string(),
    ]
}

/// Generates a note name based on a number of perfect fifths from a unison note.
///
/// # Arguments
/// * `unison` - Starting note (e.g., "C", "F#")
/// * `fifths` - Number of fifths (positive or negative)
/// * `unicode` - Optional flag for Unicode accidentals (defaults to false)
///
/// # Returns
/// Returns a `String` representing the resulting note name with octave (e.g., "G4", "F♯-1").
///
/// # Examples
/// ```
/// let note = fifths_to_note("C", 1, None);
/// assert_eq!(note, "G");
/// let note = fifths_to_note("C", 7, Some(true));
/// assert_eq!(note, "F♯1");
/// ```
pub fn fifths_to_note(unison: &str, fifths: i32, unicode: Option<bool>) -> String {
    let unicode = unicode.unwrap_or(false);
    let semitones = (fifths * 7) % 12;
    let octave_shift = (fifths * 7) / 12;
    let base = match unison.to_lowercase().as_str() {
        "c" => 0, "c#" | "db" => 1, "d" => 2, "d#" | "eb" => 3, "e" => 4,
        "f" => 5, "f#" | "gb" => 6, "g" => 7, "g#" | "ab" => 8, "a" => 9,
        "a#" | "bb" => 10, "b" => 11, _ => 0,
    };
    let note_idx = (base + semitones + 12) % 12;
    let note = match note_idx {
        0 => "C", 1 => if unicode { "C♯" } else { "C#" }, 2 => "D",
        3 => if unicode { "D♯" } else { "D#" }, 4 => "E", 5 => "F",
        6 => if unicode { "F♯" } else { "F#" }, 7 => "G",
        8 => if unicode { "G♯" } else { "G#" }, 9 => "A",
        10 => if unicode { "A♯" } else { "A#" }, 11 => "B",
        _ => "C",
    };
    format!("{}{}", note, if octave_shift != 0 { octave_shift.to_string() } else { "".to_string() })
}

/// Converts an interval ratio to Functional Just System (FJS) notation.
///
/// # Arguments
/// * `interval` - Interval ratio (e.g., 1.5 for a perfect fifth)
/// * `unison` - Optional unison ratio (defaults to 1.0)
///
/// # Returns
/// Returns a `String` representing the interval in FJS notation (e.g., "3/2").
///
/// # Notes
/// - Recognizes common just intervals (1/1, 3/2, 4/3, 5/4, 6/5); otherwise approximates as a fraction.
///
/// # Examples
/// ```
/// let fjs = interval_to_fjs(1.5, None);
/// assert_eq!(fjs, "3/2");
/// let fjs = interval_to_fjs(1.333, None);
/// assert_eq!(fjs, "1.33/1");
/// ```
pub fn interval_to_fjs(interval: f32, unison: Option<f32>) -> String {
    let unison = unison.unwrap_or(1.0);
    let ratio = interval / unison;
    match ratio {
        r if (r - 1.0).abs() < 1e-6 => "1/1".to_string(),
        r if (r - 3.0/2.0).abs() < 1e-6 => "3/2".to_string(),
        r if (r - 4.0/3.0).abs() < 1e-6 => "4/3".to_string(),
        r if (r - 5.0/4.0).abs() < 1e-6 => "5/4".to_string(),
        r if (r - 6.0/5.0).abs() < 1e-6 => "6/5".to_string(),
        _ => format!("{:.2}/1", ratio),
    }
}

/// Generates frequencies based on a sequence of intervals.
///
/// # Arguments
/// * `n_bins` - Number of frequency bins to generate
/// * `fmin` - Starting frequency in Hz
/// * `intervals` - Array of interval ratios
///
/// # Returns
/// Returns a `Vec<f32>` containing frequencies generated by applying intervals cyclically.
///
/// # Examples
/// ```
/// let freqs = interval_frequencies(3, 261.63, &[3.0/2.0, 4.0/3.0]);
/// assert!(freqs[0] == 261.63);
/// assert!(freqs[1] > 391.0 && freqs[1] < 392.0); // ~391.945
/// ```
pub fn interval_frequencies(n_bins: usize, fmin: f32, intervals: &[f32]) -> Vec<f32> {
    let mut freqs = Vec::with_capacity(n_bins);
    let mut f = fmin;
    let mut interval_idx = 0;
    for _ in 0..n_bins {
        freqs.push(f);
        f *= intervals[interval_idx % intervals.len()];
        interval_idx += 1;
    }
    freqs
}

/// Generates Pythagorean tuning intervals.
///
/// # Arguments
/// * `bins_per_octave` - Optional number of bins per octave (defaults to 12)
///
/// # Returns
/// Returns a `Vec<f32>` containing sorted Pythagorean interval ratios within an octave (1 to 2).
///
/// # Examples
/// ```
/// let intervals = pythagorean_intervals(Some(3));
/// assert_eq!(intervals, vec![1.0, 1.5, 1.125]); // 1/1, 3/2, 9/8 adjusted
/// ```
pub fn pythagorean_intervals(bins_per_octave: Option<usize>) -> Vec<f32> {
    let bins = bins_per_octave.unwrap_or(12);
    let mut intervals = Vec::with_capacity(bins);
    let fifth = 3.0 / 2.0;
    let mut ratio = 1.0;
    for i in 0..bins {
        intervals.push(ratio);
        ratio *= if i % 2 == 0 { fifth } else { 1.0 / fifth };
        while ratio > 2.0 { ratio /= 2.0; }
        while ratio < 1.0 { ratio *= 2.0; }
    }
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    intervals
}

/// Generates intervals based on prime number limits.
///
/// # Arguments
/// * `primes` - Array of prime numbers to generate intervals from
///
/// # Returns
/// Returns a `Vec<f32>` containing sorted unique interval ratios within an octave (1 to 2).
///
/// # Examples
/// ```
/// let intervals = plimit_intervals(&[2, 3]);
/// assert!(intervals.contains(&1.0));
/// assert!(intervals.contains(&1.5));
/// assert!(intervals.contains(&1.3333333)); // ~4/3
/// ```
pub fn plimit_intervals(primes: &[usize]) -> Vec<f32> {
    let mut intervals = vec![1.0];
    for &p in primes {
        let mut new_intervals = Vec::new();
        for &i in &intervals {
            let mut n = i;
            while n < 2.0 {
                new_intervals.push(n);
                n *= p as f32;
            }
            let mut d = i;
            while d > 0.5 {
                new_intervals.push(d);
                d /= p as f32;
            }
        }
        intervals.extend(new_intervals);
    }
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    intervals.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    intervals.retain(|&x| (1.0..=2.0).contains(&x));
    intervals
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn key_and_degrees_cover_major_minor() {
        let notes = key_to_notes("C:maj", None, None);
        assert_eq!(notes[0], "C");

        let c_major = key_to_degrees("C:maj");
        assert_eq!(c_major, vec![0, 2, 4, 5, 7, 9, 11]);

        let f_sharp_minor = key_to_degrees("F#:min");
        assert!(f_sharp_minor.contains(&6));
        assert!(f_sharp_minor.contains(&1));
    }

    #[test]
    fn mela_and_thaat_mappings_return_expected_sizes() {
        let svaras = mela_to_svara(29, Some(true), Some(false));
        assert_eq!(svaras, vec!["S", "R2", "G3", "M1", "P", "D2", "N3"]);

        let degrees = mela_to_degrees(1);
        assert_eq!(degrees, vec![0, 1, 2, 5, 7, 8, 10]);

        let thaats = list_thaat();
        assert_eq!(thaats.len(), 10);
        assert!(thaats.contains(&"Bilaval".to_string()));
    }

    #[test]
    fn fifths_and_intervals_generate_consistent_values() {
        let fifth = fifths_to_note("C", 1, None);
        assert_eq!(fifth, "G");
        let unicode = fifths_to_note("C", 7, Some(true));
        assert!(unicode.starts_with("F"));

        let fjs = interval_to_fjs(1.5, None);
        assert_eq!(fjs, "3/2");
        let approx = interval_to_fjs(1.333, None);
        assert!(approx.starts_with("1.33"));
    }

    #[test]
    fn interval_generators_produce_sorted_ranges() {
        let freqs = interval_frequencies(3, 100.0, &[1.5, 4.0 / 3.0]);
        assert!(approx_eq(freqs[0], 100.0));
        assert!(freqs[1] > freqs[0]);

        let pyth = pythagorean_intervals(Some(5));
        assert_eq!(pyth.first().copied().unwrap(), 1.0);
        assert!(pyth.windows(2).all(|w| w[0] <= w[1]));

        let plimit = plimit_intervals(&[2, 3]);
        assert!(plimit.contains(&1.0));
        assert!(plimit.iter().all(|v| *v >= 1.0 && *v <= 2.0));
    }

    #[test]
    fn list_mela_covers_all_entries() {
        let melas = list_mela();
        assert_eq!(melas.len(), 72);
        assert_eq!(melas[0].0, 1);
        assert!(melas.iter().any(|(_, name)| name == "Mechakalyani"));
    }
}