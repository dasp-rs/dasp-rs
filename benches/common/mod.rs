//! Shared deterministic workloads. Input construction is outside measurements.
#![allow(dead_code)]
use dasp_rs::{
    pitch::{pyin, yin},
    proc::{cqt, icqt, resample, stft},
};
use ndarray::Array2;
use num_complex::Complex;
use std::hint::black_box;

#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub operation: &'static str,
    pub sr: u32,
    pub samples: usize,
    pub frame: usize,
    pub hop: usize,
    pub bins: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub target_sr: u32,
}
impl Case {
    pub fn json(self) -> String {
        format!(
            "{{\"name\":\"{}\",\"operation\":\"{}\",\"sample_rate\":{},\"samples\":{},\"frame_length\":{},\"hop_length\":{},\"bins\":{},\"fmin\":{},\"fmax\":{},\"target_sample_rate\":{},\"audio_seconds\":{},\"signal\":\"integer_prng_plus_220hz_harmonics_v1\"}}",
            self.name,
            self.operation,
            self.sr,
            self.samples,
            self.frame,
            self.hop,
            self.bins,
            self.fmin,
            self.fmax,
            self.target_sr,
            self.samples as f64 / f64::from(self.sr)
        )
    }
}

#[allow(clippy::too_many_lines)] // Explicit workload configurations are kept together for review.
pub fn cases() -> Vec<Case> {
    let base = Case {
        name: "",
        operation: "",
        sr: 16000,
        samples: 1600,
        frame: 512,
        hop: 128,
        bins: 36,
        fmin: 100.0,
        fmax: 600.0,
        target_sr: 24000,
    };
    vec![
        Case {
            name: "stft_short_512",
            operation: "stft",
            ..base
        },
        Case {
            name: "stft_long_2048",
            operation: "stft",
            sr: 48000,
            samples: 480_000,
            frame: 2048,
            hop: 512,
            ..base
        },
        Case {
            name: "resample_short_16k_24k",
            operation: "resample",
            ..base
        },
        Case {
            name: "resample_long_48k_16k",
            operation: "resample",
            sr: 48000,
            samples: 480_000,
            target_sr: 16000,
            ..base
        },
        Case {
            name: "cqt_short_36bins",
            operation: "cqt",
            ..base
        },
        Case {
            name: "cqt_long_84bins",
            operation: "cqt",
            sr: 22050,
            samples: 44100,
            hop: 256,
            bins: 84,
            fmin: 32.703_197,
            ..base
        },
        Case {
            name: "cqt_short_long_filter",
            operation: "cqt",
            sr: 44100,
            samples: 33,
            hop: 512,
            bins: 1,
            fmin: 1.0,
            ..base
        },
        Case {
            name: "icqt_short_24bins",
            operation: "icqt",
            sr: 8000,
            samples: 800,
            hop: 64,
            bins: 24,
            ..base
        },
        Case {
            name: "icqt_long_48bins",
            operation: "icqt",
            sr: 16000,
            samples: 16000,
            hop: 128,
            bins: 48,
            fmin: 65.406_395,
            ..base
        },
        Case {
            name: "yin_short_512",
            operation: "yin",
            sr: 8000,
            samples: 1600,
            fmin: 80.0,
            ..base
        },
        Case {
            name: "yin_long_1024",
            operation: "yin",
            samples: 80000,
            frame: 1024,
            hop: 256,
            fmin: 80.0,
            ..base
        },
        Case {
            name: "pyin_short_512",
            operation: "pyin",
            sr: 8000,
            samples: 1600,
            fmin: 80.0,
            ..base
        },
        Case {
            name: "pyin_medium_512",
            operation: "pyin",
            sr: 8000,
            samples: 8000,
            fmin: 80.0,
            ..base
        },
    ]
}

pub struct Prepared {
    pub case: Case,
    signal: Vec<f32>,
    spectrum: Option<Array2<Complex<f32>>>,
}
impl Prepared {
    pub fn new(case: Case) -> Self {
        let mut state = 0x1234_5678_u32;
        let signal: Vec<_> = (0..case.samples)
            .map(|n| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = f64::from(state >> 8) / 16_777_216.0 - 0.5;
                let phase = std::f64::consts::TAU * 220.0 * n as f64 / f64::from(case.sr);
                (0.5 * phase.sin() + 0.2 * (2.0 * phase + 0.3).sin() + 0.01 * noise) as f32
            })
            .collect();
        let spectrum = if case.operation == "icqt" {
            Some(
                cqt(&signal, case.sr)
                    .hop_length(case.hop)
                    .fmin(case.fmin)
                    .n_bins(case.bins)
                    .compute()
                    .unwrap(),
            )
        } else {
            None
        };
        Self {
            case,
            signal,
            spectrum,
        }
    }
    pub fn run(&self) {
        let c = self.case;
        let y = black_box(self.signal.as_slice());
        match c.operation {
            "stft" => {
                black_box(stft(y).n_fft(c.frame).hop_length(c.hop).compute().unwrap());
            }
            "resample" => {
                black_box(resample(y, c.sr, c.target_sr).unwrap());
            }
            "cqt" => {
                black_box(
                    cqt(y, c.sr)
                        .hop_length(c.hop)
                        .fmin(c.fmin)
                        .n_bins(c.bins)
                        .compute()
                        .unwrap(),
                );
            }
            "icqt" => {
                black_box(
                    icqt(black_box(self.spectrum.as_ref().unwrap()))
                        .sample_rate(c.sr)
                        .hop_length(c.hop)
                        .fmin(c.fmin)
                        .compute()
                        .unwrap(),
                );
            }
            "yin" => {
                black_box(
                    yin(y, c.fmin, c.fmax)
                        .sample_rate(c.sr)
                        .frame_length(c.frame)
                        .hop_length(c.hop)
                        .compute()
                        .unwrap(),
                );
            }
            "pyin" => {
                black_box(
                    pyin(y, c.fmin, c.fmax)
                        .sample_rate(c.sr)
                        .frame_length(c.frame)
                        .hop_length(c.hop)
                        .compute_with_voicing()
                        .unwrap(),
                );
            }
            _ => unreachable!(),
        }
    }
}

pub struct Options {
    pub case: String,
    pub repetitions: usize,
    pub warmup: usize,
}
pub fn options() -> Option<Options> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--list") {
        for case in cases() {
            println!("{}", case.json());
        }
        return None;
    }
    // Custom bench binaries are invoked by cargo test --all-targets too. Never
    // run work unless explicitly requested by the benchmark orchestrator.
    if !args.iter().any(|a| a == "--run") {
        println!("Skipped benchmark; pass --run --case NAME explicitly.");
        return None;
    }
    let value = |flag: &str, default: &str| {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map_or_else(|| default.to_owned(), |w| w[1].clone())
    };
    let options = Options {
        case: value("--case", "stft_short_512"),
        repetitions: value("--repetitions", "15")
            .parse()
            .expect("positive repetitions"),
        warmup: value("--warmup", "3").parse().expect("positive warmup"),
    };
    assert!(options.repetitions > 0 && options.warmup > 0);
    Some(options)
}
pub fn prepare(options: &Options) -> Prepared {
    Prepared::new(
        cases()
            .into_iter()
            .find(|c| c.name == options.case)
            .expect("unknown benchmark case"),
    )
}
