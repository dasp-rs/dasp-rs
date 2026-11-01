//! Shared reader and numerical error reporting for checked-in independent references.
#![allow(dead_code)] // Each integration target uses the relevant subset of helpers.
use ndarray::Array2;
use num_complex::Complex;
use std::collections::BTreeMap;

pub type References = BTreeMap<String, Array2<Complex<f64>>>;

/// CSV columns: name,row,col,real,imag; comments start with '#'.
pub fn load(text: &str) -> References {
    let mut entries: BTreeMap<String, Vec<(usize, usize, Complex<f64>)>> = BTreeMap::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("name,") && !l.is_empty())
    {
        let c: Vec<_> = line.split(',').collect();
        assert_eq!(c.len(), 5, "Invalid fixture row: {line}");
        entries.entry(c[0].into()).or_default().push((
            c[1].parse().unwrap(),
            c[2].parse().unwrap(),
            Complex::new(c[3].parse().unwrap(), c[4].parse().unwrap()),
        ));
    }
    entries
        .into_iter()
        .map(|(name, values)| {
            let rows = values.iter().map(|v| v.0).max().unwrap() + 1;
            let cols = values.iter().map(|v| v.1).max().unwrap() + 1;
            assert_eq!(values.len(), rows * cols, "Incomplete fixture {name}");
            let mut result = Array2::from_elem((rows, cols), Complex::new(f64::NAN, f64::NAN));
            for (row, col, value) in values {
                assert!(
                    result[[row, col]].re.is_nan(),
                    "Duplicate fixture element {name}"
                );
                assert!(
                    value.re.is_finite() && value.im.is_finite(),
                    "Nonfinite reference {name}"
                );
                result[[row, col]] = value;
            }
            (name, result)
        })
        .collect()
}

pub fn real(data: &References, name: &str) -> Array2<f32> {
    data[name].mapv(|z| {
        assert_eq!(z.im, 0.0);
        z.re as f32
    })
}

pub fn signal(data: &References, name: &str) -> Vec<f32> {
    assert_eq!(data[name].nrows(), 1);
    real(data, name).into_raw_vec_and_offset().0
}

pub fn check_real(name: &str, actual: &Array2<f32>, data: &References, atol: f64, rtol: f64) {
    check_complex(
        name,
        &actual.mapv(|v| Complex::new(v, 0.0)),
        data,
        atol,
        rtol,
    );
}

pub fn check_complex(
    name: &str,
    actual: &Array2<Complex<f32>>,
    data: &References,
    atol: f64,
    rtol: f64,
) {
    let expected = &data[name];
    assert_eq!(actual.dim(), expected.dim(), "Shape mismatch: {name}");
    assert!(atol >= 0.0 && rtol >= 0.0);
    let mut max_abs = 0.0_f64;
    let mut sum_squared = 0.0;
    let mut worst_ratio = 0.0_f64;
    let mut worst_index = 0;
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.re.is_finite() && a.im.is_finite(),
            "{name}[{index}] is not finite: {a}"
        );
        let error = (Complex::new(f64::from(a.re), f64::from(a.im)) - b).norm();
        let bound = atol + rtol * b.norm();
        let ratio = if error == 0.0 { 0.0 } else { error / bound };
        max_abs = max_abs.max(error);
        sum_squared += error * error;
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_index = index;
        }
    }
    let rmse = (sum_squared / actual.len() as f64).sqrt();
    eprintln!(
        "REFERENCE {name}: count={} max_abs={max_abs:.6e} rmse={rmse:.6e} tolerance_ratio={worst_ratio:.6e} atol={atol:.2e} rtol={rtol:.2e}",
        actual.len()
    );
    assert!(
        worst_ratio <= 1.0,
        "{name}: error exceeds atol={atol} + rtol={rtol} * |reference| at flat index {worst_index}; max_abs={max_abs}, RMSE={rmse}, ratio={worst_ratio}"
    );
}
