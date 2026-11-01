#!/usr/bin/env python3
"""Independent inverse-CQT SVD characterization; exact public complex64 inputs.

Reproduce with NumPy 2.3.5, SciPy 1.18.1, librosa 0.11.0:
    python tests/fixtures/generate_cqt_inverse_reference.py

librosa supplies L1-normalized, centered, zero-padded wavelets. Dense analysis
columns are independently built from those filters, not from Rust output. NumPy
LAPACK SVD solves the real least-squares system [Re(A); Im(A)] x = [Re(c); Im(c)].
Every target is quantized to complex64 BEFORE solving; outputs are float64.
Varied hops, lengths, sample rates, coverage and inconsistent targets exercise
more than the prior hop-one consistent example. The 72-sample case deliberately
exceeds the 32-step CG budget: its SVD result is diagnostic, not waveform parity.

https://librosa.org/doc/0.11.0/generated/librosa.filters.wavelet.html
https://numpy.org/doc/2.3/reference/generated/numpy.linalg.lstsq.html
"""
from pathlib import Path
import csv
import numpy as np
import scipy
import librosa

assert (np.__version__, scipy.__version__, librosa.__version__) == ('2.3.5', '1.18.1', '0.11.0')
ALPHA = (2**(2/12)-1)/(2**(2/12)+1)
CASES = [
    ('short_hop2', 2048, 12, 2, 64., 48, 0.),
    ('inconsistent_hop3', 2048, 24, 3, 64., 48, .02),
    ('short_hop5', 4096, 30, 5, 128., 48, 0.),
    ('partial_inconsistent', 2048, 24, 2, 128., 36, .01),
    ('budget72', 2048, 72, 6, 64., 48, 0.),
]
rng = np.random.default_rng(20261117)
entries = {}
for name, sr, length, hop, fmin, bins, perturbation in CASES:
    basis, lengths = librosa.filters.wavelet(
        freqs=librosa.cqt_frequencies(n_bins=bins, fmin=fmin), sr=sr,
        gamma=0, alpha=ALPHA, norm=1, pad_fft=True, dtype=np.complex128)
    frames = length//hop
    matrix = np.zeros((bins, frames, length), dtype=np.complex128)
    for frame in range(frames):
        indices = np.arange(length)-frame*hop+basis.shape[1]//2
        valid = (indices >= 0) & (indices < basis.shape[1])
        matrix[:, frame, valid] = np.conj(basis[:, indices[valid]])*np.sqrt(lengths[:, None])
    matrix = matrix.reshape(bins*frames, length)
    original = rng.normal(scale=.2, size=length)
    coefficients = matrix @ original
    if perturbation:
        coefficients += perturbation*(rng.normal(size=bins*frames)+1j*rng.normal(size=bins*frames))
    coefficients = coefficients.astype(np.complex64).astype(np.complex128)
    real_matrix = np.vstack([matrix.real, matrix.imag])
    target = np.r_[coefficients.real, coefficients.imag]
    solution, _, rank, singular = np.linalg.lstsq(real_matrix, target, rcond=None)
    residual = matrix @ solution - coefficients
    # metadata: sample rate, length, hop, fmin, bins, rank, condition,
    # minimum residual squared, target norm squared, minimizer norm squared.
    entries[name+'_metadata'] = [sr, length, hop, fmin, bins, rank,
        singular[0]/singular[-1], np.vdot(residual,residual).real,
        np.vdot(coefficients,coefficients).real, np.dot(solution,solution)]
    entries[name+'_input'] = coefficients.reshape(bins,frames)
    entries[name+'_solution'] = solution
    print(name, 'rank', rank, 'condition', singular[0]/singular[-1],
          'optimal_relative_residual', np.linalg.norm(residual)/np.linalg.norm(coefficients))

out = Path(__file__).with_name('cqt_inverse_reference.csv')
with out.open('w', newline='') as file:
    file.write('# numpy 2.3.5; scipy 1.18.1; librosa 0.11.0; seed 20261117\n')
    file.write('# Independent real least-squares SVD; inputs quantized to complex64 before solving\n')
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['name','row','col','real','imag'])
    for name, values in entries.items():
        for (r,c), value in np.ndenumerate(np.atleast_2d(values)):
            writer.writerow([name,r,c,format(np.real(value),'.17g'),format(np.imag(value),'.17g')])
print(out)
