#!/usr/bin/env python3
"""Independent CQT filterbank fixtures: librosa 0.11.0, NumPy 2.3.5, SciPy 1.18.1.

Install requirements-reference.txt, then run this file to regenerate.
No Rust code is called. librosa supplies the independent filters and SciPy performs
correlation. This intentionally isolates a full-rate CQT from librosa.cqt's
recursive resampling / positive-spectrum approximation / optional sparsification.
Definitions: https://librosa.org/doc/0.11.0/generated/librosa.filters.wavelet.html
             https://librosa.org/doc/0.11.0/generated/librosa.cqt.html
             https://librosa.org/doc/0.11.0/generated/librosa.vqt.html
"""
from pathlib import Path
import csv
import numpy as np
import scipy
from scipy.signal import correlate
import librosa

assert (librosa.__version__, np.__version__, scipy.__version__) == ('0.11.0', '2.3.5', '1.18.1')
ALPHA = (2**(2/12)-1)/(2**(2/12)+1)

def analysis(y, sr, hop, fmin, bins, gamma=0):
    frequencies = librosa.cqt_frequencies(n_bins=bins, fmin=fmin)
    basis, lengths = librosa.filters.wavelet(freqs=frequencies, sr=sr, gamma=gamma,
        alpha=ALPHA, norm=1, pad_fft=True, dtype=np.complex128)
    padded = np.pad(y, (basis.shape[1]//2, basis.shape[1]//2))
    return np.array([correlate(padded, b, mode='valid', method='direct')[::hop]
                     for b in basis]) * np.sqrt(lengths[:, None])

entries = {}
# A valid low-rate VQT yields a singleton Hann wavelet. SciPy/librosa define
# Hann(1)=1, whereas substituting n=0 into the ordinary formula gives zero.
entries['input_singleton'] = np.array([1., -.5, .25, 0.])
entries['vqt_singleton'] = analysis(entries['input_singleton'],32,1,1,1,24)
sr, hop, fmin, bins = 4096, 32, 128, 24
t = np.arange(1024)/sr
# Float32 input quantization is explicit, before independent double calculations.
rng = np.random.default_rng(171)
y = (.55*np.cos(2*np.pi*(fmin*2**(7/12))*t+.37) + .2*np.sin(2*np.pi*320*t)
     + .03*rng.normal(size=t.size)).astype(np.float32).astype(float)
entries['input_mix'] = y
for label, gamma in [('cqt_mix', 0), ('vqt_mix', 24)]:
    entries[label] = analysis(y,sr,hop,fmin,bins,gamma)
# Interior and boundary impulses expose centering, phase origin, and padding.
for index, label in [(0,'edge'),(511,'impulse')]:
    z = np.zeros(1024); z[index] = 1
    entries[f'input_{label}'] = z
    entries[f'cqt_{label}'] = analysis(z,sr,hop,fmin,bins)
    entries[f'vqt_{label}'] = analysis(z,sr,hop,fmin,bins,24)
# Longer tone provides stable selectivity and calibrated phase progression.
t = np.arange(4096)/sr
z = (.8*np.cos(2*np.pi*fmin*2**(7/12)*t+.37)).astype(np.float32).astype(float)
entries['input_tone'] = z
entries['cqt_tone'] = analysis(z,sr,256,fmin,bins)
entries['vqt_tone'] = analysis(z,sr,256,fmin,bins,24)
# Store production librosa outputs separately. They are comparison measurements,
# not pointwise parity targets for a different computational algorithm.
entries['librosa_cqt_tone'] = librosa.cqt(z,sr=sr,hop_length=256,fmin=fmin,n_bins=bins,
    tuning=0, norm=1, sparsity=0, scale=True, pad_mode='constant',dtype=np.complex128)
entries['librosa_vqt_tone'] = librosa.vqt(z,sr=sr,hop_length=256,fmin=fmin,n_bins=bins,
    tuning=0, gamma=24, norm=1, sparsity=0, scale=True, pad_mode='constant',dtype=np.complex128)
# An independently solved real least-squares inverse for a small full-column-rank
# system; construct columns through SciPy correlations, then use LAPACK SVD.
for label, invfmin, invbins in [('inverse',64,48),('narrow_inverse',256,12)]:
    ns, invsr, invhop = 17, 2048, 1
    matrix = np.column_stack([analysis(np.eye(ns)[i],invsr,invhop,invfmin,invbins)[:,:ns].ravel() for i in range(ns)])
    a = np.vstack((matrix.real,matrix.imag))
    x = np.sin(np.arange(ns)*.7)*.3
    c = (matrix @ x).reshape(invbins, ns)
    # Public coefficients are complex32; inverse reference solves exactly that input.
    c = c.astype(np.complex64).astype(np.complex128)
    solution, _, rank, singular = np.linalg.lstsq(a,np.r_[c.ravel().real,c.ravel().imag],rcond=None)
    print(f'{label} rank={rank}, condition={singular[0]/singular[-1]:.6g}')
    entries[label+'_input'] = c
    entries[label+'_solution'] = solution
out = Path(__file__).with_name('cqt_reference.csv')
with out.open('w') as f:
    f.write('# Generated with librosa 0.11.0, NumPy 2.3.5, SciPy 1.18.1\n')
    f.write('# Full-rate wavelet correlation; librosa recursive outputs are separately labelled.\n')
    writer = csv.writer(f,lineterminator='\n'); writer.writerow(('name','row','col','real','imag'))
    for name, data in entries.items():
        for (r,c),value in np.ndenumerate(np.atleast_2d(data)):
            writer.writerow((name,r,c,format(value.real,'.17g'),format(value.imag,'.17g')))
print(out)
