#!/usr/bin/env python3
"""Independent forward wavelet sweep, not Rust-generated golden output.

Pinned: librosa 0.11.0, NumPy 2.3.5, SciPy 1.18.1.
Create a virtual environment and install the pinned reference dependencies:
    python -m pip install -r tests/fixtures/requirements-reference.txt
    python tests/fixtures/generate_cqt_sweep_reference.py

Filters come from librosa.filters.wavelet; scipy.signal.correlate computes the
full-rate reference. Only selected frames are stored, including both boundaries
and adjacent interior frames. Input quantization occurs before any reference
calculation. Production recursive librosa.cqt/vqt outputs are explicitly separate
comparison-only data. No tolerances are selected from observed Rust errors.

https://librosa.org/doc/0.11.0/generated/librosa.filters.wavelet.html
https://librosa.org/doc/0.11.0/generated/librosa.cqt.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
"""
from pathlib import Path
import csv
import numpy as np
import scipy
from scipy.signal import correlate, chirp
import librosa

assert (librosa.__version__, np.__version__, scipy.__version__) == ('0.11.0','2.3.5','1.18.1')
ALPHA = (2**(2/12)-1)/(2**(2/12)+1)
# name,sr,length,fmin,bins,hop,kind,amplitude,tone_bin
CASES = [
    ('singleton_8k',8000,1,220,1,1,'impulse',1,-1),
    ('odd_short_8k',8000,63,100.3,24,17,'mixed',.8,-1),
    ('chirp_8k',8000,4097,55,60,128,'chirp',.7,-1),
    ('noise_16k',16000,4096,110,48,96,'noise',.2,-1),
    ('tone_16k',16000,8193,32.7,84,256,'tone',.6,31),
    ('tiny_16k',16000,35,440,12,64,'mixed',1e-18,-1),
    ('transient_44k',44100,513,32.7,84,512,'transient',.8,-1),
    ('tone_44k',44100,16385,220,60,257,'tone',.7,23),
    ('chirp_44k',44100,2049,65.4,36,73,'chirp',.6,-1),
    ('tone_48k',48000,4097,1000,36,128,'tone',.5,13),
    ('large_48k',48000,513,55,24,255,'noise',1e20,-1),
    ('silence_48k',48000,1,200,12,1,'silence',0,-1),
    ('subnormal_8k',8000,65,220,12,7,'mixed',1e-40,-1),
    ('tone_single_bin',16000,2049,440,1,31,'tone',.9,0),
]
entries = {}
for index,(name,sr,n,fmin,bins,hop,kind,amplitude,tone_bin) in enumerate(CASES):
    # fmin is an f32 public argument: preserve its exact value in the reference.
    fmin = float(np.float32(fmin))
    t = np.arange(n)/sr
    if kind == 'tone':
        y = amplitude*np.cos(2*np.pi*fmin*2**(tone_bin/12)*t+.37)
    elif kind == 'mixed':
        y = amplitude*(.7*np.cos(2*np.pi*fmin*2**(5/12)*t+.21)+.3*np.sin(2*np.pi*fmin*2**(9/12)*t))
    elif kind == 'chirp':
        y = amplitude*chirp(t, f0=fmin*.75, f1=fmin*2**((bins-1)/12)*1.1, t1=t[-1],method='logarithmic',phi=23)
    elif kind == 'noise':
        y = amplitude*np.random.default_rng(9700+index).uniform(-1,1,n)
    elif kind == 'transient':
        y = np.zeros(n); y[[0,n//2,n-1]] = amplitude*np.array([1,-.7,.5])
    elif kind == 'impulse':
        y = np.zeros(n); y[0] = amplitude
    else:
        y = np.zeros(n)
    y = y.astype(np.float32).astype(float)
    frames = 1+n//hop
    selected = np.array(sorted({0,min(1,frames-1),max(0,frames//2-1),frames//2,
                                min(frames-1,frames//2+1),max(0,frames-2),frames-1}))
    entries[name+'_config'] = np.array([sr,fmin,bins,hop,amplitude,tone_bin,.37])
    entries[name+'_input'] = y
    entries[name+'_frames'] = selected
    freqs = librosa.cqt_frequencies(n_bins=bins,fmin=fmin)
    for transform,gamma in [('cqt',0),('vqt',24)]:
        basis,lengths = librosa.filters.wavelet(freqs=freqs,sr=sr,gamma=gamma,
            alpha=ALPHA,norm=1,pad_fft=True,dtype=np.complex128)
        padded = np.pad(y,(basis.shape[1]//2,basis.shape[1]//2))
        # SciPy FFT correlation keeps regeneration fast for 44.1k/C1 filters.
        # Its f64 rounding is far below the tested public f32 accuracy.
        result = np.array([correlate(padded,b,mode='valid',method='fft')[selected*hop]
                           for b in basis])*np.sqrt(lengths[:,None])
        entries[f'{transform}_{name}'] = result
        if kind == 'tone':
            # Independent closed-form positive-frequency sinusoid response.
            # Small negative-frequency leakage is separately bounded in Rust.
            middle = frames//2
            ideal = amplitude*.5*np.sqrt(lengths[tone_bin])*np.exp(1j*(2*np.pi*freqs[tone_bin]*middle*hop/sr+.37))
            entries[f'{transform}_{name}_calibration'] = np.array([ideal])
        # Recursive implementations require enough powers of two in hop_length.
        if name in ('tone_16k','tone_48k','noise_16k','transient_44k','chirp_8k'):
            kwargs = dict(sr=sr,hop_length=hop,fmin=fmin,n_bins=bins,tuning=0,
                          norm=1,sparsity=0,scale=True,pad_mode='constant',dtype=np.complex128)
            recursive = (librosa.cqt(y,**kwargs) if gamma==0 else librosa.vqt(y,gamma=gamma,**kwargs))
            entries[f'recursive_librosa_{transform}_{name}'] = recursive[:,selected]
            # Interior means every nonzero full-rate wavelet tap is inside the
            # signal, not merely that the frame is different from first/last.
            # The short transient and low CQT bands can have no such frame.
            nonzero = np.flatnonzero(np.any(basis != 0,axis=0))
            first_offset = nonzero[0]-basis.shape[1]//2
            last_offset = nonzero[-1]-basis.shape[1]//2
            interior = ((selected*hop+first_offset >= 0) & (selected*hop+last_offset < n))
            entries[f'{transform}_{name}_interior_mask'] = interior.astype(float)
    print(f'{name}: sr={sr}, n={n}, bins={bins}, hop={hop}, frames={frames}, stored={len(selected)}')

out = Path(__file__).with_name('cqt_sweep_reference.csv')
with out.open('w') as f:
    f.write('# librosa 0.11.0 / NumPy 2.3.5 / SciPy 1.18.1; independent selected-frame sweep\n')
    f.write('# Config vector: sample_rate,fmin,bins,hop,amplitude,tone_bin,phase\n')
    writer = csv.writer(f,lineterminator='\n'); writer.writerow(('name','row','col','real','imag'))
    for name,data in entries.items():
        for (row,col),value in np.ndenumerate(np.atleast_2d(data)):
            writer.writerow((name,row,col,format(value.real,'.17g'),format(value.imag,'.17g')))
print(f'{out}: {out.stat().st_size} bytes')
