"""Independent transform references; run with numpy 2.3.5, scipy 1.18.1, librosa 0.11.0.

https://librosa.org/doc/0.11.0/generated/librosa.stft.html
https://librosa.org/doc/0.11.0/generated/librosa.istft.html
https://librosa.org/doc/0.11.0/generated/librosa.phase_vocoder.html
https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html

All inputs are first rounded to float32 and serialized exactly (17 digits).
References use float64 arithmetic on those same samples. Inverse/vocoder input
spectra are also rounded to complex64 before reference evaluation. Windows are
periodic; shorter windows are centered within the FFT; centered analysis uses
NumPy reflect padding. Explicit FFT lengths preserve odd-size conventions.
Resampling uses SciPy's default Kaiser FIR and zero extension; rubato uses a
different filter, so tests measure passband/interior and boundary errors separately.
"""
from pathlib import Path
import warnings
from functools import partial
from unittest.mock import patch
import librosa
import numpy as np
import scipy
from scipy.signal import chirp, resample_poly

assert (np.__version__, scipy.__version__, librosa.__version__) == ('2.3.5', '1.18.1', '0.11.0')
lines = ['# numpy=2.3.5 scipy=1.18.1 librosa=0.11.0; exact f32 inputs, float64 reference arithmetic',
         '# periodic windows; center=True uses reflect padding; explicit odd/even n_fft',
         'name,row,col,real,imag']

def put(name, value):
    a = np.atleast_2d(value)
    for (i, j), z in np.ndenumerate(a):
        lines.append(f'{name},{i},{j},{float(np.real(z)):.17g},{float(np.imag(z)):.17g}')

t = np.arange(257, dtype=np.float64)
y = (0.31*np.sin(2*np.pi*0.071*t + 0.2) + 0.17*np.cos(2*np.pi*0.183*t)
     + 0.001*((t*37)%19-9)).astype(np.float32)
y[[0, 79, 256]] += np.array([0.2, 0.6, -0.3], dtype=np.float32)
put('input', y)
# name, FFT, hop, window length, window, centered
cases = [('even',64,16,64,'hann',True), ('odd',63,13,63,'hann',True),
         ('short_window',64,11,39,'hamming',True),
         ('noncenter',64,16,64,'hamming',False),
         ('odd_noncenter',63,13,33,'hann',False)]
for name,n_fft,hop,win,window,center in cases:
    kwargs = dict(n_fft=n_fft, hop_length=hop, win_length=win, window=window, center=center)
    d = librosa.stft(y.astype(np.float64), pad_mode='reflect', **kwargs)
    put('stft_'+name, d)
    d32 = d.astype(np.complex64)
    put('spectrum_'+name, d32)
    for suffix,length in [('natural',None), ('length',len(y))]:
        put('istft_'+name+'_'+suffix, librosa.istft(d32.astype(np.complex128), length=length, **kwargs))
    if name in ('even','odd'):
        for rate in (0.75, 1.3, 2.0):
            r = float(np.float32(rate))
            tag = str(rate).replace('.','_')
            put('pv_'+name+'_'+tag, librosa.phase_vocoder(d32.astype(np.complex128), rate=r, hop_length=hop, n_fft=n_fft))
            # effects forwards kwargs to ISTFT too, which rejects pad_mode.
            # Adapt only the STFT default to our documented reflect convention.
            with patch.object(librosa.effects.core, 'stft', partial(librosa.stft, pad_mode='reflect')):
                put('stretch_'+name+'_'+tag, librosa.effects.time_stretch(y.astype(np.float64), rate=r, **kwargs))

short = np.array([0.2,-0.1,0.5,0.7,-0.4,0.3,-0.9,0.6,0.1,-0.2,0.8],dtype=np.float32)
put('input_short',short)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    put('stft_short_signal',librosa.stft(short.astype(np.float64),n_fft=64,hop_length=8,pad_mode='reflect'))

# A large Hann window exposes inverse normalization near a tiny nonzero endpoint.
large = (0.4 + 0.2*np.sin(np.arange(2048)*0.073)).astype(np.float32)
put('input_large',large)
d = librosa.stft(large.astype(np.float64),n_fft=2048,hop_length=512,center=False).astype(np.complex64)
put('spectrum_large',d)
put('istft_large',librosa.istft(d.astype(np.complex128),n_fft=2048,hop_length=512,center=False))

for name,orig,target in [('tone',8000,12000),('chirp',12000,8000),('multitone',44100,16000),('stopband',48000,16000)]:
    n=4097
    t=np.arange(n,dtype=np.float64)/orig
    if name=='tone': x=0.7*np.sin(2*np.pi*631*t+0.3)
    elif name=='chirp': x=0.7*chirp(t,f0=170,t1=t[-1],f1=2600,method='linear')
    elif name=='multitone': x=0.4*np.sin(2*np.pi*350*t+0.4)+0.3*np.cos(2*np.pi*2700*t)
    else: x=0.7*np.sin(2*np.pi*17760*t+0.3)
    x=x.astype(np.float32)
    put('resample_input_'+name,x)
    put('resample_'+name,resample_poly(x.astype(np.float64),target,orig,padtype='constant'))
impulse=np.zeros(257,dtype=np.float32)
impulse[128]=1
put('resample_input_impulse',impulse)
put('resample_impulse',resample_poly(impulse.astype(np.float64),3,2,padtype='constant'))
Path(__file__).with_name('transforms_reference.csv').write_text('\n'.join(lines)+'\n')
print(f'Generated {len(lines)-3} reference elements')
