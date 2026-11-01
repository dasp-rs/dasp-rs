"""Independent reassignment raster, numpy2.3.5/scipy1.18.1/librosa0.11.0.

Reference equations: https://librosa.org/doc/0.11.0/generated/librosa.reassigned_spectrogram.html
All input samples are first quantized to float32, then analyzed in float64.
librosa STFTs use SciPy periodic Hann, its analytic derivative, and a sample-
centered time-weighted window. Centering is reflect padding with hop=max(N//4,1).
This API returns a MAGNITUDE RASTER, not librosa's frequency/time coordinate arrays.
Each nonzero coefficient contributes its magnitude to the nearest valid frequency
bin and time frame. Half-grid ties round upward; a 64*eps64 relative neighborhood
of exact half-grid coordinates is treated as a tie to stabilize FFT roundoff.
No absolute amplitude threshold discards quiet coefficients. Out-of-grid estimates
are clamped. Periodic Hann's analytic derivative differs from librosa's cyclic
finite difference by sinc(2*pi/N); the time origin differs by 0.5 samples for even
N. A separate librosa.reassigned_spectrogram call checks those known conversions.
"""
from pathlib import Path
import warnings
import numpy as np
import scipy
from scipy.signal import get_window, chirp
from scipy.fft import rfft
import librosa
assert (np.__version__,scipy.__version__,librosa.__version__)==('2.3.5','1.18.1','0.11.0')
entries={}
sr=1024
n=np.arange(257,dtype=float)
mix=.3*np.sin(2*np.pi*91*n/sr+.4)+.2*np.cos(2*np.pi*237*n/sr+.2)
impulse=np.zeros(257);impulse[73]=.8
edges=np.zeros(257);edges[[0,256]]=[.7,-.4]
half=np.zeros(257);half[8]=1
signals=[('mix',mix,64),('odd',mix,63),('chirp',.6*chirp(n/sr,f0=61,f1=371,t1=n[-1]/sr),64),
         ('tone',.7*np.cos(2*np.pi*128*n/sr+.3),64),('half_bin',.7*np.cos(2*np.pi*136*n/sr+.3),64),
         ('impulse',impulse,64),('edges',edges,64),('half_time',half,64),('silence',np.zeros(257),64),
         ('quiet',mix*1e-30,64),('loud',mix*1e30,64),('subnormal',mix*1e-40,64),
         ('tiny_fft',np.array([.2,-.4,.7,-.1,.3,-.8,.5,.2,-.3]),2),
         ('dc',np.full(257,.3),64),('nyquist',.3*(-1.)**n,64)]

def nearest(value,size):
    value=np.clip(value,0,size-1)
    half=np.floor(2*value+.5)/2
    value=np.where(abs(value-half)<=64*np.finfo(float).eps*np.maximum(1,abs(value)),half,value)
    return np.minimum(np.floor(value+.5).astype(int),size-1)

for name,original,fft in signals:
    y=original.astype(np.float32).astype(float)
    entries['input_'+name]=y
    hop=max(fft//4,1)
    w=get_window('hann',fft,fftbins=True)
    dw=np.pi/fft*np.sin(2*np.pi*np.arange(fft)/fft)
    tw=(np.arange(fft)-fft//2)*w
    kwargs=dict(n_fft=fft,hop_length=hop,center=True,pad_mode='reflect')
    if fft == 2:
        # librosa0.11's optimized final-frame pad reflects a one-sample tail
        # as a singleton here. Use the explicit NumPy reflect definition,
        # followed by independent SciPy FFT, to preserve the declared boundary.
        frames=np.lib.stride_tricks.sliding_window_view(np.pad(y,fft//2,mode='reflect'),fft)[::hop]
        d,dt,df=(rfft(frames*window,axis=1).T for window in (w,tw,dw))
    else:
        d=librosa.stft(y,window=w,**kwargs)
        dt=librosa.stft(y,window=tw,**kwargs)
        df=librosa.stft(y,window=dw,**kwargs)
    magnitude=abs(d)
    frequencies=np.broadcast_to(np.arange(d.shape[0])[:,None],d.shape).astype(float).copy()
    times=np.broadcast_to(np.arange(d.shape[1])[None,:],d.shape).astype(float).copy()
    supported=magnitude>0
    frequencies[supported]-=np.imag(df[supported]/d[supported])*fft/(2*np.pi)
    times[supported]+=np.real(dt[supported]/d[supported])/hop
    raster=np.zeros(d.shape)
    np.add.at(raster,(nearest(frequencies,d.shape[0])[supported],nearest(times,d.shape[1])[supported]),magnitude[supported])
    entries['raster_'+name]=raster
    entries['magnitude_'+name]=magnitude
    # Independently check the parameter conversions against librosa's public
    # coordinate-returning API on well-supported coefficients (not null bins).
    if fft>2:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lf,lt,lm=librosa.reassigned_spectrogram(y,sr=sr,ref_power=0,fill_nan=True,clip=False,**kwargs)
        strong=magnitude>max(magnitude.max()*1e-8,np.finfo(float).tiny)
        grid=np.arange(d.shape[0])[:,None]*sr/fft
        factor=np.sin(2*np.pi/fft)/(2*np.pi/fft)
        corrected_f=grid+(lf-grid)/factor
        corrected_t=lt-(.5/sr if fft%2==0 else 0)
        assert np.allclose(corrected_f[strong],frequencies[strong]*sr/fft,rtol=1e-8,atol=1e-7),name
        assert np.allclose(corrected_t[strong],times[strong]*hop/sr,rtol=1e-8,atol=1e-8),name

lines=['# numpy=2.3.5 scipy=1.18.1 librosa=0.11.0; float32 inputs, float64 reference',
       '# periodic Hann; analytic derivatives; reflect center; nearest-grid magnitude raster',
       'name,row,col,real,imag']
for name,value in entries.items():
    for (row,col),v in np.ndenumerate(np.atleast_2d(value)):
        lines.append(f'{name},{row},{col},{v:.17g},0')
Path(__file__).with_name('reassignment_reference.csv').write_text('\n'.join(lines)+'\n')
print(f'Generated {len(lines)-3} reassignment reference values')
