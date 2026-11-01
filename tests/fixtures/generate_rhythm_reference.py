#!/usr/bin/env python3
"""Pinned references: numpy==2.3.5 scipy==1.18.1 librosa==0.11.0.
Run python tests/fixtures/generate_rhythm_reference.py; no Rust code is called.

Conventions deliberately differ from default librosa rhythm analysis:
* Onsets: centered periodic-Hann STFT with reflect padding (zero padding when
  len(y)<=n_fft//2), sum of positive adjacent log10(power) differences; floor1e-10.
* Multiband: average positive dB differences in equal integer-width linear bins;
  last band absorbs remainder. Frame zero is zero, matching the single envelope.
* Tempogram: symmetric Hann, zero-centered padding, unnormalized SciPy correlation,
  only lags0..win//2. A singleton Hann window is unity.
* Fourier tempogram: magnitude STFT of the above envelope, periodic Hann,
  centered reflect padding, hop1. It is not librosa's default onset pipeline.
* Peak picking: seven-frame clipped max/mean, inclusive threshold, positive
  evidence, minimum spacing >=wait. Sparse pulses allow exact analytical timing.
References:
https://librosa.org/doc/0.11.0/generated/librosa.stft.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html
"""
from pathlib import Path
import csv
import numpy as np
import scipy
from scipy.signal import correlate, windows
import librosa
assert (np.__version__,scipy.__version__,librosa.__version__)==('2.3.5','1.18.1','0.11.0')
rng=np.random.default_rng(711)
entries={}
def f32(x):return np.asarray(x,dtype=np.float32).astype(np.float64)
def stft(y,n,hop):return librosa.stft(y,n_fft=n,hop_length=hop,pad_mode='reflect' if len(y)>n//2 else 'constant')
def onset(y,n,hop):
    power=np.abs(stft(y,n,hop))**2
    logs=np.log10(np.maximum(power,float(np.float32(1e-10))))
    flux=np.maximum(np.diff(logs,axis=1),0)
    single=np.r_[0,flux.sum(axis=0)]
    multi=np.zeros((3,len(single)))
    width=logs.shape[0]//3
    for band in range(3):
        lo=band*width;hi=logs.shape[0] if band==2 else (band+1)*width
        multi[band,1:]=10*flux[lo:hi].mean(axis=0)
    return single,multi
base=rng.normal(scale=.15,size=1024)
base[300:400]+=np.sin(np.arange(100)*.4)
for name,y,n in [('even',base,128),('odd',base,127),('quiet',base*1e-6,128),('huge',base*1e20,128),('silence',np.zeros(1024),128)]:
    y=f32(y);entries[name+'_input']=y
    entries[name+'_onset'],entries[name+'_multi']=onset(y,n,32)
envelope=f32(np.abs(rng.normal(size=23)))
for scale,name in [(1.,'normal'),(1e-12,'quiet'),(1e10,'loud')]:
    env=f32(envelope*scale);entries['env_'+name]=env
    for win in [1,9,10]:
        padded=np.pad(env,(win//2,win//2))
        out=[]
        for t in range(len(env)):
            frame=padded[t:t+win]*windows.hann(win,sym=True)
            out.append(correlate(frame,frame,mode='full',method='direct')[win-1:win+win//2])
        entries[f'tgram_{name}_{win}']=np.array(out).T
long=f32(rng.normal(scale=.1,size=4096));long[1500:1900]*=5
entries['fourier_input']=long
env,_=onset(long,2048,128)
for win in [9,10]:entries[f'fourier_{win}']=np.abs(stft(env,win,1))
# PLP's existing local autocorrelation-weighted cosine lobes, not librosa's
# Fourier-phase PLP. This fixture checks that declared curve, not beat quality.
for win in [32,64]:
    padded=np.pad(env,(win//2,win//2));pulse=np.zeros(len(env))
    lags=np.arange(1,win//2+1);bpms=60*4096/(128*lags)
    valid=lags[(bpms>=30)&(bpms<=300)]
    for t in range(len(env)):
        frame=padded[t:t+win]*windows.hann(win,sym=True)
        ac=correlate(frame,frame,mode='full',method='direct')[win-1:]
        best=valid[np.argmax(ac[valid][::-1])*-1-1]
        for delta in range(int(np.ceil(best/2))+1):
            lobe=ac[best]*max(0,np.cos(np.pi*delta/best))
            if t+delta<len(env):pulse[t+delta]+=lobe
            if delta>0 and t>=delta:pulse[t-delta]+=lobe
    entries[f'plp_{win}']=pulse/max(pulse.max(),1e-10)
pulses=np.zeros(4096);positions=np.arange(256,4096,512);pulses[positions]=1
entries['pulses_input']=pulses
entries['pulse_onsets']=(positions-32)//32
entries['pulse_times']=(positions-32)/4096
# Regular long click trains; BPM is exact on the chosen sample/frame grid.
for period,name in [(2048,'120'),(3072,'80')]:
    clicks=np.zeros(period*12+1024);clicks[np.arange(1024,len(clicks)-512,period)]=1
    entries['clicks_'+name]=clicks
out=Path(__file__).with_name('rhythm_reference.csv')
with out.open('w',newline='') as file:
    file.write('# numpy2.3.5 scipy1.18.1 librosa0.11.0 seed711; exactf32 inputs\n')
    writer=csv.writer(file,lineterminator='\n');writer.writerow(['name','row','col','real','imag'])
    for name,array in entries.items():
        for (r,c),v in np.ndenumerate(np.atleast_2d(array)):writer.writerow([name,r,c,format(float(v),'.17g'),0])
print(out)
