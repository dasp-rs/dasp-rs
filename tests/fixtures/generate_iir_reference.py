#!/usr/bin/env python3
"""Independent Butterworth/SOS and causal 12-semitone RMS filterbank references.

Install tests/fixtures/requirements-reference.txt in a Python virtual environment,
then run: python tests/fixtures/generate_iir_reference.py
Pinned NumPy 2.3.5, SciPy 1.18.1. No Rust implementation is invoked.

Contract: centers 32.7*2**(b/12), b=0..11, half-semitone critical frequencies;
scipy.signal.butter(4, [low, high], fs=sr, btype='bandpass', output='sos').
Filter the entire signal causally from zero state, then take uncentered frame RMS.
This is explicitly different from librosa.iirt's multirate zero-phase power output.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz_sos.html
"""
from pathlib import Path
import csv
import numpy as np
import scipy
from scipy.signal import butter,sosfilt,freqz_sos,chirp
assert (np.__version__,scipy.__version__) == ('2.3.5','1.18.1')
entries={}
impulse=np.zeros(8192);impulse[0]=1
indices=np.array(list(range(16))+[31,63,127,255,511,1023,2047,4095,8191])
entries['impulse_indices']=indices
for name,sr,low,high,order in [('narrow',44100,31.7,33.7,4),('wide',8000,300,900,4),
                              ('nyquist',8000,2800,3600,2),('odd_order',16000,120,150,3)]:
    sos=butter(order,[low,high],fs=sr,btype='bandpass',output='sos')
    frequencies=np.array([0,low*.5,low,np.sqrt(low*high),high,high*1.1,(high+sr/2)/2,sr/2])
    _,response=freqz_sos(sos,worN=frequencies,fs=sr)
    entries[name+'_config']=np.array([sr,low,high,order])
    entries[name+'_sos']=sos
    entries[name+'_frequencies']=frequencies
    entries[name+'_response']=response
    entries[name+'_impulse']=sosfilt(sos,impulse)[indices]

cases=[('silence',44100,2048,2048,512),('tone',8000,64000,8000,8000),
       ('chirp',44100,8193,2048,511),('transient',16000,4097,1024,333),
       ('noise',48000,8192,2048,512),('low_rate',128,1024,128,64)]
for name,sr,n,win,hop in cases:
    t=np.arange(n)/sr
    if name=='tone': y=.6*np.cos(2*np.pi*32.7*2**(5/12)*t+.37)
    elif name=='chirp': y=.5*chirp(t,f0=20,f1=100,t1=t[-1],method='linear',phi=23)
    elif name=='transient':
        y=np.zeros(n);y[[0,1023,1024,n-1]]=[1,-.5,.25,1]
    elif name=='noise':y=np.random.default_rng(318).uniform(-.3,.3,n)
    elif name=='low_rate':y=.7*np.cos(2*np.pi*32.7*2**(11/12)*t+.1)
    else:y=np.zeros(n)
    y=y.astype(np.float32).astype(float)
    result=[]
    for b in range(12):
        center=32.7*2**(b/12)
        sos=butter(4,[center*2**(-1/24),center*2**(1/24)],fs=sr,btype='bandpass',output='sos')
        filtered=sosfilt(sos,y)
        result.append([np.sqrt(np.mean(filtered[start:start+win]**2)) for start in range(0,n-win+1,hop)])
    entries[name+'_iirt_config']=np.array([sr,win,hop])
    entries[name+'_input']=y
    entries[name+'_iirt']=np.array(result)

out=Path(__file__).with_name('iir_reference.csv')
with out.open('w') as f:
    f.write('# NumPy 2.3.5 / SciPy 1.18.1; independent SOS design, impulse, response, causal RMS\n')
    writer=csv.writer(f,lineterminator='\n');writer.writerow(('name','row','col','real','imag'))
    for name,data in entries.items():
        for (row,col),value in np.ndenumerate(np.atleast_2d(data)):
            writer.writerow((name,row,col,format(value.real,'.17g'),format(value.imag,'.17g')))
print(f'{out}: {out.stat().st_size} bytes')
