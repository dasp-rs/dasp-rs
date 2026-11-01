#!/usr/bin/env python3
"""Independent STFT phase-recovery and NMF reconstruction references.

Install requirements-reference.txt, then run this file. No Rust output is used.
Griffin-Lim uses librosa STFT/ISTFT with the crate's deterministic initial
broadband noise, reflect padding, periodic Hann windows, zero momentum and explicit
output length. This matches the declared iteration, not librosa's default
random initialization/momentum. Target magnitudes are quantized to float32.
NMF fixtures specify known nonnegative low-rank products; tests check objectives
and scale behavior rather than requiring a particular nonunique factorization.

https://librosa.org/doc/0.11.0/generated/librosa.griffinlim.html
https://papers.nips.cc/paper_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html
"""
import csv
from pathlib import Path
import librosa
import numpy as np
import scipy

assert (np.__version__, scipy.__version__, librosa.__version__) == ('2.3.5', '1.18.1', '0.11.0')
rng = np.random.default_rng(77419)
entries = {}
for name, fft, hop, length in [('small',16,4,64), ('ordinary',64,16,256), ('short',64,4,16)]:
    t = np.arange(length)
    source = (.3*np.sin(t*.21)+.1*np.cos(t*.44)+rng.normal(scale=.03,size=length)).astype(np.float32).astype(float)
    magnitude = np.abs(librosa.stft(source,n_fft=fft,hop_length=hop,pad_mode='reflect')).astype(np.float32).astype(float)
    entries[name+'_metadata'] = [fft,hop,length]
    entries[name+'_input'] = magnitude
    # Match only the documented initial state, not Rust transform arithmetic.
    state = 0x12345678
    initial = []
    for _ in range(length):
        state = (1664525*state+1013904223) & 0xffffffff
        initial.append(np.float32(np.float32(state >> 8)*np.float32(2**-24)-np.float32(.5))*np.float32(.02))
    y = np.array(initial,dtype=float)
    for iteration in range(1,13):
        spectrum = librosa.stft(y,n_fft=fft,hop_length=hop,pad_mode='reflect')
        phase = np.ones_like(spectrum)
        np.divide(spectrum,np.abs(spectrum),out=phase,where=np.abs(spectrum)>0)
        y = librosa.istft(magnitude*phase,hop_length=hop,n_fft=fft,length=length)
        if iteration in (1,4,12):
            entries[f'{name}_iteration{iteration}'] = y
left = rng.uniform(.1,1,(9,3)); right = rng.uniform(.1,1,(3,15))
entries['nmf_dense_input'] = (left@right).astype(np.float32).astype(float)
out = Path(__file__).with_name('recovery_reference.csv')
with out.open('w',newline='') as file:
    file.write('# NumPy2.3.5 SciPy1.18.1 librosa0.11.0; seed77419; matched deterministic Griffin-Lim iteration\n')
    writer=csv.writer(file,lineterminator='\n');writer.writerow(['name','row','col','real','imag'])
    for name,value in entries.items():
        for (r,c),v in np.ndenumerate(np.atleast_2d(value)):
            writer.writerow([name,r,c,format(v,'.17g'),'0'])
print(out)
