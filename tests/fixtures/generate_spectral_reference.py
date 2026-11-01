#!/usr/bin/env python3
"""Regenerate: python -m pip install numpy==2.3.5 scipy==1.18.1 librosa==0.11.0
python tests/fixtures/generate_spectral_reference.py

Independent references; no Rust implementation is called. Inputs are first rounded
once to float32 and stored as exact float64 decimal values; reference algorithms
then run in float64 to avoid sharing Rust's roundoff. Seed: 20261101.

Conventions:
- Slaney mel scale and area normalization, power spectrogram; reflect-centered
  periodic Hann STFT, including odd n_fft. Identity spectra expose each mel weight.
- MFCC is scipy.fft.dct(log(max(mel, float32(1e-10))), type=2, norm='ortho').
  DASP intentionally uses natural log (NOT librosa's default dB) and always ortho.
- Features use magnitude; flatness uses power=1 and amin=float32(1e-10).
- Delta uses librosa.feature.delta(order=1, mode='nearest'), not default 'interp'.
- Inverse MFCC is exp(scipy.fft.idct(..., norm='ortho', n=n_mels)), with zero
  padding/truncation and type I endpoint normalization defined by SciPy.

- Waveform RMS: librosa.feature.rms(center=False), unwindowed. Spectrum RMS:
  librosa.feature.rms(S=magnitude), Parseval weighting of DC/Nyquist bins.
- Contrast: the crate's existing inclusive [0,200],[200,400],... octave bands;
  average upper/lower round-half-up 2% ranks and 10*log10((peak+1e-10)/(valley+1e-10)).
  Unlike librosa: no adjacent-bin expansion, final-band extension to Nyquist,
  hard dB floor, or 80 dB clipping. Quiet-input contrast is intentionally regularized.
  NumPy sorting/means evaluate this declared definition independently in float64.
- HPSS: SciPy generic_filter with NumPy upper median, ignoring NaN pads to match
  clipped boundary neighborhoods; librosa.util.softmask(power=1, split_zeros=True).
  The crate uses power-spectrum input on its waveform path, not librosa's usual
  magnitude/power-2 masks. Indeterminate masks split energy equally.

References:
https://librosa.org/doc/0.11.0/generated/librosa.feature.rms.html
https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_contrast.html
https://librosa.org/doc/0.11.0/generated/librosa.util.softmask.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html
https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html
https://librosa.org/doc/0.11.0/generated/librosa.stft.html
https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_centroid.html
https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_bandwidth.html
https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_rolloff.html
https://librosa.org/doc/0.11.0/generated/librosa.feature.spectral_flatness.html
https://librosa.org/doc/0.11.0/generated/librosa.feature.delta.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idct.html
"""
from pathlib import Path
import csv
import numpy as np
import scipy
from scipy.fft import dct, idct
from scipy.ndimage import generic_filter
import librosa

assert (np.__version__, scipy.__version__, librosa.__version__) == ('2.3.5', '1.18.1', '0.11.0')

def f32(value):
    return np.asarray(value, dtype=np.float32).astype(np.float64)

def contrast_reference(magnitude, sr, n_fft, n_bands=4):
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edges = np.r_[0., 200. * 2.**np.arange(n_bands+1)]
    result = np.zeros((n_bands+1, magnitude.shape[1]))
    for band, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        values = np.sort(magnitude[(frequencies >= low) & (frequencies <= high)], axis=0)
        if len(values):
            count = max(1, int(np.floor(float(np.float32(.02))*len(values)+.5)))
            epsilon = float(np.float32(1e-10))
            result[band] = 10*np.log10((values[-count:].mean(axis=0)+epsilon)/(values[:count].mean(axis=0)+epsilon))
    return result


def hpss_reference(spectrum, harmonic_width, percussive_width):
    def upper_median(values):
        return np.nanquantile(values, .5, method='higher')
    harmonic = generic_filter(spectrum, upper_median, size=(1, harmonic_width), mode='constant', cval=np.nan)
    percussive = generic_filter(spectrum, upper_median, size=(percussive_width, 1), mode='constant', cval=np.nan)
    mask = librosa.util.softmask(harmonic, percussive, power=1., split_zeros=True)
    other_mask = librosa.util.softmask(percussive, harmonic, power=1., split_zeros=True)
    return spectrum*mask, spectrum*other_mask

rng = np.random.default_rng(20261101)
noise = rng.standard_normal(384)
t = np.arange(384) / 8000
impulse = np.zeros(384)
impulse[[0, 79, 192, 383]] = [1, -.5, .75, -.25]
cases = [
    ('noise_even', noise, 128, 0., 4000.),
    ('noise_odd', noise, 127, 73., 3200.),
    ('tone_even', .7*np.sin(2*np.pi*631*t)+.2*np.cos(2*np.pi*1793*t), 128, 150., 3500.),
    ('impulse_odd', impulse, 127, 0., 4000.),
    ('quiet_even', noise*1e-12, 128, 0., 4000.),
    ('scaled_odd', noise*300., 127, 73., 3200.),
    ('silence_even', np.zeros(384), 128, 0., 4000.),
]
path = Path(__file__).with_name('spectral_reference.csv')
with path.open('w', newline='') as file:
    file.write('# numpy 2.3.5; scipy 1.18.1; librosa 0.11.0; seed 20261101\n')
    file.write('# Generated by generate_spectral_reference.py; exact f32 inputs, float64 independent outputs\n')
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['name', 'row', 'col', 'real', 'imag'])
    def save(name, array):
        array = np.atleast_2d(array)
        for (r, c), value in np.ndenumerate(array):
            writer.writerow([name, r, c, format(float(value), '.17g'), 0])
    for name, y, n_fft, fmin, fmax in cases:
        y = f32(y)
        save(name+'_input', y)
        filters = librosa.filters.mel(sr=8000, n_fft=n_fft, n_mels=12, fmin=fmin, fmax=fmax, dtype=np.float64)
        save(name+'_filters', filters)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=48, window='hann', center=True, pad_mode='reflect')
        magnitude = f32(np.abs(stft))
        save(name+'_magnitude', magnitude)
        save(name+'_rms_audio', librosa.feature.rms(y=y, frame_length=n_fft, hop_length=48, center=False, dtype=np.float64))
        save(name+'_rms_spectrum', librosa.feature.rms(S=magnitude, frame_length=n_fft, dtype=np.float64))
        save(name+'_contrast', contrast_reference(magnitude, 8000, n_fft))
        power = f32(np.abs(stft)**2)
        save(name+'_power', power)
        mel = filters @ power
        save(name+'_mel', mel)
        save(name+'_mel_audio', filters @ np.abs(stft)**2)
        mel_input = f32(mel)
        save(name+'_mel_input', mel_input)
        save(name+'_mfcc', dct(np.log(np.maximum(mel_input, float(np.float32(1e-10)))), type=2, axis=0, norm='ortho'))
        freq = librosa.fft_frequencies(sr=8000, n_fft=n_fft)
        save(name+'_centroid', librosa.feature.spectral_centroid(S=magnitude, freq=freq))
        for p in (1, 2, 3):
            save(name+f'_bandwidth{p}', librosa.feature.spectral_bandwidth(S=magnitude, freq=freq, p=p))
        save(name+'_rolloff', librosa.feature.spectral_rolloff(S=magnitude, freq=freq, roll_percent=float(np.float32(.85))))
        save(name+'_flatness', librosa.feature.spectral_flatness(S=magnitude, power=1., amin=float(np.float32(1e-10))))
    delta = f32(rng.normal(size=(9, 13)))
    save('delta_input', delta)
    for axis in (0, 1):
        for width in (3, 5, 9):
            save(f'delta_axis{axis}_width{width}', librosa.feature.delta(delta, width=width, order=1, axis=axis, mode='nearest'))
    coefficients = f32(rng.normal(scale=.7, size=(8, 4)))
    save('inverse_input', coefficients)
    for kind in (1, 2, 3, 4):
        for count in (5, 8, 12):
            save(f'inverse_type{kind}_mels{count}', np.exp(idct(coefficients, type=kind, n=count, axis=0, norm='ortho')))
    # Endpoint-only spectra isolate even/odd Parseval weights without FFT noise.
    for n_fft in (127, 128):
        bins = np.zeros((n_fft//2+1, 3))
        bins[0, 0] = n_fft
        bins[-1, 1] = n_fft
        bins[1, 2] = n_fft
        save(f'rms_bins{n_fft}_input', f32(bins))
        save(f'rms_bins{n_fft}', librosa.feature.rms(S=f32(bins), frame_length=n_fft, dtype=np.float64))
    # Additional amplitudes expose underflow/overflow in sums of squares.
    for name, scale in [('tiny', 1e-30), ('huge', 1e25)]:
        y = f32(noise * scale)
        save('rms_'+name+'_input', y)
        save('rms_'+name, librosa.feature.rms(y=y, frame_length=128, hop_length=48, center=False, dtype=np.float64))
    extreme = f32(np.linspace(.1, 1., 513)[:, None] * np.array([1., .8, .5]) * 3e38)
    save('contrast_extreme_input', extreme)
    save('contrast_extreme', contrast_reference(extreme, 8000, 1024))
    mixed = rng.uniform(.01, .2, size=(9, 11))
    mixed[3, :] += 3.
    mixed[:, 5] += 4.
    sparse = np.zeros((9, 11))
    sparse[4, 5] = 2.
    extreme_ratio = np.full((9, 11), 1e-30)
    extreme_ratio[4, :] = 1e30
    for name, spectrum in [('ratio', extreme_ratio), ('mixed', mixed), ('quiet', mixed*1e-30), ('loud', mixed*1e30),
                           ('huge', np.full((9, 11), 3e38)), ('sparse', sparse), ('silence', sparse*0.)]:
        spectrum = f32(spectrum)
        save('hpss_'+name+'_input', spectrum)
        for harmonic_width, percussive_width in [(3, 5), (31, 31)]:
            harmonic, percussive = hpss_reference(spectrum, harmonic_width, percussive_width)
            prefix = f'hpss_{name}_{harmonic_width}_{percussive_width}'
            save(prefix+'_harmonic', harmonic)
            save(prefix+'_percussive', percussive)
print(path)
