"""Regenerate the pitch fixture with librosa==0.11.0 (optional developer tool).

Reference: https://librosa.org/doc/0.11.0/generated/librosa.pyin.html
The Rust tests need only the checked-in CSV, not Python or librosa.
Our fixed half-frame difference window differs from librosa 0.11's full-frame
formulation; tests compare pitch within 15 cents, not bitwise output/probability parity.
YIN intentionally reports zero for silence instead of a fallback frequency.
"""
import librosa
import numpy as np
from pathlib import Path
assert librosa.__version__ == "0.11.0", "Use librosa==0.11.0 to regenerate this fixture"
sr, frame, hop, length = 8000, 512, 128, 2048
t = np.arange(length, dtype=np.float64) / sr
cases = {
 'tone_100': 0.5*np.sin(2*np.pi*100*t),
 'tone_220': 0.5*np.sin(2*np.pi*220*t),
 'tone_440': 0.5*np.sin(2*np.pi*440*t),
 'quiet_220': 1e-10*np.sin(2*np.pi*220*t),
 'missing_150': 0.5*np.sin(2*np.pi*300*t)+0.3*np.sin(2*np.pi*450*t),
 'silence': np.zeros(length),
}
lines = ['# librosa 0.11.0; sr=8000; frame_length=512; hop_length=128; center=False; fmin=80; fmax=600',
 '# Default librosa thresholds/HMM settings. All input signals are cast to float32.',
 'case,frame,yin_hz,pyin_hz,voiced']
for name,y in cases.items():
 y=y.astype(np.float32)
 kwargs=dict(sr=sr,fmin=80,fmax=600,frame_length=frame,hop_length=hop,center=False)
 a=librosa.yin(y,**kwargs)
 b,v,p=librosa.pyin(y,fill_na=0.0,**kwargs)
 for i in range(len(a)):
  lines.append(f'{name},{i},{a[i]:.9f},{b[i]:.9f},{int(v[i])}')
Path(__file__).with_name('pitch_librosa_0_11.csv').write_text('\n'.join(lines)+'\n')
print(f'Generated {len(lines)-3} reference frames with librosa {librosa.__version__}')
