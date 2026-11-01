#!/usr/bin/env python3
"""Prepare or verify three small, attributed, redistributable audio fixtures.

  python scripts/prepare_audio_corpus.py --verify  # stdlib only, no network
  python scripts/prepare_audio_corpus.py --prepare --cache-dir CACHE [--download]
  python scripts/prepare_audio_corpus.py --check-references  # pinned audio environment

Preparation requires numpy2.3.5/scipy1.18.1/librosa0.11.0/soundfile0.14.0 and
libsndfile1.2.2. Downloads occur ONLY with explicit --download, and every cached or
downloaded full source/notice/license must match its pinned SHA256. The checked-in
FLAC clips, not an upstream decoder, define Rust test inputs. --verify has no
third-party imports and needs neither the original sources nor internet access.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'tests/fixtures/audio'
BASE='https://librosa.org/data/audio/'
SOURCES=[
 dict(key='speech',example='libri1',stem='5703-47212-0000',author='Garth Comira',
      title='Ashiel Mystery - A Detective Story, chapter 2',category='speech narration',
      original_url='https://www.openslr.org/12/',license='CC-BY-4.0',start_seconds=.5,
      sha256='a284612b46af0535f7e1873758c4387bb8369f6dbbe192ffdec1f171108f98dd',
      notice_sha256='1c5bd5eb792ff0d2fbe895e3a3dd0acf3beb48ee364e64e12cde3a646304a9f4'),
 dict(key='trumpet',example='trumpet',stem='sorohanro_-_solo-trumpet-06',author='Mihai Sorohan',
      title='Solo trumpet 06, Jazz Trumpet Loops Pack in F 90 bpm',category='monophonic instrumental music',
      original_url='https://freesound.org/s/77711/',license='CC-BY-3.0',start_seconds=0.,
      sha256='8374466fd3951d24509da6e799b132a0db0bdeda69d99c69d989a6888d3d727d',
      notice_sha256='750a191b9d0cc94b2f19cfbf11acc13783f75bedcccfcf562f5b076efd068aba'),
 dict(key='pitched_percussion',example='vibeace',stem='Kevin_MacLeod_-_Vibe_Ace',author='Kevin MacLeod',
      title='Vibe Ace',category='instrumental music with pitched percussion (vibraphone/piano/bass)',
      original_url='https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace',license='CC-BY-3.0',start_seconds=2.,
      sha256='6c23aed3dd5aa57f2b1652ecab68d15d9b82ad257f54e639eb2880ca09bc118a',
      notice_sha256='6c71e0525cb0452ea74c6d6f5fde6fa1e221223db7b2aa35b9914b98367ee7b9'),
]
LICENSES={
 'CC-BY-3.0':('https://creativecommons.org/licenses/by/3.0/legalcode.txt','e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76'),
 'CC-BY-4.0':('https://creativecommons.org/licenses/by/4.0/legalcode.txt','9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    manifest=(DEST/'manifest.json').read_bytes()
    expected=(DEST/'manifest.sha256').read_text().split()[0]
    if digest(manifest)!=expected:
        raise ValueError('manifest.json hash differs from manifest.sha256')
    data=json.loads(manifest)
    for name,expected in data['artifact_sha256'].items():
        if digest((DEST/name).read_bytes())!=expected:
            raise ValueError(f'Corpus hash mismatch: {name}')
    for clip in data['clips']:
        path=DEST/clip['file']
        payload=path.read_bytes()
        if digest(payload)!=clip['clip_sha256']:
            raise ValueError(f'Clip hash mismatch: {path}')
        if path.stat().st_size!=clip['clip_bytes'] or payload[:4]!=b'fLaC':
            raise ValueError(f'Invalid FLAC file/size: {path}')
    print(f"Verified {len(data['clips'])} clips and {len(data['artifact_sha256'])} artifacts; no downloads")
    return data


def libraries():
    import numpy as np
    import scipy
    import librosa
    import soundfile as sf
    expected=('2.3.5','1.18.1','0.11.0','0.14.0','1.2.2')
    actual=(np.__version__,scipy.__version__,librosa.__version__,sf.__version__,sf.__libsndfile_version__)
    if actual!=expected:
        raise ValueError(f'Use pinned numpy/scipy/librosa/soundfile/libsndfile versions {expected}; found {actual}')
    return np,scipy,librosa,sf


def cached(cache,name,url,sha,download):
    path=cache/name
    if not path.exists():
        if not download:
            raise ValueError(f'Missing cached {path}; use --download explicitly to fetch {url}')
        import requests
        response=requests.get(url,timeout=60)
        response.raise_for_status()
        if digest(response.content)!=sha:
            raise ValueError(f'Download hash mismatch: {url}')
        cache.mkdir(parents=True,exist_ok=True)
        path.write_bytes(response.content)
    if digest(path.read_bytes())!=sha:
        raise ValueError(f'Cached source hash mismatch: {path}')
    return path


def references():
    np,scipy,librosa,sf=libraries()
    from scipy.fft import dct
    from scipy.signal import resample_poly
    entries={}
    frames=[0,31,89,160,250]
    points=[0,1,17,511,5000,31999,63999]
    for source in SOURCES:
        key=source['key']
        y,sr=sf.read(DEST/f'{key}.flac',dtype='float32')
        assert sr==16000 and len(y)==64000
        y=y.astype(np.float64)
        entries[key+'_decoded_samples']=y[points]
        entries[key+'_waveform_summary']=np.array([y.min(),y.max(),y.mean(),np.sqrt(np.mean(y*y)),np.mean(abs(y))])
        d=librosa.stft(y,n_fft=1024,hop_length=256,window='hann',center=True,pad_mode='reflect')
        power=abs(d)**2
        mel=librosa.feature.melspectrogram(S=power,sr=sr,n_fft=1024,n_mels=40,fmin=0,fmax=8000,htk=False,norm='slaney')
        # dasp MFCC uses natural-log mel power, not librosa's default dB input.
        cepstra=dct(np.log(np.maximum(mel,1e-10)),type=2,norm='ortho',axis=0)[:13]
        entries[key+'_mel_selected']=mel[:,frames]
        entries[key+'_mfcc_selected']=cepstra[:,frames]
        entries[key+'_mel_mean']=mel.mean(axis=1)
        entries[key+'_mfcc_mean']=cepstra.mean(axis=1)
        down=resample_poly(y,3,4,padtype='constant')
        entries[key+'_resampled_rms']=np.array([np.sqrt(np.mean(down[256:-256]**2))])
    out=io.StringIO()
    out.write('# numpy2.3.5 scipy1.18.1 librosa0.11.0; inputs decoded from checked-in PCM16 FLAC\n')
    out.write('# periodic Hann1024, hop256, reflect center; Slaney40 mel; natural-log ortho DCT-II13\n')
    writer=csv.writer(out,lineterminator='\n')
    writer.writerow(('name','row','col','real','imag'))
    for name,value in entries.items():
        for (r,c),v in np.ndenumerate(np.atleast_2d(value)):
            writer.writerow((name,r,c,format(v,'.17g'),0))
    return out.getvalue().encode()


def prepare(args):
    np,scipy,librosa,sf=libraries()
    from scipy.signal import resample_poly
    DEST.mkdir(parents=True,exist_ok=True)
    clips=[]
    attribution=['# Audio corpus attribution and licenses','',
      'These recordings retain their Creative Commons licenses; the project MIT license does not replace them.',
      'Each four-second excerpt was cropped, downmixed by channel mean if needed, resampled to 16 kHz with SciPy resample_poly (default Kaiser FIR, zero padding), and encoded as mono PCM16 FLAC. No level normalization or denoising was applied.',
      'This collection has no pitch, onset, beat, source-separation, or transcript ground-truth annotations. The third clip is instrumental music with pitched percussion, not isolated drums. Upstream pack titles/BPM are attribution metadata, not validated labels.',
      'The clips exercise decoding and DSP workflows; three excerpts do not establish performance on a general real-world population.','']
    for source in SOURCES:
        key,stem=source['key'],source['stem']
        original=cached(args.cache_dir,stem+'.ogg',BASE+stem+'.ogg',source['sha256'],args.download)
        notice=cached(args.cache_dir,stem+'.txt',BASE+stem+'.txt',source['notice_sha256'],args.download)
        shutil.copyfile(notice,DEST/f'{key}.source-notice.txt')
        audio,sr=sf.read(original,dtype='float32',always_2d=True)
        start=round(source['start_seconds']*sr);count=4*sr
        segment=audio[start:start+count].astype(np.float64).mean(axis=1)
        if len(segment)!=count:
            raise ValueError(f'Source too short: {original}')
        clip=resample_poly(segment,16000,sr,padtype='constant')
        if np.max(abs(clip))>=1:
            raise ValueError(f'Clip would require PCM16 clipping: {key}')
        path=DEST/f'{key}.flac'
        sf.write(path,clip,16000,format='FLAC',subtype='PCM_16')
        decoded,_=sf.read(path,dtype='float32')
        info=sf.info(original)
        record={**source,'source_sha256':source['sha256'],'source_format':info.format,
                'source_subtype':info.subtype,'source_url':BASE+stem+'.ogg','source_notice_url':BASE+stem+'.txt',
                'license_url':f"https://creativecommons.org/licenses/by/{source['license'].rsplit('-',1)[1]}/",
                'source_sample_rate':sr,'source_channels':audio.shape[1],'source_frames':len(audio),
                'source_start_sample':start,'source_samples_used':count,
                'file':path.name,'clip_sha256':digest(path.read_bytes()),'clip_bytes':path.stat().st_size,
                'clip_sample_rate':16000,'clip_channels':1,'clip_samples':len(decoded),'clip_duration_seconds':4,
                'clip_subtype':'PCM_16','decoded_f32le_sha256':digest(decoded.astype('<f4').tobytes())}
        del record['sha256']
        clips.append(record)
        attribution += [f"## {path.name}",f"{source['title']} — {source['author']}",
          f"Category: {source['category']}",f"Original source: {source['original_url']}",
          f"Retrieved via librosa example `{source['example']}`: {BASE+stem+'.ogg'}",
          f"License: {source['license']} — {record['license_url']}",
          f"Excerpt: {source['start_seconds']:.1f}–{source['start_seconds']+4:.1f} seconds; modifications are listed above.",
          f"Complete upstream notice is retained verbatim in `{key}.source-notice.txt`.",'']
    for name,(url,sha) in LICENSES.items():
        license_path=cached(args.cache_dir,name+'.txt',url,sha,args.download)
        shutil.copyfile(license_path,DEST/(name+'.txt'))
    (DEST/'ATTRIBUTION.md').write_text('\n'.join(attribution)+'\n')
    (DEST/'corpus_reference.csv').write_bytes(references())
    files=[p for p in DEST.iterdir() if p.name not in ('manifest.json','manifest.sha256') and p.is_file()]
    manifest={'schema_version':1,'corpus':'three attributed four-second mono16k excerpts',
        'generation_versions':{'numpy':np.__version__,'scipy':scipy.__version__,'librosa':librosa.__version__,
                               'soundfile':sf.__version__,'libsndfile':sf.__libsndfile_version__},
        'conversion':'float32 source decode; float64 channel mean; scipy resample_poly default Kaiser; PCM16 FLAC; no normalization',
        'reference_conventions':{'n_fft':1024,'hop_length':256,'window':'periodic Hann','center':'reflect',
          'n_mels':40,'mel':'Slaney area normalization','mfcc':'13 natural-log mel coefficients, DCT-II ortho',
          'selected_frames':[0,31,89,160,250],'selected_samples':[0,1,17,511,5000,31999,63999]},
        'licenses':{k:{'legalcode_url':v[0],'legalcode_sha256':v[1]} for k,v in LICENSES.items()},
        'clips':clips,'artifact_sha256':{p.name:digest(p.read_bytes()) for p in sorted(files)},
        'limitations':['No annotated pitch, beat, onset, transcript, or separation ground truth.',
                       'Pitched-percussion example is a music mix, not an isolated-drum recording.',
                       'Small codec/workflow regression corpus, not a population accuracy benchmark.']}
    payload=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
    (DEST/'manifest.json').write_bytes(payload)
    (DEST/'manifest.sha256').write_text(digest(payload)+'  manifest.json\n')
    verify()


def check_references():
    verify()
    expected=list(csv.DictReader(line for line in (DEST/'corpus_reference.csv').read_text().splitlines() if not line.startswith('#')))
    actual=list(csv.DictReader(line for line in references().decode().splitlines() if not line.startswith('#')))
    if len(expected)!=len(actual):raise ValueError('Reference count changed')
    import math
    for a,b in zip(expected,actual):
        if any(a[k]!=b[k] for k in ('name','row','col')) or not math.isclose(float(a['real']),float(b['real']),rel_tol=1e-10,abs_tol=5e-12):
            raise ValueError(f'Independent reference drift: {a} != {b}')
    print(f'Independent corpus reference summaries verified: {len(actual)} elements')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--verify',action='store_true',help='Offline stdlib-only hash verification (default)')
    mode.add_argument('--prepare',action='store_true',help='Rebuild clips/notices/manifest/references from verified cached sources')
    mode.add_argument('--check-references',action='store_true',help='Recompute reference summaries without modifying checked-in files')
    parser.add_argument('--download',action='store_true',help='Permit explicit preparation to download missing pinned sources')
    parser.add_argument('--cache-dir',type=Path,default=Path.home()/'.cache/librosa')
    args=parser.parse_args()
    if args.download and not args.prepare:parser.error('--download requires --prepare')
    if args.prepare:prepare(args)
    elif args.check_references:check_references()
    else:verify()

if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as error:
        print(error,file=sys.stderr)
        raise SystemExit(1) from error
