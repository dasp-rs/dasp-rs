# Audio corpus attribution and licenses

These recordings retain their Creative Commons licenses; the project MIT license does not replace them.
Each four-second excerpt was cropped, downmixed by channel mean if needed, resampled to 16 kHz with SciPy resample_poly (default Kaiser FIR, zero padding), and encoded as mono PCM16 FLAC. No level normalization or denoising was applied.
This collection has no pitch, onset, beat, source-separation, or transcript ground-truth annotations. The third clip is instrumental music with pitched percussion, not isolated drums. Upstream pack titles/BPM are attribution metadata, not validated labels.
The clips exercise decoding and DSP workflows; three excerpts do not establish performance on a general real-world population.

## speech.flac
Ashiel Mystery - A Detective Story, chapter 2 — Garth Comira
Category: speech narration
Original source: https://www.openslr.org/12/
Retrieved via librosa example `libri1`: https://librosa.org/data/audio/5703-47212-0000.ogg
License: CC-BY-4.0 — https://creativecommons.org/licenses/by/4.0/
Excerpt: 0.5–4.5 seconds; modifications are listed above.
Complete upstream notice is retained verbatim in `speech.source-notice.txt`.

## trumpet.flac
Solo trumpet 06, Jazz Trumpet Loops Pack in F 90 bpm — Mihai Sorohan
Category: monophonic instrumental music
Original source: https://freesound.org/s/77711/
Retrieved via librosa example `trumpet`: https://librosa.org/data/audio/sorohanro_-_solo-trumpet-06.ogg
License: CC-BY-3.0 — https://creativecommons.org/licenses/by/3.0/
Excerpt: 0.0–4.0 seconds; modifications are listed above.
Complete upstream notice is retained verbatim in `trumpet.source-notice.txt`.

## pitched_percussion.flac
Vibe Ace — Kevin MacLeod
Category: instrumental music with pitched percussion (vibraphone/piano/bass)
Original source: https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace
Retrieved via librosa example `vibeace`: https://librosa.org/data/audio/Kevin_MacLeod_-_Vibe_Ace.ogg
License: CC-BY-3.0 — https://creativecommons.org/licenses/by/3.0/
Excerpt: 2.0–6.0 seconds; modifications are listed above.
Complete upstream notice is retained verbatim in `pitched_percussion.source-notice.txt`.

