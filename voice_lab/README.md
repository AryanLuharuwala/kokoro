# voice_lab

Tools for building a custom Kokoro voice. Two paths:

1. **Blend** existing voices → instant, no data required.
2. **Fit** a voice via gradient descent against a small dataset of
   `(audio, transcript)` pairs.

The Kokoro model is **frozen** in both cases. We only produce a new voice
pack file (a `[510, 256]` tensor saved as `.pt`) that the existing
inference pipeline already knows how to load.

## Install

```bash
pip install -e '.[training]'   # adds soundfile + resampy for audio I/O
```

For Hindi (`lang_code="h"`) you also need `espeak-ng` on the system:

```bash
# Debian/Ubuntu
sudo apt-get install espeak-ng
# macOS
brew install espeak-ng
```

## Quickstart: blend a Shinchan-flavoured voice

```bash
python -m voice_lab blend \
    --weights hm_psi=0.5,hf_alpha=0.4,hf_beta=0.1 \
    --out voices/shinchan_seed.pt
```

Try it:

```bash
python -m voice_lab speak \
    --voice voices/shinchan_seed.pt \
    --lang h \
    --text "नमस्ते, मैं शिनचान हूँ।" \
    --out shinchan_hello.wav
```

The Kokoro Hindi voice inventory is small (`hf_alpha`, `hf_beta`, `hm_omega`,
`hm_psi`); blending male+female lifts pitch toward a child voice. This is a
seed, not a finished Shinchan — for a real likeness you need step 2.

## Fitting from data

### Prepare a dataset

Create a folder and a JSONL manifest:

```
data/shinchan/
├── manifest.jsonl
└── clips/
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

`manifest.jsonl` — one JSON object per line:

```json
{"audio": "clips/0001.wav", "text": "ओहो, ये क्या हो गया!"}
{"audio": "clips/0002.wav", "text": "मम्मी, मुझे चॉकलेट चाहिए।"}
```

Notes:

* Audio: any format `soundfile` reads (WAV/FLAC). Resampled to 24 kHz
  automatically. Mono is fine; stereo is averaged.
* Length: short clips work best (2–10s). Trim silence at the edges.
* Transcripts: in the target language (Hindi for `--lang h`). Accuracy
  matters — use Whisper if you don't have transcripts:

  ```bash
  pip install openai-whisper
  whisper clips/0001.wav --language Hindi --output_format json
  ```

* Quantity: 30 clips / ~3 minutes of audio is a reasonable start.

### Run fitting

```bash
python -m voice_lab fit \
    --manifest data/shinchan/manifest.jsonl \
    --lang h \
    --init-voice hm_psi \
    --epochs 50 \
    --lr 1e-2 \
    --out voices/shinchan.pt
```

Then synthesise as before:

```bash
python -m voice_lab speak \
    --voice voices/shinchan.pt \
    --lang h \
    --text "मम्मी, मैं अभी होमवर्क नहीं करूँगा!" \
    --out shinchan_test.wav
```

## How it works

* `blend.py` — load voice packs from HF Hub, weighted average, save to disk.
* `losses.py` — multi-resolution log-mel L1 loss (no torchaudio dep).
  Compares the overlapping prefix, so it tolerates duration drift.
* `data.py` — JSONL manifest → audio + text iterator. Resamples to 24 kHz.
* `optimize.py` — replicates `KModel.forward_with_tokens` without the
  `@torch.no_grad()` decorator, freezes the entire model, and trains only
  the style vector with AdamW.

### Parameterisation

* `--parameterization shared` (default): one 256-d vector broadcast across
  all 510 length buckets. Small (1 KB), fast to fit, matches the
  single-speaker assumption.
* `--parameterization per-length`: full `[510, 256]` tensor. Higher
  capacity, but probably overkill for one voice.

## Limitations

* The model can never produce a phoneme it wasn't trained to produce. Hindi
  voices in stock Kokoro are limited; some Shinchan-isms may not transfer.
* Fitting only the style vector cannot change duration distributions much,
  because duration is jointly determined by text + style + frozen weights.
  Severe pacing differences would need actual fine-tuning of the model
  weights (see project root TODO for the architecture-shrink path).
* If your transcripts are wrong, the loss will fight the text encoder. Use
  clean transcripts.
