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
pip install -e '.[dataset]'    # also adds yt-dlp + librosa + faster-whisper
                               # for the YouTube -> manifest pipeline
```

You also need ``ffmpeg`` (for yt-dlp) and ``espeak-ng`` (for Hindi G2P) on
the system: ``sudo apt-get install ffmpeg espeak-ng``.


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

## YouTube -> voice pack (one command)

For Shinchan specifically there's an end-to-end pipeline that:

1. downloads each YouTube URL to 24 kHz mono WAV (`yt-dlp` + `ffmpeg`)
2. splits each clip into speech regions (Silero VAD)
3. keeps only regions whose **median pitch is in Shinchan's range** —
   ~220–450 Hz, well above adult speakers (default `min_in_range_ratio = 0.6`)
4. transcribes each kept region with `faster-whisper` (Hindi)
5. writes a JSONL manifest
6. fits a voice pack via the same gradient-descent optimizer as below

Edit `voice_lab/examples/shinchan_hindi.toml` (replace the placeholder
URLs), then:

```bash
python -m voice_lab pipeline --config voice_lab/examples/shinchan_hindi.toml
```

Stages cache to `data/shinchan/{raw,segments,manifest.jsonl}`, so reruns
only redo what's missing. Run just the data side without fitting:

```bash
python -m voice_lab build-dataset --config voice_lab/examples/shinchan_hindi.toml
```

Tuning notes:

* If almost no segments survive: lower `min_in_range_ratio` or widen
  `[min_f0, max_f0]`. Use `method = "all"` to disable the pitch filter
  entirely and inspect what VAD produced.
* If too many adult-speaker clips slip through: raise `min_f0` (e.g. 250)
  or `min_in_range_ratio` (e.g. 0.75).
* If transcripts look wrong: bump the Whisper model (`small` -> `medium`
  -> `large-v3`).

## Fitting from data manually

If you already have a curated dataset (no YouTube needed):

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
* `data_pipeline/` — YouTube → manifest pipeline:
  * `download.py` — yt-dlp + ffmpeg → 24 kHz mono WAV (cached by video id)
  * `segment.py` — Silero VAD + librosa.pyin pitch filter for Shinchan range
  * `transcribe.py` — faster-whisper, cached per-segment as JSON
  * `pipeline.py` — orchestrator (idempotent, incremental)
  * `config.py` — TOML config schema

### Parameterisation

* `--parameterization shared` (default): one 256-d vector broadcast across
  all 510 length buckets. Small (1 KB), fast to fit, matches the
  single-speaker assumption.
* `--parameterization per-length`: full `[510, 256]` tensor. Higher
  capacity, but probably overkill for one voice.

## Fine-tuning model weights (heavier path)

`voice_lab fit` only optimizes the 256-d style vector. To actually update
the Kokoro-82M model weights:

```bash
python -m voice_lab finetune \
    --manifest data/shinchan/manifest.jsonl \
    --out-dir checkpoints/shinchan \
    --lang h \
    --preset voice \
    --epochs 20 \
    --val-sentence "नमस्ते, मैं शिनचान हूँ।"
```

Or set `[finetune].enabled = true` in the TOML config and rerun
`voice_lab pipeline`.

**Presets** (which weights are trainable):

* `style_only` — same as `voice_lab fit`; only the 256-d style.
* `voice` — recommended. Trains `bert_encoder`, `text_encoder`, decoder,
  and the F0/N branches of the prosody predictor. Keeps BERT frozen.
* `full` — every parameter (note: duration LSTM/proj will still receive
  zero gradient, see below).

**Loss:** multi-resolution log-mel L1 + multi-resolution STFT magnitude
loss (Yamamoto-style spectral convergence + log-mag L1). The STFT loss is
weighted lower (default 0.5) but matters: without a vocoder discriminator
it's the main thing keeping the decoder from drifting toward muffled
output.

**Outputs:**

* `<out-dir>/sample_epochNNN.wav` after each validation epoch — listen!
* `<out-dir>/ckpt_epochNNN.pth` every `save_every_epochs`.
* `<out-dir>/kokoro-finetuned.pth` + `config.json` at the end. Load with
  `KModel(config="<out-dir>/config.json", model="<out-dir>/kokoro-finetuned.pth")`.

### Known limits of this fine-tuner

* **Duration won't change.** The forward pass uses `torch.round` on
  predicted duration before building the alignment matrix, which blocks
  gradient flow into the duration LSTM and projection. Cadence stays close
  to whatever the pretrained model produces. (Original StyleTTS2 trains
  duration with MFA-aligned targets — out of scope here.)
* **No discriminator.** Mel + STFT-magnitude losses preserve spectral
  shape but don't enforce perceptual realism the way HiFi-GAN-style
  adversarial training does. Keep epochs modest (10–30) and watch the
  validation samples for muffling.
* **No SLM / WavLM loss.** Same reason — the auxiliary models from
  Kokoro's original training pipeline aren't open-sourced.
* **Limited Hindi support upstream.** Kokoro ships only `hf_alpha`,
  `hf_beta`, `hm_omega`, `hm_psi` for Hindi. Fine-tuning extends one of
  them toward Shinchan; some phonemes may still be weak.

### When to use a heavier setup instead

If this fine-tuner's quality ceiling isn't enough, the community paths
that use the full StyleTTS2 training stack (with discriminator, SLM, JDC,
ASR aligner) are:

* [`semidark/kokoro-deutsch`](https://github.com/semidark/kokoro-deutsch)
  — a Kokoro-specific recipe with weight conversion and two-stage
  training. Closest to "real Kokoro fine-tuning".
* [`IIEleven11/StyleTTS2FineTune`](https://github.com/IIEleven11/StyleTTS2FineTune)
  — single-voice wrapper around StyleTTS2's official `train_finetune.py`.
  Loses Kokoro's prior (starts from StyleTTS2-LibriTTS) but is the
  best-documented end-to-end recipe.

## Limitations of the embedding path

* The model can never produce a phoneme it wasn't trained to produce. Hindi
  voices in stock Kokoro are limited; some Shinchan-isms may not transfer.
* Fitting only the style vector cannot change duration distributions much,
  because duration is jointly determined by text + style + frozen weights.
  Severe pacing differences would need actual fine-tuning of the model
  weights (see "Fine-tuning model weights" above).
* If your transcripts are wrong, the loss will fight the text encoder. Use
  clean transcripts.
