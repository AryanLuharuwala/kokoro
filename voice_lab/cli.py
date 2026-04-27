"""CLI entrypoints for voice_lab.

Subcommands::

    python -m voice_lab blend  --weights hm_psi=0.5,hf_alpha=0.4,hf_beta=0.1 \
                              --out voices/shinchan_seed.pt
    python -m voice_lab fit    --manifest data/shinchan/manifest.jsonl \
                              --lang h --init-voice hm_psi --epochs 50 \
                              --out voices/shinchan.pt
    python -m voice_lab speak  --voice voices/shinchan.pt --lang h \
                              --text "नमस्ते, मैं शिनचान हूँ।" \
                              --out shinchan_hello.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict


def _parse_weights(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise argparse.ArgumentTypeError(f"expected NAME=WEIGHT, got {piece!r}")
        name, w = piece.split("=", 1)
        out[name.strip()] = float(w)
    if not out:
        raise argparse.ArgumentTypeError("no weights supplied")
    return out


def cmd_blend(args: argparse.Namespace) -> int:
    from .blend import blend_voices, save_voice_pack, SHINCHAN_HINDI_PRESET

    weights = (
        _parse_weights(args.weights) if args.weights else SHINCHAN_HINDI_PRESET
    )
    pack = blend_voices(weights)
    out = save_voice_pack(pack, args.out)
    print(f"wrote {out}  ({tuple(pack.shape)}, weights={weights})")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from kokoro import KPipeline

    from .data import VoiceDataset
    from .optimize import OptimizerConfig, StyleOptimizer

    pipeline = KPipeline(lang_code=args.lang, device=args.device)
    dataset = VoiceDataset(args.manifest)
    cfg = OptimizerConfig(
        lr=args.lr,
        epochs=args.epochs,
        parameterization=args.parameterization,
        init_voice=args.init_voice,
        init_blend=_parse_weights(args.init_blend) if args.init_blend else None,
    )
    opt = StyleOptimizer(pipeline, dataset, cfg)
    opt.fit()
    out = opt.export(args.out)
    print(f"wrote {out}")
    return 0


def cmd_speak(args: argparse.Namespace) -> int:
    import soundfile as sf

    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=args.lang, device=args.device)
    audio_chunks = []
    for _, _, audio in pipeline(args.text, voice=args.voice):
        if audio is not None:
            audio_chunks.append(audio.detach().cpu().numpy())
    if not audio_chunks:
        print("no audio generated", file=sys.stderr)
        return 1
    import numpy as np

    wav = np.concatenate(audio_chunks)
    sf.write(args.out, wav, 24000)
    print(f"wrote {args.out}  ({len(wav) / 24000:.2f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice_lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("blend", help="Blend existing voices into a .pt pack")
    b.add_argument(
        "--weights",
        type=str,
        default=None,
        help="NAME=WEIGHT[,NAME=WEIGHT...]; defaults to the Shinchan-Hindi preset",
    )
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(func=cmd_blend)

    f = sub.add_parser("fit", help="Optimize a voice pack against audio+transcripts")
    f.add_argument("--manifest", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--lang", type=str, default="h")
    f.add_argument("--lr", type=float, default=1e-2)
    f.add_argument("--epochs", type=int, default=50)
    f.add_argument(
        "--parameterization", choices=["shared", "per-length"], default="shared"
    )
    f.add_argument("--init-voice", type=str, default=None)
    f.add_argument("--init-blend", type=str, default=None)
    f.add_argument("--device", type=str, default=None)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("speak", help="Synthesize text using a voice pack")
    s.add_argument("--voice", type=str, required=True, help="voice name or .pt path")
    s.add_argument("--text", type=str, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--lang", type=str, default="h")
    s.add_argument("--device", type=str, default=None)
    s.set_defaults(func=cmd_speak)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
