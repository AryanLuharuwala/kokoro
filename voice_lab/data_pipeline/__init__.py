"""Dataset pipeline: YouTube -> segments -> transcripts -> manifest -> fit.

Entry points:

* ``voice_lab.data_pipeline.config.load_config(path)``
* ``voice_lab.data_pipeline.pipeline.build_dataset(cfg)``
* ``voice_lab.data_pipeline.pipeline.fit_voice(cfg)``
* ``voice_lab.data_pipeline.pipeline.run(cfg)`` (build + fit)

CLI: ``python -m voice_lab build-dataset --config dataset.toml``
     ``python -m voice_lab pipeline      --config dataset.toml``
"""

from .config import PipelineConfig, load_config
from .pipeline import build_dataset, finetune_model, fit_voice, run

__all__ = [
    "PipelineConfig",
    "load_config",
    "build_dataset",
    "fit_voice",
    "finetune_model",
    "run",
]
