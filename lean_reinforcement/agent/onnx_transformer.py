"""
ONNX Runtime-based Transformer for faster, lower-memory tactic generation.

This module provides :class:`ONNXTransformer`, a drop-in replacement for
:class:`Transformer` that uses ONNX Runtime (via HuggingFace Optimum)
instead of native PyTorch for inference.  Benefits:

- ~30 % lower CPU memory (no PyTorch autograd graph in memory)
- Potentially faster inference via ORT's graph optimizations & operator fusion
- Same beam-search interface via ``optimum``'s ``ORTModelForSeq2SeqLM``

Usage::

    # First, export the model (one-time):
    python -m lean_reinforcement.agent.onnx_transformer --export

    # Then use it wherever you used Transformer:
    from lean_reinforcement.agent.onnx_transformer import ONNXTransformer
    t = ONNXTransformer()
    tactics = t.generate_tactics("⊢ ∀ n, 0 + n = n", n=8)

Requirements (install via ``pip install optimum[onnxruntime-gpu]``)::

    pip install optimum onnxruntime-gpu

If only CPU inference is needed::

    pip install optimum onnxruntime
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Tuple, cast

import torch
from loguru import logger

from transformers import AutoTokenizer

from lean_reinforcement.utilities.memory import periodic_cache_cleanup

_ONNX_AVAILABLE = False
_ORTModelForSeq2SeqLM = None
try:
    from importlib import import_module

    _ORTModelForSeq2SeqLM = import_module("optimum.onnxruntime").ORTModelForSeq2SeqLM
    _ONNX_AVAILABLE = True
except Exception:
    _ORTModelForSeq2SeqLM = None

# Default cache location for ONNX-exported models
_ONNX_CACHE_DIR = Path(
    os.environ.get(
        "LEAN_RL_ONNX_CACHE",
        Path.home() / ".cache" / "lean-reinforcement" / "onnx_models",
    )
)


def is_onnx_available() -> bool:
    """Return True if ``optimum`` + ``onnxruntime`` are installed."""
    return _ONNX_AVAILABLE


def _require_onnx() -> Any:
    if not _ONNX_AVAILABLE or _ORTModelForSeq2SeqLM is None:
        raise ImportError(
            "Install `optimum` and `onnxruntime-gpu` to use ONNX Runtime:\n"
            "  pip install optimum onnxruntime-gpu"
        )
    return _ORTModelForSeq2SeqLM


def export_model(
    model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small",
    output_dir: Path | None = None,
) -> Path:
    """Export a HuggingFace seq2seq model to ONNX format.

    Args:
        model_name: HuggingFace model name or local path.
        output_dir: Where to save the ONNX model.  Defaults to
            ``~/.cache/lean-reinforcement/onnx_models/<model_name>``.

    Returns:
        The directory containing the ONNX model files.
    """
    if not _ONNX_AVAILABLE:
        raise ImportError(
            "Install `optimum` and `onnxruntime-gpu` to export ONNX models:\n"
            "  pip install optimum onnxruntime-gpu"
        )

    if output_dir is None:
        safe_name = model_name.replace("/", "--")
        output_dir = _ONNX_CACHE_DIR / safe_name

    if (output_dir / "encoder_model.onnx").exists():
        logger.info(f"ONNX model already exported at {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting {model_name} to ONNX at {output_dir} ...")

    ort_model = _require_onnx()
    model = ort_model.from_pretrained(
        model_name,
        export=True,
    )
    model.save_pretrained(output_dir)

    # Also save the tokenizer alongside
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    logger.success(f"ONNX export complete: {output_dir}")
    return output_dir


class ONNXTransformer:
    """Drop-in replacement for ``Transformer`` using ONNX Runtime.

    Implements :class:`TransformerProtocol`.  If the ONNX model hasn't been
    exported yet, falls back to auto-exporting on first use.
    """

    def __init__(
        self,
        model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small",
        onnx_dir: Path | str | None = None,
    ):
        if not _ONNX_AVAILABLE:
            raise ImportError(
                "Install `optimum` and `onnxruntime-gpu` to use ONNXTransformer:\n"
                "  pip install optimum onnxruntime-gpu"
            )

        if onnx_dir is not None:
            onnx_path = Path(onnx_dir)
        else:
            safe_name = model_name.replace("/", "--")
            onnx_path = _ONNX_CACHE_DIR / safe_name

        # Auto-export if needed
        if not (onnx_path / "encoder_model.onnx").exists():
            logger.info("ONNX model not found, exporting ...")
            export_model(model_name, onnx_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        provider = (
            "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        )
        ort_model = _require_onnx()
        self.model = ort_model.from_pretrained(onnx_path, provider=provider)
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        self._generate_call_count = 0
        logger.info(f"ONNXTransformer loaded from {onnx_path} ({provider})")

    # -- TransformerProtocol implementation ----------------------------

    def generate_tactics(self, state: str, n: int = 1) -> List[str]:
        tokenized = self.tokenizer(
            state, return_tensors="pt", truncation=True, max_length=2048
        )
        input_length = tokenized.input_ids.shape[1]

        output_ids = self.model.generate(
            **tokenized,
            max_length=input_length + 512,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            length_penalty=0.0,
        )
        result = cast(
            List[str], self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        )
        self._generate_call_count = periodic_cache_cleanup(self._generate_call_count)
        return result

    def generate_tactics_with_probs(
        self, state: str, n: int = 1
    ) -> List[Tuple[str, float]]:
        tokenized = self.tokenizer(
            state, return_tensors="pt", truncation=True, max_length=2048
        )
        input_length = tokenized.input_ids.shape[1]

        outputs = self.model.generate(
            **tokenized,
            max_length=input_length + 512,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        tactics = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        probs = torch.softmax(outputs.sequences_scores, dim=0).tolist()
        result = list(zip(tactics, probs))
        self._generate_call_count = periodic_cache_cleanup(self._generate_call_count)
        return result

    def generate_tactics_batch(self, states: List[str], n: int = 1) -> List[List[str]]:
        if not states:
            return []
        tokenized = self.tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_length = tokenized.input_ids.shape[1]

        output_ids = self.model.generate(
            **tokenized,
            max_length=input_length + 512,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            length_penalty=0.0,
        )
        all_tactics = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        result = [all_tactics[i : i + n] for i in range(0, len(all_tactics), n)]
        self._generate_call_count = periodic_cache_cleanup(self._generate_call_count)
        return result

    def generate_tactics_with_probs_batch(
        self, states: List[str], n: int = 1
    ) -> List[List[Tuple[str, float]]]:
        if not states:
            return []
        tokenized = self.tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_length = tokenized.input_ids.shape[1]

        outputs = self.model.generate(
            **tokenized,
            max_length=input_length + 512,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        all_tactics = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        scores = outputs.sequences_scores

        results = []
        for i in range(len(states)):
            s, e = i * n, (i + 1) * n
            batch_probs = torch.softmax(scores[s:e], dim=0).tolist()
            results.append(list(zip(all_tactics[s:e], batch_probs)))
        self._generate_call_count = periodic_cache_cleanup(self._generate_call_count)
        return results


# -- CLI entry point for exporting ------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--export", action="store_true", help="Export model")
    parser.add_argument(
        "--model",
        default="kaiyuy/leandojo-lean4-tacgen-byt5-small",
        help="HuggingFace model name",
    )
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    if args.export:
        out = Path(args.output) if args.output else None
        export_model(args.model, out)
    else:
        parser.print_help()
