"""LoRA actor helpers used by PPO variants."""

from __future__ import annotations

from typing import Any, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def build_lora_actor(
    model_name: str,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> Tuple[nn.Module, AutoTokenizer]:
    """Create a frozen seq2seq model with trainable LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "k", "v", "o"],
    )
    peft_model = get_peft_model(base_model, lora_config)
    return cast(nn.Module, peft_model), tokenizer


def get_action_log_probs(
    model: nn.Module,
    encoder_input_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-sequence log-probabilities for decoder action tokens."""
    outputs = cast(
        Any,
        model(
            input_ids=encoder_input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        ),
    )
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    target_ids = decoder_input_ids[:, 1:]
    aligned_log_probs = log_probs[:, :-1, :]
    token_log_probs = aligned_log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs.sum(dim=-1)
