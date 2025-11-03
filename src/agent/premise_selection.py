"""
Premise selection module using a ByT5-based text encoder from ReProver.
"""

import torch
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForTextEncoding


class PremiseSelector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-byt5-small"
        )
        self.model = AutoModelForTextEncoding.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-byt5-small"
        )
        # Cache for premise embeddings to avoid recomputation
        self._premise_cache = {}

    @torch.no_grad()
    def _encode(self, s: Union[str, List[str]]) -> torch.Tensor:
        """Encode texts into feature vectors."""
        if isinstance(s, str):
            s = [s]
            should_squeeze = True
        else:
            should_squeeze = False
        tokenized_s = self.tokenizer(s, return_tensors="pt", padding=True)
        hidden_state = self.model(tokenized_s.input_ids).last_hidden_state
        lens = tokenized_s.attention_mask.sum(dim=1)
        features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)
        if should_squeeze:
            features = features.squeeze()
        return features

    @torch.no_grad()
    def retrieve(self, state: str, premises: List[str], k: int) -> List[str]:
        """Retrieve the top-k premises from a list given a state."""
        state_emb = self._encode(state)

        # Create cache key from premises tuple
        cache_key = tuple(premises)

        # Check if we've already encoded these premises
        if cache_key in self._premise_cache:
            premise_embs = self._premise_cache[cache_key]
        else:
            premise_embs = self._encode(premises)
            self._premise_cache[cache_key] = premise_embs

        scores = state_emb @ premise_embs.T
        topk = scores.topk(k).indices.tolist()
        return [premises[i] for i in topk]

    def clear_cache(self):
        """Clear the premise embedding cache to free memory."""
        self._premise_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
