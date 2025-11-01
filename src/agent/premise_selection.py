"""
Premise selection module using a ByT5-based text encoder from ReProver.
"""

from typing_extensions import Self
import torch
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForTextEncoding
import torch.nn as nn

from lean_dojo import Theorem
from ReProver.common import Pos
from ..utilities.dataloader import DataLoader


class PremiseSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-byt5-small"
        )
        self.model = AutoModelForTextEncoding.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-byt5-small"
        )
        self.dataloader = DataLoader()

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
        premise_embs = self._encode(premises)
        scores = state_emb @ premise_embs.T
        topk = scores.topk(k).indices.tolist()
        return [premises[i] for i in topk]

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        self.model.eval()
        return self
