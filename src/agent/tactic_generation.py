from typing import List, Optional
from typing_extensions import Self
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch.nn as nn


class TacticGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
        )


    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in folder/filename
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


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1024, 1)

    def forward(self, x):
        return self.dense(x)

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
        self.dense.train(mode)
        return self

    def eval(self) -> Self:
        super().eval()
        self.dense.eval()
        return self
