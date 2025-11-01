from typing_extensions import Self
import torch
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTextEncoding


class TacticGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
        )
        # Set to evaluation mode
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    @torch.no_grad()
    def generate_tactics(
        self, state: str, retrieved_premises: List[str], n: int = 1
    ) -> List[str]:
        """
        Generates n tactics using beam search.
        """
        input = "\n\n".join(retrieved_premises + [state])
        tokenized_input = self.tokenizer(
            input, return_tensors="pt", max_length=2300, truncation=True
        )

        if torch.cuda.is_available():
            tokenized_input = tokenized_input.to("cuda")

        # Generate n tactics using beam search
        tactic_ids = self.model.generate(
            tokenized_input.input_ids,
            max_length=1024,
            num_beams=n,
            num_return_sequences=n,
            early_stopping=True,
        )
        tactics = self.tokenizer.batch_decode(tactic_ids, skip_special_tokens=True)
        return tactics

    @torch.no_grad()
    def generate_tactics_with_probs(
        self, state: str, retrieved_premises: List[str], n: int = 1
    ) -> List[tuple[str, float]]:
        """
        Generates n tactics and their probabilities using beam search.
        """
        input_str = "\n\n".join(retrieved_premises + [state])
        tokenized_input = self.tokenizer(
            input_str, return_tensors="pt", max_length=2300, truncation=True
        )

        if torch.cuda.is_available():
            tokenized_input = tokenized_input.to("cuda")

        # Generate n tactics using beam search and get scores
        outputs = self.model.generate(
            tokenized_input.input_ids,
            max_length=1024,
            num_beams=n,
            num_return_sequences=n,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # The scores are log-probabilities. We can convert them to probabilities.
        # The `sequences_scores` are the sum of log-softmax scores for each token.
        sequence_scores = outputs.sequences_scores
        probs = torch.softmax(sequence_scores, dim=0)

        # Decode the tactics
        tactics = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        return list(zip(tactics, probs.tolist()))


class ValueHead(nn.Module):
    """
    A value head that uses a pre-trained encoder to predict the
    value (win probability) of a given proof state.
    """

    def __init__(
        self, encoder_name: str = "kaiyuy/leandojo-lean4-retriever-byt5-small"
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModelForTextEncoding.from_pretrained(encoder_name)

        # Freeze the pre-trained encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # The new value head that will be trained
        # 1472 is the hidden size of the encoder
        self.value_head = nn.Sequential(
            nn.Linear(1472, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        if torch.cuda.is_available():
            self.to("cuda")

    def _encode(self, s: List[str]) -> torch.Tensor:
        """Encode a batch of texts into feature vectors."""
        tokenized_s = self.tokenizer(
            s, return_tensors="pt", padding=True, truncation=True, max_length=2300
        )
        if torch.cuda.is_available():
            tokenized_s = tokenized_s.to("cuda")

        hidden_state = self.encoder(tokenized_s.input_ids).last_hidden_state
        lens = tokenized_s.attention_mask.sum(dim=1)
        features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)
        return features

    @torch.no_grad()
    def predict(self, state_str: str, premises: List[str]) -> float:
        """
        Predicts the value of a single state.
        Returns a float between -1.0 and 1.0.
        """
        self.eval()  # Set to evaluation mode
        input_str = "\n\n".join(premises + [state_str])

        # Encode the input string (pass as a batch of 1)
        features = self._encode([input_str])

        # Get value prediction
        value = self.value_head(features).squeeze()

        # Apply tanh to squash the value between -1 and 1
        return torch.tanh(value).item()

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
        return super().train(mode)

    def eval(self) -> Self:
        return super().eval()
