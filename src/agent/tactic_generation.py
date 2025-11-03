"""
Tactic generation module using a ByT5-based text encoder-decoder from ReProver.
"""

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TacticGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
        )
        # TODO: add training mode for AlphaZero style training
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

        # Explicitly delete tensors to free GPU memory
        del tokenized_input
        del tactic_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        result = list(zip(tactics, probs.tolist()))

        # Explicitly delete tensors to free GPU memory
        del tokenized_input
        del outputs
        del sequence_scores
        del probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
