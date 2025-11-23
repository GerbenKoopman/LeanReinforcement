"""
A transformer class that loads the ReProver model and provides easy
tactic generation.
"""

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Transformer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "kaiyuy/leandojo-lean4-tacgen-byt5-small"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "kaiyuy/leandojo-lean4-tacgen-byt5-small"
        ).to(self.device)

    @torch.no_grad()
    def generate_tactics(self, state: str, n: int = 1) -> List[str]:
        tokenized_state = self.tokenizer(state, return_tensors="pt").to(self.device)

        tactics_ids = self.model.generate(
            tokenized_state.input_ids,
            max_length=1024,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
        )
        tactics = self.tokenizer.batch_decode(tactics_ids, skip_special_tokens=True)

        return tactics

    @torch.no_grad()
    def generate_tactics_with_probs(
        self, state: str, n: int = 1
    ) -> List[tuple[str, float]]:
        tokenized_state = self.tokenizer(state, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            tokenized_state.input_ids,
            max_length=1024,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        tactics = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        sequence_scores = outputs.sequences_scores
        probs = torch.softmax(sequence_scores, dim=0)

        return list(zip(tactics, probs.tolist()))

    @torch.no_grad()
    def generate_tactics_batch(self, states: List[str], n: int = 1) -> List[List[str]]:
        """
        Generates tactics for a batch of states.
        Returns a list of lists of tactics, one list per state.
        """
        if not states:
            return []

        tokenized_states = self.tokenizer(
            states, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        tactics_ids = self.model.generate(
            tokenized_states.input_ids,
            max_length=1024,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
        )

        # tactics_ids shape: (batch_size * n, sequence_length)
        all_tactics = self.tokenizer.batch_decode(tactics_ids, skip_special_tokens=True)

        # Reshape result to List[List[str]]
        results = []
        for i in range(len(states)):
            results.append(all_tactics[i * n : (i + 1) * n])

        return results

    @torch.no_grad()
    def generate_tactics_with_probs_batch(
        self, states: List[str], n: int = 1
    ) -> List[List[tuple[str, float]]]:
        """
        Generates tactics with probabilities for a batch of states.
        """
        if not states:
            return []

        tokenized_states = self.tokenizer(
            states, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            tokenized_states.input_ids,
            max_length=1024,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        all_tactics = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        # Calculate probabilities
        # sequences_scores shape: (batch_size * n,)
        sequence_scores = outputs.sequences_scores

        results = []
        for i in range(len(states)):
            start_idx = i * n
            end_idx = (i + 1) * n

            batch_tactics = all_tactics[start_idx:end_idx]
            batch_scores = sequence_scores[start_idx:end_idx]
            batch_probs = torch.softmax(batch_scores, dim=0).tolist()

            results.append(list(zip(batch_tactics, batch_probs)))

        return results
