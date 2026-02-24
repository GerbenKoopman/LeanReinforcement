"""
A transformer class that loads the ReProver model and provides easy
tactic generation.
"""

import torch
from typing import List, Protocol, Tuple, runtime_checkable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@runtime_checkable
class TransformerProtocol(Protocol):
    def generate_tactics(self, state: str, n: int = 1) -> List[str]: ...

    def generate_tactics_with_probs(
        self, state: str, n: int = 1
    ) -> List[Tuple[str, float]]: ...

    def generate_tactics_batch(
        self, states: List[str], n: int = 1
    ) -> List[List[str]]: ...

    def generate_tactics_with_probs_batch(
        self, states: List[str], n: int = 1
    ) -> List[List[Tuple[str, float]]]: ...


class Transformer:
    def __init__(self, model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self._generate_call_count = 0

    def _periodic_cache_cleanup(self) -> None:
        """Clear GPU cache periodically instead of on every call."""
        self._generate_call_count += 1
        if self._generate_call_count % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_tactics(self, state: str, n: int = 1) -> List[str]:
        tokenized_state = self.tokenizer(
            state, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        # max_length is TOTAL (input + output), so add to input length
        input_length = tokenized_state.input_ids.shape[1]
        tactics_ids = self.model.generate(
            tokenized_state.input_ids,
            max_length=input_length + 512,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        tactics: List[str] = self.tokenizer.batch_decode(
            tactics_ids, skip_special_tokens=True
        )

        del tokenized_state
        del tactics_ids

        # Periodic KV-cache cleanup
        self._periodic_cache_cleanup()

        assert isinstance(
            tactics, list
        ), f"Expected list of tactics, got {type(tactics)}"
        return tactics

    @torch.no_grad()
    def generate_tactics_with_probs(
        self, state: str, n: int = 1
    ) -> List[tuple[str, float]]:
        tokenized_state = self.tokenizer(
            state, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        input_length = tokenized_state.input_ids.shape[1]
        outputs = self.model.generate(
            tokenized_state.input_ids,
            max_length=input_length + 512,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        tactics = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        sequence_scores = outputs.sequences_scores
        probs = torch.softmax(sequence_scores, dim=0)

        result = list(zip(tactics, probs.tolist()))

        del tokenized_state
        del outputs
        del sequence_scores

        # Periodic KV-cache cleanup
        self._periodic_cache_cleanup()

        return result

    @torch.no_grad()
    def generate_tactics_batch(self, states: List[str], n: int = 1) -> List[List[str]]:
        """
        Generates tactics for a batch of states.
        Returns a list of lists of tactics, one list per state.
        """
        if not states:
            return []

        tokenized_states = self.tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        input_length = tokenized_states.input_ids.shape[1]
        tactics_ids = self.model.generate(
            tokenized_states.input_ids,
            attention_mask=tokenized_states.attention_mask,
            max_length=input_length + 512,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        # tactics_ids shape: (batch_size * n, sequence_length)
        tactics = self.tokenizer.batch_decode(tactics_ids, skip_special_tokens=True)

        del tokenized_states
        del tactics_ids

        # Reshape the flat list of tactics into a list of lists
        result = []
        for i in range(0, len(tactics), n):
            result.append(tactics[i : i + n])

        # Periodic KV-cache cleanup
        self._periodic_cache_cleanup()

        return result

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
            states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        input_length = tokenized_states.input_ids.shape[1]
        outputs = self.model.generate(
            tokenized_states.input_ids,
            attention_mask=tokenized_states.attention_mask,
            max_length=input_length + 512,
            num_beams=n,
            do_sample=False,
            num_return_sequences=n,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
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

        del tokenized_states
        del outputs
        del sequence_scores

        # Periodic KV-cache cleanup
        self._periodic_cache_cleanup()

        return results
