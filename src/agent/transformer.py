import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Transformer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "kaiyuy/leandojo-lean4-tacgen-byt5-small"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "kaiyuy/leandojo-lean4-tacgen-byt5-small"
        )

    @torch.no_grad()
    def generate_tactics(self, state: str, n: int = 1) -> List[str]:
        tokenized_state = self.tokenizer(state, return_tensors="pt")

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
        tokenized_state = self.tokenizer(state, return_tensors="pt")

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
