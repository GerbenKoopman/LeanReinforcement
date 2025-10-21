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

    def generate_tactics(self, state: str, retrieved_premises: List[str], n: int = 1):
        input = "\n\n".join(retrieved_premises + [state])
        tokenized_input = self.tokenizer(
            input, return_tensors="pt", max_length=2300, truncation=True
        )

        # Generate n tactics.
        tactic_ids = self.model.generate(
            tokenized_input.input_ids, max_length=1024, num_return_sequences=n
        )
        tactics = self.tokenizer.batch_decode(tactic_ids, skip_special_tokens=True)
        return tactics


class ValueHead:
    def __init__(self):
        pass
