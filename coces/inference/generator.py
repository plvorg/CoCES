from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coces.config import GenerationConfig
from coces.data.schema import KGQAExample


class EvidenceGenerator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    @torch.no_grad()
    def generate(
        self,
        example: KGQAExample,
        answer: str,
        path_indices: list[int],
    ) -> str:
        evidence = "\n".join(
            f"- {example.paths[index].verbalize()}" for index in path_indices
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied knowledge-graph evidence. "
                    "State the answer first, followed by a brief evidence-grounded explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {example.question}\n"
                    f"Predicted answer: {answer}\n"
                    f"Evidence paths:\n{evidence}"
                ),
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs: dict[str, Any] = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 1e-5),
        )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

