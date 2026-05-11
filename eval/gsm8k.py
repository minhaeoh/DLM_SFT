import torch
import numpy as np
from datasets import load_dataset
from parsers import Parser

DEFAULT_PROMPT = """
Please reason step by step, and put your final answer within \\boxed{}.
"""

FORMAT_PROMPT = """
Please reason step by step and respond in the following format, with the final answer inside \\boxed{}:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

ANSWER_FIRST_PROMPT = """
Please reason step by step, but respond with the final answer first inside \\boxed{}, followed by the reasoning:

<answer>
...
</answer>
<reasoning>
...
</reasoning>
"""

PROMPT_STYLE_TO_TEXT = {
    "default": DEFAULT_PROMPT,
    "format": FORMAT_PROMPT,
    "answer_first": ANSWER_FIRST_PROMPT,
}

PROMPT_STYLE_ALIASES = {
    "raw": "default",
    "raw_long_cot": "default",
    "xml": "format",
    "answer-first": "answer_first",
    "answerfirst": "answer_first",
}


def _normalize_prompt_style(prompt_style: str) -> str:
    normalized_prompt_style = str(prompt_style or "default").strip().lower()
    normalized_prompt_style = PROMPT_STYLE_ALIASES.get(normalized_prompt_style, normalized_prompt_style)
    if normalized_prompt_style not in PROMPT_STYLE_TO_TEXT:
        valid = ", ".join(sorted(PROMPT_STYLE_TO_TEXT))
        raise ValueError(f"Unsupported prompt_style `{prompt_style}`. Expected one of: {valid}.")
    return normalized_prompt_style


def _render_chat_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"
    return f"User: {user_content}\nAssistant:\n"


class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        prompt_style="default",
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.prompt_style = _normalize_prompt_style(prompt_style)
        self.load_test_dataset()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test")

    def create_prompt(self, input_text):
        prompt_text = PROMPT_STYLE_TO_TEXT[self.prompt_style].strip()
        user_prompt = f"Question:\n{input_text}\n\n{prompt_text}"
        return _render_chat_prompt(self.tokenizer, user_prompt)

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["question"]
        answer = Parser.extract_answer_gsm8k(self.dataset[self.subsample[idx].item()]["answer"])
        prompt = self.create_prompt(question)
        return prompt, question, answer

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}
