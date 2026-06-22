import torch
import numpy as np
from datasets import load_dataset
from parsers import Parser

# Must match the train-time prompt built in long_cot_data_utils._build_math_user_prompt.
INSTRUCTION_PROMPT = """
Please reason step by step with the final answer inside \\boxed{}.
"""


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
        start_index=0,
        end_index=-1,
    ):
        self.tokenizer = tokenizer
        # Retained for output metadata only; the prompt is now style-independent
        # and always matches the train template.
        self.prompt_style = str(prompt_style or "default").strip().lower()
        self.load_test_dataset()

        total_examples = len(self.dataset)
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}.")
        if start_index >= total_examples:
            raise ValueError(
                f"start_index {start_index} is out of range for dataset of size {total_examples}."
            )
        if end_index != -1 and end_index < start_index:
            raise ValueError(
                f"end_index must be >= start_index (got start_index={start_index}, end_index={end_index})."
            )
        if end_index >= total_examples:
            raise ValueError(
                f"end_index {end_index} is out of range for dataset of size {total_examples}."
            )

        self.start_index = int(start_index)
        self.requested_end_index = int(end_index)
        self.effective_end_index = total_examples - 1 if end_index == -1 else int(end_index)

        candidate_indices = np.arange(self.start_index, self.effective_end_index + 1)
        assert subsample <= len(candidate_indices), "Subsample size is greater than selected dataset range"
        self.subsample = (
            np.random.choice(candidate_indices, subsample, replace=False)
            if subsample != -1
            else candidate_indices
        )
        print(
            f"evaluating {len(self.subsample)} examples "
            f"(dataset indices: {self.start_index}-{self.effective_end_index})"
        )

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test")

    def create_prompt(self, input_text):
        user_prompt = f"Question:\n{input_text}\n\n{INSTRUCTION_PROMPT.strip()}"
        return _render_chat_prompt(self.tokenizer, user_prompt)

    def __getitem__(self, idx):
        dataset_index = int(self.subsample[idx])
        question = self.dataset[dataset_index]["question"]
        answer = Parser.extract_answer_gsm8k(self.dataset[dataset_index]["answer"])
        prompt = self.create_prompt(question)
        return prompt, question, answer, dataset_index

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        dataset_indices = [item[3] for item in batch]
        tokenized = self.tokenizer(
            prompts,
            padding_side="left",
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        input_ids = tokenized.input_ids
        prompt_token_lengths = [int(length) for length in tokenized.attention_mask.sum(dim=1).tolist()]
        return {
            "input_ids": input_ids,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
            "dataset_indices": dataset_indices,
            "prompt_token_lengths": prompt_token_lengths,
        }
