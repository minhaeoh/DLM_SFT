import torch
import numpy as np
import random
from datasets import load_dataset
from parsers import Parser

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

RAW_SYSTEM_PROMPT = """
Please reason step by step, and put your final answer within \\boxed{}.
"""

FORMAT_SYSTEM_PROMPT = """
Please reason step by step and respond in the following format, with the final answer inside \boxed{}:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

ANSWER_FIRST_SYSTEM_PROMPT = """
Please reason step by step, but respond with the final answer first inside \boxed{}, followed by the reasoning:

<answer>
...
</answer>
<reasoning>
...
</reasoning>
"""


class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=GSM_SYSTEM_PROMPT,
        prompt_style="xml",
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.prompt_style = str(prompt_style or "xml").strip().lower()
        self.load_test_dataset()
        self.create_few_shot_prompt()

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
        if self.prompt_style == "raw":
            user_prompt = (
                f"{input_text}"
                f"{RAW_SYSTEM_PROMPT}\n\n"
            )
            messages = [{"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False) + "\n"
        if self.prompt_style == "format":
            user_prompt = (
                f"{input_text}"
                f"{FORMAT_SYSTEM_PROMPT}\n\n"
            )
            messages = [{"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False) + "\n"
        if self.prompt_style == "answer_first":
            user_prompt = (
                f"{input_text}"
                f"{ANSWER_FIRST_SYSTEM_PROMPT}\n\n"
            )
            messages = [{"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False) + "\n"

        # Legacy XML prompt used by the original eval scripts.
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + prompt}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<reasoning>"
        return user_input

    def load_few_shot_examples(self):
        if self.num_examples <= 0:
            return []

        train_data = load_dataset("gsm8k", "main", split="train")
        examples = random.sample(range(len(train_data)), self.num_examples)
        return [train_data[example] for example in examples]

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples"""
        few_shot_examples = self.load_few_shot_examples()

        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            answer = example["answer"]
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")
        self.few_shot_prompt = "\n\n".join(formatted_examples)

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
