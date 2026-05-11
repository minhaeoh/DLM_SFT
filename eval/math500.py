from gsm8k import GSM8KDataset
from datasets import load_dataset

class MATH500Dataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        prompt_style="default",
        subsample=-1,
    ):
        super().__init__(tokenizer, prompt_style=prompt_style, subsample=subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["problem"]
        answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer
