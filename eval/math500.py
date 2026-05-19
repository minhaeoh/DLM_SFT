from gsm8k import GSM8KDataset
from datasets import load_dataset


class MATH500Dataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        prompt_style="default",
        subsample=-1,
        start_index=0,
        end_index=-1,
    ):
        super().__init__(
            tokenizer,
            prompt_style=prompt_style,
            subsample=subsample,
            start_index=start_index,
            end_index=end_index,
        )

    def load_test_dataset(self):
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    def __getitem__(self, idx):
        dataset_index = int(self.subsample[idx])
        question = self.dataset[dataset_index]["problem"]
        answer = self.dataset[dataset_index]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer, dataset_index
