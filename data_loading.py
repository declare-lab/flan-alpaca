import json
import random
from pathlib import Path
from typing import List, Optional, Dict

from datasets import load_dataset
from fire import Fire
from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding, AutoTokenizer


class TokensLengthAnalyzer(BaseModel, arbitrary_types_allowed=True):
    name: str
    tokenizer: Optional[PreTrainedTokenizer]

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.name, model_max_length=99999
            )

    def run(self, texts: List[str], limit: int = 0) -> Dict[str, float]:
        if limit:
            texts = texts[:limit]

        self.load()
        tokens = self.tokenizer(texts).input_ids
        lengths = sorted(len(lst) for lst in tokens)
        info = dict(min=lengths[0], max=lengths[-1], median=lengths[len(lengths) // 2])
        info.update({"95_percentile": lengths[round(len(lengths) * 0.95)]})
        return info


class TextToTextSample(BaseModel):
    source: str
    target: str


class TextToTextData(BaseModel):
    samples: List[TextToTextSample]

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            all_lines = tqdm(f.readlines(), desc=path)
            samples = [TextToTextSample(**json.loads(line)) for line in all_lines]
        return cls(samples=samples)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for sample in self.samples:
                print(sample.json(), file=f)

    def analyze(self, num: int = 10, tokenizer_name: str = "t5-base"):
        random.seed(num)
        for sample in random.sample(self.samples, k=num):
            print(sample.json(indent=2))

        token_checker = TokensLengthAnalyzer(name=tokenizer_name)
        info = dict(
            total_samples=len(self.samples),
            source=str(token_checker.run([sample.source for sample in self.samples])),
            target=str(token_checker.run([sample.target for sample in self.samples])),
        )
        print(json.dumps(info, indent=2))


class AlpacaSample(BaseModel):
    instruction: str
    input: str
    output: str


class AlpacaData(BaseModel):
    samples: List[AlpacaSample]

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            raw = json.load(f)
            return cls(samples=[AlpacaSample(**r) for r in raw])

    def save(self, path: str):
        raw = [sample.dict() for sample in self.samples]
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(raw, f)

    def as_data(self) -> TextToTextData:
        self.analyze()
        samples = []
        for raw in self.samples:
            source = raw.instruction.strip()
            if raw.input.strip():
                source = source + "\n" + raw.input
            samples.append(TextToTextSample(source=source, target=raw.output))
        return TextToTextData(samples=samples)

    def analyze(self):
        info = dict(
            alpaca_samples=len(self.samples),
            with_context=sum(sample.input.strip() != "" for sample in self.samples),
        )
        print(json.dumps(info, indent=2))


class TextToTextDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int,
        max_target_length: int,
    ):
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.data = TextToTextData.load(path)

    def __len__(self) -> int:
        return len(self.data.samples)

    def tokenize(self, text: str, is_source: bool) -> BatchEncoding:
        return self.tokenizer(
            text,
            max_length=self.max_source_length if is_source else self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, i: int) -> dict:
        x = self.tokenize(self.data.samples[i].source, is_source=True)
        y = self.tokenize(self.data.samples[i].target, is_source=False)

        return {
            "source_ids": x.input_ids.squeeze(),
            "source_mask": x.attention_mask.squeeze(),
            "target_ids": y.input_ids.squeeze(),
            "target_mask": y.attention_mask.squeeze(),
        }

    def to_human_readable(self, raw: dict) -> dict:
        source = self.tokenizer.decode(raw["source_ids"])
        target = self.tokenizer.decode(raw["target_ids"])
        return dict(source=source, target=target)


def preprocess_alpaca(
    path_in: str = "data/alpaca.json", path_out: str = "data/train.json"
):
    data = AlpacaData.load(path_in).as_data()
    data.analyze()
    data.save(path_out)


def clean_gpt4all_text(text: str) -> str:
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    text = text.replace("<pre><code>", "")
    text = text.replace("</code></pre>", "")
    return text


def preprocess_gpt4all(
    path_in: str = "nomic-ai/gpt4all_prompt_generations",
    path_out="data/train_gpt4all.json",
):
    data = []
    for raw in tqdm(load_dataset(path_in, split="train"), desc=path_in):
        prompt = clean_gpt4all_text(raw["prompt"])
        response = clean_gpt4all_text(raw["response"])
        data.append(dict(source=prompt, target=response))

    random.seed(0)
    TextToTextData(
        samples=[TextToTextSample(**raw) for raw in random.sample(data, 1000)]
    ).analyze()

    with open(path_out, "w") as f:
        for raw in tqdm(data, desc=path_out):
            print(json.dumps(raw), file=f)


if __name__ == "__main__":
    Fire()
