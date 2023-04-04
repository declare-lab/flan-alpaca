import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict

from datasets import load_dataset
from fire import Fire
from pydantic import BaseModel, Field
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
        x = self.tokenizer(
            text,
            max_length=self.max_source_length if is_source else self.max_target_length,
            padding="max_length",
            truncation=not is_source,
            return_tensors="pt",
        )

        """
        T5 truncates on right by default, but we can easily truncate on left
        for the encoder input as there is no special token on the left side
        """
        if is_source:
            assert x.input_ids.ndim == 2
            assert x.input_ids.shape == x.attention_mask.shape
            length = x.input_ids.shape[1]
            start = max(length - self.max_source_length, 0)
            x.input_ids = x.input_ids[:, start:]
            x.attention_mask = x.attention_mask[:, start:]
            assert x.input_ids.shape[1] == self.max_source_length

        return x

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


class ShareGPTConversation(BaseModel):
    speaker: str = Field(alias="from")
    value: str


class ShareGPTSample(BaseModel):
    id: str
    conversations: List[ShareGPTConversation]

    def contains_texts(self, texts: List[str], do_lower: bool = True) -> bool:
        for c in self.conversations:
            for t in texts:
                if do_lower and t.lower() in c.value.lower():
                    return True
                elif not do_lower and t in c.value:
                    return True
        return False

    def has_empty_text(self) -> bool:
        for c in self.conversations:
            lst = [char for char in c.value if c.value.isalnum()]
            if len(lst) == 0:
                return True
        return False


class ShareGPTData(BaseModel):
    samples: List[ShareGPTSample]

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            samples = [ShareGPTSample(**raw) for raw in json.load(f)]
        return cls(samples=samples)

    def analyze(self):
        speakers = [conv.speaker for s in self.samples for conv in s.conversations]
        info = dict(samples=len(self.samples), speakers=Counter(speakers))
        print(json.dumps(info, indent=2))

    def clean(self, phrases: List[str] = None):
        """
        ~100k ShareGPT conversations narrowed down to 48k by:

        Removing non-english conversations
        Removing excessive unicode (indicative of Chinese or Korean text, usually)
        Removing excessive repeated characters
        Removing various instances "AI Moralizing". Conversations with these phrases were removed:
        """
        if phrases is None:
            phrases = [
                "prioritize human safety",
                "ethical principles",
                "harmful to human beings",
                "September 2021",
                "as a language model",
                "ethical guidelines",
                "as an AI language model",
                "my guidelines",
                "As an AI",
                "prioritize user safety",
                "adhere to ethical guidelines",
                "harmful consequences",
                "potentially harmful",
                "dangerous activities",
                "promote safety",
                "well-being of all users",
                "responsible information sharing",
                "jeopardize the safety",
                "illegal actions or intentions",
                "undermine the stability",
                "promote the well-being",
                "illegal activities or actions",
                "adherence to the law",
                "potentially be harmful",
                "illegal substances or activities",
                "committed to promoting",
                "safe information",
                "lawful information",
                "cannot provide guidance",
                "cannot provide information",
                "unable to offer assistance",
                "cannot engage in discussions",
                "programming prohibits",
                "follow ethical guidelines",
                "ensure the safety",
                "involves an illegal subject",
                "prioritize safety",
                "illegal subject",
                "prioritize user well-being",
                "cannot support or promote",
                "activities that could harm",
                "pose a risk to others",
                "against my programming",
                "activities that could undermine",
                "potentially dangerous",
                "not within the scope",
                "designed to prioritize safety",
                "not able to provide",
                "maintain user safety",
                "adhere to safety guidelines",
                "dangerous or harmful",
                "cannot provide any information",
                "focus on promoting safety",
            ]

        self.samples = [
            s for s in tqdm(self.samples, desc="clean empty") if s.has_empty_text()
        ]
        self.analyze()

        self.samples = [
            s
            for s in tqdm(self.samples, desc="clean phrases")
            if not s.contains_texts(phrases, do_lower=True)
        ]
        self.analyze()

    def as_data(self) -> TextToTextData:
        samples = []
        for s in tqdm(self.samples, desc="as_data"):
            for i, conv in enumerate(s.conversations):
                prev = s.conversations[max(i - 1, 0)]
                if conv.speaker == "gpt" and prev.speaker == "human":
                    target = conv.value
                    source = "\n\n".join([conv.value for conv in s.conversations[:i]])
                    samples.append(TextToTextSample(source=source, target=target))

        return TextToTextData(samples=samples)


def preprocess_sharegpt(
    path_in: str = "data/sharegpt.json",
    path_out="data/train_sharegpt.json",
):
    # See: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    raw = ShareGPTData.load(path_in)
    raw.analyze()
    raw.clean()

    data = raw.as_data()
    data.analyze()
    data.save(path_out)


if __name__ == "__main__":
    Fire()
