import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor

from data_loading import TextToTextDataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def init_args(raw_args):
    # Training args should follow FlanT5 (Scaling Instruction-Finetuned Language Models)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--max_source_length", type=int, default=40)
    parser.add_argument("--max_target_length", type=int, default=160)
    parser.add_argument("--data_path", type=str, default="data/train.json")
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="")

    args = parser.parse_args(raw_args)
    return args


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        print(self.hparams)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name_or_path
        )
        if torch.__version__.startswith("2"):
            print("Torch 2.0 compile")
            self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss, on_step=True)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = self.trainer.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # noinspection PyTypeChecker
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            relative_step=False,
        )
        return [optimizer]

    def train_dataloader(self):
        dataset = TextToTextDataset(
            path=self.hparams.data_path,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
            tokenizer=self.tokenizer,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            shuffle=True,
        )


def main(raw_args=None):
    torch.set_float32_matmul_precision("high")
    args = init_args(raw_args)
    seed_everything(args.seed)
    model = LightningModel(args)

    saver = ModelCheckpoint(
        verbose=True,
        dirpath=args.output_dir,
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        precision="bf16-mixed",
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=args.gradient_accumulation_steps,
        default_root_dir=args.output_dir,
        gradient_clip_val=1.0,
        max_epochs=args.train_epochs,
        callbacks=[saver],
        logger=False,
    )

    trainer.fit(model)


"""
p training.py --output_dir outputs/model/base

p training.py --output_dir outputs/model/xl \
--model_name_or_path "google/flan-t5-xl" \
--train_batch_size 1 \
--gradient_accumulation_steps 64

"""


if __name__ == "__main__":
    main()
