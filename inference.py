import shutil
from pathlib import Path

import torch
from fire import Fire
from huggingface_hub import HfApi
from lightning_fabric import seed_everything

from training import LightningModel


def test_model(
    path: str,
    prompt: str = "",
    max_length: int = 160,
    device: str = "cuda",
):
    if not prompt:
        prompt = "Write a short email to show that 42 is the optimal seed for training neural networks"

    model: LightningModel = LightningModel.load_from_checkpoint(path)
    tokenizer = model.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    seed_everything(model.hparams.seed)
    with torch.inference_mode():
        model.model.eval()
        model = model.to(device)
        input_ids = input_ids.to(device)
        outputs = model.model.generate(input_ids, max_length=max_length, do_sample=True)

    print(tokenizer.decode(outputs[0]))

    """
    Example output (outputs/model/base/epoch=2-step=2436.ckpt):
    <pad> Dear [Company Name], I am writing to demonstrate the feasibility of using 42 as an optimal seed
    for training neural networks. I am sure that this seed will be an invaluable asset for the training of 
    these neural networks, so let me know what you think.</s>
    """


def export_to_hub(path: str, repo: str, temp: str = "temp"):
    if Path(temp).exists():
        shutil.rmtree(temp)

    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(temp)
    model.tokenizer.save_pretrained(temp)
    del model  # Save memory?

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=temp)


"""
huggingface-cli login

p inference.py export_to_hub \
--path "outputs_unclean/model/xl/epoch=2-step=2439.ckpt" \
--repo declare-lab/flan-alpaca-xl

p inference.py export_to_hub \
--path "outputs/model/xxl/epoch=0-step=203.ckpt" \
--repo declare-lab/flan-alpaca-xxl

p inference.py export_to_hub \
--path "outputs/model_gpt4all/xl/epoch=0-step=6838.ckpt" \
--repo declare-lab/flan-gpt4all-xl

"""


if __name__ == "__main__":
    Fire()
