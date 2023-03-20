import torch
from fire import Fire
from lightning_fabric import seed_everything

from training import LightningModel


def test_model(path: str, prompt: str = "", max_length: int = 160):
    if not prompt:
        prompt = "Write a short email to show that 42 is the optimal seed for training neural networks"

    model: LightningModel = LightningModel.load_from_checkpoint(path)
    tokenizer = model.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    seed_everything(model.hparams.seed)
    with torch.inference_mode():
        model.model.eval()
        outputs = model.model.generate(input_ids, max_length=max_length, do_sample=True)

    print(tokenizer.decode(outputs[0]))

    """
    Example output (outputs/model/base/epoch=2-step=2436.ckpt):
    <pad> Dear [Company Name], I am writing to demonstrate the feasibility of using 42 as an optimal seed
    for training neural networks. I am sure that this seed will be an invaluable asset for the training of 
    these neural networks, so let me know what you think.</s>
    """


if __name__ == "__main__":
    Fire()
