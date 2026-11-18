from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .data import CharTokenizer
from .model import GPT, GPTConfig


def save_checkpoint(path: str | Path, model: GPT, tokenizer: CharTokenizer, step: int, val_loss: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.cfg),
            "tokenizer": tokenizer.to_dict(),
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[GPT, CharTokenizer, dict]:
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {"step": ckpt["step"], "val_loss": ckpt["val_loss"]}
    return model, CharTokenizer.from_dict(ckpt["tokenizer"]), meta
