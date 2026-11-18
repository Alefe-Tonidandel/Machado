"""Bits por caractere (bpc) em um texto longo, com janelas deslizantes."""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .data import load_corpus
from .model import GPT
from .utils import plan_windows


@torch.no_grad()
def bits_per_char(model: GPT, ids: torch.Tensor, stride: int | None = None, batch_size: int = 64) -> float:
    """Entropia cruzada média por caractere, em bits. Menor é melhor.

    Cada caractere é pontuado uma única vez (ver utils.plan_windows), e a maioria é
    prevista com contexto de pelo menos block_size - stride caracteres.
    """
    model.eval()
    device = next(model.parameters()).device
    block = model.cfg.block_size
    plan = plan_windows(len(ids), block, stride or block // 2)

    total_nll, total = 0.0, 0
    for i in range(0, len(plan), batch_size):
        chunk = plan[i : i + batch_size]
        x = torch.stack([ids[s : s + block] for s, _ in chunk]).to(device)
        y = torch.stack([ids[s + 1 : s + block + 1] for s, _ in chunk]).to(device)
        skip = torch.tensor([k for _, k in chunk], device=device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        nll = nll.view(x.size(0), block)
        keep = torch.arange(block, device=device)[None, :] >= skip[:, None]
        total_nll += (nll * keep).sum().item()
        total += int(keep.sum().item())

    if total != len(ids) - 1:
        raise RuntimeError(f"{total} alvos pontuados, esperado {len(ids) - 1}")
    return total_nll / total / math.log(2)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Avalia um checkpoint no livro de teste")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--test-book", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    model, tok, meta = load_checkpoint(args.ckpt, args.device)
    corpus = load_corpus(args.data_dir, args.test_book)
    ids = torch.tensor(tok.encode(corpus.test), dtype=torch.long)
    print(f"teste ({args.test_book}): {bits_per_char(model, ids):.3f} bpc | checkpoint no passo {meta['step']}")


if __name__ == "__main__":
    main()
