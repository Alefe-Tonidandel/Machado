"""Treino do mini-GPT. Uso: python -m machadinho.train --test-book papeis_avulsos --out runs/small"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import CharTokenizer, load_corpus
from .evaluate import bits_per_char
from .model import GPT, GPTConfig
from .utils import lr_at


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina um mini-GPT em português")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--test-book", required=True, help="nome (sem .txt) do livro reservado para teste")
    p.add_argument("--out", required=True, help="pasta da execução, ex.: runs/small")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", help="auto, cpu ou cuda")
    return p.parse_args(argv)


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str, gen: torch.Generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, torch.Tensor], args: argparse.Namespace, device: str) -> dict[str, float]:
    """Perda média em lotes sorteados com semente fixa, para comparar avaliações entre si."""
    model.eval()
    out = {}
    for name, data in splits.items():
        gen = torch.Generator().manual_seed(1234)
        losses = [
            model(*get_batch(data, args.block_size, args.batch_size, device, gen))[1].item()
            for _ in range(args.eval_iters)
        ]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    corpus = load_corpus(args.data_dir, args.test_book)
    tok = CharTokenizer.from_text(corpus.train)
    to_ids = lambda s: torch.tensor(tok.encode(s), dtype=torch.long)  # noqa: E731
    train_ids, val_ids, test_ids = to_ids(corpus.train), to_ids(corpus.val), to_ids(corpus.test)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizer(args.lr, args.weight_decay)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"dispositivo={device} | vocabulário={tok.vocab_size} | parâmetros={model.num_params():,} | "
        f"treino={len(train_ids):,} val={len(val_ids):,} teste={len(test_ids):,} caracteres"
    )

    gen = torch.Generator().manual_seed(args.seed)
    best_val, start = math.inf, time.time()
    model.train()
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.max_steps + 1):
            lr = lr_at(step, args.max_steps, args.lr, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if step % args.eval_interval == 0 or step == args.max_steps:
                losses = estimate_loss(model, {"train": train_ids, "val": val_ids}, args, device)
                record = {"step": step, "lr": lr, **losses, "elapsed_s": round(time.time() - start, 1)}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(f"passo {step:5d} | treino {losses['train']:.3f} | val {losses['val']:.3f} | lr {lr:.2e}")
                if losses["val"] < best_val:  # guarda o melhor ponto: o modelo pequeno sobreajusta
                    best_val = losses["val"]
                    save_checkpoint(out / "best.pt", model, tok, step, best_val)

            if step == args.max_steps:
                break

            x, y = get_batch(train_ids, args.block_size, args.batch_size, device, gen)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

    train_seconds = time.time() - start
    best_model, _, meta = load_checkpoint(out / "best.pt", device)
    results = {
        "config": asdict(cfg),
        "n_params": best_model.num_params(),
        "best_step": meta["step"],
        "best_val_loss": best_val,
        "val_bpc": bits_per_char(best_model, val_ids),
        "test_bpc": bits_per_char(best_model, test_ids),
        "test_book": args.test_book,
        "train_seconds": round(train_seconds, 1),
        "args": vars(args),
        "device": device,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"melhor passo {meta['step']} | val {results['val_bpc']:.3f} bpc | teste {results['test_bpc']:.3f} bpc")


if __name__ == "__main__":
    main()
