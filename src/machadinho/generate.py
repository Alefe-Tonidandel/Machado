from __future__ import annotations

import argparse

import torch

from .checkpoint import load_checkpoint


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Gera texto com um checkpoint treinado")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = guloso")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    model, tok, _ = load_checkpoint(args.ckpt, args.device)
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k or None)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
