"""Roda a grade de experimentos, uma configuração por vez.

Uso:
    python scripts/sweep.py --test-book papeis_avulsos
    python scripts/sweep.py --test-book papeis_avulsos --only tiny small
    python scripts/sweep.py --test-book papeis_avulsos --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CONFIGS: dict[str, dict[str, object]] = {
    "tiny": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64},
    "small": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 128},
    "base": {"n_layer": 6, "n_head": 8, "n_embd": 256, "block_size": 256},
    # ablações a partir do "small": uma mudança por vez
    "small-ctx64": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 64},
    "small-ctx256": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 256},
    "small-nodrop": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 128, "dropout": 0.0},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-book", required=True)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--only", nargs="*", help="nomes das configurações a rodar")
    p.add_argument("--force", action="store_true", help="refaz mesmo se results.json já existir")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    names = args.only or list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"configurações desconhecidas: {unknown}; disponíveis: {list(CONFIGS)}")

    for name in names:
        out = Path(args.runs_dir) / name
        if (out / "results.json").exists() and not args.force:
            print(f"pulando {name} (já tem results.json; use --force para refazer)")
            continue
        cmd = [
            sys.executable, "-m", "machadinho.train",
            "--data-dir", args.data_dir, "--test-book", args.test_book,
            "--out", str(out), "--max-steps", str(args.max_steps),
        ]
        for key, value in CONFIGS[name].items():
            cmd += [f"--{key.replace('_', '-')}", str(value)]
        print("\n$", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
