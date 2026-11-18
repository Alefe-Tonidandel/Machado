"""Baseline sem rede neural: modelo de n-gramas de caracteres com interpolação.

Serve de régua. Se o Transformer não ganha disso, ele não aprendeu nada além de contagens.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .data import load_corpus


class NgramBaseline:
    def __init__(self, order: int = 4, floor: float = 1e-3) -> None:
        if order < 1:
            raise ValueError("order >= 1")
        self.order = order
        self.floor = floor  # massa mínima dividida entre todos os símbolos (evita prob. 0)
        self._counts: list[defaultdict[str, Counter]] = [defaultdict(Counter) for _ in range(order)]
        self._totals: list[Counter] = [Counter() for _ in range(order)]  # evita somar a cada consulta
        self._vocab_size = 1

    def fit(self, text: str) -> "NgramBaseline":
        self._vocab_size = len(set(text)) + 1  # +1 para símbolos desconhecidos
        for i, ch in enumerate(text):
            for k in range(self.order):  # k = tamanho do contexto (0 = unigrama)
                if i - k < 0:
                    break
                ctx = text[i - k : i]
                self._counts[k][ctx][ch] += 1
                self._totals[k][ctx] += 1
        return self

    def prob(self, context: str, ch: str) -> float:
        """Mistura as ordens cujo contexto foi visto, dando mais peso às ordens maiores."""
        num = den = 0.0
        for k in range(self.order):
            if k > len(context):
                break
            ctx = context[len(context) - k :] if k else ""
            counter = self._counts[k].get(ctx)
            if not counter:
                continue
            weight = 2.0**k
            num += weight * counter[ch] / self._totals[k][ctx]
            den += weight
        mixed = num / den if den else 0.0
        return (1.0 - self.floor) * mixed + self.floor / self._vocab_size

    def bits_per_char(self, text: str) -> float:
        if len(text) < 2:
            raise ValueError("texto curto demais")
        total = 0.0
        for i in range(1, len(text)):
            context = text[max(0, i - (self.order - 1)) : i]
            total -= math.log2(self.prob(context, text[i]))
        return total / (len(text) - 1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Baseline de n-gramas de caracteres")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--test-book", required=True)
    p.add_argument("--orders", default="2,3,4,5,6", help="ordens a testar, separadas por vírgula")
    p.add_argument("--out", default="runs/baseline.json")
    args = p.parse_args(argv)

    corpus = load_corpus(args.data_dir, args.test_book)
    rows = []
    for order in (int(x) for x in args.orders.split(",")):
        model = NgramBaseline(order).fit(corpus.train)
        row = {"order": order, "val_bpc": model.bits_per_char(corpus.val), "test_bpc": model.bits_per_char(corpus.test)}
        rows.append(row)
        print(f"ordem {order}: val {row['val_bpc']:.3f} bpc | teste {row['test_bpc']:.3f} bpc")

    best = min(rows, key=lambda r: r["val_bpc"])  # a ordem é escolhida na validação, nunca no teste
    uniform = math.log2(len(set(corpus.train)) + 1)
    result = {"best": best, "all": rows, "uniform_bpc": uniform, "test_book": args.test_book}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"melhor ordem pela validação: {best['order']} | uniforme: {uniform:.3f} bpc | salvo em {args.out}")


if __name__ == "__main__":
    main()
