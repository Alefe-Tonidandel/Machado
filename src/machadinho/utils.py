"""Funções puras (sem PyTorch), fáceis de testar."""
from __future__ import annotations

import math


def lr_at(step: int, max_steps: int, lr: float, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Aquecimento linear e depois decaimento por cosseno até lr * min_lr_ratio."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def plan_windows(n_tokens: int, block_size: int, stride: int) -> list[tuple[int, int]]:
    """Plano de janelas deslizantes para medir a perda em um texto longo.

    Devolve [(início, descartar)]: cada janela cobre ids[início : início+block_size+1]
    (entrada e alvo deslocado em 1). Os primeiros `descartar` alvos da janela já foram
    pontuados por janelas anteriores e não entram na soma. Assim cada alvo é contado
    exatamente uma vez, e a maioria deles é prevista com bastante contexto.
    """
    if not 1 <= stride <= block_size:
        raise ValueError("stride deve estar entre 1 e block_size")
    if n_tokens < block_size + 1:
        raise ValueError(f"texto com {n_tokens} tokens é menor que block_size + 1 ({block_size + 1})")

    last_start = n_tokens - block_size - 1
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    plan: list[tuple[int, int]] = []
    covered = 0  # os alvos de índice 1..covered já foram pontuados
    for start in starts:
        plan.append((start, max(0, covered - start)))
        covered = start + block_size
    return plan
