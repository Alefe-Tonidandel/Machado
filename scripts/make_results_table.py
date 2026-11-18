"""Gera RESULTS.md a partir de runs/*/results.json (e runs/baseline.json, se existir)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_table(runs_dir: Path) -> str:
    rows = []
    for path in sorted(runs_dir.glob("*/results.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        c = r["config"]
        rows.append((r["test_bpc"], path.parent.name, c, r))
    if not rows:
        raise SystemExit(f"nenhum results.json em {runs_dir}/*/")
    rows.sort(key=lambda x: x[0])
    test_book = rows[0][3]["test_book"]

    lines = [
        "# Resultados",
        "",
        f"Livro de teste (nunca visto no treino nem na escolha de hiperparâmetros): `{test_book}`.",
        "Métrica: bits por caractere (bpc); menor é melhor. Uma semente por configuração.",
        "",
        "| Execução | Camadas | d_model | Contexto | Dropout | Parâmetros | Val bpc | Teste bpc | Tempo (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for test_bpc, name, c, r in rows:
        lines.append(
            f"| {name} | {c['n_layer']} | {c['n_embd']} | {c['block_size']} | {c['dropout']} | "
            f"{r['n_params']:,} | {r['val_bpc']:.3f} | {test_bpc:.3f} | {r['train_seconds']:.0f} |"
        )

    baseline_path = runs_dir / "baseline.json"
    if baseline_path.exists():
        b = json.loads(baseline_path.read_text(encoding="utf-8"))
        best = b["best"]
        lines += [
            "",
            "## Referências sem rede neural",
            "",
            "| Modelo | Val bpc | Teste bpc |",
            "|---|---|---|",
            f"| Uniforme (chute aleatório entre os símbolos) | {b['uniform_bpc']:.3f} | {b['uniform_bpc']:.3f} |",
            f"| N-grama de caracteres (ordem {best['order']}, escolhida na validação) | "
            f"{best['val_bpc']:.3f} | {best['test_bpc']:.3f} |",
        ]
    lines += [
        "",
        "Leitura: a diferença entre `Val bpc` e `Teste bpc` mostra o quanto o modelo depende do estilo dos "
        "livros de treino. Diferenças pequenas entre execuções podem ser ruído de uma única semente.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="RESULTS.md")
    args = p.parse_args()
    text = build_table(Path(args.runs_dir))
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
