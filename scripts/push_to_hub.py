"""Publica um checkpoint no Hugging Face Hub com um model card gerado dos resultados reais.

Uso:
    huggingface-cli login            # ou: export HF_TOKEN=...  (nunca coloque o token no código)
    python scripts/push_to_hub.py --run runs/small --repo-id SEU_USUARIO/machadinho-small
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CARD = """---
language: pt
license: mit
library_name: pytorch
pipeline_tag: text-generation
tags:
  - text-generation
  - from-scratch
  - character-level
  - portuguese
---

# {name}

Mini-GPT (Transformer decoder-only) treinado **do zero** em PyTorch, em nível de caractere, com romances de
Machado de Assis (domínio público, Projeto Gutenberg). Projeto de estudo; não é um modelo para uso em produção.

Código, treino e avaliação: {code_url}

## Arquitetura

{n_layer} camadas, {n_head} cabeças, d_model = {n_embd}, contexto de {block_size} caracteres,
dropout {dropout}, {n_params:,} parâmetros, vocabulário de {vocab} símbolos.

## Avaliação

Medida em bits por caractere (bpc; menor é melhor) no livro `{test_book}`, **que o modelo nunca viu**
durante o treino nem na escolha de hiperparâmetros.

| | bpc |
|---|---|
| Validação (final dos livros de treino) | {val_bpc:.3f} |
| Teste (`{test_book}`) | {test_bpc:.3f} |
{baseline_rows}
## Como usar

O modelo usa a arquitetura do repositório de código (não é compatível com `AutoModel`):

```bash
git clone {code_url}
cd machadinho && pip install -e .
python -m machadinho.generate --ckpt best.pt --prompt "Capitu" --temperature 0.8
```

## Limitações

- Corpus pequeno (poucos romances): o modelo tende a sobreajustar e pode reproduzir trechos dos livros de treino.
- As edições do Gutenberg usam a ortografia da época; o modelo aprende essa ortografia, não a atual.
- Nível de caractere: o texto imita o estilo, mas não mantém coerência de longo prazo.
- Uma única semente por configuração.
"""


def build_model_card(results: dict, name: str, code_url: str, baseline: dict | None = None) -> str:
    c = results["config"]
    rows = ""
    if baseline:
        b = baseline["best"]
        rows = (
            f"| Uniforme | {baseline['uniform_bpc']:.3f} |\n"
            f"| N-grama de caracteres (ordem {b['order']}), teste | {b['test_bpc']:.3f} |\n"
        )
    return CARD.format(
        name=name, code_url=code_url, n_layer=c["n_layer"], n_head=c["n_head"], n_embd=c["n_embd"],
        block_size=c["block_size"], dropout=c["dropout"], n_params=results["n_params"],
        vocab=c["vocab_size"], test_book=results["test_book"], val_bpc=results["val_bpc"],
        test_bpc=results["test_bpc"], baseline_rows=rows,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="pasta da execução, ex.: runs/small")
    p.add_argument("--repo-id", required=True, help="USUARIO/nome-do-modelo")
    p.add_argument("--code-url", default="https://github.com/Alefe-Tonidandel/machadinho")
    p.add_argument("--baseline", default="runs/baseline.json")
    args = p.parse_args()

    from huggingface_hub import HfApi  # import tardio

    run = Path(args.run)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    card = build_model_card(results, args.repo_id.split("/")[-1], args.code_url, baseline)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(run / "best.pt"), path_in_repo="best.pt", repo_id=args.repo_id)
    api.upload_file(path_or_fileobj=str(run / "results.json"), path_in_repo="results.json", repo_id=args.repo_id)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo_id)
    print(f"publicado: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
