"""Baixa livros de Machado de Assis do Projeto Gutenberg (domínio público) para data/raw/.

Uso:
    python scripts/download_corpus.py
    python scripts/download_corpus.py --add memorias_postumas=ID   # ID = número na URL do livro no gutenberg.org
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

# IDs conferidos nas páginas do Projeto Gutenberg.
BOOKS = {
    "dom_casmurro": 55752,
    "quincas_borba": 55682,
    "papeis_avulsos": 57001,
}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw")
    p.add_argument("--add", action="append", default=[], metavar="NOME=ID", help="livro extra, pode repetir")
    args = p.parse_args()

    books = dict(BOOKS)
    for item in args.add:
        name, _, book_id = item.partition("=")
        if not name or not book_id.isdigit():
            raise SystemExit(f"formato inválido: {item!r}; use NOME=ID (ex.: helena=12345)")
        books[name] = int(book_id)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, book_id in books.items():
        target = out / f"{name}.txt"
        if target.exists():
            print(f"já existe: {target}")
            continue
        req = urllib.request.Request(URL.format(id=book_id), headers={"User-Agent": "machadinho-corpus-downloader"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        if "PROJECT GUTENBERG" not in text.upper():
            print(f"aviso: {name} (id {book_id}) não parece um texto do Gutenberg; confira o ID")
        target.write_text(text, encoding="utf-8")
        print(f"baixado: {target} ({len(text):,} caracteres)")
        time.sleep(1)  # educação com o servidor


if __name__ == "__main__":
    main()
