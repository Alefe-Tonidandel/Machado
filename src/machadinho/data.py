"""Limpeza do corpus, divisão treino/validação/teste e tokenizador de caracteres."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

UNK = "\ufffd"  # caractere de substituição para símbolos fora do vocabulário

_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.I | re.S)


def strip_gutenberg(text: str) -> str:
    """Remove o cabeçalho e o rodapé de licença do Projeto Gutenberg, se existirem."""
    m = _START.search(text)
    if m:
        text = text[m.end():]
    m = _END.search(text)
    if m:
        text = text[: m.start()]
    return text


def clean_text(raw: str) -> str:
    """Normaliza o texto: sem cabeçalho, em NFC, com parágrafos desquebrados.

    Os livros do Gutenberg vêm quebrados em linhas de ~70 colunas. Juntamos as linhas de
    cada parágrafo para o modelo não gastar capacidade aprendendo a quebra de linha.
    """
    text = strip_gutenberg(raw).replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(p for p in paragraphs if p) + "\n"


@dataclass(frozen=True)
class Corpus:
    train: str
    val: str
    test: str
    books: tuple[str, ...]
    test_book: str | None


def load_corpus(raw_dir: str | Path, test_book: str | None = None, val_frac: float = 0.05) -> Corpus:
    """Lê data/raw/*.txt. O livro `test_book` fica inteiro de fora do treino.

    Os demais livros cedem o final (val_frac) para validação, usada na parada antecipada.
    O livro de teste nunca é visto durante o treino nem para escolher hiperparâmetros.
    """
    files = sorted(Path(raw_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"nenhum .txt em {raw_dir}; rode scripts/download_corpus.py")
    names = [f.stem for f in files]
    if test_book is not None and test_book not in names:
        raise ValueError(f"livro de teste {test_book!r} não encontrado; disponíveis: {names}")

    train_parts: list[str] = []
    val_parts: list[str] = []
    test = ""
    for f in files:
        text = clean_text(f.read_text(encoding="utf-8"))
        if f.stem == test_book:
            test = text
            continue
        cut = int(len(text) * (1.0 - val_frac))
        boundary = text.find("\n\n", cut)  # corta em fronteira de parágrafo
        if boundary != -1:
            cut = boundary + 2
        train_parts.append(text[:cut])
        val_parts.append(text[cut:])

    val = "".join(val_parts)
    if not train_parts or not val:
        raise ValueError("é preciso ao menos um livro de treino com texto suficiente para validação")
    return Corpus("".join(train_parts), val, test, tuple(names), test_book)


class CharTokenizer:
    """Um id por caractere. O vocabulário é construído só com o texto de treino."""

    def __init__(self, chars: list[str]) -> None:
        if UNK in chars:
            chars = [c for c in chars if c != UNK]
        self.itos = [UNK, *chars]  # id 0 = desconhecido
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_dict(self) -> dict:
        return {"chars": self.itos[1:]}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(list(d["chars"]))
