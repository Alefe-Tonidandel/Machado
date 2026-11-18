import tempfile
from pathlib import Path

import pytest

from machadinho.data import CharTokenizer, clean_text, load_corpus, strip_gutenberg

RAW = """The Project Gutenberg eBook of Exemplo

Title: Exemplo

*** START OF THE PROJECT GUTENBERG EBOOK EXEMPLO ***

CAPITULO I

Era uma vez um bruxo
que morava em Cosme Velho.

Outro   paragrafo
com espaços   extras.

*** END OF THE PROJECT GUTENBERG EBOOK EXEMPLO ***
Licença do Gutenberg aqui.
"""


def test_strip_gutenberg_removes_header_and_footer():
    body = strip_gutenberg(RAW)
    assert "CAPITULO I" in body
    assert "Title: Exemplo" not in body
    assert "Licença do Gutenberg" not in body


def test_strip_gutenberg_leaves_other_texts_untouched():
    assert strip_gutenberg("texto simples") == "texto simples"


def test_clean_text_unwraps_paragraphs_and_collapses_spaces():
    cleaned = clean_text(RAW)
    assert "Era uma vez um bruxo que morava em Cosme Velho." in cleaned
    assert "Outro paragrafo com espaços extras." in cleaned
    assert cleaned.count("\n\n") == 2  # três parágrafos
    assert cleaned.endswith("\n") and not cleaned.endswith("\n\n")


def test_clean_text_normalizes_unicode_to_nfc():
    decomposed = "cafe\u0301"  # e + acento combinante
    assert clean_text(decomposed).strip() == "café"


def test_tokenizer_roundtrip_and_unknown_char():
    tok = CharTokenizer.from_text("abc ção")
    assert tok.decode(tok.encode("ação")) == "ação"
    unk_ids = tok.encode("xyz")
    assert unk_ids == [0, 0, 0]
    assert tok.decode(unk_ids) == "\ufffd\ufffd\ufffd"


def test_tokenizer_serialization():
    tok = CharTokenizer.from_text("machado")
    again = CharTokenizer.from_dict(tok.to_dict())
    assert again.itos == tok.itos
    assert again.encode("macho") == tok.encode("macho")


def _write_books(directory: Path, books: dict[str, str]) -> None:
    for name, text in books.items():
        (directory / f"{name}.txt").write_text(text, encoding="utf-8")


def test_load_corpus_keeps_test_book_out_of_train_and_vocab():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        para = "palavras de treino aqui.\n\n" * 40
        _write_books(base, {"a": para, "b": para, "teste": "ZZZ unico livro de teste.\n\n" * 5})
        corpus = load_corpus(base, test_book="teste")
        assert "ZZZ" not in corpus.train and "ZZZ" not in corpus.val
        assert "ZZZ" in corpus.test
        tok = CharTokenizer.from_text(corpus.train)
        assert "Z" not in tok.stoi  # o vocabulário não vaza do livro de teste
        assert corpus.test_book == "teste"


def test_load_corpus_validation_is_the_tail_of_each_book():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        text = "".join(f"paragrafo {i:03d}.\n\n" for i in range(100))
        _write_books(base, {"livro": text, "outro": text})
        corpus = load_corpus(base, test_book="outro")
        assert "paragrafo 099" in corpus.val
        assert "paragrafo 099" not in corpus.train
        assert corpus.train.startswith("paragrafo 000")


def test_load_corpus_errors():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        with pytest.raises(FileNotFoundError):
            load_corpus(base)
        _write_books(base, {"a": "x\n\n" * 50})
        with pytest.raises(ValueError):
            load_corpus(base, test_book="inexistente")
