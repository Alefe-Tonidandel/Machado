import math

import pytest

from machadinho.baseline import NgramBaseline

TEXT = "o rato roeu a roupa do rei de roma. " * 200


def test_probabilities_sum_to_one_over_the_vocabulary():
    model = NgramBaseline(order=3).fit(TEXT)
    chars = sorted(set(TEXT))
    total = sum(model.prob("ro", c) for c in chars)
    # a massa do piso também cobre símbolos desconhecidos, então a soma é < 1 por pouco
    assert 0.99 < total <= 1.0 + 1e-9


def test_beats_uniform_on_repetitive_text():
    model = NgramBaseline(order=4).fit(TEXT)
    uniform = math.log2(len(set(TEXT)) + 1)
    assert model.bits_per_char(TEXT) < 0.5 * uniform


def test_higher_order_helps_on_the_training_text():
    low = NgramBaseline(order=1).fit(TEXT).bits_per_char(TEXT)
    high = NgramBaseline(order=4).fit(TEXT).bits_per_char(TEXT)
    assert high < low


def test_unseen_symbol_gets_nonzero_probability():
    model = NgramBaseline(order=3).fit(TEXT)
    assert model.prob("ro", "Q") > 0


def test_invalid_order_and_short_text():
    with pytest.raises(ValueError):
        NgramBaseline(order=0)
    with pytest.raises(ValueError):
        NgramBaseline(order=2).fit(TEXT).bits_per_char("a")
