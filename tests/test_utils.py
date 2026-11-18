import pytest

from machadinho.utils import lr_at, plan_windows


def test_every_target_is_scored_exactly_once():
    for n in (101, 129, 200, 1000, 1234):
        for block in (8, 64, 100):
            if n < block + 1:
                continue
            for stride in (1, block // 2 or 1, block):
                plan = plan_windows(n, block, stride)
                assert sum(block - skip for _, skip in plan) == n - 1, (n, block, stride)
                assert all(start + block + 1 <= n for start, _ in plan)
                assert all(0 <= skip < block for _, skip in plan)


def test_windows_skip_only_what_was_already_scored():
    plan = plan_windows(50, 10, 5)
    assert plan[0] == (0, 0)
    assert plan[1] == (5, 5)  # metade da janela já foi pontuada


def test_windows_reject_bad_arguments():
    with pytest.raises(ValueError):
        plan_windows(5, 10, 5)  # texto menor que a janela
    with pytest.raises(ValueError):
        plan_windows(100, 10, 0)
    with pytest.raises(ValueError):
        plan_windows(100, 10, 11)


def test_lr_warms_up_then_decays_to_floor():
    lrs = [lr_at(s, 1000, 1e-3, 100) for s in range(1001)]
    assert lrs[0] < lrs[50] < lrs[99]
    assert lrs[100] == pytest.approx(1e-3)
    assert all(a >= b - 1e-12 for a, b in zip(lrs[100:], lrs[101:]))
    assert lrs[-1] == pytest.approx(1e-4)  # min_lr_ratio = 0.1


def test_lr_without_warmup():
    assert lr_at(0, 100, 1e-3, 0) == pytest.approx(1e-3)
