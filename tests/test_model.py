"""Testes do modelo. Exigem PyTorch; são pulados quando ele não está instalado."""
import math
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from machadinho.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from machadinho.data import CharTokenizer  # noqa: E402
from machadinho.evaluate import bits_per_char  # noqa: E402
from machadinho.model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=11, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def make_model() -> GPT:
    torch.manual_seed(0)
    return GPT(CFG)


def test_forward_shapes_and_initial_loss_is_near_uniform():
    model = make_model()
    idx = torch.randint(0, CFG.vocab_size, (4, 16))
    targets = torch.randint(0, CFG.vocab_size, idx.shape)
    logits, loss = model(idx, targets)
    assert logits.shape == (4, 16, CFG.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3  # sem treino ~ chute uniforme


def test_attention_is_causal():
    """Mudar um token do futuro não pode alterar as previsões das posições anteriores."""
    model = make_model().eval()
    a = torch.randint(0, CFG.vocab_size, (1, 16))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % CFG.vocab_size
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)
    assert not torch.allclose(la[:, -1], lb[:, -1], atol=1e-5)


def test_weights_are_tied():
    model = make_model()
    assert model.lm_head.weight is model.wte.weight


def test_can_overfit_a_tiny_batch():
    model = make_model().train()
    opt = model.configure_optimizer(lr=3e-3, weight_decay=0.0)
    # Padrão determinístico (o próximo token é sempre o atual + 1): se o modelo não
    # consegue aprender isso em poucos passos, há um bug no treino ou na arquitetura.
    x = (torch.arange(16)[None, :] + torch.arange(8)[:, None]) % CFG.vocab_size
    y = (x + 1) % CFG.vocab_size
    first = model(x, y)[1].item()
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_generate_length_range_and_determinism_at_zero_temperature():
    model = make_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    out1 = model.generate(idx, 40, temperature=0.0)  # passa de block_size: o contexto é cortado
    out2 = model.generate(idx, 40, temperature=0.0)
    assert out1.shape == (1, 43)
    assert torch.equal(out1, out2)
    assert int(out1.max()) < CFG.vocab_size and int(out1.min()) >= 0


def test_sequence_longer_than_block_size_is_rejected():
    with pytest.raises(ValueError):
        make_model()(torch.zeros((1, CFG.block_size + 1), dtype=torch.long))


def test_bits_per_char_of_untrained_model_is_near_uniform_and_covers_all_targets():
    model = make_model()
    ids = torch.randint(0, CFG.vocab_size, (203,))
    bpc = bits_per_char(model, ids, stride=8, batch_size=5)  # levanta erro se contar alvos a mais ou a menos
    assert abs(bpc - math.log2(CFG.vocab_size)) < 0.5


def test_bits_per_char_does_not_depend_on_stride_for_a_context_free_model():
    """Com pesos zerados o modelo é uniforme: qualquer stride deve dar exatamente log2(V)."""
    model = make_model()
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    ids = torch.randint(0, CFG.vocab_size, (100,))
    for stride in (1, 4, 16):
        assert bits_per_char(model, ids, stride=stride) == pytest.approx(math.log2(CFG.vocab_size), abs=1e-4)


def test_checkpoint_roundtrip_keeps_outputs():
    model = make_model().eval()
    tok = CharTokenizer.from_text("abcdefghi")
    cfg_vocab = tok.vocab_size
    model = GPT(GPTConfig(vocab_size=cfg_vocab, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)).eval()
    idx = torch.randint(0, cfg_vocab, (2, 8))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.pt"
        save_checkpoint(path, model, tok, step=7, val_loss=1.23)
        loaded, tok2, meta = load_checkpoint(path)
    assert meta["step"] == 7
    assert tok2.itos == tok.itos
    assert torch.allclose(model(idx)[0], loaded(idx)[0], atol=1e-6)
