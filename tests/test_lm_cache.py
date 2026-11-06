"""O cache KV do receptor deve dar o mesmo estado oculto que a passada completa."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from assim.lm import _KVRunner  # noqa: E402

START = 0


@pytest.fixture(scope="module")
def base():
    cfg = transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).eval().base_model


def _full(base, seqs):
    """Referência: uma passada completa por sequência, sem preenchimento nem cache."""
    out = []
    for s in seqs:
        with torch.no_grad():
            h = base(input_ids=torch.tensor([[START] + list(s)])).last_hidden_state[0, -1]
        out.append(h)
    return torch.stack(out)


def _runner(base, budget=1 << 30):
    return _KVRunner(torch, base, START, "cpu", budget)


def _tree():
    # comprimentos diferentes no mesmo lote, para exercitar o preenchimento do meio
    level0 = [[]]
    level1 = [[5], [9], [17]]
    level2 = [[5, 7], [5, 8], [9, 3], [17, 4]]
    level3 = [[5, 7, 1], [9, 3, 2]]
    return level0, level1, level2, level3


def test_cached_matches_full_pass(base):
    r = _runner(base)
    for level in _tree():
        got = r.last_hidden(level)
        assert torch.allclose(got, _full(base, level), atol=1e-5)
    assert r.hits > 0


def test_mixed_lengths_in_one_batch(base):
    r = _runner(base)
    l0, l1, l2, _ = _tree()
    r.last_hidden(l0)
    r.last_hidden(l1)
    r.last_hidden(l2[:2])
    batch = [[5, 7, 4], [9, 6], [17, 4, 8, 2]]  # os pais [5,7], [9] e [17,4,8] podem faltar
    got = r.last_hidden(batch)
    assert torch.allclose(got, _full(base, batch), atol=1e-5)


def test_parent_batch_with_different_lengths_and_order(base):
    r = _runner(base)
    r.last_hidden([[]])
    r.last_hidden([[1], [2]])
    r.last_hidden([[1, 3], [2, 4], [2, 5]])
    r.last_hidden([[1, 3, 6]])
    batch = [[1, 3, 6, 7], [2, 5, 1], [1, 3, 2]]  # pais de 3, 2 e 2 tokens
    got = r.last_hidden(batch)
    assert torch.allclose(got, _full(base, batch), atol=1e-5)


def test_tiny_budget_evicts_and_falls_back(base):
    r = _runner(base, budget=1)  # guarda só a última entrada: quase tudo cai no recálculo
    for level in _tree():
        got = r.last_hidden(level)
        assert torch.allclose(got, _full(base, level), atol=1e-5)
    assert r.misses > 1
    assert len(r._kv) == 1


def test_cache_does_not_keep_views_of_the_batch(base):
    r = _runner(base)
    r.last_hidden([[]])
    r.last_hidden([[1], [2], [3]])
    r.last_hidden([[1, 4], [2, 4], [3, 4]])
    for kv in r._kv.values():
        assert kv.is_contiguous()
        assert kv._base is None  # cópia própria, não view do tensor do lote
