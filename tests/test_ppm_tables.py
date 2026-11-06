import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_classical as bc  # noqa: E402
from assim.ngram import ByteNgramLM  # noqa: E402
from sender_memory import ppm_tables as pt  # noqa: E402


def _o2_dict():
    return {
        b"": {b"a": 9, b"b": 5, b"c": 2},
        b"a": {b"b": 4, b"c": 1},
        b"b": {b"a": 3},
        b"ab": {b"c": 1, b"a": 2},
        b"ba": {b"b": 1, b"c": 1},
        b"ca": {b"a": 3},
        b"bc": {b"a": 2, b"b": 5},
    }


def _reference_prune(c, target):
    """Naive rule of the spec (stage 2): remove order-2 entries by (count, ctx+sym) until <= target."""
    work = {k: dict(v) for k, v in c.items()}
    if pt.table_bytes(work) <= target:
        return work, True, 0
    order = sorted(((n, ctx + sym, ctx, sym) for ctx, d in c.items() if len(ctx) == 2 for sym, n in d.items()))
    for removed, (_, _, ctx, sym) in enumerate(order, start=1):
        del work[ctx][sym]
        if not work[ctx]:
            del work[ctx]
        if pt.table_bytes(work) <= target:
            return work, True, removed
    return work, False, len(order)


def test_table_bytes_formula():
    c = {b"": {b"a": 3, b"b": 1}, b"a": {b"b": 300}}
    # order 0: offsets 2x1 + 2 entries x (1+1) = 6; order 1: ctx 1 + offsets 2x1 + 1 x (1+2) = 6
    assert pt.table_bytes(c) == 12


def test_restrict_does_not_change_ppm_bits():
    lm = ByteNgramLM(max_context=5)
    lm.add(b"the cat sat\nthe hat\n")
    full = bc.nested_counts(lm)
    for o in (1, 2, 3):
        r = pt.restrict(full, o)
        assert all(len(k) <= o for k in r)
        assert bc.ppm_bits(r, b"the mat", o) == bc.ppm_bits(full, b"the mat", o)


def test_prune_matches_bruteforce_for_every_target():
    c = _o2_dict()
    full = pt.table_bytes(c)
    floor = pt.table_bytes({k: v for k, v in c.items() if len(k) < 2})
    for target in range(floor - 2, full + 2):
        assert pt.prune_o2(c, target) == _reference_prune(c, target), target


def test_prune_tiebreak_is_lexicographic_on_ctx_plus_sym():
    # count-1 entries: ab->c, ba->b, ba->c; lexicographic order removes "abc" first
    c = _o2_dict()
    pruned, reached, removed = pt.prune_o2(c, pt.table_bytes(c) - 1)
    assert (reached, removed) == (True, 1)
    assert pruned[b"ab"] == {b"a": 2}
    assert pruned[b"ba"] == {b"b": 1, b"c": 1}


def test_prune_unreachable_keeps_orders_0_1_and_reports_false():
    c = _o2_dict()
    pruned, reached, removed = pt.prune_o2(c, 1)
    assert reached is False
    assert removed == sum(len(v) for k, v in c.items() if len(k) == 2)
    assert pruned == {k: v for k, v in c.items() if len(k) < 2}


def test_prune_noop_when_already_under_target():
    c = _o2_dict()
    assert pt.prune_o2(c, 10**9) == (c, True, 0)


def test_prune_does_not_mutate_input():
    c = _o2_dict()
    pt.prune_o2(c, 1)
    assert c == _o2_dict()


def test_emit_header_rejects_context_total_that_overflows_the_coder(tmp_path):
    # the 40-bit WNC coder needs range * total < 2^64, i.e. n + t < 2^23 in every context
    import pytest
    with pytest.raises(ValueError):
        pt.emit_header({b"": {b"a": 2**23}}, 0, tmp_path / "h.h")
    assert pt.emit_header({b"": {b"a": 2**23 - 2}}, 0, tmp_path / "h.h") > 0


def test_counts_sha_ignores_insertion_order():
    a = {b"": {b"x": 1, b"y": 2}}
    b = {b"": {b"y": 2, b"x": 1}}
    assert pt.counts_sha256(a) == pt.counts_sha256(b)
    assert pt.counts_sha256(a) != pt.counts_sha256({b"": {b"x": 1, b"y": 3}})
