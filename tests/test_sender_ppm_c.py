import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import baseline_classical as bc  # noqa: E402
import taskmaster_hybrid_prereg as hyb  # noqa: E402
from assim.ngram import ByteNgramLM  # noqa: E402
from sender_memory import build  # noqa: E402
from sender_memory import ppm_tables as pt  # noqa: E402


def _model(text: bytes, order: int):
    lm = ByteNgramLM(max_context=order)
    lm.add(text)
    return pt.restrict(bc.nested_counts(lm), order)


def _dev_messages() -> list[bytes]:
    out = []
    for name in ("taskmaster_en_30.txt", "taskmaster_hybrid_30.txt"):
        out += [m for m in (ROOT / "data" / name).read_bytes().splitlines() if m]
    return out


@pytest.mark.parametrize("order", [1, 2, 3])
def test_roundtrip_and_bound_on_synthetic(tmp_path, order):
    c = _model(b"the cat sat on the mat\nthe hat\n" * 20, order)
    exe = build.build_ppm_cli(c, order, tmp_path)
    msgs = [b"the cat", b"a mat sat on the hat", b"zzz", b"t"]
    res = build.run_ppm_cli(exe, msgs)
    assert len(res) == len(msgs)
    for m, (nbits, ok) in zip(msgs, res):
        assert ok, m
        assert nbits <= bc.ppm_bits(c, m, order) + 2, (m, nbits, bc.ppm_bits(c, m, order))


def test_unseen_bytes_use_uniform_fallback(tmp_path):
    c = _model(b"hello world\n" * 5, 2)
    exe = build.build_ppm_cli(c, 2, tmp_path)
    m = "café 👍".encode()
    [(nbits, ok)] = build.run_ppm_cli(exe, [m])
    assert ok
    assert nbits <= bc.ppm_bits(c, m, 2) + 2


def test_corrupted_code_is_detected_by_roundtrip_flag(tmp_path):
    # the CLI must really decode: a one-bit flip in the encoder output breaks the roundtrip
    c = _model(b"the cat sat on the mat\n" * 20, 2)
    exe = build.build_ppm_cli(c, 2, tmp_path, flip_first_bit=True)
    [(_, ok)] = build.run_ppm_cli(exe, [b"the cat sat"])
    assert not ok


@pytest.mark.skipif(not hyb.RAW_PATH.exists(), reason="raw Taskmaster absent")
@pytest.mark.parametrize("order", [1, 2])
def test_dev_messages_within_2_bits_of_ppm_bits(tmp_path, order):
    raw = hyb.load_raw()
    lines = hyb.train_lines(raw, hyb.load_ids(hyb.TRAIN_IDS), hyb.dev_conversations(raw))
    c = _model(b"\n".join(lines) + b"\n", order)
    exe = build.build_ppm_cli(c, order, tmp_path)
    msgs = _dev_messages()
    res = build.run_ppm_cli(exe, msgs)
    assert len(res) == 60 and all(ok for _, ok in res)
    excess = [nb - bc.ppm_bits(c, m, order) for m, (nb, _) in zip(msgs, res)]
    assert max(excess) <= 2, max(excess)
