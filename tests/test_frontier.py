import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import baseline_classical as bc  # noqa: E402
import indomain_frontier as fr  # noqa: E402
import taskmaster_hybrid_prereg as hyb  # noqa: E402
from assim.ngram import ByteNgramLM  # noqa: E402
from sender_memory import vendor  # noqa: E402

LINES = [b"book a table for two", b"book a cab", b"i want a table for four tonight"] * 20


@pytest.fixture(scope="module")
def points():
    try:
        vendor.ensure()
    except RuntimeError as exc:
        pytest.skip(f"vendor sources unavailable: {exc}")
    return fr.build_points(LINES)


def test_points_are_the_pre_registered_set(points):
    assert tuple(points) == fr.POINTS


def test_raw_and_ppm_bits_accounting(points):
    lm = ByteNgramLM(max_context=5)
    lm.add(b"\n".join(LINES) + b"\n")
    ctx = bc.nested_counts(lm)
    m = b"book a table"
    assert points["raw"].bits(m) == 8 * len(m) + 40
    for name, order in (("ppm_o1", 1), ("ppm_o2", 2), ("ppm_o3", 3)):
        assert points[name].bits(m) == pytest.approx(bc.ppm_bits(ctx, m, order) + 40)


def test_per_message_has_term_columns_only_for_ppm(points):
    [row] = fr.per_message([b"book a cab"], points)
    assert row["bytes"] == 10
    assert row["bits_ppm_o1_term"] == pytest.approx(row["bits_ppm_o1"] + 2)
    assert "bits_raw_term" not in row and "bits_unishox2_term" not in row


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_test_mode_refuses_without_committed_prereg(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    doc = tmp_path / "PRE.md"
    doc.write_text("draft\n")
    with pytest.raises(SystemExit):
        fr.require_prereg_committed(doc)  # untracked
    _git(tmp_path, "add", "PRE.md")
    _git(tmp_path, "commit", "-q", "-m", "prereg")
    fr.require_prereg_committed(doc)  # committed and clean
    doc.write_text("edited after the commit\n")
    with pytest.raises(SystemExit):
        fr.require_prereg_committed(doc)  # modified
    with pytest.raises(SystemExit):
        fr.require_prereg_committed(tmp_path / "MISSING.md")


@pytest.mark.skipif(not hyb.RAW_PATH.exists(), reason="raw Taskmaster absent")
def test_dev_reproduces_exploratory_numbers():
    try:
        vendor.ensure()
    except RuntimeError as exc:
        pytest.skip(f"vendor sources unavailable: {exc}")
    raw = hyb.load_raw()
    pts = fr.build_points(hyb.train_lines(raw, hyb.load_ids(hyb.TRAIN_IDS), hyb.dev_conversations(raw)))
    msgs = [m for m in hyb.DEV_LIST.read_bytes().splitlines() if m]
    rows = fr.per_message(msgs, pts)
    mean = {p: sum(r[f"bits_{p}"] for r in rows) / len(rows) for p in ("ppm_o1", "ppm_o2", "ppm_o3")}
    # scripts/analysis/indomain_frontier/result.txt, tm30
    assert mean["ppm_o1"] == pytest.approx(177.5, abs=0.05)
    assert mean["ppm_o2"] == pytest.approx(142.2, abs=0.05)
    assert mean["ppm_o3"] == pytest.approx(114.3, abs=0.05)
    o2p = {p: sum(r[f"bits_{p}"] for r in rows) / len(rows) for p in ("ppm_o2_16k", "ppm_o2_32k", "ppm_o2_64k")}
    # The pruning rule of the design recomputes the PPM-C escape on the pruned dictionary: a context
    # left with few, frequent symbols makes every novel byte cost ~13 bits. At 16 KiB (142 order-2
    # entries) the point is dominated by o1 (measured 2026-11-05, DECISOES 5.59); 32 and 64 KiB are not.
    assert o2p["ppm_o2_16k"] == pytest.approx(264.3, abs=0.05)
    assert o2p["ppm_o2_32k"] == pytest.approx(144.5, abs=0.05)
    assert o2p["ppm_o2_64k"] == pytest.approx(142.0, abs=0.05)
