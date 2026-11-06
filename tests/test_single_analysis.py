import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import baseline_classical as bc  # noqa: E402
import single_analysis as sa  # noqa: E402
import taskmaster_hybrid_prereg as hyb  # noqa: E402
from assim.ngram import ByteNgramLM  # noqa: E402


def test_wilson_matches_spec_forecast():
    lo, hi = sa.wilson(27, 30)
    assert (round(lo, 3), round(hi, 3)) == (0.744, 0.965)
    assert sa.wilson(30, 30)[1] == pytest.approx(1.0)
    assert sa.wilson(0, 30)[0] == pytest.approx(0.0)


def test_labels_are_exhaustive_and_closed():
    assert sa.labels(-10, -1) == ("menos", "nao_equivalente")
    assert sa.labels(-3, 5) == ("indeterminado", "equivalente")
    assert sa.labels(1, 9) == ("mais", "nao_equivalente")
    assert sa.labels(-8, 8) == ("indeterminado", "equivalente")
    assert sa.labels(-8.01, 0.5) == ("indeterminado", "nao_equivalente")


def test_bootstrap_deterministic_and_degenerate():
    d = np.arange(100.0)
    assert sa.paired_bootstrap_ci(d, 1000, 7) == sa.paired_bootstrap_ci(d, 1000, 7)
    assert sa.paired_bootstrap_ci(d, 1000, 7) != sa.paired_bootstrap_ci(d, 1000, 8)
    lo, hi = sa.paired_bootstrap_ci(d, 1000, 7)
    assert lo < d.mean() < hi
    assert sa.paired_bootstrap_ci(np.full(50, 3.0), 1000, 7) == (3.0, 3.0)


def test_normal_ci():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    lo, hi = sa.normal_ci(d)
    half = sa.Z * d.std(ddof=1) / 2
    assert (lo, hi) == pytest.approx((2.5 - half, 2.5 + half))


def test_cost_all_delivered_is_scheme_bits_and_failure_adds_nack_and_fallback():
    assert sa.cost_bits(140, True, 999) == 140
    assert sa.cost_bits(140, False, 8 * 35 + 40) == 140 + 16 + 320
    assert sa.fallback_bits(b"x" * 35, "raw") == 320


def test_unishox_failure_propagates(monkeypatch):
    def boom(m):
        raise ValueError("roundtrip")
    monkeypatch.setattr(sa, "unishox2", boom)
    with pytest.raises(ValueError):
        sa.fallback_bits(b"x" * 30, "unishox2")


def test_correction7():
    r = {"sucesso": "False", "teto_etiqueta_atingido": "True", "testes_etiqueta": "1024", "max_tag_checks": "1024"}
    out = sa.correction7(r)
    assert out["cap_truncou_lista"] == "True" and out["busca_esgotada"] is True
    assert sa.correction7({**r, "testes_etiqueta": "40"})["busca_esgotada"] is False
    assert sa.correction7({**r, "sucesso": "True", "testes_etiqueta": "3"})["busca_esgotada"] == ""


def test_link_cost_failure_pays_nack_frame():
    for link in sa.LINKS:
        ok = sa.cost_link(link, 143, True, 352)
        bad = sa.cost_link(link, 143, False, 352)
        assert ok == pytest.approx(sa.link_cost(link, 18))
        assert bad == pytest.approx(ok + sa.link_cost(link, 2) + sa.link_cost(link, 44))
    # US915 DR0 carries 11 B per packet: 18 B -> 2 packets
    import lorawan_fragmentation as lf
    assert sa.link_cost("us915_dr0", 18) == pytest.approx(lf.send(18, 11, 10)[1])


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_summary_without_failures_has_no_nan(tmp_path):
    rows = [{"msg_sha256": f"{i:016x}", "bytes": 30, "assim_bits": 130, "delivered": True, "cost_raw": 130,
             "cost_unishox2": 130, "segundos": 1.0, "bits_ppm_o1": 150.0 + i} for i in range(20)]
    p = tmp_path / "a.csv"
    _write_csv(p, rows)
    s = sa.summarize_csv(p, seed=1, points=("ppm_o1",), reps=500)
    assert s["k"] == 20 and s["wilson"][1] == pytest.approx(1.0)
    d = s["diff"][("raw", "ppm_o1")]
    assert not any(math.isnan(x) for x in (d["mean"], *d["boot"], *d["normal"]))
    assert d["mean"] == pytest.approx(-29.5)


MSGS = [b"book a table for two people", b"call a taxi to the airport now", b"what time does it open today"]


def _fixture(tmp_path, primary_ok=(True, False, False)):
    lst = tmp_path / "list.txt"
    lst.write_bytes(b"\n".join(MSGS) + b"\n")
    prim = []
    for m, ok in zip(MSGS, primary_ok):
        prim.append({"msg_sha256": hashlib.sha256(m).hexdigest()[:16], "sucesso": ok, "correto": ok,
                     "assim_bits": 130, "segundos": 10.0, "teto_etiqueta_atingido": False,
                     "testes_etiqueta": 1 if ok else 1024, "max_tag_checks": 1024, "tentativa": 1,
                     "rss_max_bytes": 1000, "mps_driver_bytes": 2000})
    front = [{"msg_sha256": r["msg_sha256"], **{f"bits_{p}": 150.0 for p in sa.POINTS},
              **{f"bits_{p}_term": 152.0 for p in sa.POINTS if p.startswith("ppm")}} for r in prim]
    _write_csv(tmp_path / "p.csv", prim)
    _write_csv(tmp_path / "f.csv", front)
    return lst, prim


def test_build_rows_refuses_incomplete_retry(tmp_path):
    lst, prim = _fixture(tmp_path)
    att = [{**prim[1], "tentativa": 1, "sucesso": False, "correto": False},  # msg 1 failed at 256: needs attempt 2
           {**prim[2], "tentativa": 1, "sucesso": True, "correto": True}]
    _write_csv(tmp_path / "r.csv", att)
    with pytest.raises(SystemExit):
        sa.build_rows(tmp_path / "p.csv", tmp_path / "f.csv", lst, retry_csv=tmp_path / "r.csv")
    att.append({**prim[1], "tentativa": 2, "sucesso": False, "correto": False})
    _write_csv(tmp_path / "r.csv", att)
    rows = sa.build_rows(tmp_path / "p.csv", tmp_path / "f.csv", lst, retry_csv=tmp_path / "r.csv")
    assert [r["delivered_retry"] for r in rows] == [True, False, True]


def test_build_rows_refuses_missing_ngram_rows(tmp_path):
    lst, prim = _fixture(tmp_path)
    _write_csv(tmp_path / "n.csv", prim[:2])
    with pytest.raises(SystemExit):
        sa.build_rows(tmp_path / "p.csv", tmp_path / "f.csv", lst, ngram_csv=tmp_path / "n.csv")


def test_aprs_raw_text_goes_as_characters_as_in_5_41():
    import airtime as at
    import random
    # raw text + 32-bit tag in Base91 (5 chars), as scripts/airtime.py does for "texto cru + etiqueta 32"
    want = at.aprs_airtime_s(at.random_printable(37 + 5 + at.APRS_MSG_WRAPPER_CHARS, random.Random(0)))
    assert sa.raw_link_cost("aprs", 37) == pytest.approx(want)
    assert sa.raw_link_cost("meshtastic", 37) == pytest.approx(sa.link_cost("meshtastic", 37 + 5))
    bad = sa.cost_link("aprs", 143, False, 8 * 37 + 40, raw_len=37)
    ok = sa.cost_link("aprs", 143, True, 8 * 37 + 40, raw_len=37)
    assert bad == pytest.approx(ok + sa.link_cost("aprs", 2) + want)


def test_summary_has_term_sensitivity_receiver_memory_and_variant_airtime(tmp_path):
    lst, prim = _fixture(tmp_path)
    _write_csv(tmp_path / "n.csv", prim)
    sa.write_rows(tmp_path / "a.csv", sa.build_rows(tmp_path / "p.csv", tmp_path / "f.csv", lst,
                                                    ngram_csv=tmp_path / "n.csv"))
    s = sa.summarize_csv(tmp_path / "a.csv", seed=1, reps=200)
    assert ("raw", "ppm_o1_term") in s["diff"]
    assert s["diff"][("raw", "ppm_o1_term")]["mean"] == pytest.approx(s["diff"][("raw", "ppm_o1")]["mean"] - 2)
    assert s["receiver"]["rss_max_bytes"] == 1000 and s["receiver"]["mps_driver_bytes"] == 2000
    for v in ("raw", "unishox2", "raw_ngram"):
        assert ("meshtastic", f"scheme_{v}") in s["air"]
    assert s["correction7"]["busca_esgotada"] == 2


def test_wrong_accept_is_refused(tmp_path):
    lst, prim = _fixture(tmp_path)
    prim[0]["correto"] = False  # accepted a wrong message
    _write_csv(tmp_path / "p.csv", prim)
    with pytest.raises(SystemExit):
        sa.build_rows(tmp_path / "p.csv", tmp_path / "f.csv", lst)


@pytest.mark.skipif(not hyb.RAW_PATH.exists(), reason="raw Taskmaster absent")
def test_reproduces_crit_numbers_on_tm30():
    raw = hyb.load_raw()
    lines = hyb.train_lines(raw, hyb.load_ids(hyb.TRAIN_IDS), hyb.dev_conversations(raw))
    lm = ByteNgramLM(max_context=1)
    lm.add(b"\n".join(lines) + b"\n")
    ctx = bc.nested_counts(lm)
    res = {r["msg_sha256"]: r for r in csv.DictReader((ROOT / "results/taskmaster_en_2026-10-26.csv").open())}
    msgs = [m for m in hyb.DEV_LIST.read_bytes().splitlines() if m]
    cost, ppm = [], []
    for m in msgs:
        r = res[hashlib.sha256(m).hexdigest()[:16]]
        cost.append(sa.cost_bits(int(r["assim_bits"]), r["sucesso"] == "True", sa.fallback_bits(m, "raw")))
        ppm.append(bc.ppm_bits(ctx, m, 1) + 40)
    d = np.array(cost) - np.array(ppm)
    # scripts/analysis/design_review/crit.txt, ov40
    assert np.mean(cost) == pytest.approx(179.9, abs=0.05)
    assert np.mean(ppm) == pytest.approx(177.5, abs=0.05)
    assert d.std(ddof=1) == pytest.approx(112.3, abs=0.05)
