#!/usr/bin/env python3
"""Pre-registered analysis of the single article (design, stages 4 and 6). Frozen before the run.

Cost to deliver 100% per message: scheme bits + (0 if delivered, else 2-byte NACK + fallback), with
the fallback being raw text (8L + 40) or Unishox2 (8*bytes + 40). Primary endpoint: paired difference
of that cost against the in-domain PPM o1, read by the two labels (sign, equivalence within 8 bits)
on a paired percentile bootstrap (10,000 resamples, seed from the pre-registration); the normal
approximation is reported beside it as sensitivity. p comes with a Wilson 95% interval.
Correction 7 (DECISOES 5.57 E) is applied here, never in the published CSVs.

  python -I scripts/analysis/single_analysis.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sender_memory.tiny_codecs import unishox2  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_air = _load("airtime", "scripts/airtime.py")
_lf = _load("lorawan_fragmentation", "scripts/analysis/lorawan_fragmentation.py")

OVERHEAD = 40
NACK_BITS = 16
NACK_BYTES = 2
MARGIN_BITS = 8.0
BOOT_REPS = 10_000
BOOT_SEED = 20261106
Z = 1.959963984540054
POINTS = ("raw", "smaz2", "unishox2", "ppm_o1", "ppm_o2_16k", "ppm_o2_32k", "ppm_o2_64k", "ppm_o2", "ppm_o3")
LINKS = ("meshtastic", "us915_dr0", "eu868_dr0", "aprs", "rockblock")
OUT_CSV = ROOT / "docs" / "medidas" / "single_analysis.csv"
OUT_TXT = ROOT / "docs" / "medidas" / "single_summary.txt"


def fallback_bits(msg: bytes, kind: str) -> int:
    if kind == "raw":
        return 8 * len(msg) + OVERHEAD
    if kind == "unishox2":
        return 8 * len(unishox2(msg)) + OVERHEAD
    raise ValueError(kind)


def cost_bits(scheme_bits: int, delivered: bool, fallback: int) -> int:
    return scheme_bits + (0 if delivered else NACK_BITS + fallback)


def paired_bootstrap_ci(diffs, reps: int, seed: int) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(reps, d.size))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def normal_ci(diffs) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=np.float64)
    half = Z * d.std(ddof=1) / math.sqrt(d.size)
    return float(d.mean() - half), float(d.mean() + half)


def wilson(k: int, n: int) -> tuple[float, float]:
    ph = k / n
    den = 1 + Z * Z / n
    c = (ph + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(ph * (1 - ph) / n + Z * Z / (4 * n * n)) / den
    return c - h, c + h


def labels(lo: float, hi: float, margin: float = MARGIN_BITS) -> tuple[str, str]:
    sign = "menos" if hi < 0 else "mais" if lo > 0 else "indeterminado"
    equiv = "equivalente" if -margin <= lo and hi <= margin else "nao_equivalente"
    return sign, equiv


def correction7(row: dict) -> dict:
    """DECISOES 5.57 E: `cap_truncou_lista` = teto_etiqueta_atingido; `busca_esgotada` only on failures."""
    out = dict(row)
    out["cap_truncou_lista"] = row["teto_etiqueta_atingido"]
    failed = row["sucesso"] != "True"
    out["busca_esgotada"] = (int(row["testes_etiqueta"]) == int(row["max_tag_checks"])) if failed else ""
    return out


def link_cost(link: str, nbytes: int) -> float:
    """Seconds on air (RockBLOCK: credits) of one payload of `nbytes`, frame models of 5.28 and 5.41."""
    if link == "meshtastic":
        return _air.meshtastic_airtime_s(nbytes)
    if link == "us915_dr0":
        return _lf.send(nbytes, 11, 10)[1]
    if link == "eu868_dr0":
        return _lf.send(nbytes, 51, 12)[1]
    if link == "aprs":
        chars = _air.base91_chars(8 * nbytes) + _air.APRS_MSG_WRAPPER_CHARS
        return _air.aprs_airtime_s(_air.random_printable(chars, random.Random(0)))
    if link == "rockblock":
        return float(_air.rockblock_credits(nbytes))
    raise ValueError(link)


def raw_link_cost(link: str, msg_len: int) -> float:
    """Raw text + 40-bit overhead. On APRS the text goes as characters and only the 32-bit tag is Base91
    (5 chars), as in scripts/airtime.py for "texto cru + etiqueta 32" (5.28/5.41); elsewhere L + 5 bytes."""
    if link == "aprs":
        chars = msg_len + _air.base91_chars(32) + _air.APRS_MSG_WRAPPER_CHARS
        return _air.aprs_airtime_s(_air.random_printable(chars, random.Random(0)))
    return link_cost(link, msg_len + OVERHEAD // 8)


def cost_link(link: str, scheme_bits: int, delivered: bool, fallback_bits_: int, raw_len: int | None = None) -> float:
    """Air cost to deliver 100%: scheme frame, plus NACK and fallback frames on failure. With `raw_len`
    the fallback is raw text (costed by `raw_link_cost`); otherwise a binary payload of `fallback_bits_`."""
    t = link_cost(link, math.ceil(scheme_bits / 8))
    if not delivered:
        fb = raw_link_cost(link, raw_len) if raw_len is not None else link_cost(link, math.ceil(fallback_bits_ / 8))
        t += link_cost(link, NACK_BYTES) + fb
    return t


def _ok(r: dict, where: str = "") -> bool:
    if r["sucesso"] == "True" and r["correto"] != "True":
        raise SystemExit(f"wrong message accepted {where} {r['msg_sha256']}: the run is invalid")
    return r["sucesso"] == "True"


def build_rows(primary_csv: Path, frontier_csv: Path, list_path: Path,
               ngram_csv: Path | None = None, retry_csv: Path | None = None) -> list[dict]:
    msgs = {hashlib.sha256(m).hexdigest()[:16]: m for m in list_path.read_bytes().splitlines() if m}
    prim = {r["msg_sha256"]: r for r in csv.DictReader(primary_csv.open(encoding="utf-8"))}
    front = {r["msg_sha256"]: r for r in csv.DictReader(frontier_csv.open(encoding="utf-8"))}
    ngram = {r["msg_sha256"]: r for r in csv.DictReader(ngram_csv.open(encoding="utf-8"))} if ngram_csv else {}
    retry: dict[str, list[dict]] = {}
    if retry_csv:
        for r in csv.DictReader(retry_csv.open(encoding="utf-8")):
            retry.setdefault(r["msg_sha256"], []).append(r)
    if set(prim) != set(msgs) or set(front) != set(msgs):
        raise SystemExit("primary CSV, frontier CSV and list do not hold the same messages")
    if ngram_csv and set(ngram) != set(msgs):
        raise SystemExit("secondary A CSV does not hold every message of the list")
    rows = []
    for sha, m in msgs.items():
        p = correction7(prim[sha])
        ok = _ok(p, "in the primary")
        fb_raw, fb_usx = fallback_bits(m, "raw"), fallback_bits(m, "unishox2")
        s = int(p["assim_bits"])
        row = {"msg_sha256": sha, "bytes": len(m), "assim_bits": s, "delivered": ok,
               "fallback_raw_bits": fb_raw, "fallback_unishox2_bits": fb_usx,
               "cost_raw": cost_bits(s, ok, fb_raw), "cost_unishox2": cost_bits(s, ok, fb_usx),
               "segundos": float(p["segundos"]), "cap_truncou_lista": p["cap_truncou_lista"],
               "busca_esgotada": p["busca_esgotada"], "rss_max_bytes": p.get("rss_max_bytes", ""),
               "mps_driver_bytes": p.get("mps_driver_bytes", "")}
        if ngram_csv:
            a = correction7(ngram[sha])
            ok_a = _ok(a, "in secondary A")
            row.update({"delivered_ngram": ok_a, "cost_raw_ngram": cost_bits(s, ok_a, fb_raw),
                        "segundos_ngram": float(a["segundos"]), "busca_esgotada_ngram": a["busca_esgotada"]})
        if retry_csv:
            att = {int(x["tentativa"]): x for x in retry.get(sha, [])}
            if ok and att:
                raise SystemExit(f"retry ran on a delivered message {sha}")
            ok_b = ok
            if not ok:
                # protocol: attempt 1 (beam 256) always; attempt 2 (beam 1024) only if attempt 1 failed
                if 1 not in att or (not _ok(att[1], "in secondary B") and 2 not in att):
                    raise SystemExit(f"secondary B incomplete for {sha}: attempts {sorted(att)}")
                ok_b = any(_ok(x, "in secondary B") for x in att.values())
            row.update({"delivered_retry": ok_b, "cost_raw_retry": cost_bits(s, ok_b, fb_raw),
                        "segundos_retry": sum(float(x["segundos"]) for x in att.values())})
        for pt in POINTS:
            row[f"bits_{pt}"] = float(front[sha][f"bits_{pt}"])
            if f"bits_{pt}_term" in front[sha]:
                row[f"bits_{pt}_term"] = float(front[sha][f"bits_{pt}_term"])
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def summarize_csv(path: Path, seed: int = BOOT_SEED, points=POINTS, reps: int = BOOT_REPS) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    n = len(rows)
    truth = lambda v: str(v) == "True"  # noqa: E731
    k = sum(truth(r["delivered"]) for r in rows)
    out = {"n": n, "k": k, "wilson": wilson(k, n), "cost": {}, "diff": {}, "bits": {}, "air": {}}
    variants = [c[len("cost_"):] for c in rows[0] if c.startswith("cost_")]
    # +2 bits of arithmetic-coder termination on the PPM points: pre-registered sensitivity
    compared = list(points) + [f"{pt}_term" for pt in points if f"bits_{pt}_term" in rows[0]]
    for v in variants:
        c = np.array([float(r[f"cost_{v}"]) for r in rows])
        out["cost"][v] = {"mean": float(c.mean()), "boot": paired_bootstrap_ci(c, reps, seed)}
        for pt in compared:
            d = c - np.array([float(r[f"bits_{pt}"]) for r in rows])
            boot = paired_bootstrap_ci(d, reps, seed)
            out["diff"][(v, pt)] = {"mean": float(d.mean()), "boot": boot, "normal": normal_ci(d),
                                    "labels": labels(*boot)}
    for pt in points:
        out["bits"][pt] = statistics.mean(float(r[f"bits_{pt}"]) for r in rows)
    for kind in ("delivered_ngram", "delivered_retry"):
        if kind in rows[0]:
            kk = sum(truth(r[kind]) for r in rows)
            out[kind] = {"k": kk, "wilson": wilson(kk, n)}
    out["segundos"] = statistics.mean(float(r["segundos"]) for r in rows)
    if "segundos_retry" in rows[0]:
        out["segundos_retry_extra"] = statistics.mean(float(r["segundos_retry"]) for r in rows)
    if "rss_max_bytes" in rows[0]:
        mem = lambda col: max((int(r[col]) for r in rows if r.get(col)), default=None)  # noqa: E731
        out["receiver"] = {"rss_max_bytes": mem("rss_max_bytes"), "mps_driver_bytes": mem("mps_driver_bytes")}
    if "busca_esgotada" in rows[0]:
        out["correction7"] = {"cap_truncou_lista": sum(truth(r["cap_truncou_lista"]) for r in rows),
                              "busca_esgotada": sum(truth(r["busca_esgotada"]) for r in rows)}
        if "busca_esgotada_ngram" in rows[0]:
            out["correction7"]["busca_esgotada_ngram"] = sum(truth(r["busca_esgotada_ngram"]) for r in rows)
    if "fallback_raw_bits" in rows[0]:
        # variant -> (delivered column, fallback): raw fallback goes as text (raw_len), Unishox2 as binary
        air_variants = {"raw": ("delivered", "raw"), "unishox2": ("delivered", "unishox2"),
                        "raw_ngram": ("delivered_ngram", "raw"), "raw_retry": ("delivered_retry", "raw")}
        for link in LINKS:
            for v, (col, fb) in air_variants.items():
                if col not in rows[0]:
                    continue
                out["air"][(link, f"scheme_{v}")] = statistics.mean(
                    cost_link(link, int(r["assim_bits"]), truth(r[col]), int(r[f"fallback_{fb}_bits"]),
                              raw_len=int(r["bytes"]) if fb == "raw" else None) for r in rows)
            for pt in points:
                out["air"][(link, pt)] = statistics.mean(
                    raw_link_cost(link, int(r["bytes"])) if pt == "raw"
                    else link_cost(link, math.ceil(float(r[f"bits_{pt}"]) / 8)) for r in rows)
    return out


def render(s: dict) -> str:
    f = lambda x: f"{x:.1f}"  # noqa: E731
    lines = [f"n={s['n']} delivered={s['k']} p={s['k'] / s['n']:.4f} Wilson95 [{s['wilson'][0]:.4f}, {s['wilson'][1]:.4f}]",
             f"mean decode seconds {s['segundos']:.2f}"]
    for kind in ("delivered_ngram", "delivered_retry"):
        if kind in s:
            lines.append(f"{kind}: {s[kind]['k']}/{s['n']} Wilson95 [{s[kind]['wilson'][0]:.4f}, {s[kind]['wilson'][1]:.4f}]")
    if "receiver" in s:
        lines.append(f"receiver memory: max RSS {s['receiver']['rss_max_bytes']} B, max MPS driver {s['receiver']['mps_driver_bytes']} B")
    if "correction7" in s:
        lines.append("correction 7: " + ", ".join(f"{k} {v}" for k, v in s["correction7"].items()))
    if "segundos_retry_extra" in s:
        lines.append(f"retry extra receiver seconds per message (all messages) {s['segundos_retry_extra']:.2f}")
    for v, c in s["cost"].items():
        lines.append(f"cost_{v}: mean {f(c['mean'])} boot95 [{f(c['boot'][0])}; {f(c['boot'][1])}]")
    lines.append("bits per point: " + ", ".join(f"{pt} {f(b)}" for pt, b in s["bits"].items()))
    for (v, pt), d in s["diff"].items():
        lines.append(f"diff cost_{v} - {pt}: {f(d['mean'])} boot95 [{f(d['boot'][0])}; {f(d['boot'][1])}] "
                     f"normal95 [{f(d['normal'][0])}; {f(d['normal'][1])}] -> {d['labels'][0]}, {d['labels'][1]}")
    for (link, what), t in s["air"].items():
        lines.append(f"air {link:10s} {what:12s} {t:.4f}")
    return "\n".join(lines) + "\n"


def analyze(primary_csv: Path, frontier_csv: Path, list_path: Path, seed: int = BOOT_SEED,
            ngram_csv: Path | None = None, retry_csv: Path | None = None,
            out_csv: Path = OUT_CSV, out_txt: Path = OUT_TXT) -> dict:
    write_rows(out_csv, build_rows(primary_csv, frontier_csv, list_path, ngram_csv, retry_csv))
    s = summarize_csv(out_csv, seed)
    out_txt.write_text(render(s))
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary", type=Path, default=ROOT / "results" / "single_primary.csv")
    p.add_argument("--ngram", type=Path, default=ROOT / "results" / "single_receiver_ngram.csv")
    p.add_argument("--retry", type=Path, default=ROOT / "results" / "single_retry.csv")
    p.add_argument("--frontier", type=Path, default=ROOT / "docs" / "medidas" / "frontier_test.csv")
    p.add_argument("--list", type=Path, default=ROOT / "data" / "taskmaster_single_1000.txt")
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    a = p.parse_args()
    s = analyze(a.primary, a.frontier, a.list, a.seed, a.ngram if a.ngram.exists() else None,
                a.retry if a.retry.exists() else None)
    print(render(s))


if __name__ == "__main__":
    main()
