#!/usr/bin/env python3
"""In-domain classical senders for the single article (design, stage 2). No decoding.

Bits per message, all with 8 size bits + 32 tag bits (overhead 40): raw text, smaz2, Unishox2 and
PPM-C (o1; o2 pruned to 16/32/64 KiB of tables; o2 full; o3), trained on the official Taskmaster
train split without the development conversations. PPM bits are ideal code lengths (`ppm_bits`);
the `_term` columns add 2 bits of arithmetic-coder termination as sensitivity. The PPM models are
the same objects measured in `docs/medidas/sender_memory.csv` (`sender_memory.measure.ppm_models`).

  python -I scripts/analysis/indomain_frontier.py --dev
  python -I scripts/analysis/indomain_frontier.py --test data/taskmaster_single_1000.txt \
      --prereg docs/PREREGISTRO_ARTIGO_UNICO.md      # refuses before the pre-registration is committed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_classical as bc  # noqa: E402
import taskmaster_hybrid_prereg as hyb  # noqa: E402
from assim.ngram import ByteNgramLM  # noqa: E402
from sender_memory import measure, tiny_codecs  # noqa: E402

OVERHEAD = 40
TERM_BITS = 2
POINTS = ("raw", "smaz2", "unishox2", "ppm_o1", "ppm_o2_16k", "ppm_o2_32k", "ppm_o2_64k", "ppm_o2", "ppm_o3")
TARGETS = measure.O2_TARGETS
DEV_CSV = ROOT / "docs" / "medidas" / "frontier_dev.csv"
TEST_CSV = ROOT / "docs" / "medidas" / "frontier_test.csv"
DEV_IDS = ROOT / "data" / "taskmaster_dev_ids.txt"


@dataclass
class Point:
    name: str
    fn: Callable[[bytes], float]
    counts_sha256: str | None = None
    reached: bool = True

    def bits(self, msg: bytes) -> float:
        return self.fn(msg)


def _ppm(c, order):
    return lambda m: bc.ppm_bits(c, m, order) + OVERHEAD


def build_points(lines: list[bytes]) -> dict[str, Point]:
    from sender_memory import ppm_tables

    lm = ByteNgramLM(max_context=3)
    lm.add(b"\n".join(lines) + b"\n")
    models = measure.ppm_models(bc.nested_counts(lm))
    pts = {
        "raw": Point("raw", lambda m: 8 * len(m) + OVERHEAD),
        "smaz2": Point("smaz2", lambda m: 8 * len(tiny_codecs.smaz2(m)) + OVERHEAD),
        "unishox2": Point("unishox2", lambda m: 8 * len(tiny_codecs.unishox2(m)) + OVERHEAD),
    }
    for name, (c, order, reached) in models.items():
        pts[name] = Point(name, _ppm(c, order), ppm_tables.counts_sha256(c), reached)
    return {p: pts[p] for p in POINTS}


def per_message(msgs: list[bytes], points: dict[str, Point]) -> list[dict]:
    rows = []
    for m in msgs:
        row = {"msg_sha256": hashlib.sha256(m).hexdigest()[:16], "bytes": len(m)}
        for name, pt in points.items():
            row[f"bits_{name}"] = pt.bits(m)
            if name.startswith("ppm"):
                row[f"bits_{name}_term"] = row[f"bits_{name}"] + TERM_BITS
        rows.append(row)
    return rows


def require_prereg_committed(prereg: Path) -> None:
    """Competitor bits on the test list exist only after the pre-registration is committed and clean."""
    prereg = Path(prereg).resolve()
    if not prereg.exists():
        raise SystemExit(f"pre-registration {prereg} does not exist")
    repo = str(prereg.parent)
    head = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%H", "--", prereg.name],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", prereg.name],
                           capture_output=True, text=True).stdout.strip()
    if not head or dirty:
        raise SystemExit(f"pre-registration {prereg.name} is not committed and clean: refusing the test list")


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def _summary(label: str, rows: list[dict], points: dict[str, Point]) -> list[str]:
    out = [f"[{label}] n={len(rows)}"]
    for name, pt in points.items():
        mean = sum(r[f"bits_{name}"] for r in rows) / len(rows)
        extra = f" reached={pt.reached} counts {pt.counts_sha256[:12]}" if pt.counts_sha256 else ""
        out.append(f"  {name:12s} {mean:7.1f}{extra}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev", action="store_true", help="the 60 development messages (exploratory)")
    p.add_argument("--test", type=Path, help="test list; needs --prereg committed")
    p.add_argument("--prereg", type=Path)
    a = p.parse_args()
    if bool(a.dev) == bool(a.test):
        p.error("use exactly one of --dev or --test")
    if a.test:
        require_prereg_committed(a.prereg or p.error("--test needs --prereg"))
    raw = hyb.load_raw()
    train_ids = hyb.load_ids(hyb.TRAIN_IDS)
    lines = hyb.train_lines(raw, train_ids, hyb.dev_conversations(raw))
    points = build_points(lines)
    if a.dev:
        out, rows = [], []
        for name, path in (("tm30", hyb.DEV_LIST), ("hyb30", hyb.LIST_PATH)):
            r = per_message([m for m in path.read_bytes().splitlines() if m], points)
            for x in r:
                x["set"] = name
            out += _summary(name, r, points)
            rows += r
        _write(DEV_CSV, rows)
        print("\n".join(out) + f"\nCSV: {DEV_CSV}")
        return
    dev_ids = set(hyb.load_ids(DEV_IDS))
    if dev_ids & set(train_ids):
        raise SystemExit("dev split and train split share conversations")
    msgs = [m for m in a.test.read_bytes().splitlines() if m]
    leaked = set(msgs) & set(lines)
    if leaked:
        raise SystemExit(f"{len(leaked)} test messages are identical to training lines")
    rows = per_message(msgs, points)
    _write(TEST_CSV, rows)
    print("\n".join(_summary("test", rows, points)) + f"\nCSV: {TEST_CSV}")


if __name__ == "__main__":
    main()
