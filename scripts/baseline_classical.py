#!/usr/bin/env python3
"""Classical baselines for the asymmetric short-message scheme (line A, 2026-10-26).

Compares, on the same messages and with the same 8-bit size + 32-bit tag
accounting, four families:

  * the asymmetric scheme, read from the existing result CSVs;
  * a Witten-Bell byte n-gram with context 3 (the project's current comparator);
  * PPM-C with exclusion, orders 1..5, trained on the same Wikipedia corpus;
  * zstd with dictionaries trained on the same Wikipedia corpus (4/16/64/256 KiB).

PPM/n-gram costs are ideal code lengths (-log2 p), without the ~2 bits of
arithmetic-coder termination. zstd costs are actual frame bytes. The emitter of
every classical method must store its model or dictionary; the asymmetric
emitter stores only the hash parameters.

  python -I scripts/baseline_classical.py --self-test
  python -I scripts/baseline_classical.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import time
from pathlib import Path

import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assim.emitter import LENGTH_FIELD_BITS  # noqa: E402
from assim.ngram import ByteNgramLM  # noqa: E402

TAG_BITS = 32
OVERHEAD = LENGTH_FIELD_BITS + TAG_BITS
PPM_ORDERS = (1, 2, 3, 4, 5)
DEFAULT_DICT_SIZES = (4096, 16384, 65536, 262144)

DATASETS = [
    {
        "name": "clean",
        "path": "data/amostra_pt_en.txt",
        "langs": ["PT", "EN"],
        "pt_count": 15,
        "schemes": {
            "PT": ("g522_pt_b1024_k65536.csv", "10:4"),
            "EN": ("g514_b64_k32768.csv", "10:4"),
        },
    },
    {
        "name": "noisy",
        "path": "data/confirmacao_ruido.txt",
        "langs": ["PT", "EN"],
        "pt_count": 15,
        "schemes": {
            "PT": ("c525_pt_b1024_k65536.csv", "10:4"),
            "EN": ("c525_en_b64_k32768.csv", "10:4"),
        },
    },
    {
        "name": "b2w",
        "path": "data/confirmacao_academica_pt.txt",
        "langs": ["PT"],
        "pt_count": 15,
        "schemes": {"PT": ("b540_pt_b1024_k65536.csv", "10:4")},
    },
    {
        "name": "len_en_k2",
        "path": "data/comprimento_en_k2.txt",
        "langs": ["EN"],
        "pt_count": 0,
        "schemes": {"EN": ("l537_en_k2_b64_k32768.csv", "10:4")},
    },
    {
        "name": "len_en_k3",
        "path": "data/comprimento_en_k3.txt",
        "langs": ["EN"],
        "pt_count": 0,
        "schemes": {"EN": ("l537_en_k3_b64_k32768.csv", "10:4")},
    },
    {
        "name": "len_en_k4",
        "path": "data/comprimento_en_k4.txt",
        "langs": ["EN"],
        "pt_count": 0,
        "schemes": {"EN": ("l537_en_k4_b64_k32768.csv", "10:4")},
    },
    {
        "name": "len_pt_k2",
        "path": "data/comprimento_pt_k2.txt",
        "langs": ["PT"],
        "pt_count": 0,
        "schemes": {"PT": ("l537_pt_k2_b1024_k65536.csv", "10:4")},
    },
    {
        "name": "taskmaster",
        "path": "data/taskmaster_en_30.txt",
        "langs": ["EN"],
        "pt_count": 0,
        "schemes": {"EN": ("taskmaster_en_2026-10-26.csv", "10:4")},
    },
]


def load_messages(path: Path) -> list[bytes]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip().encode("utf-8") for ln in lines if ln.strip() and not ln.startswith("#")]


def split_messages(msgs: list[bytes], langs: list[str], pt_count: int) -> dict[str, list[bytes]]:
    if len(langs) == 2:
        return {"PT": msgs[:pt_count], "EN": msgs[pt_count:]}
    return {langs[0]: msgs}


def load_scheme(path: Path, lang: str, scheme: str) -> dict[str, dict]:
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row["idioma"] != lang or row["esquema"] != scheme:
            continue
        out[row["msg_sha256"]] = row
    return out


def nested_counts(lm: ByteNgramLM) -> dict[bytes, dict[bytes, int]]:
    out: dict[bytes, dict[bytes, int]] = {}
    for key, count in lm._n.items():
        ctx, sym = key[:-1], key[-1:]
        out.setdefault(ctx, {})[sym] = count
    return out


def ppm_prob(
    ctx_counts: dict[bytes, dict[bytes, int]], ctx: bytes, sym: bytes, order: int
) -> float:
    """PPM-C with exclusion; returns P(symbol | context) for one byte."""
    excluded: set[bytes] = set()
    p = 1.0
    for k in range(min(order, len(ctx)), -1, -1):
        c = ctx[len(ctx) - k :] if k else b""
        counts = ctx_counts.get(c)
        if not counts:
            continue
        avail = {s: n for s, n in counts.items() if s not in excluded}
        if not avail:
            continue
        n = sum(avail.values())
        t = len(avail)
        if n <= 0:
            continue
        denom = n + t
        if sym in avail:
            return p * (avail[sym] / denom)
        p *= t / denom
        excluded.update(avail)
    rest = 256 - len(excluded)
    if rest <= 0:
        rest = 256
    return p * (1.0 / rest)


def ppm_bits(ctx_counts: dict[bytes, dict[bytes, int]], message: bytes, order: int) -> float:
    seq = b"\n" + message
    total = 0.0
    for i in range(1, len(seq)):
        ctx = seq[max(0, i - order) : i]
        total -= math.log2(ppm_prob(ctx_counts, ctx, seq[i : i + 1], order))
    return total


def order_stats(lm: ByteNgramLM, order: int) -> tuple[int, int]:
    entries = sum(1 for key in lm._n if len(key) <= order + 1)
    contexts = sum(1 for key in lm._total if len(key) <= order)
    return entries, contexts


class Model:
    def __init__(self, lang: str, corpora: Path, heldout_bytes: int, max_order: int, dict_sizes):
        self.lang = lang
        raw = (corpora / lang.lower() / "text.txt").read_bytes()
        cut = raw.rfind(b"\n", 0, len(raw) - heldout_bytes)
        self.train, held = raw[:cut], raw[cut + 1 :]
        self.lm = ByteNgramLM(max_context=max_order)
        self.lm.add(self.train)
        self.ctx_counts = nested_counts(self.lm)
        self.held_lines = [ln for ln in held.split(b"\n") if ln][:300]
        self.held_bytes = sum(len(ln) for ln in self.held_lines)
        self.order_entries = {o: order_stats(self.lm, o) for o in range(1, max_order + 1)}
        self.ppm_held = {
            o: sum(ppm_bits(self.ctx_counts, ln, o) for ln in self.held_lines) / self.held_bytes
            for o in PPM_ORDERS
            if o <= max_order
        }
        self.ppm_best_order = min(self.ppm_held, key=self.ppm_held.get)
        self.dicts: dict[int, dict] = {}
        samples = [self.train[i : i + 1024] for i in range(0, len(self.train), 1024) if len(self.train[i : i + 1024]) >= 32]
        for size in dict_sizes:
            try:
                d = zstd.train_dictionary(size, samples)
                raw_dict = d.as_bytes()
                try:
                    cctx = zstd.ZstdCompressor(level=19, dict_data=d, write_dict_id=False)
                except TypeError:
                    cctx = zstd.ZstdCompressor(level=19, dict_data=d)
                self.dicts[size] = {"dict": d, "raw": raw_dict, "cctx": cctx, "requested": size}
            except Exception as exc:  # training can reject a size/sample combination
                self.dicts[size] = {"error": f"{type(exc).__name__}: {exc}", "requested": size}

    def ngram_bits(self, message: bytes, order: int = 3) -> float:
        return self.lm.bits(message, order) + OVERHEAD

    def ppm_order_bits(self, message: bytes, order: int) -> float:
        return ppm_bits(self.ctx_counts, message, order) + OVERHEAD

    def zstd_bits(self, message: bytes, size: int) -> tuple[float, int]:
        item = self.dicts[size]
        if "error" in item:
            return math.nan, 0
        frame = item["cctx"].compress(message)
        return len(frame) * 8 + OVERHEAD, len(frame)


def summarize(rows: list[dict], dict_sizes: tuple[int, ...]) -> dict:
    n = len(rows)
    success = sum(r["sucesso"] for r in rows)
    scheme_mean = sum(r["assim_bits"] for r in rows) / n
    scheme_per_delivered = scheme_mean / (success / n) if success else math.inf
    out = {
        "n": n,
        "scheme_success": success,
        "scheme_bits_mean": scheme_mean,
        "scheme_bits_per_delivered": scheme_per_delivered,
        "ngram_c3_bits_mean": sum(r["ngram_c3_bits"] for r in rows) / n,
        "ppm_best_order": rows[0]["ppm_best_order"],
        "ppm_best_bits_mean": sum(r[f"ppm_o{rows[0]['ppm_best_order']}_bits"] for r in rows) / n,
    }
    for o in PPM_ORDERS:
        out[f"ppm_o{o}_bits_mean"] = sum(r[f"ppm_o{o}_bits"] for r in rows) / n
    for size in dict_sizes:
        label = size // 1024
        out[f"zstd_{label}k_bits_mean"] = sum(r[f"zstd_{label}k_bits"] for r in rows) / n
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpora", default=str(ROOT / "corpora"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--prefix", default="classical_baseline_2026-10-26")
    p.add_argument("--heldout-bytes", type=int, default=200_000)
    p.add_argument("--max-order", type=int, default=5)
    p.add_argument("--dict-sizes", default=",".join(str(x) for x in DEFAULT_DICT_SIZES))
    p.add_argument("--datasets", default="", help="comma-separated dataset names; default all")
    p.add_argument("--self-test", action="store_true", help="run only on heldout text, never on test messages")
    a = p.parse_args()

    dict_sizes = tuple(int(x) for x in a.dict_sizes.split(",") if x.strip())
    corpora = Path(a.corpora)
    data_dir = Path(a.data_dir)
    results = Path(a.results)
    results.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    models = {lang: Model(lang, corpora, a.heldout_bytes, a.max_order, dict_sizes) for lang in ("PT", "EN")}
    for lang, m in models.items():
        print(
            f"[{lang}] treino {len(m.train)} B, retido {m.held_bytes} B; "
            f"PPM melhor ordem no retido: {m.ppm_best_order} "
            f"({m.ppm_held[m.ppm_best_order]:.3f} b/byte); "
            f"zstd: " + ", ".join(
                f"{s//1024}k=" + (f"{len(d['raw'])}B" if "raw" in d else "ERRO") for s, d in m.dicts.items()
            )
        )

    if a.self_test:
        # Instrument check on heldout text only. No test message is touched.
        for lang, m in models.items():
            sample = m.held_lines[:20]
            print(f"\n[{lang}] self-test on {len(sample)} heldout lines ({sum(len(x) for x in sample)} B)")
            for order in PPM_ORDERS:
                bits = sum(m.ppm_order_bits(x, order) for x in sample)
                print(f"  PPM o{order}: {bits / sum(len(x) for x in sample):.3f} bits/byte (com 40 bits de overhead)")
            for size in dict_sizes:
                bits = sum(m.zstd_bits(x, size)[0] for x in sample)
                print(f"  zstd {size//1024}k: {bits / sum(len(x) for x in sample):.3f} bits/byte (com 40 bits de overhead)")
        print(f"\nself-test ok em {time.time() - t0:.1f}s")
        return

    wanted = {x.strip() for x in a.datasets.split(",") if x.strip()}
    per_rows: list[dict] = []
    summary_rows: list[dict] = []
    for ds in DATASETS:
        if wanted and ds["name"] not in wanted:
            continue
        path = data_dir / Path(ds["path"]).name
        if not path.exists():
            print(f"[{ds['name']}] SKIP: {path} ausente")
            continue
        msgs = load_messages(path)
        by_lang = split_messages(msgs, ds["langs"], ds["pt_count"])
        for lang in ds["langs"]:
            messages = by_lang[lang]
            scheme_name, scheme_label = ds["schemes"][lang]
            scheme = load_scheme(results / scheme_name, lang, scheme_label)
            model = models[lang]
            rows = []
            for msg in messages:
                sha = hashlib.sha256(msg).hexdigest()[:16]
                row = scheme.get(sha)
                if row is None:
                    raise SystemExit(f"[{ds['name']}/{lang}] mensagem {sha} ausente em {scheme_name}")
                success = row["sucesso"] == "True" and row.get("correto", "True") == "True"
                out = {
                    "dataset": ds["name"],
                    "idioma": lang,
                    "msg_sha256": sha,
                    "bytes": len(msg),
                    "esquema": row["esquema"],
                    "assim_bits": float(row["assim_bits"]),
                    "sucesso": success,
                    "ngram_c3_bits": model.ngram_bits(msg, 3),
                    "ppm_best_order": model.ppm_best_order,
                }
                for order in PPM_ORDERS:
                    out[f"ppm_o{order}_bits"] = model.ppm_order_bits(msg, order)
                for size in dict_sizes:
                    bits, frame_bytes = model.zstd_bits(msg, size)
                    out[f"zstd_{size//1024}k_bits"] = bits
                    out[f"zstd_{size//1024}k_frame_bytes"] = frame_bytes
                rows.append(out)
            per_rows.extend(rows)
            summary = summarize(rows, dict_sizes)
            summary.update({"dataset": ds["name"], "idioma": lang})
            for order in PPM_ORDERS:
                entries, contexts = model.order_entries[order]
                summary[f"ppm_o{order}_entries"] = entries
                summary[f"ppm_o{order}_contexts"] = contexts
                summary[f"ppm_o{order}_mem6_kib"] = (entries * 6 + contexts * 4) / 1024
                summary[f"ppm_o{order}_mem4_kib"] = (entries * 4 + contexts * 4) / 1024
            for size, item in model.dicts.items():
                summary[f"zstd_{size//1024}k_dict_bytes"] = len(item["raw"]) if "raw" in item else 0
            ngram_entries = sum(1 for key in model.lm._n if len(key) <= 4)
            ngram_contexts = sum(1 for key in model.lm._total if len(key) <= 3)
            summary["ngram_c3_entries"] = ngram_entries
            summary["ngram_c3_contexts"] = ngram_contexts
            summary["ngram_c3_mem6_kib"] = (ngram_entries * 6 + ngram_contexts * 4) / 1024
            summary["ngram_c3_mem4_kib"] = (ngram_entries * 4 + ngram_contexts * 4) / 1024
            summary_rows.append(summary)
            print(
                f"[{ds['name']}/{lang}] n={summary['n']} sucesso={summary['scheme_success']}/{summary['n']} "
                f"esquema={summary['scheme_bits_mean']:.1f} bits ({summary['scheme_bits_per_delivered']:.1f}/entregue) "
                f"n-grama c3={summary['ngram_c3_bits_mean']:.1f} "
                f"PPM melhor o{summary['ppm_best_order']}={summary['ppm_best_bits_mean']:.1f} "
                f"zstd64k={summary['zstd_64k_bits_mean']:.1f}"
            )

    per_path = results / f"{a.prefix}_per_message.csv"
    sum_path = results / f"{a.prefix}_summary.csv"
    with per_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_rows)
    with sum_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nper-message: {per_path}\nsummary: {sum_path}\ntempo: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
