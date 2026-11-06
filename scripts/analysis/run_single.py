#!/usr/bin/env python3
"""The single pre-registered run of the single article (design, stage 5). Run once; never to improve.

Stages (all with the frozen `10:4` hashes, chunk 2, head 8, 32-bit tag, 1,024 tag checks per attempt):
  primary         Qwen2.5-0.5B (pinned revision), beam 64, topk 32,768, on the N messages
  receiver_ngram  secondary A: byte n-gram order 5 from the train split as receiver, beam 64, topk 256
  retry           secondary B: only primary failures; beam 256, then beam 1,024 if that fails
                  (with the primary: 3 x 1,024 x 2^-32 ~ 7.2e-7 per message)

Each message row is written atomically as soon as it is decoded. `--resume` keeps the complete rows
already on disk (after checking their parameters) and decodes only the rest; without `--resume` an
existing output is refused.

  python scripts/analysis/run_single.py --stage primary --list data/taskmaster_single_1000.txt \
      --out results/single_primary.csv --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from assim.codec import decode, encode  # noqa: E402
from assim.hashing import HashScheme  # noqa: E402

TAG_KEY = b"chave-de-teste-nao-usar-em-producao"
SCHEME = HashScheme(10, 8, 4)
MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REV = "060db6499f32faf8b98477b0a26969ef7d8b9987"
NGRAM_MODEL = "ngram-o5-taskmaster-train"
FROZEN = {"esquema": "10:4", "chunk_bytes": 2, "head_bytes": 8, "tag_bits": 32, "max_tag_checks": 1024,
          "beam": 64, "topk": 32768, "modelo": MODEL, "modelo_rev": MODEL_REV}
STAGES = {
    "primary": [{"beam": 64, "topk": 32768}],
    "receiver_ngram": [{"beam": 64, "topk": 256}],
    "retry": [{"beam": 256, "topk": 32768}, {"beam": 1024, "topk": 32768}],
}
COLUMNS = ["id", "idioma", "bytes", "msg_sha256", "estagio", "tentativa", "modelo", "modelo_rev", "versoes",
           "beam", "topk", "chunk_bytes", "head_bytes", "tag_bits", "max_tag_checks", "esquema", "assim_bits",
           "sucesso", "correto", "candidatos", "testes_etiqueta", "teto_etiqueta_atingido", "segundos",
           "rss_max_bytes", "mps_driver_bytes", "dispositivo"]


def sha16(m: bytes) -> str:
    return hashlib.sha256(m).hexdigest()[:16]


def _write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (int(r["id"]), int(r["tentativa"]))))
    os.replace(tmp, path)


def device_of(lm) -> str:
    return str(getattr(lm, "device", "cpu"))


def load_done(out: Path, stage: str, model_name: str, model_rev: str = "", versions: str = "",
              device: str = "cpu") -> dict[tuple[str, int], dict]:
    """Complete rows on disk keyed by (msg_sha256, tentativa); SystemExit if any row differs from the frozen
    setup or from this process (model revision, library versions, device): a resume never mixes runs."""
    done = {}
    for r in csv.DictReader(out.open(encoding="utf-8")):
        t = int(r["tentativa"])
        want = {k: str(v) for k, v in FROZEN.items() if k not in ("beam", "topk", "modelo", "modelo_rev")}
        want.update({"estagio": stage, "modelo": model_name, "modelo_rev": model_rev, "versoes": versions,
                     "dispositivo": device, **{k: str(v) for k, v in STAGES[stage][t - 1].items()}})
        bad = {k: r.get(k) for k, v in want.items() if r.get(k) != v}
        if bad:
            raise SystemExit(f"{out}: row {r['msg_sha256']} differs from the frozen setup: {bad}")
        done[(r["msg_sha256"], t)] = r
    return done


def _memory(lm) -> tuple[int, str]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    torch = getattr(lm, "torch", None)
    mps = ""
    if torch is not None and getattr(lm, "device", "") == "mps":
        mps = str(torch.mps.driver_allocated_memory())
    return rss, mps


def _decode_row(i: int, m: bytes, stage: str, attempt: int, lm, model_name: str, model_rev: str, versions: str) -> dict:
    params = STAGES[stage][attempt - 1]
    enc = encode(m, FROZEN["chunk_bytes"], SCHEME, TAG_KEY, FROZEN["tag_bits"])
    out, st = decode(enc, lm, TAG_KEY, beam_width=params["beam"], topk=params["topk"],
                     max_tag_checks=FROZEN["max_tag_checks"])
    rss, mps = _memory(lm)
    return {"id": i, "idioma": "EN", "bytes": len(m), "msg_sha256": sha16(m), "estagio": stage, "tentativa": attempt,
            "modelo": model_name, "modelo_rev": model_rev, "versoes": versions, "beam": params["beam"],
            "topk": params["topk"], "chunk_bytes": FROZEN["chunk_bytes"], "head_bytes": FROZEN["head_bytes"],
            "tag_bits": FROZEN["tag_bits"], "max_tag_checks": FROZEN["max_tag_checks"], "esquema": FROZEN["esquema"],
            "assim_bits": enc.total_bits(), "sucesso": out is not None, "correto": out == m,
            "candidatos": st.candidates, "testes_etiqueta": st.tag_checks, "teto_etiqueta_atingido": st.tag_cap_hit,
            "segundos": round(st.seconds, 3), "rss_max_bytes": rss, "mps_driver_bytes": mps, "dispositivo": device_of(lm)}


def run(stage: str, msgs: list[bytes], lm, out: Path, resume: bool, primary_csv: Path | None = None,
        model_name: str = MODEL, model_rev: str = "", versions: str = "") -> list[dict]:
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage}")
    if model_name == MODEL and device_of(lm) != "mps":
        raise SystemExit(f"the pre-registered receiver runs on MPS fp16; this process got {device_of(lm)}")
    if out.exists() and not resume:
        raise SystemExit(f"{out} exists; use --resume to continue it (never to re-run)")
    done = load_done(out, stage, model_name, model_rev, versions, device_of(lm)) if out.exists() else {}
    if stage == "retry":
        if primary_csv is None:
            raise SystemExit("retry needs --primary-csv")
        by_sha = {sha16(m): m for m in msgs}
        todo = [(int(r["id"]), by_sha[r["msg_sha256"]]) for r in csv.DictReader(primary_csv.open(encoding="utf-8"))
                if r["sucesso"] != "True"]
    else:
        todo = list(enumerate(msgs))
    rows = list(done.values())
    for i, m in todo:
        for attempt in range(1, len(STAGES[stage]) + 1):
            row = done.get((sha16(m), attempt))
            if row is None:
                row = _decode_row(i, m, stage, attempt, lm, model_name, model_rev, versions)
                rows.append(row)
                _write(out, rows)
                print(f"  {stage} {i} attempt {attempt} ({len(m)} B) ok={row['sucesso']}", flush=True)
            if str(row["sucesso"]) == "True":
                break
    if rows:
        _write(out, rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--primary-csv", type=Path)
    p.add_argument("--toy", action="store_true", help="byte bigram receiver; smoke tests only")
    a = p.parse_args()
    msgs = [m for m in a.list.read_bytes().splitlines() if m]
    versions, rev = "", ""
    if a.toy:
        from assim.lm import ByteBigramLM

        lm, name = ByteBigramLM(msgs), "toy-bigram"
    elif a.stage == "receiver_ngram":
        import taskmaster_hybrid_prereg as hyb
        from run_hybrid import NgramAdapter

        raw = hyb.load_raw()
        lm = NgramAdapter(hyb.train_ngram(hyb.train_lines(raw, hyb.load_ids(hyb.TRAIN_IDS), hyb.dev_conversations(raw))))
        name = NGRAM_MODEL
    else:
        import torch
        import transformers
        from assim.lm import HFLM

        lm = HFLM(MODEL, revision=MODEL_REV)
        name, rev = MODEL, lm.revision or MODEL_REV
        versions = f"transformers {transformers.__version__}; torch {torch.__version__}"
    run(a.stage, msgs, lm, a.out, a.resume, a.primary_csv, name, rev, versions)


if __name__ == "__main__":
    main()
