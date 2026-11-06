#!/usr/bin/env python3
"""Receiver-side hybrid: in-domain n-gram first, Qwen LLM as fallback.

Contract v2: the emitter sends only the `10:4` hashes and the 32-bit tag. The
receiver runs a byte n-gram (order 5, in-domain Taskmaster train split) first and
the Qwen LLM only when the n-gram fails. No return channel.

One execution of the pre-registered test (`docs/PREREGISTRO_HIBRIDO_2026-10-26.md`).
Do not re-run to improve the result.

  python -I scripts/analysis/run_hybrid.py
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from assim.codec import decode, encode  # noqa: E402
from assim.lm import HFLM  # noqa: E402

from taskmaster_hybrid_prereg import (  # noqa: E402
    BEAM,
    CHUNK_BYTES,
    LIST_PATH,
    MAX_TAG_CHECKS,
    NGRAM_TOPK,
    LLM_TOPK,
    SCHEME,
    TAG_BITS,
    TRAIN_IDS,
    dev_conversations,
    load_ids,
    load_raw,
    train_lines,
    train_ngram,
)

TAG_KEY = b"chave-de-teste-nao-usar-em-producao"
MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_REV = "060db6499f32faf8b98477b0a26969ef7d8b9987"
OUT = ROOT / "results" / "taskmaster_hybrid_2026-10-26.csv"


class NgramAdapter:
    """Adapta o ByteNgramLM byte a byte à interface que `decode` espera."""

    def __init__(self, lm):
        self.lm = lm
        self.token_bytes = [bytes([i]) for i in range(256)]
        self.start_id = 256

    def next_topk(self, seqs, k):
        k = min(k, 256)
        vals = np.empty((len(seqs), k), dtype=np.float64)
        idx = np.empty((len(seqs), k), dtype=np.int64)
        for i, s in enumerate(seqs):
            ctx = (b"\n" + bytes(s))[-self.lm.max_context:]
            lp = np.array([math.log2(self.lm.prob(ctx, bytes([b]))) for b in range(256)])
            order = np.argsort(-lp)[:k]
            vals[i] = lp[order]
            idx[i] = order
        return vals, idx


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    raw = load_raw()
    dev = dev_conversations(raw)
    lines = train_lines(raw, load_ids(TRAIN_IDS), dev)
    lm_ngram = train_ngram(lines)
    adapter = NgramAdapter(lm_ngram)
    msgs = LIST_PATH.read_bytes().splitlines()
    lm = HFLM(MODEL)
    rows: list[dict] = []
    for i, msg in enumerate(msgs):
        enc = encode(msg, CHUNK_BYTES, SCHEME, TAG_KEY, TAG_BITS)
        t0 = time.perf_counter()
        out_ng, st_ng = decode(enc, adapter, TAG_KEY, beam_width=BEAM, topk=NGRAM_TOPK,
                               max_tag_checks=MAX_TAG_CHECKS)
        ng_seconds = time.perf_counter() - t0
        via = "ngram" if out_ng == msg else ""
        out_llm, st_llm, llm_seconds = None, None, 0.0
        if out_ng != msg:
            t0 = time.perf_counter()
            out_llm, st_llm = decode(enc, lm, TAG_KEY, beam_width=BEAM, topk=LLM_TOPK,
                                     max_tag_checks=MAX_TAG_CHECKS)
            llm_seconds = time.perf_counter() - t0
            if out_llm == msg:
                via = "llm"
        out = out_ng if out_ng == msg else out_llm
        delivered = out is not None
        correct = out == msg
        ngram_checks = st_ng.tag_checks
        llm_checks = st_llm.tag_checks if st_llm is not None else 0
        rows.append({
            "id": i,
            "idioma": "EN",
            "bytes": len(msg),
            "msg_sha256": hashlib.sha256(msg).hexdigest()[:16],
            "esquema": "10:4",
            "chunk_bytes": CHUNK_BYTES,
            "head_bytes": SCHEME.head_bytes,
            "tag_bits": TAG_BITS,
            # teto total: duas buscas, 1.024 testes cada; 2.048 * 2^-32 < 1e-6
            "max_tag_checks": 2 * MAX_TAG_CHECKS,
            "ngram_beam": BEAM,
            "ngram_topk": NGRAM_TOPK,
            "llm_beam": BEAM,
            "llm_topk": LLM_TOPK,
            "assim_bits": enc.total_bits(),
            "via": via or "none",
            "sucesso": delivered,
            "correto": correct,
            "testes_etiqueta": ngram_checks + llm_checks,
            "ngram_testes": ngram_checks,
            "llm_testes": llm_checks,
            "ngram_candidatos": st_ng.candidates,
            "llm_candidatos": st_llm.candidates if st_llm is not None else "",
            "ngram_segundos": round(ng_seconds, 3),
            "llm_segundos": round(llm_seconds, 3),
            "segundos": round(ng_seconds + llm_seconds, 3),
            "teto_etiqueta_atingido": st_ng.tag_cap_hit or (st_llm.tag_cap_hit if st_llm is not None else False),
            "modelo": MODEL,
            "modelo_rev": lm.revision or MODEL_REV,
        })
        write_rows(OUT, rows)
        print(f"  mensagem {i + 1}/{len(msgs)} ({len(msg)} bytes) via={via or 'none'}")

    ok = sum(r["correto"] for r in rows)
    bits = st.mean(r["assim_bits"] for r in rows)
    secs = st.mean(r["segundos"] for r in rows)
    print(f"\nresultado: {ok}/{len(rows)} corretas, {bits:.1f} bits/mensagem, {secs:.2f} s/mensagem")
    print(f"CSV: {OUT}")
    print(f"via n-grama: {sum(r['via'] == 'ngram' for r in rows)} | via LLM: {sum(r['via'] == 'llm' for r in rows)}")


if __name__ == "__main__":
    main()
