#!/usr/bin/env python3
"""Pre-registration for the Taskmaster-1 test set (2026-10-26).

The selection rule and the frozen parameters are fixed before any decoding. This
script only reads the corpus and computes the ideal cost (no beam search). It
writes:

  data/taskmaster_en_30.txt      selected messages, one per line (canonical list)
  docs/taskmaster_prereg.csv     per-message ideal bits, scheme bits, ratio, forecast

Raw corpus (not versioned; 63 MB, CC BY 4.0):

  https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/self-dialogs.json

  python scripts/analysis/taskmaster_prereg.py --download
  python scripts/analysis/taskmaster_prereg.py --check

`--check` recomputes the selection and the envelope and verifies the pinned
SHAs; it does not load the model. The gate imports the same functions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from en_dev_thresholds import base_envelope, observations  # noqa: E402

RAW_URL = "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/self-dialogs.json"
RAW_PATH = ROOT / "data" / "taskmaster_self_dialogs.json"
LIST_PATH = ROOT / "data" / "taskmaster_en_30.txt"
PREREG_CSV = ROOT / "docs" / "taskmaster_prereg.csv"

# Pinned after the first selection run (2026-10-26), before any decoding.
RAW_SHA256 = "1e590ed0ccee279e40c2fb9e083d3b9417477c6bfe35ce5b2277167698dd858d"
LIST_SHA256 = "fa23c3fe87451b6d6cbfc0e1f9dcd0f10ceff9208c59315d22efd1a67958370f"

SEED = 20261026
N = 30
MIN_BYTES = 28
MAX_BYTES = 49
TAG_BITS = 32
LENGTH_FIELD_BITS = 8
SCHEME = (10, 8, 4)  # head_bits, head_bytes, tail_bits
CHUNK_BYTES = 2
BEAM = 64
TOPK = 32768
MAX_TAG_CHECKS = 1024
MODEL = "Qwen/Qwen2.5-0.5B"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download() -> None:
    if RAW_PATH.exists() and sha256_file(RAW_PATH) == RAW_SHA256:
        print(f"raw já presente e conferido: {RAW_PATH}")
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"baixando {RAW_URL}")
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    got = sha256_file(RAW_PATH)
    if got != RAW_SHA256:
        raise SystemExit(f"SHA-256 do bruto não bate: esperado {RAW_SHA256}, obtido {got}")
    print(f"raw conferido: {got}")


def load_conversations(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"formato inesperado em {path}: topo não é uma lista")
    return data


def candidates(conversations: list[dict]) -> list[dict]:
    """Falas do usuário na faixa, distintas, na ordem do arquivo.

    As restrições de nova linha, `#` inicial e strip não são filtro de limpeza: são
    o formato de uma linha por mensagem lido por `run_experiment.py` (que faz
    `strip()` e descarta linhas começando com `#`).
    """
    seen: set[bytes] = set()
    out: list[dict] = []
    for ci, conv in enumerate(conversations):
        for ui, utt in enumerate(conv.get("utterances", [])):
            if utt.get("speaker") != "USER":
                continue
            text = (utt.get("text") or "").strip()
            msg = text.encode("utf-8")
            if not (MIN_BYTES <= len(msg) <= MAX_BYTES):
                continue
            if "\n" in text or "\r" in text or text.startswith("#"):
                continue
            if msg in seen:
                continue
            seen.add(msg)
            out.append({"conversation": ci, "utterance": ui, "text": text, "msg": msg})
    return out


def selection_digest(msg: bytes, seed: int = SEED) -> bytes:
    return hashlib.sha256(f"taskmaster-1-2019|seed={seed}|".encode() + msg).digest()


def select(cands: list[dict], n: int = N, seed: int = SEED) -> list[dict]:
    """The n smallest SHA-256 digests of `seed` + message, listed by digest.

    Hash ordering is deterministic across Python versions and does not depend on
    the random module. If fewer than n candidates exist, all are used and the
    caller reports the actual N.
    """
    chosen = sorted(cands, key=lambda c: selection_digest(c["msg"], seed))[:n]
    return sorted(chosen, key=lambda c: selection_digest(c["msg"], seed))


def list_bytes(selected: list[dict]) -> bytes:
    return b"\n".join(c["msg"] for c in selected) + b"\n"


def check_list(path: Path = LIST_PATH, expected_sha: str | None = None) -> list[bytes]:
    data = path.read_bytes()
    got = sha256_bytes(data)
    expected_sha = LIST_SHA256 if expected_sha is None else expected_sha
    if expected_sha and got != expected_sha:
        raise SystemExit(f"SHA-256 da lista não bate: esperado {expected_sha}, obtido {got}")
    msgs = data.splitlines()
    if not msgs:
        raise SystemExit(f"lista vazia: {path}")
    return msgs


def envelope() -> tuple[float, float]:
    return base_envelope(observations())


def forecast(ratio: float, lo: float, hi: float) -> str:
    if ratio < lo:
        return "deliver"
    if ratio > hi:
        return "fail"
    return "uncertain"


def scheme_bits(length: int) -> int:
    from assim.hashing import HashScheme

    scheme = HashScheme(*SCHEME)
    return LENGTH_FIELD_BITS + TAG_BITS + sum(scheme.bits_for(length, CHUNK_BYTES))


def ideal_bits(lm, msg: bytes) -> float:
    # mesmo arredondamento de `simetrico_ideal` em run_experiment.py
    return round(lm.sequence_bits(msg) + TAG_BITS, 1)


def write_predictions(msgs: list[bytes], lo: float, hi: float) -> tuple[list[dict], object]:
    from assim.lm import HFLM

    lm = HFLM(MODEL)
    rows = []
    for i, msg in enumerate(msgs):
        ideal = ideal_bits(lm, msg)
        bits = scheme_bits(len(msg))
        ratio = ideal / bits
        rows.append({
            "index": i,
            "msg_sha256": hashlib.sha256(msg).hexdigest()[:16],
            "msg_sha256_full": hashlib.sha256(msg).hexdigest(),
            "bytes": len(msg),
            "ideal_bits": f"{ideal:.1f}",
            "assim_bits": bits,
            "ratio": f"{ratio:.6f}",
            "prediction": forecast(ratio, lo, hi),
        })
    PREREG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PREREG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows, lm


def revision_check(lm) -> None:
    """Item 11 do adendo: o custo ideal com a revisão fixada bate com os CSVs antigos?

    Se bater, a inferência de que as rodadas usaram o único snapshot no cache fica
    fechada sem re-rodar a busca. Se não bater, fica registrado que não fecha.
    """
    clean = [ln.strip().encode() for ln in (ROOT / "data" / "amostra_pt_en.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")][15:]
    noisy = [ln.strip().encode() for ln in (ROOT / "data" / "confirmacao_ruido.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")][15:]
    for label, msgs, name in (("limpo", clean, "g514_b64_k32768.csv"),
                              ("ruidoso", noisy, "c525_en_b64_k32768.csv")):
        path = ROOT / "results" / name
        if not path.exists():
            print(f"[revisão] {label}: {name} ausente, conferência pulada")
            continue
        old = {}
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["idioma"] == "EN" and r["esquema"] == "10:4" and r["beam"] == "64" and r["topk"] == "32768":
                    old[r["msg_sha256"]] = float(r["simetrico_ideal"])
        diffs = []
        for m in msgs:
            sha = hashlib.sha256(m).hexdigest()[:16]
            if sha not in old:
                continue
            diffs.append(abs(ideal_bits(lm, m) - old[sha]))
        if diffs:
            print(f"[revisão] {label}: {len(diffs)} mensagens, |dif| máx {max(diffs):.3f} bits, "
                  f"iguais após arredondar {sum(d == 0 for d in diffs)}/{len(diffs)}")
        else:
            print(f"[revisão] {label}: nenhuma mensagem casou com {name}")


def print_summary(selected: list[dict], rows: list[dict], lo: float, hi: float) -> None:
    pred = [r["prediction"] for r in rows]
    d, u, f_ = pred.count("deliver"), pred.count("uncertain"), pred.count("fail")
    print(f"\nbruto: {RAW_PATH} ({sha256_file(RAW_PATH)})")
    print(f"lista: {LIST_PATH} ({sha256_bytes(list_bytes(selected))})")
    print(f"envelope de desenvolvimento: entrega se razão < {lo:.6f}; falha se > {hi:.6f}; "
          f"incerta entre {lo:.6f} e {hi:.6f}")
    print(f"previsão agregada: entregues {d}, incertas {u}, falhas {f_}; "
          f"faixa de sucesso prevista [{d}/{len(rows)}, {(d + u)}/{len(rows)}]")
    print(f"previsões: {PREREG_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true", help="baixa o self-dialogs.json e confere o SHA-256")
    p.add_argument("--check", action="store_true", help="só recomputa a seleção e o envelope, sem modelo")
    p.add_argument("--allow-fewer", action="store_true", help="aceita N < 30 (plano B do pré-registro)")
    a = p.parse_args()

    if a.download:
        download()
    if not RAW_PATH.exists():
        raise SystemExit(f"bruto ausente: {RAW_PATH} (rode com --download)")
    got = sha256_file(RAW_PATH)
    if got != RAW_SHA256:
        raise SystemExit(f"SHA-256 do bruto não bate: esperado {RAW_SHA256}, obtido {got}")

    convs = load_conversations(RAW_PATH)
    cands = candidates(convs)
    selected = select(cands)
    if len(selected) < N and not a.allow_fewer:
        raise SystemExit(f"plano B: só {len(selected)} candidatas na faixa; use --allow-fewer para aceitar N menor")
    data = list_bytes(selected)
    list_sha = sha256_bytes(data)
    print(f"conversas {len(convs)} | candidatas distintas {len(cands)} | selecionadas {len(selected)}")
    print(f"SHA-256 da lista: {list_sha}")
    print(f"SHA-256 da lista pinado no script: {LIST_SHA256 or '(vazio)'}")
    if LIST_SHA256 and list_sha != LIST_SHA256:
        raise SystemExit("a seleção reproduzida não bate com LIST_SHA256")

    if a.check:
        lo, hi = envelope()
        print(f"envelope: {lo:.6f} .. {hi:.6f}")
        if LIST_PATH.exists():
            on_disk = LIST_PATH.read_bytes()
            print(f"SHA-256 do arquivo existente: {sha256_bytes(on_disk)}")
            if on_disk != data:
                raise SystemExit("a lista versionada não é a seleção reproduzida")
            if sha256_bytes(on_disk) != list_sha:
                raise SystemExit("SHA-256 do arquivo existente difere da seleção")
        return

    LIST_PATH.write_bytes(data)
    if not LIST_SHA256:
        print("AVISO: LIST_SHA256 está vazio; fixe o valor impresso acima antes de commitar")
    lo, hi = envelope()
    rows, lm = write_predictions([c["msg"] for c in selected], lo, hi)
    print_summary(selected, rows, lo, hi)
    revision_check(lm)


if __name__ == "__main__":
    main()
