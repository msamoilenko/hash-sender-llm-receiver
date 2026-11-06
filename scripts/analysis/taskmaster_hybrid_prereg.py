#!/usr/bin/env python3
"""Pre-registration for the receiver-side hybrid (n-gram + LLM), 2026-10-26.

Contract v2: the emitter still sends only the `10:4` hashes and the 32-bit tag;
the receiver may run an in-domain byte n-gram first and the Qwen LLM as a
fallback. No return channel. The 30 Taskmaster messages already used are
development and are not reused as test.

This script only selects the new test set and trains the n-gram; it does not
decode. It writes:

  data/taskmaster_train_ids.txt   official Taskmaster train split (normalized)
  data/taskmaster_test_ids.txt    official Taskmaster test split (normalized)
  data/taskmaster_hybrid_30.txt   new test list, one message per line

  python -I scripts/analysis/taskmaster_hybrid_prereg.py
  python -I scripts/analysis/taskmaster_hybrid_prereg.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from assim.hashing import HashScheme  # noqa: E402
from assim.ngram import ByteNgramLM  # noqa: E402

RAW_PATH = ROOT / "data" / "taskmaster_self_dialogs.json"
DEV_LIST = ROOT / "data" / "taskmaster_en_30.txt"
TRAIN_IDS = ROOT / "data" / "taskmaster_train_ids.txt"
TEST_IDS = ROOT / "data" / "taskmaster_test_ids.txt"
LIST_PATH = ROOT / "data" / "taskmaster_hybrid_30.txt"

SEED = 20261027
N = 30
MIN_BYTES = 28
MAX_BYTES = 49
NGRAM_ORDER = 5
SCHEME = HashScheme(10, 8, 4)
CHUNK_BYTES = 2
BEAM = 64
NGRAM_TOPK = 256
LLM_TOPK = 32768
MAX_TAG_CHECKS = 1024
TAG_BITS = 32


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_ids(path: Path) -> list[str]:
    return [ln.strip().rstrip(",") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_raw() -> list[dict]:
    with RAW_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit(f"formato inesperado em {RAW_PATH}")
    return raw


def dev_conversations(raw: list[dict]) -> set[int]:
    """Conversas que contêm alguma das 30 mensagens de desenvolvimento."""
    dev = set(DEV_LIST.read_bytes().splitlines())
    out = set()
    for ci, conv in enumerate(raw):
        for u in conv.get("utterances", []):
            if u.get("speaker") == "USER" and (u.get("text") or "").strip().encode() in dev:
                out.add(ci)
    return out


def test_candidates(raw: list[dict], test_ids: list[str], dev_convs: set[int]) -> list[dict]:
    """Every eligible message in the official test conversations, in file order (2.443; 5.59)."""
    by_id = {conv.get("conversation_id"): (ci, conv) for ci, conv in enumerate(raw)}
    seen: set[bytes] = set()
    cands: list[dict] = []
    for cid in test_ids:
        item = by_id.get(cid)
        if item is None:
            continue
        ci, conv = item
        if ci in dev_convs:
            continue
        for ui, u in enumerate(conv.get("utterances", [])):
            if u.get("speaker") != "USER":
                continue
            text = (u.get("text") or "").strip()
            msg = text.encode("utf-8")
            if not (MIN_BYTES <= len(msg) <= MAX_BYTES):
                continue
            if "\n" in text or "\r" in text or text.startswith("#"):
                continue
            if msg in seen:
                continue
            seen.add(msg)
            digest = hashlib.sha256(f"taskmaster-1-2019|hybrid|seed={SEED}|".encode() + msg).digest()
            cands.append({"conversation": ci, "utterance": ui, "msg": msg, "digest": digest})
    return cands


def select_test(raw: list[dict], test_ids: list[str], dev_convs: set[int]) -> list[dict]:
    """30 menores SHA-256(seed + mensagem) entre as candidatas das conversas oficiais de teste."""
    return sorted(test_candidates(raw, test_ids, dev_convs), key=lambda c: c["digest"])[:N]


def list_bytes(selected: list[dict]) -> bytes:
    return b"\n".join(c["msg"] for c in selected) + b"\n"


def train_lines(raw: list[dict], train_ids: list[str], dev_convs: set[int]) -> list[bytes]:
    by_id = {conv.get("conversation_id"): (ci, conv) for ci, conv in enumerate(raw)}
    lines: list[bytes] = []
    for cid in train_ids:
        item = by_id.get(cid)
        if item is None:
            continue
        ci, conv = item
        if ci in dev_convs:
            continue
        for u in conv.get("utterances", []):
            if u.get("speaker") != "USER":
                continue
            msg = (u.get("text") or "").strip().encode("utf-8")
            if msg:
                lines.append(msg)
    return lines


def train_ngram(lines: list[bytes]) -> ByteNgramLM:
    lm = ByteNgramLM(max_context=NGRAM_ORDER)
    lm.add(b"\n".join(lines) + b"\n")
    return lm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write-ids", action="store_true", help="escreve os IDs normalizados dos splits em data/")
    p.add_argument("--check", action="store_true", help="verifica a lista existente contra a seleção")
    a = p.parse_args()

    raw = load_raw()
    dev_convs = dev_conversations(raw)
    train_ids = load_ids(TRAIN_IDS) if TRAIN_IDS.exists() else []
    test_ids = load_ids(TEST_IDS) if TEST_IDS.exists() else []
    if not train_ids or not test_ids:
        raise SystemExit(f"faltam {TRAIN_IDS} ou {TEST_IDS}; rode --write-ids com os splits oficiais em data/")
    cands = test_candidates(raw, test_ids, dev_convs)
    selected = select_test(raw, test_ids, dev_convs)
    data = list_bytes(selected)
    list_sha = sha256_bytes(data)
    lines = train_lines(raw, train_ids, dev_convs)
    lm = train_ngram(lines)
    entries, contexts = len(lm._n), len(lm._total)
    print(f"raw {RAW_PATH} {sha256_file(RAW_PATH)}")
    print(f"train ids {TRAIN_IDS} {sha256_file(TRAIN_IDS)} ({len(train_ids)} conversas)")
    print(f"test ids {TEST_IDS} {sha256_file(TEST_IDS)} ({len(test_ids)} conversas)")
    print(f"dev conversations excluded {len(dev_convs)}")
    print(f"test candidates {len(cands)} | selected {len(selected)}")
    print(f"lista {LIST_PATH} {list_sha}")
    print(f"ngram order {NGRAM_ORDER} | train lines {len(lines)} | train bytes {sum(len(x) for x in lines) + len(lines)}")
    print(f"ngram entries {entries} | contexts {contexts} | est MiB {(entries * 6 + contexts * 4) / 1024 / 1024:.2f}")

    if a.check:
        on_disk = LIST_PATH.read_bytes() if LIST_PATH.exists() else b""
        print(f"lista em disco {sha256_bytes(on_disk)}")
        if on_disk != data:
            raise SystemExit("a lista versionada não é a seleção reproduzida")
        return

    LIST_PATH.write_bytes(data)
    if a.write_ids:
        TRAIN_IDS.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
        TEST_IDS.write_text("\n".join(test_ids) + "\n", encoding="utf-8")
        print("IDs normalizados escritos")


if __name__ == "__main__":
    main()
