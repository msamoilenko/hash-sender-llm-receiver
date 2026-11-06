#!/usr/bin/env python3
"""Selection for the single article (design, stage 4). Selects; never decodes.

Pool: the 770 conversations of the official Taskmaster-1 `dev` split (train-dev-test/dev.csv), minus
the development conversations among them; user utterances, strip(), 28-49 bytes, no \\n/\\r, no
leading '#', exact duplicates removed, and no utterance identical to a training line. Test list:
the N smallest SHA-256(DIGEST_PREFIX + message), in digest order.

  python -I scripts/analysis/single_prereg.py           # writes data/taskmaster_single_1000.txt
  python -I scripts/analysis/single_prereg.py --check   # recomputes and compares, writes nothing
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import taskmaster_hybrid_prereg as hyb  # noqa: E402
from single_analysis import BOOT_SEED  # noqa: E402,F401  (re-exported for the gate)

SEED = 20261105
N = 1000
MIN_BYTES, MAX_BYTES = 28, 49
DIGEST_PREFIX = f"taskmaster-1-2019|single|seed={SEED}|".encode()
DEV_IDS = ROOT / "data" / "taskmaster_dev_ids.txt"
LIST_PATH = ROOT / "data" / "taskmaster_single_1000.txt"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dev_split_matches(raw: list[dict], train_ids: list[str], test_ids: list[str], dev_ids: list[str]) -> bool:
    used = set(train_ids) | set(test_ids)
    neither = {c.get("conversation_id") for c in raw if c.get("conversation_id") not in used}
    return neither == set(dev_ids)


def pool(raw: list[dict], dev_ids: list[str], dev_convs: set[int], train_set: set[bytes]) -> tuple[list[bytes], int, int]:
    """(kept, eligible, identical_to_train) in the order of the dev ID list."""
    by_id = {conv.get("conversation_id"): (ci, conv) for ci, conv in enumerate(raw)}
    seen: set[bytes] = set()
    kept: list[bytes] = []
    identical = 0
    for cid in dev_ids:
        item = by_id.get(cid)
        if item is None:
            continue
        ci, conv = item
        if ci in dev_convs:
            continue
        for u in conv.get("utterances", []):
            if u.get("speaker") != "USER":
                continue
            text = (u.get("text") or "").strip()
            msg = text.encode("utf-8")
            if not (MIN_BYTES <= len(msg) <= MAX_BYTES) or "\n" in text or "\r" in text or text.startswith("#"):
                continue
            if msg in seen:
                continue
            seen.add(msg)
            if msg in train_set:
                identical += 1
                continue
            kept.append(msg)
    return kept, len(seen), identical


def select(pool_msgs: list[bytes], n: int = N) -> list[bytes]:
    return sorted(pool_msgs, key=lambda m: hashlib.sha256(DIGEST_PREFIX + m).digest())[:n]


def list_bytes(msgs: list[bytes]) -> bytes:
    return b"\n".join(msgs) + b"\n"


def compute() -> dict:
    """Everything the pre-registration states about the selection and the pruned counts."""
    import baseline_classical as bc
    from assim.ngram import ByteNgramLM
    from sender_memory import measure, ppm_tables

    raw = hyb.load_raw()
    train_ids, test_ids, dev_ids = hyb.load_ids(hyb.TRAIN_IDS), hyb.load_ids(hyb.TEST_IDS), hyb.load_ids(DEV_IDS)
    dev_convs = hyb.dev_conversations(raw)
    lines = hyb.train_lines(raw, train_ids, dev_convs)
    kept, eligible, identical = pool(raw, dev_ids, dev_convs, set(lines))
    chosen = select(kept)
    lm = ByteNgramLM(max_context=3)
    lm.add(b"\n".join(lines) + b"\n")
    models = measure.ppm_models(bc.nested_counts(lm))
    pruned = {name: (ppm_tables.counts_sha256(c), ppm_tables.table_bytes(c), reached)
              for name, (c, _, reached) in models.items() if name in measure.O2_TARGETS}
    dev_in_dev_split = sum(1 for ci in dev_convs if raw[ci].get("conversation_id") in set(dev_ids))
    return {"raw_sha": hyb.sha256_file(hyb.RAW_PATH), "dev_ids_sha": hyb.sha256_file(DEV_IDS),
            "dev_split_ok": dev_split_matches(raw, train_ids, test_ids, dev_ids), "dev_ids": len(dev_ids),
            "dev_convs_in_dev_split": dev_in_dev_split, "eligible": eligible, "identical": identical,
            "pool": len(kept), "list": list_bytes(chosen), "pruned": pruned}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    c = compute()
    if not c["dev_split_ok"]:
        raise SystemExit("data/taskmaster_dev_ids.txt is not the set of conversations outside train and test")
    print(f"raw {c['raw_sha']}\ndev ids {c['dev_ids_sha']} ({c['dev_ids']} conversations; "
          f"{c['dev_convs_in_dev_split']} development conversations inside)")
    print(f"eligible {c['eligible']} | identical to train {c['identical']} | pool {c['pool']} | N {N}")
    print(f"list {LIST_PATH.relative_to(ROOT)} {sha256_bytes(c['list'])}")
    for name, (sha, tb, reached) in c["pruned"].items():
        print(f"{name} counts {sha} table_bytes {tb} reached {reached}")
    if a.check:
        if not LIST_PATH.exists() or LIST_PATH.read_bytes() != c["list"]:
            raise SystemExit("the versioned list is not the reproduced selection")
        print("check ok")
        return
    LIST_PATH.write_bytes(c["list"])


if __name__ == "__main__":
    main()
