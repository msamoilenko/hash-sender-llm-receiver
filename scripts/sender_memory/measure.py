#!/usr/bin/env python3
"""Sender memory, one definition, one platform (design, stage 1).

Flash = code + read-only data of the complete sender, tag included: sections of a dead-stripped
executable (driver.c + sender objects) minus those of the empty driver. Static RAM = __data,
__bss, __common. Stack bound = sum of every frame of the sender objects (-fstack-usage), an
upper bound when there is no recursion (not verified for third-party code).

Platform: host arm64 (Mach-O), Apple clang, flags from build.cc_flags(). Excluded on purpose:
__stubs, __got, __unwind_info, __eh_frame, __compact_unwind and the dynamic libc (memcpy, memset).

  python -I scripts/sender_memory/measure.py docs/medidas/sender_memory.csv
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    sys.path.insert(0, str(ROOT / "src"))
    from sender_memory import build, ppm_tables, vendor  # noqa: E402
else:
    from . import build, ppm_tables, vendor

PLATFORM = "host arm64 (Mach-O)"
LINK = ["-Wl,-dead_strip"]
FLASH = {("__TEXT", "__text"), ("__TEXT", "__const"), ("__TEXT", "__cstring"), ("__TEXT", "__literal4"),
         ("__TEXT", "__literal8"), ("__TEXT", "__literal16"), ("__DATA_CONST", "__const"), ("__DATA", "__const")}
RAM = {("__DATA", "__data"), ("__DATA", "__bss"), ("__DATA", "__common")}
COLUMNS = ["emitter", "platform", "compiler", "flags", "flash_bytes", "text_bytes", "const_bytes",
           "static_ram_bytes", "stack_bound_bytes", "table_bytes", "counts_sha256", "reached"]
O2_TARGETS = {"ppm_o2_16k": 16384, "ppm_o2_32k": 32768, "ppm_o2_64k": 65536}

Source = tuple[Path, list[str]]
_SEGMENT = re.compile(r"^Segment\s*(\w*)\s*:")
_SECTION_OBJ = re.compile(r"Section \((\w+), (\w+)\): (\d+)")  # object file: "Section (__TEXT, __text): 32"
_SECTION_EXE = re.compile(r"Section (\w+): (\d+)")  # executable: under "Segment __TEXT:", "Section __text: 9108"


def parse_size_m(text: str) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    seg = ""
    for ln in text.splitlines():
        s = ln.strip()
        if m := _SEGMENT.match(s):
            seg = m.group(1)
        elif m := _SECTION_OBJ.match(s):
            key, n = (m.group(1), m.group(2)), int(m.group(3))
            out[key] = out.get(key, 0) + n
        elif m := _SECTION_EXE.match(s):
            key, n = (seg, m.group(1)), int(m.group(2))
            out[key] = out.get(key, 0) + n
    return out


def flash_bytes(s: dict) -> int:
    return sum(n for k, n in s.items() if k in FLASH)


def text_bytes(s: dict) -> int:
    return s.get(("__TEXT", "__text"), 0)


def static_ram_bytes(s: dict) -> int:
    return sum(n for k, n in s.items() if k in RAM)


def compiler() -> str:
    return subprocess.run(["cc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]


def tag_only_sources() -> list[Source]:
    inc, src = build.blake2_sources()
    return [(Path(src[0]), inc), (HERE / "tag.c", inc)]


def scheme_sources() -> list[Source]:
    inc, _ = build.blake2_sources()
    return tag_only_sources() + [(HERE / "scheme_enc.c", inc)]


def ppm_sources(ctx_counts, order: int, workdir: Path) -> list[Source]:
    header = workdir / f"ppm_tables_{ppm_tables.counts_sha256(ctx_counts)[:12]}_o{order}.h"
    ppm_tables.emit_header(ctx_counts, order, header)
    return tag_only_sources() + [(HERE / "ppm_enc.c", ["-include", str(header), "-I", str(HERE)])]


def smaz2_sources() -> list[Source]:
    return tag_only_sources() + [(vendor.VENDOR / "smaz2" / "smaz2.c", [])]


def unishox2_sources() -> list[Source]:
    return tag_only_sources() + [(vendor.VENDOR / "Unishox2" / "unishox2.c", [])]


def _sections(exe: Path) -> dict:
    res = subprocess.run(["size", "-m", str(exe)], capture_output=True, text=True, check=True)
    return parse_size_m(res.stdout)


def _link(name: str, objs: list[Path], method: str, workdir: Path) -> Path:
    drv = workdir / f"driver_{name}.o"
    build.cc(["-c", f"-DMETHOD_{method.upper()}", str(HERE / "driver.c"), "-o", str(drv)])
    exe = workdir / f"exe_{name}"
    build.cc([*LINK, str(drv), *map(str, objs), "-o", str(exe)])
    return exe


def _stack_bound(objs: list[Path]) -> int:
    total = 0
    for o in objs:
        for ln in o.with_suffix(".su").read_text().splitlines():
            _, n, qual = ln.split("\t")
            if qual not in ("static", "dynamic,bounded"):
                raise ValueError(f"unbounded stack in {o.name}: {ln}")
            total += int(n)
    return total


def measure(name: str, sources: list[Source], method: str, workdir: Path, extra: dict | None = None) -> dict:
    objdir = workdir / name
    objdir.mkdir(parents=True, exist_ok=True)
    objs = []
    for i, (src, flags) in enumerate(sources):
        obj = objdir / f"{i}_{src.stem}.o"
        build.cc(["-c", "-fstack-usage", *flags, str(src), "-o", str(obj)])
        objs.append(obj)
    base = _sections(_link(f"{name}_empty", [], "empty", objdir))
    s = _sections(_link(name, objs, method, objdir))
    delta = {k: s.get(k, 0) - base.get(k, 0) for k in set(s) | set(base)}
    flash, text = flash_bytes(delta), text_bytes(delta)
    row = {"emitter": name, "platform": PLATFORM, "compiler": compiler(),
           "flags": " ".join(build.cc_flags() + LINK), "flash_bytes": flash, "text_bytes": text,
           "const_bytes": flash - text, "static_ram_bytes": static_ram_bytes(delta),
           "stack_bound_bytes": _stack_bound(objs), "table_bytes": "", "counts_sha256": "", "reached": ""}
    row.update(extra or {})
    return row


def _train_counts() -> dict:
    import baseline_classical as bc
    import taskmaster_hybrid_prereg as hyb
    from assim.ngram import ByteNgramLM

    raw = hyb.load_raw()
    lines = hyb.train_lines(raw, hyb.load_ids(hyb.TRAIN_IDS), hyb.dev_conversations(raw))
    lm = ByteNgramLM(max_context=3)
    lm.add(b"\n".join(lines) + b"\n")
    return bc.nested_counts(lm)


def ppm_models(counts) -> dict[str, tuple[dict, int, bool]]:
    """name -> (ctx_counts, order, reached) for every PPM point of the frontier."""
    o2 = ppm_tables.restrict(counts, 2)
    out = {"ppm_o1": (ppm_tables.restrict(counts, 1), 1, True)}
    for name, target in O2_TARGETS.items():
        pruned, reached, _ = ppm_tables.prune_o2(o2, target)
        out[name] = (pruned, 2, reached)
    out["ppm_o2"] = (o2, 2, True)
    out["ppm_o3"] = (ppm_tables.restrict(counts, 3), 3, True)
    return out


def measure_all(out: Path | None, workdir: Path) -> list[dict]:
    vendor.ensure()
    rows = [measure("tag_only", tag_only_sources(), "tag", workdir),
            measure("scheme", scheme_sources(), "scheme", workdir)]
    for name, (c, order, reached) in ppm_models(_train_counts()).items():
        rows.append(measure(name, ppm_sources(c, order, workdir), "ppm", workdir,
                            {"table_bytes": ppm_tables.table_bytes(c), "counts_sha256": ppm_tables.counts_sha256(c),
                             "reached": reached}))
    rows.append(measure("smaz2", smaz2_sources(), "smaz2", workdir))
    rows.append(measure("unishox2", unishox2_sources(), "unishox2", workdir))
    rows = [{k: str(r[k]) for k in COLUMNS} for r in rows]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "medidas" / "sender_memory.csv"
    with tempfile.TemporaryDirectory() as tmp:
        for r in measure_all(target, Path(tmp)):
            print(f"{r['emitter']:12s} flash {int(r['flash_bytes']):8d} text {int(r['text_bytes']):6d} "
                  f"const {int(r['const_bytes']):8d} ram {r['static_ram_bytes']:>5s} stack<= {r['stack_bound_bytes']:>6s} "
                  f"table {r['table_bytes']:>7s} reached {r['reached']}")
