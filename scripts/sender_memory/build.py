"""Build helpers for the C sender instrument. `cc_flags()` is the single source of the flags."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from . import ppm_tables, vendor

HERE = Path(__file__).resolve().parent


def cc_flags() -> list[str]:
    return ["-Os", "-std=c11", "-Wall", "-Werror"]


def cc(args: list[str]) -> None:
    res = subprocess.run(["cc", *cc_flags(), *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cc failed ({res.returncode}): {' '.join(args)}\n{res.stderr}")


def build_ppm_cli(ctx_counts, order: int, workdir: Path, flip_first_bit: bool = False) -> Path:
    tag = hashlib.sha256(f"{ppm_tables.counts_sha256(ctx_counts)}|{order}|{flip_first_bit}".encode()).hexdigest()[:12]
    header = workdir / f"ppm_tables_{tag}.h"
    ppm_tables.emit_header(ctx_counts, order, header)
    exe = workdir / f"ppm_cli_{tag}"
    extra = ["-DFLIP_FIRST_BIT"] if flip_first_bit else []
    cc([*extra, "-include", str(header), "-I", str(HERE), str(HERE / "ppm_enc.c"), str(HERE / "ppm_dec.c"),
        str(HERE / "ppm_cli.c"), "-o", str(exe)])
    return exe


def blake2_sources() -> tuple[list[str], list[str]]:
    """(include flags, sources) of the BLAKE2b reference implementation."""
    ref = vendor.VENDOR / "BLAKE2" / "ref"
    return ["-I", str(ref)], [str(ref / "blake2b-ref.c")]


def build_scheme_cli(workdir: Path) -> Path:
    vendor.ensure()
    inc, src = blake2_sources()
    exe = workdir / "scheme_cli"
    cc([*inc, *src, str(HERE / "tag.c"), str(HERE / "scheme_enc.c"), str(HERE / "scheme_cli.c"), "-o", str(exe)])
    return exe


def run_scheme_cli(exe: Path, msgs: list[bytes]) -> list[tuple[list[int], int]]:
    res = subprocess.run([str(exe)], input="".join(m.hex() + "\n" for m in msgs), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{exe.name} exited {res.returncode}: {res.stderr}")
    out = []
    for ln in res.stdout.splitlines():
        hashes, tag = ln.split("|")
        out.append(([int(h) for h in hashes.split()], int(tag)))
    return out


def run_ppm_cli(exe: Path, msgs: list[bytes]) -> list[tuple[int, bool]]:
    res = subprocess.run([str(exe)], input="".join(m.hex() + "\n" for m in msgs), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{exe.name} exited {res.returncode}: {res.stderr}")
    out = []
    for ln in res.stdout.splitlines():
        nbits, ok = ln.split()
        out.append((int(nbits), ok == "1"))
    return out
