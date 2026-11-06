"""smaz2 and Unishox2 (upstream C at the commits of vendor.lock, default settings) via ctypes.

Same build as the 5.56 measurement (`-O2 -shared -fPIC`); the compressed bytes do not depend on the
optimization level. Every call checks the roundtrip and raises ValueError on a mismatch or overflow,
so a failing codec can never be skipped silently.
"""
from __future__ import annotations

import ctypes
import subprocess
from functools import lru_cache

from . import vendor

BUILD = vendor.VENDOR / "build"


def _dylib(name: str, source: str) -> ctypes.CDLL:
    vendor.ensure()
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / f"lib{name}.dylib"
    src = vendor.VENDOR / source
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        res = subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(out), str(src)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"building {out.name} failed:\n{res.stderr}")
    return ctypes.CDLL(str(out))


@lru_cache(maxsize=None)
def _usx() -> ctypes.CDLL:
    lib = _dylib("unishox2", "Unishox2/unishox2.c")
    for fn in (lib.unishox2_compress_simple, lib.unishox2_decompress_simple):
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
        fn.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=None)
def _smz() -> ctypes.CDLL:
    lib = _dylib("smaz2", "smaz2/smaz2.c")
    for fn in (lib.smaz2_compress, lib.smaz2_decompress):
        fn.argtypes = [ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p, ctypes.c_ulong]
        fn.restype = ctypes.c_ulong
    return lib


def _unishox2_decompress(c: bytes, n: int) -> bytes:
    back = ctypes.create_string_buffer(4 * n + 64)
    k = _usx().unishox2_decompress_simple(c, len(c), back)
    return back.raw[:k]


def _smaz2_decompress(c: bytes, n: int) -> bytes:
    back = ctypes.create_string_buffer(4 * n + 64)
    k = _smz().smaz2_decompress(back, len(back), c, len(c))
    return back.raw[:k]


def unishox2(m: bytes) -> bytes:
    out = ctypes.create_string_buffer(4 * len(m) + 64)
    n = _usx().unishox2_compress_simple(m, len(m), out)
    if not 0 < n < len(out):
        raise ValueError(f"unishox2 output size {n} out of range for {m!r}")
    c = out.raw[:n]
    if _unishox2_decompress(c, len(m)) != m:
        raise ValueError(f"unishox2 roundtrip failed for {m!r}")
    return c


def smaz2(m: bytes) -> bytes:
    out = ctypes.create_string_buffer(4 * len(m) + 64)
    n = _smz().smaz2_compress(out, len(out), m, len(m))
    if not 0 < n < len(out):
        raise ValueError(f"smaz2 output size {n} out of range for {m!r}")
    c = out.raw[:n]
    if _smaz2_decompress(c, len(m)) != m:
        raise ValueError(f"smaz2 roundtrip failed for {m!r}")
    return c
