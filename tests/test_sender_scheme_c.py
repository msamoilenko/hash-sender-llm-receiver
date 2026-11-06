import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assim.emitter import encode  # noqa: E402
from assim.hashing import HashScheme  # noqa: E402
from sender_memory import build, vendor  # noqa: E402

TAG_KEY = b"chave-de-teste-nao-usar-em-producao"
LENGTHS = [1, 2, 7, 8, 9, 49, 255]


@pytest.fixture(scope="module")
def vendored():
    try:
        return vendor.ensure()
    except RuntimeError as exc:  # no network and no vendor/ checkout
        pytest.skip(f"vendor sources unavailable: {exc}")


def _dev_messages() -> list[bytes]:
    out = []
    for name in ("taskmaster_en_30.txt", "taskmaster_hybrid_30.txt"):
        out += [m for m in (ROOT / "data" / name).read_bytes().splitlines() if m]
    return out


def test_lock_pins_full_commits_and_licenses():
    entries = vendor.entries()
    assert {e.name for e in entries} == {"BLAKE2", "Unishox2", "smaz2"}
    assert all(len(e.commit) == 40 and e.license for e in entries)


def test_c_scheme_and_tag_match_python(vendored, tmp_path):
    exe = build.build_scheme_cli(tmp_path)
    msgs = _dev_messages() + [bytes((i % 251) + 1 for i in range(n)) for n in LENGTHS]
    res = build.run_scheme_cli(exe, msgs)
    assert len(res) == len(msgs)
    for m, (hashes, tag) in zip(msgs, res):
        enc = encode(m, 2, HashScheme(10, 8, 4), TAG_KEY, 32)
        assert hashes == enc.hashes, m
        assert tag == enc.tag, m
