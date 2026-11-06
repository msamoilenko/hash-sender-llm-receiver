import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sender_memory import tiny_codecs, vendor  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def vendored():
    try:
        vendor.ensure()
    except RuntimeError as exc:
        pytest.skip(f"vendor sources unavailable: {exc}")


def test_failed_roundtrip_raises(monkeypatch):
    monkeypatch.setattr(tiny_codecs, "_unishox2_decompress", lambda c, n: b"garbage")
    with pytest.raises(ValueError):
        tiny_codecs.unishox2(b"book a table for two at seven")
    monkeypatch.setattr(tiny_codecs, "_smaz2_decompress", lambda c, n: b"garbage")
    with pytest.raises(ValueError):
        tiny_codecs.smaz2(b"book a table for two at seven")
