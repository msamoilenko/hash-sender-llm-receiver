import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("airtime", Path(__file__).resolve().parents[1] / "scripts" / "airtime.py")
airtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(airtime)


def test_lora_formula_matches_worked_example():
    # LongFast (SF11, 250 kHz, 4/5, preâmbulo 16), 16 bytes: 23 símbolos de payload e ~354,3 ms
    # (exemplo de uma página que reproduz a fórmula da Semtech; re-derivado à mão)
    t = airtime.lora_airtime_s(16)
    assert math.isclose(t, 0.354304, rel_tol=1e-9)
    assert math.isclose((16 + 4.25) * 2**11 / 250_000, 0.165888)  # o firmware anota ~165 ms de preâmbulo


def test_lora_airtime_is_monotonic_and_steps_by_symbols():
    times = [airtime.lora_airtime_s(n) for n in range(1, 120)]
    assert all(b >= a for a, b in zip(times, times[1:]))


def test_bit_stuffing_inserts_after_five_ones():
    assert airtime.stuffed_bits(bytes([0xFF])) == 9  # 8 uns seguidos: um zero depois dos cinco primeiros
    assert airtime.stuffed_bits(bytes([0x00])) == 8
    assert airtime.stuffed_bits(bytes([0xFF, 0xFF])) == 16 + 3  # 16 uns: zeros após o 5º, 10º e 15º


def test_rockblock_credits():
    assert [airtime.rockblock_credits(n) for n in (1, 50, 51, 100, 101)] == [1, 1, 2, 2, 3]
