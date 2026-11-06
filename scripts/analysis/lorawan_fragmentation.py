#!/usr/bin/env python3
"""Fragmentação no LoRaWAN DR0: texto cru, n-grama e esquema `10:4` (DECISOES 5.41).

Cada pacote paga 13 bytes de cabeçalho e rodapé do MAC (MHDR 1 + DevAddr 4 + FCtrl 1 + FCnt 2 +
FPort 1 + MIC 4; conferido contra a tabela da TTN: 59 de MAC payload menos 51 de aplicação = 8, mais
MHDR e MIC). EU868 DR0 = SF12/125 kHz com 51 B de aplicação; US915 DR0 = SF10/125 kHz com 11 B
(documentação da TTN). CRC ligado e cabeçalho explícito são suposição. Tamanhos pelas fórmulas
lineares de 5.13 (esquema 2 L + 64 bits; n-grama 3,07 L + 40 bits), extrapoladas acima de 53 bytes.

  python -I scripts/analysis/lorawan_fragmentation.py
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("airtime", ROOT / "scripts" / "airtime.py")
at = importlib.util.module_from_spec(spec)
spec.loader.exec_module(at)
MAC = 13


def frame_s(app: int, sf: int) -> float:
    return at.lora_airtime_s(MAC + app, sf=sf, bw_hz=125_000, cr_code=1, preamble=8, crc=True, explicit_header=True)


def send(total: int, cap: int, sf: int) -> tuple[int, float]:
    full, rest = divmod(total, cap)
    return math.ceil(total / cap), full * frame_s(cap, sf) + (frame_s(rest, sf) if rest else 0.0)


def main() -> None:
    for length in (60, 120, 150, 170):
        # CLAUDE.md: a etiqueta de 32 bits entra em todos os métodos; "texto cru" é o texto + 4 bytes de etiqueta
        sizes = {"texto cru + etiqueta": length + 4, "texto cru (hoje)": length,
                 "n-grama": math.ceil((3.07 * length + 40) / 8), "esquema": math.ceil((2 * length + 64) / 8)}
        for name, cap, sf in (("EU868 DR0", 51, 12), ("US915 DR0", 11, 10)):
            cells = []
            for label, nbytes in sizes.items():
                n, t = send(nbytes, cap, sf)
                cells.append(f"{label} {nbytes:3d} B = {n} pacote(s), {t:5.2f} s")
            print(f"L={length:3d} {name}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
