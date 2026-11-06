#!/usr/bin/env python3
"""Quanto 18 bits a menos valem em tempo no ar e em custo, por protocolo (DECISOES 5.28).

Só aritmética de overhead; não usa o modelo. Cada constante cita a fonte e marca o que é suposição.

  python -I scripts/airtime.py
"""
from __future__ import annotations

import math
import random

# --- LoRa (fórmula de tempo no ar da Semtech: AN1200.13 / datasheets SX127x) ---
# A fórmula foi conferida contra um exemplo trabalhado (LongFast, 16 bytes, 23 símbolos, 354,3 ms)
# e re-derivada à mão; o PDF original da Semtech não foi lido, só uma página que o reproduz.
# Meshtastic LongFast (firmware, MeshRadio.h): BW 250 kHz, SF 11, CR 4/5; preâmbulo de 16 símbolos
# (RadioInterface.h); cabeçalho de 16 bytes (MESHTASTIC_HEADER_LENGTH). CRC ligado e cabeçalho
# explícito são SUPOSIÇÃO (padrão do chip; não verificado no firmware).
MESH_HEADER_BYTES = 16
# Envelope protobuf do texto (portnum + payload, comprimento < 128): 2 + 2 bytes, derivado das regras
# de codificação protobuf, NÃO verificado no código do Meshtastic; versões novas podem somar um campo.
MESH_PROTOBUF_BYTES = 4


def lora_airtime_s(payload_bytes: int, sf: int = 11, bw_hz: int = 250_000, cr_code: int = 1,
                   preamble: int = 16, crc: bool = True, explicit_header: bool = True) -> float:
    """`cr_code` 1..4 é 4/5..4/8. Otimização de baixa taxa ligada se o símbolo passa de 16 ms."""
    tsym = 2**sf / bw_hz
    de = 1 if tsym > 0.016 else 0
    ih = 0 if explicit_header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * int(crc) - 20 * ih
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr_code + 4), 0)
    return (preamble + 4.25) * tsym + n_payload * tsym


def meshtastic_airtime_s(text_bytes: int) -> float:
    return lora_airtime_s(MESH_HEADER_BYTES + MESH_PROTOBUF_BYTES + text_bytes)


# --- AX.25 / APRS a 1200 bit/s (AFSK 1200/2200 Hz) ---
# AX.25 v2.2: bandeira 1 octeto; endereços de 7 octetos por subcampo (destino, origem, até 2
# repetidores); controle 1; PID 1; FCS 2; inserção de um 0 após cinco 1 seguidos (3.6).
# APRS101: mensagem = ':' + 9 caracteres do destinatário + ':' + texto (≤ 67) + '{' + número (1 a 5).
# TXDELAY é configuração do TNC; 300 ms é o padrão citado por um guia do XRouter (não do Dire Wolf).
AFSK_BPS = 1200
APRS_MSG_WRAPPER_CHARS = 11 + 3  # ':' + 9 + ':' mais '{' e número de 2 dígitos (SUPOSIÇÃO)
BASE91_BITS_PER_CHAR = math.log2(91)


def stuffed_bits(frame: bytes) -> int:
    """Bits na linha com inserção de zero (bits enviados do menos para o mais significativo)."""
    ones, out = 0, 0
    for byte in frame:
        for i in range(8):
            out += 1
            if (byte >> i) & 1:
                ones += 1
                if ones == 5:
                    out += 1
                    ones = 0
            else:
                ones = 0
    return out


def aprs_airtime_s(info: bytes, digipeaters: int = 2, txdelay_s: float = 0.3) -> float:
    addr = 7 * (2 + digipeaters)
    header = bytes([0x82, 0xA0, 0xB4, 0x60, 0x60, 0x60, 0x60]) + bytes([0xA0] * 7) + bytes([0xAE] * 7 * digipeaters)
    assert len(header) == addr
    frame = header + bytes([0x03, 0xF0]) + info + bytes([0x12, 0x34])  # FCS fictício
    bits = 8 + stuffed_bits(frame) + 8  # bandeiras de abertura e fechamento
    return txdelay_s + bits / AFSK_BPS


def base91_chars(bits: float) -> int:
    return math.ceil(bits / BASE91_BITS_PER_CHAR)


def random_printable(n: int, rng: random.Random) -> bytes:
    # caracteres imprimíveis exceto '|' e '~' (reservados), como em APRS101
    pool = [c for c in range(33, 127) if chr(c) not in "|~"]
    return bytes(rng.choice(pool) for _ in range(n))


# --- RockBLOCK (Iridium SBD): 1 crédito por 50 bytes ou fração (docs.groundcontrol.com) ---
def rockblock_credits(nbytes: int) -> int:
    return max(1, math.ceil(nbytes / 50))


# Cenários por mensagem média (DECISOES 5.13, 5.15, 5.23): bytes de texto, bits do esquema `10:4`
# e do n-grama de contexto 3, ambos com 8 (tamanho) + 32 (etiqueta).
SCENARIOS = {
    "PT": {"text_bytes": 40.3, "asym_bits": 145.9, "ngram_bits": 163.6},
    "EN": {"text_bytes": 36.5, "asym_bits": 138.1, "ngram_bits": 153.7},
}


def main() -> None:
    rng = random.Random(0)
    for lang, s in SCENARIOS.items():
        text_b = round(s["text_bytes"])
        sizes = {
            "texto cru (sem etiqueta)": text_b,
            "texto cru + etiqueta 32": math.ceil(s["text_bytes"] + 4),
            "n-grama no emissor": math.ceil(s["ngram_bits"] / 8),
            "esquema assimétrico": math.ceil(s["asym_bits"] / 8),
        }
        print(f"\n=== {lang}: mensagem média de {s['text_bytes']} bytes ===")
        print(f"{'carga':28s} {'bytes':>5s} | {'Meshtastic LongFast':>20s} | {'APRS 2 digis, TXD 0.3 s':>24s} | {'APRS sem TXDELAY':>17s} | créditos RockBLOCK")
        base = {}
        for name, nb in sizes.items():
            mesh = meshtastic_airtime_s(nb)
            if name.startswith("texto"):
                # texto plano; a etiqueta de 32 bits, em campo imprimível, ocupa 5 caracteres Base91 (não 4 bytes)
                chars = nb + (1 if name == "texto cru + etiqueta 32" else 0)
                info_len = chars + APRS_MSG_WRAPPER_CHARS
            else:
                chars = base91_chars(8 * nb)
                info_len = chars + APRS_MSG_WRAPPER_CHARS
            info = random_printable(info_len, rng)
            aprs = aprs_airtime_s(info)
            aprs0 = aprs_airtime_s(info, txdelay_s=0.0)
            base[name] = (mesh, aprs, aprs0)
            print(f"{name:28s} {nb:5d} | {mesh * 1000:17.1f} ms | {aprs:21.3f} s | {aprs0:14.3f} s | {rockblock_credits(nb)}  (APRS: {chars} chars)")
        a, n = base["esquema assimétrico"], base["n-grama no emissor"]
        t = base["texto cru (sem etiqueta)"]
        tt = base["texto cru + etiqueta 32"]
        # CLAUDE.md: toda comparação inclui a etiqueta de 32 bits em todos os métodos; a linha sem etiqueta é "o que se envia hoje"
        print("  redução do esquema sobre o texto cru + etiqueta (convenção): "
              f"Meshtastic {100 * (1 - a[0] / tt[0]):.0f}%, APRS (TXD 0.3 s) {100 * (1 - a[1] / tt[1]):.0f}%, APRS sem TXDELAY {100 * (1 - a[2] / tt[2]):.0f}%")
        print("  redução do esquema sobre o texto cru sem etiqueta (o que se envia hoje): "
              f"Meshtastic {100 * (1 - a[0] / t[0]):.0f}%, APRS (TXD 0.3 s) {100 * (1 - a[1] / t[1]):.0f}%, APRS sem TXDELAY {100 * (1 - a[2] / t[2]):.0f}%")
        print("  redução do esquema sobre o n-grama:      "
              f"Meshtastic {100 * (1 - a[0] / n[0]):.1f}%, APRS (TXD 0.3 s) {100 * (1 - a[1] / n[1]):.1f}%, APRS sem TXDELAY {100 * (1 - a[2] / n[2]):.1f}%")

    print("\n=== RockBLOCK: onde o esquema muda a conta (aproximação linear, só PT; não medido) ===")
    # bits do esquema 10:4 com chunk 2 e head 8: 40 + 2*(L-8) de hash; n-grama ~3,07 bits/byte + 40
    for length in (50, 100, 118, 150, 168, 169, 200, 255):
        asym_bits = 40 + 40 + 2 * (length - 8)
        ngram_bits = 3.07 * length + 40
        print(f"  L={length:3d} B texto: cru {rockblock_credits(length)} crédito(s), n-grama {rockblock_credits(math.ceil(ngram_bits / 8))}, esquema {rockblock_credits(math.ceil(asym_bits / 8))}")


if __name__ == "__main__":
    main()
