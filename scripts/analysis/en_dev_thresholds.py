#!/usr/bin/env python3
"""Faixas de razão para EN a partir só dos dados de desenvolvimento (DECISOES 5.53).

Razão = `simetrico_ideal` ÷ `assim_bits` (ideal do Qwen com etiqueta de 32 bits ÷ bits do esquema com tamanho
e etiqueta), como em 5.37 e 5.40. Usa todas as linhas EN com a configuração congelada (`10:4`, feixe 64,
topk 32.768). Regra mecânica, fixada antes de olhar o conjunto de teste:
  entrega  se razão <  menor razão entre as falhas
  falha    se razão >  maior razão entre os sucessos
  incerta  no intervalo entre as duas
Também compara as rodadas repetidas da mesma mensagem (m546 contra g514/c525) para medir o não determinismo.

  python -I scripts/analysis/en_dev_thresholds.py
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = {"esquema": "10:4", "beam": "64", "topk": "32768"}
SETS = {
    "limpo": ["g514_b64_k32768.csv", "m546_en_limpo_b64_k32768.csv"],
    "ruidoso": ["c525_en_b64_k32768.csv", "m546_en_ruido_b64_k32768.csv"],
    "juncao_k2": ["l537_en_k2_b64_k32768.csv"],
    "juncao_k3": ["l537_en_k3_b64_k32768.csv"],
    "juncao_k4": ["l537_en_k4_b64_k32768.csv"],
}


def load(name: str) -> list[dict]:
    with open(ROOT / "results" / name, newline="") as h:
        rows = list(csv.DictReader(h))
    return [r for r in rows if r["idioma"] == "EN" and all(r[k] == v for k, v in FROZEN.items())]


def observations() -> list[tuple]:
    """(conjunto, csv, sha, bytes, razão, sucesso, teto esgotado) de todas as linhas EN congeladas."""
    obs = []
    for label, files in SETS.items():
        for f in files:
            for r in load(f):
                obs.append((label, f, r["msg_sha256"], int(r["bytes"]),
                            float(r["simetrico_ideal"]) / float(r["assim_bits"]),
                            # `teto_etiqueta_atingido` só diz que havia mais candidatos que o teto
                            # (handoff da sessão 2, item 7); esgotar o teto é testar todos os N
                            r["sucesso"] == "True", r["testes_etiqueta"] == r["max_tag_checks"]))
    return obs


def base_envelope(obs: list[tuple]) -> tuple[float, float]:
    """Envelope das mensagens base (limpo + ruidoso), não um limiar calibrado (5.53, adendo da sessão 4).

    Devolve (menor razão entre as falhas, maior razão entre os sucessos). Fora do envelope a razão
    prevê; dentro dele, quem decide é a busca e a previsão é incerta.
    """
    base = [o for o in obs if o[0] in ("limpo", "ruidoso")]
    lo = min(o[4] for o in base if not o[5])
    hi = max(o[4] for o in base if o[5])
    return lo, hi


def main() -> None:
    obs = observations()

    print("conjunto     csv                                linhas  sucessos  bytes      razao sucessos   razao falhas")
    for label, files in SETS.items():
        for f in files:
            sub = [o for o in obs if o[1] == f]
            ok = [o[4] for o in sub if o[5]]
            bad = [o[4] for o in sub if not o[5]]
            rng = lambda v: f"{min(v):.2f}-{max(v):.2f}" if v else "-"
            print(f"{label:12s} {f:34s} {len(sub):6d}  {len(ok):8d}  {min(o[3] for o in sub):3d}-{max(o[3] for o in sub):3d}    "
                  f"{rng(ok):15s}  {rng(bad)}")

    def zones(sub: list, tag: str) -> None:
        ok = [o[4] for o in sub if o[5]]
        bad = [o for o in sub if not o[5]]
        lo = min(o[4] for o in bad)
        hi = max(ok)
        print(f"\n[{tag}] linhas {len(sub)}, mensagens distintas {len({o[2] for o in sub})}, "
              f"bytes {min(o[3] for o in sub)}-{max(o[3] for o in sub)}")
        print(f"  entrega se razao < {lo:.3f} | falha se razao > {hi:.3f} | incerta entre {min(lo, hi):.3f} e {max(lo, hi):.3f}")
        print("  falhas (razao, bytes, teto atingido, conjunto): "
              + "; ".join(f"{o[4]:.3f} {o[3]}B {'teto' if o[6] else '-'} {o[0]}" for o in sorted(bad, key=lambda o: o[4])))
        between = [o for o in sub if min(lo, hi) <= o[4] <= max(lo, hi)]
        print(f"  linhas na faixa incerta: {len(between)} ({sum(o[5] for o in between)} sucessos)")

    zones(obs, "todas as linhas EN de desenvolvimento")
    base = [o for o in obs if o[0] in ("limpo", "ruidoso")]
    zones(base, "so as mensagens base (limpo + ruidoso), sem juncoes")

    by_msg = defaultdict(list)
    for o in base:
        by_msg[o[2]].append(o)
    pairs = [v for v in by_msg.values() if len(v) == 2]
    flips = [v for v in pairs if v[0][5] != v[1][5]]
    dr = [abs(v[0][4] - v[1][4]) for v in pairs]
    print(f"\n[repeticao m546 x g514/c525] pares {len(pairs)} | sucesso diferente em {len(flips)} | "
          f"|delta razao| max {max(dr):.4f}, media {st.mean(dr):.4f}")
    for v in flips:
        print("  muda: " + " | ".join(f"{o[1]} razao {o[4]:.3f} sucesso {int(o[5])}" for o in v))


if __name__ == "__main__":
    main()
