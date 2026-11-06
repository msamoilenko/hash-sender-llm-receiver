"""Baseline simétrico: n-grama de bytes com interpolação de Witten-Bell.

Emissor e receptor usam o mesmo modelo estático. O custo é o ideal (-log2 p), sem um codificador
aritmético real (que somaria ~2 bits de terminação). Não depende de idioma: opera em bytes.
"""
from __future__ import annotations

import math

_NEWLINE = b"\n"  # separa mensagens e dá o contexto de início


class ByteNgramLM:
    def __init__(self, max_context: int = 5):
        self.max_context = max_context
        self._n: dict[bytes, int] = {}  # contexto + byte -> contagem
        self._total: dict[bytes, int] = {}  # contexto -> soma das contagens
        self._types: dict[bytes, int] = {}  # contexto -> bytes distintos vistos depois dele

    def add(self, data: bytes, weight: int = 1) -> None:
        """Soma (ou, com peso negativo, remove) as contagens de cada linha de `data`."""
        n, total, types = self._n, self._total, self._types
        for line in data.split(_NEWLINE):
            if not line:
                continue
            seq = _NEWLINE + line + _NEWLINE
            for i in range(1, len(seq)):
                b = seq[i : i + 1]
                for k in range(min(self.max_context, i) + 1):
                    ctx = seq[i - k : i]
                    key = ctx + b
                    old = n.get(key, 0)
                    new = old + weight
                    if new > 0:
                        n[key] = new
                    else:
                        n.pop(key, None)
                    total[ctx] = total.get(ctx, 0) + weight
                    if old == 0 and new > 0:
                        types[ctx] = types.get(ctx, 0) + 1
                    elif old > 0 and new <= 0:
                        types[ctx] -= 1

    def prob(self, context: bytes, b: bytes, max_context: int | None = None) -> float:
        """P(b | context) por Witten-Bell; `max_context` usa uma ordem menor que a treinada."""
        limit = self.max_context if max_context is None else min(max_context, self.max_context)
        p = 1.0 / 256
        for k in range(min(limit, len(context)) + 1):
            ctx = context[len(context) - k :] if k else b""
            tot = self._total.get(ctx, 0)
            if tot <= 0:
                continue
            t = self._types[ctx]
            p = (self._n.get(ctx + b, 0) + t * p) / (tot + t)
        return p

    def bits(self, message: bytes, max_context: int | None = None) -> float:
        """Custo ideal da mensagem em bits, começando após um separador de linha."""
        limit = self.max_context if max_context is None else min(max_context, self.max_context)
        seq = _NEWLINE + message
        return -sum(
            math.log2(self.prob(seq[max(0, i - limit) : i], seq[i : i + 1], limit)) for i in range(1, len(seq))
        )
