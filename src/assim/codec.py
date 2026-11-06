"""Receptor: busca em feixe guiada por modelo. O emissor está em `emitter.py`."""
from __future__ import annotations

import bisect
import hmac
import time
from dataclasses import dataclass

import numpy as np

from .emitter import LENGTH_FIELD_BITS, Encoded, encode  # noqa: F401  (reexportados)
from .hashing import PrefixHasher, chunk_ends, chunk_hash
from .hashing import tag as make_tag

# Teto de candidatos completos testados contra a etiqueta. A chance de aceitar uma versão
# errada é no máximo N * 2**-bits; com 1.024 e 32 bits fica em 2,4e-7, abaixo de 1e-6.
MAX_TAG_CHECKS = 1024


@dataclass
class DecodeStats:
    steps: int = 0
    candidates: int = 0
    tag_checks: int = 0
    tag_cap_hit: bool = False  # havia mais candidatos completos do que o teto; o excesso não foi testado
    seconds: float = 0.0
    # Diagnóstico opcional (decode(..., trace_prefix=...)): posição do prefixo verdadeiro no
    # feixe a cada passo. Não é usado na decodificação normal.
    trace: list[dict] | None = None


@dataclass
class _Beam:
    data: bytes
    ids: list[int]
    score: float


def _tag_matches(data: bytes, key: bytes, enc: Encoded) -> bool:
    """Compara a etiqueta em tempo constante."""
    if enc.tag >> enc.tag_bits:
        return False
    width = (enc.tag_bits + 7) // 8
    return hmac.compare_digest(make_tag(data, key, enc.tag_bits).to_bytes(width, "big"), enc.tag.to_bytes(width, "big"))


def _hashes_ok(enc: Encoded, ends: list[int], old_len: int, data: bytes) -> bool:
    """Verifica os hashes de todos os trechos cuja fronteira foi cruzada agora."""
    for i, end in enumerate(ends):
        if end <= old_len:
            continue
        if end > len(data):
            break
        if chunk_hash(i, data[:end], enc.chunk_bits[i]) != enc.hashes[i]:
            return False
    return True


def _later_hashes_ok(enc: Encoded, ends: list[int], first: int, data: bytes) -> bool:
    """Como `_hashes_ok`, mas começa no trecho `first` (os anteriores já foram verificados)."""
    for i in range(first, len(ends)):
        end = ends[i]
        if end > len(data):
            break
        if chunk_hash(i, data[:end], enc.chunk_bits[i]) != enc.hashes[i]:
            return False
    return True


_HEAD_BYTES = 8  # o pré-filtro empacota até 8 bytes iniciais de cada token em um uint64


class _VocabIndex:
    """Comprimento e primeiros bytes de cada token, para filtrar candidatos com numpy."""

    def __init__(self, token_bytes: list[bytes]):
        n = len(token_bytes)
        self.lens = np.fromiter((len(b) for b in token_bytes), dtype=np.int64, count=n)
        self.head = np.zeros((n, _HEAD_BYTES), dtype=np.uint8)
        for i, b in enumerate(token_bytes):
            if b:
                head = b[:_HEAD_BYTES]
                self.head[i, : len(head)] = np.frombuffer(head, dtype=np.uint8)


def _vocab_index(lm) -> _VocabIndex:
    index = getattr(lm, "_assim_vocab_index", None)
    if index is None:
        index = lm._assim_vocab_index = _VocabIndex(lm.token_bytes)
    return index


def _candidate_mask(enc: Encoded, ends: list[int], vocab: _VocabIndex, data: bytes, ids) -> np.ndarray:
    """Máscara dos tokens que podem estender `data`; condição necessária, não suficiente.

    Descarta peça vazia e excesso de tamanho e, para quem cruza a próxima fronteira,
    confere o hash desse trecho uma vez por prefixo distinto de k bytes (k = bytes até
    a fronteira). Quem cruza mais fronteiras ainda passa por `_hashes_ok`, que decide.
    """
    old = len(data)
    lens = vocab.lens[ids]
    keep = (lens > 0) & (old + lens <= enc.length)
    j = bisect.bisect_right(ends, old)
    k = ends[j] - old
    if k > _HEAD_BYTES:
        return keep
    cross = np.nonzero(keep & (lens >= k))[0]
    if cross.size:
        shifts = np.arange(k - 1, -1, -1, dtype=np.uint64) * np.uint64(8)
        keys = (vocab.head[ids[cross], :k].astype(np.uint64) << shifts).sum(axis=1)
        uniq, inverse = np.unique(keys, return_inverse=True)
        hasher = PrefixHasher(j, data)
        bits, target = enc.chunk_bits[j], enc.hashes[j]
        ok = np.fromiter(
            (hasher.extended(int(u).to_bytes(k, "big"), bits) == target for u in uniq),
            dtype=bool,
            count=len(uniq),
        )
        keep[cross] = ok[inverse]
    return keep


def decode(
    enc: Encoded,
    lm,
    tag_key: bytes,
    beam_width: int = 32,
    topk: int = 32,
    max_steps: int | None = None,
    max_tag_checks: int = MAX_TAG_CHECKS,
    trace_prefix: bytes | None = None,
) -> tuple[bytes | None, DecodeStats]:
    """Receptor: busca em feixe guiada pelo modelo, podada pelos hashes por trecho.

    Só o receptor usa o modelo; por isso o resultado não depende de determinismo
    numérico entre plataformas (pode-se usar fp16, GPU, qualquer hardware).

    Os candidatos completos são testados contra a etiqueta em ordem de pontuação, até
    `max_tag_checks`; passando disso, o excesso não é testado (`tag_cap_hit`) e a decodificação
    pode falhar em vez de arriscar um falso aceite.

    `trace_prefix`, quando dado, registra em `stats.trace` o posto do prefixo verdadeiro no
    frontier a cada passo, sem alterar a busca. Para o `tested` final ter sentido, passe a
    mensagem completa.
    """
    if beam_width < 1 or topk < 1 or max_tag_checks < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("beam_width, topk, max_tag_checks e max_steps devem ser >= 1")
    # Os candidatos são agrupados pela posição em bytes já decodificada e só competem
    # com outros na mesma posição. Comparar sequências de comprimentos diferentes pela
    # log-probabilidade acumulada favoreceria as mais curtas (viés de comprimento).
    vocab = _vocab_index(lm)  # construído uma vez por modelo, fora da medição de tempo
    t0 = time.perf_counter()
    stats = DecodeStats()
    if trace_prefix is not None:
        stats.trace = []
    ends = chunk_ends(enc.length, enc.chunk_bytes)
    if max_steps is None:
        max_steps = 4 * enc.length
    frontier: dict[int, dict[bytes, _Beam]] = {0: {b"": _Beam(b"", [], 0.0)}}
    finished: dict[bytes, _Beam] = {}

    while frontier and stats.steps < max_steps:
        pos = min(frontier)
        bucket = sorted(frontier.pop(pos).values(), key=lambda c: c.score, reverse=True)
        if stats.trace is not None:
            prefix = trace_prefix[:pos]
            rank = next((i for i, b in enumerate(bucket) if b.data == prefix), None)
            stats.trace.append({"pos": pos, "bucket": len(bucket), "rank": rank,
                                "in_beam": rank is not None and rank < beam_width})
        beams = bucket[:beam_width]
        stats.steps += 1
        values, indices = lm.next_topk([b.ids for b in beams], topk)
        for bi, beam in enumerate(beams):
            mask = _candidate_mask(enc, ends, vocab, beam.data, indices[bi])
            # `first` é a primeira fronteira ainda não cruzada. Quem não a alcança não tem nada a
            # verificar; quem a cruza com k <= _HEAD_BYTES já teve esse trecho conferido pelo pré-filtro.
            first = bisect.bisect_right(ends, len(beam.data))
            boundary = ends[first]
            first += boundary - len(beam.data) <= _HEAD_BYTES
            for p in np.nonzero(mask)[0]:
                logprob, tid = values[bi][p], indices[bi][p]
                data = beam.data + lm.token_bytes[int(tid)]
                if len(data) >= boundary and not _later_hashes_ok(enc, ends, first, data):
                    continue
                stats.candidates += 1
                cand = _Beam(data, beam.ids + [int(tid)], beam.score + float(logprob))
                if len(data) == enc.length:
                    best = finished.get(data)
                    if best is None or cand.score > best.score:
                        finished[data] = cand
                    continue
                slot = frontier.setdefault(len(data), {})
                previous = slot.get(data)
                if previous is None or cand.score > previous.score:
                    slot[data] = cand

    ranked = sorted(finished.values(), key=lambda c: c.score, reverse=True)
    stats.tag_cap_hit = len(ranked) > max_tag_checks
    if stats.trace is not None:
        true_rank = next((i for i, c in enumerate(ranked) if c.data == trace_prefix), None)
        stats.trace.append({"finished": len(ranked), "true_rank": true_rank,
                            "tested": true_rank is not None and true_rank < max_tag_checks})
    for cand in ranked[:max_tag_checks]:
        stats.tag_checks += 1
        if _tag_matches(cand.data, tag_key, enc):
            stats.seconds = time.perf_counter() - t0
            return cand.data, stats

    stats.seconds = time.perf_counter() - t0
    return None, stats
