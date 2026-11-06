import random

import numpy as np
import pytest

from assim import codec
from assim.codec import decode, encode
from assim.hashing import HashScheme, PrefixHasher, chunk_ends, chunk_hash
from assim.hashing import tag as make_tag
from assim.lm import ByteBigramLM

CORPUS = [
    b"oi, tudo bem?",
    b"tudo bem por aqui",
    b"chego em casa as oito",
    b"bateria em 40%, sinal fraco",
    b"estou bem, sem novidades",
]
KEY = b"chave-de-teste"
MSG = b"oi, tudo bem?"


@pytest.fixture(scope="module")
def lm():
    return ByteBigramLM(CORPUS)


def test_chunk_ends():
    assert chunk_ends(10, 4) == [4, 8, 10]
    assert chunk_ends(8, 4) == [4, 8]
    assert chunk_ends(3, 8) == [3]


def test_roundtrip(lm):
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=16, tag_key=KEY)
    out, stats = decode(enc, lm, KEY, beam_width=256, topk=256)
    assert out == MSG
    assert stats.tag_checks >= 1


def test_trace_prefix_does_not_change_decode(lm):
    # O trace é diagnóstico: não pode mudar a busca nem os contadores.
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=16, tag_key=KEY)
    out1, s1 = decode(enc, lm, KEY, beam_width=256, topk=256)
    out2, s2 = decode(enc, lm, KEY, beam_width=256, topk=256, trace_prefix=MSG)
    assert out1 == out2 == MSG
    assert (s1.steps, s1.candidates, s1.tag_checks) == (s2.steps, s2.candidates, s2.tag_checks)
    assert s2.trace and s2.trace[0]["pos"] == 0 and s2.trace[0]["rank"] == 0
    assert s2.trace[-1]["tested"] is True


def test_total_bits_accounting():
    enc = encode(MSG, chunk_bytes=4, bits_per_chunk=12, tag_key=KEY)
    assert len(enc.hashes) == 4  # 13 bytes -> trechos de 4, 4, 4, 1
    assert enc.total_bits() == 8 + 4 * 12 + 32


def test_tampered_hash_is_rejected(lm):
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=16, tag_key=KEY)
    enc.hashes[1] ^= 1
    out, _ = decode(enc, lm, KEY, beam_width=64, topk=64)
    assert out is None


def test_wrong_key_is_rejected(lm):
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=16, tag_key=KEY)
    out, _ = decode(enc, lm, b"outra-chave", beam_width=64, topk=64)
    assert out is None


def test_scheme_bits_per_chunk():
    scheme = HashScheme(head_bits=10, head_bytes=8, tail_bits=4)
    # 13 bytes em trechos de 2: inícios 0,2,4,6 (início) e 8,10,12 (restante)
    assert scheme.bits_for(13, 2) == [10, 10, 10, 10, 4, 4, 4]
    # trecho que começa antes de head_bytes conta como início, mesmo que o cruze
    assert HashScheme(10, 8, 4).bits_for(12, 3) == [10, 10, 10, 4]
    assert HashScheme.uniform(6).bits_for(13, 2) == [6] * 7


def test_scheme_validation():
    with pytest.raises(ValueError):
        HashScheme(0, 8, 4)
    with pytest.raises(ValueError):
        HashScheme(10, 8, 65)
    with pytest.raises(ValueError):
        HashScheme(10, -1, 4)


def test_nonuniform_bits_accounting():
    scheme = HashScheme(head_bits=10, head_bytes=8, tail_bits=4)
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=scheme, tag_key=KEY)
    assert enc.chunk_bits == [10, 10, 10, 10, 4, 4, 4]
    assert enc.total_bits() == 8 + 4 * 10 + 3 * 4 + 32
    assert all(0 <= h < 2**b for h, b in zip(enc.hashes, enc.chunk_bits))


def test_uniform_int_matches_uniform_scheme():
    a = encode(MSG, chunk_bytes=2, bits_per_chunk=8, tag_key=KEY)
    b = encode(MSG, chunk_bytes=2, bits_per_chunk=HashScheme.uniform(8), tag_key=KEY)
    assert a == b


def test_roundtrip_nonuniform(lm):
    scheme = HashScheme(head_bits=16, head_bytes=4, tail_bits=8)
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=scheme, tag_key=KEY)
    out, _ = decode(enc, lm, KEY, beam_width=256, topk=256)
    assert out == MSG


def test_decode_uses_per_chunk_bits(lm):
    # com 1 bit por trecho a poda quase não age; se o decode usasse os bits errados
    # (ex.: os do início em todos os trechos), o hash de 1 bit nunca bateria
    scheme = HashScheme(head_bits=16, head_bytes=4, tail_bits=1)
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=scheme, tag_key=KEY)
    out, stats = decode(enc, lm, KEY, beam_width=256, topk=256)
    assert out == MSG
    assert stats.tag_checks >= 1


def test_tampered_head_hash_is_rejected_nonuniform(lm):
    scheme = HashScheme(head_bits=16, head_bytes=4, tail_bits=8)
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=scheme, tag_key=KEY)
    enc.hashes[0] ^= 1
    out, _ = decode(enc, lm, KEY, beam_width=64, topk=64)
    assert out is None


def _reference_decode(enc, lm, tag_key, beam_width, topk, max_tag_checks=codec.MAX_TAG_CHECKS):
    """Busca em feixe direta, um candidato por vez; é o oráculo do decode otimizado."""
    stats = codec.DecodeStats()
    ends = chunk_ends(enc.length, enc.chunk_bytes)
    max_steps = 4 * enc.length
    frontier = {0: {b"": codec._Beam(b"", [], 0.0)}}
    finished = {}
    while frontier and stats.steps < max_steps:
        pos = min(frontier)
        bucket = sorted(frontier.pop(pos).values(), key=lambda c: c.score, reverse=True)
        beams = bucket[:beam_width]
        stats.steps += 1
        values, indices = lm.next_topk([b.ids for b in beams], topk)
        for bi, beam in enumerate(beams):
            for logprob, tid in zip(values[bi], indices[bi]):
                piece = lm.token_bytes[int(tid)]
                if not piece:
                    continue
                data = beam.data + piece
                if len(data) > enc.length:
                    continue
                if not codec._hashes_ok(enc, ends, len(beam.data), data):
                    continue
                stats.candidates += 1
                cand = codec._Beam(data, beam.ids + [int(tid)], beam.score + float(logprob))
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
    for cand in ranked[:max_tag_checks]:
        stats.tag_checks += 1
        if make_tag(cand.data, tag_key, enc.tag_bits) == enc.tag:
            return cand.data, stats
    return None, stats


class _MultiByteLM:
    """Tokens de 1 a 3 bytes e pontuações determinísticas, só para exercitar o pré-filtro.

    O ByteBigramLM tem apenas tokens de 1 byte, então o caminho com k >= 2 bytes por
    token do `_candidate_mask` nunca rodaria nos testes.
    """

    def __init__(self, corpus, sizes=(2, 3)):
        grams = {bytes([i]) for i in range(256)}
        for msg in corpus:
            for n in sizes:
                grams |= {msg[i : i + n] for i in range(len(msg) - n + 1)}
        self.token_bytes = sorted(grams)

    def next_topk(self, seqs, k):
        v = len(self.token_bytes)
        rows = np.stack([np.random.default_rng(hash(tuple(s)) % 2**32).normal(size=v) for s in seqs])
        idx = np.argsort(-rows, axis=1)[:, : min(k, v)]
        return np.take_along_axis(rows, idx, axis=1), idx


@pytest.mark.parametrize("chunk_bytes", [2, 3, 4])
def test_decode_matches_reference_multibyte_tokens(chunk_bytes):
    lm = _MultiByteLM(CORPUS)
    scheme = HashScheme(4, 6, 2)
    explored = 0
    for msg in CORPUS[:2]:
        enc = encode(msg, chunk_bytes, scheme, KEY)
        got, gs = decode(enc, lm, KEY, beam_width=8, topk=200)
        want, ws = _reference_decode(enc, lm, KEY, 8, 200)
        assert got == want
        assert (gs.steps, gs.candidates, gs.tag_checks) == (ws.steps, ws.candidates, ws.tag_checks)
        explored += gs.candidates
    assert explored > 0  # o teste precisa de fato expandir candidatos


@pytest.mark.parametrize("chunk_bytes", [10, 12])
def test_decode_matches_reference_tokens_longer_than_head(chunk_bytes):
    # tokens de 9 a 12 bytes: a fronteira fica além dos 8 bytes do pré-filtro (k > _HEAD_BYTES),
    # que então não verifica o primeiro trecho e quem verifica é o decode
    lm = _MultiByteLM(CORPUS, sizes=(2, 9, 10, 11, 12))
    scheme = HashScheme(3, 0, 3)
    explored = 0
    for msg in CORPUS[:2]:
        enc = encode(msg, chunk_bytes, scheme, KEY)
        got, gs = decode(enc, lm, KEY, beam_width=8, topk=300)
        want, ws = _reference_decode(enc, lm, KEY, 8, 300)
        assert got == want
        assert (gs.steps, gs.candidates, gs.tag_checks) == (ws.steps, ws.candidates, ws.tag_checks)
        explored += gs.candidates
    assert explored > 0


def test_prefix_hasher_matches_chunk_hash():
    rng = random.Random(1)
    for _ in range(200):
        index = rng.randrange(0, 40)
        prefix = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 60)))
        suffix = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 9)))
        bits = rng.randrange(1, 65)
        assert PrefixHasher(index, prefix).extended(suffix, bits) == chunk_hash(index, prefix + suffix, bits)


@pytest.mark.parametrize("chunk_bytes", [1, 2, 3, 4, 9])
@pytest.mark.parametrize(
    "scheme", [HashScheme.uniform(3), HashScheme.uniform(8), HashScheme(10, 4, 3), HashScheme(6, 8, 2)]
)
def test_decode_matches_reference(lm, chunk_bytes, scheme):
    # mesmos candidatos, contagens e saída que a busca direta, inclusive com decodificações
    # que falham (hash adulterado) e com topk pequeno (o token certo pode ficar fora)
    for msg in CORPUS[:2]:
        for beam, topk in ((1, 8), (4, 32), (16, 128)):
            enc = encode(msg, chunk_bytes, scheme, KEY)
            for tamper in (False, True):
                if tamper:
                    enc.hashes[len(enc.hashes) // 2] ^= 1
                got, gs = decode(enc, lm, KEY, beam_width=beam, topk=topk)
                want, ws = _reference_decode(enc, lm, KEY, beam, topk)
                assert got == want
                assert (gs.steps, gs.candidates, gs.tag_checks) == (ws.steps, ws.candidates, ws.tag_checks)


def test_tag_requires_key_and_minimum_bits():
    with pytest.raises(ValueError):
        make_tag(MSG, b"", 32)
    with pytest.raises(ValueError):
        make_tag(MSG, KEY, 8)
    with pytest.raises(ValueError):
        encode(MSG, chunk_bytes=2, bits_per_chunk=8, tag_key=b"")
    with pytest.raises(ValueError):
        encode(MSG, chunk_bytes=2, bits_per_chunk=8, tag_key=KEY, tag_bits=24)


def test_tag_matches_constant_time_helper():
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=8, tag_key=KEY)
    assert codec._tag_matches(MSG, KEY, enc)
    assert not codec._tag_matches(MSG + b"!", KEY, enc)
    assert not codec._tag_matches(MSG, b"outra-chave", enc)
    enc.tag |= 1 << enc.tag_bits  # fora da faixa de bits da etiqueta
    assert not codec._tag_matches(MSG, KEY, enc)


@pytest.mark.parametrize("kwargs", [{"beam_width": 0}, {"topk": 0}, {"max_steps": 0}, {"max_tag_checks": 0}])
def test_decode_rejects_invalid_limits(lm, kwargs):
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=8, tag_key=KEY)
    with pytest.raises(ValueError):
        decode(enc, lm, KEY, **kwargs)


def test_max_steps_is_honoured(lm):
    enc = encode(MSG, chunk_bytes=2, bits_per_chunk=16, tag_key=KEY)
    out, stats = decode(enc, lm, KEY, beam_width=256, topk=256, max_steps=2)
    assert out is None
    assert stats.steps == 2


def test_tag_check_cap_is_enforced(lm):
    # 1 bit por trecho quase não poda: sobram muitos candidatos completos para o teto agir
    enc = encode(MSG, chunk_bytes=4, bits_per_chunk=1, tag_key=KEY)
    out, ds = decode(enc, lm, KEY, beam_width=256, topk=256)
    assert ds.tag_cap_hit and ds.tag_checks <= codec.MAX_TAG_CHECKS  # o teto padrão age
    assert out in (None, MSG)
    out1, cs = decode(enc, lm, KEY, beam_width=256, topk=256, max_tag_checks=1)
    assert cs.tag_checks == 1 and cs.tag_cap_hit
    assert out1 in (None, MSG)  # o teto nunca troca a mensagem por outra
    want, ws = _reference_decode(enc, lm, KEY, 256, 256, max_tag_checks=1)
    assert out1 == want and cs.tag_checks == ws.tag_checks


def test_emitter_does_not_import_numpy():
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(codec.__file__).resolve().parents[1])
    code = f"import sys; sys.path.insert(0, {src!r}); import assim.emitter; assert 'numpy' not in sys.modules"
    assert subprocess.run([sys.executable, "-I", "-c", code]).returncode == 0


def test_ngram_probabilities_are_normalised_and_trainable():
    from assim.ngram import ByteNgramLM

    lm = ByteNgramLM(max_context=3)
    lm.add(b"tudo bem\nestou bem\nchego em casa\n")
    for ctx in (b"\n", b"tudo", b"xyz", b"em ca"):
        assert abs(sum(lm.prob(ctx[-3:], bytes([b])) for b in range(256)) - 1.0) < 1e-9
    before = lm.bits(b"tudo bem")
    assert before < 8 * len(b"tudo bem")  # aprendeu algo
    lm.add(b"tudo bem\n", weight=20)
    assert lm.bits(b"tudo bem") < before
    lm.add(b"tudo bem\n", weight=-20)
    assert abs(lm.bits(b"tudo bem") - before) < 1e-9  # remover desfaz


def test_length_limit():
    with pytest.raises(ValueError):
        encode(b"x" * 256, chunk_bytes=8, bits_per_chunk=16, tag_key=KEY)


def test_chunk_hash_bits_are_nested():
    # o hash de b bits é o prefixo do de b+1 bits: base do envio incremental de bits extras (DECISOES 5.31)
    for index in (0, 3, 17):
        for bits in range(1, 64):
            assert chunk_hash(index, b"abc", bits) == chunk_hash(index, b"abc", bits + 1) >> 1


@pytest.mark.parametrize(
    "text",
    ["安全到达，电池剩40%。", "وصلت بسلام، البطارية عند 40%.", "Llegué bien, ¿batería al 40%? ñ", "ok 👍 chegamos às 18h ✓"],
)
def test_roundtrip_is_byte_exact_for_non_ascii(text):
    # o codec opera em bytes: qualquer UTF-8 deve voltar idêntico quando a etiqueta confere
    msg = text.encode("utf-8")
    lm_ = ByteBigramLM([msg])
    enc = encode(msg, 2, HashScheme.uniform(16), KEY)
    out, _ = decode(enc, lm_, KEY, beam_width=64, topk=256)
    assert out == msg
