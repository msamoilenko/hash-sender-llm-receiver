"""Emissor: só calcula hashes. Sem modelo de linguagem e sem numpy (só `hashlib`)."""
from __future__ import annotations

from dataclasses import dataclass

from .hashing import HashScheme, chunk_ends, chunk_hash
from .hashing import tag as make_tag

LENGTH_FIELD_BITS = 8  # mensagens de até 255 bytes


@dataclass
class Encoded:
    length: int
    chunk_bytes: int
    chunk_bits: list[int]  # derivado do esquema, que o receptor conhece; não é transmitido
    hashes: list[int]
    tag: int
    tag_bits: int

    def total_bits(self) -> int:
        return LENGTH_FIELD_BITS + sum(self.chunk_bits) + self.tag_bits


def encode(
    message: bytes,
    chunk_bytes: int,
    bits_per_chunk: int | HashScheme,
    tag_key: bytes,
    tag_bits: int = 32,
) -> Encoded:
    """Emissor: só calcula hashes. Não usa modelo de linguagem.

    `bits_per_chunk` inteiro é a distribuição uniforme; um `HashScheme` dá bits diferentes
    ao início e ao restante da mensagem.
    """
    if len(message) >= 2**LENGTH_FIELD_BITS:
        raise ValueError(f"mensagem com {len(message)} bytes excede o campo de tamanho")
    scheme = bits_per_chunk if isinstance(bits_per_chunk, HashScheme) else HashScheme.uniform(bits_per_chunk)
    ends = chunk_ends(len(message), chunk_bytes)
    chunk_bits = scheme.bits_for(len(message), chunk_bytes)
    hashes = [chunk_hash(i, message[:e], chunk_bits[i]) for i, e in enumerate(ends)]
    return Encoded(
        length=len(message),
        chunk_bytes=chunk_bytes,
        chunk_bits=chunk_bits,
        hashes=hashes,
        tag=make_tag(message, tag_key, tag_bits),
        tag_bits=tag_bits,
    )
