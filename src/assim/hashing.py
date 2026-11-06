"""Hashes por trecho (emissor) e etiqueta final.

ATENÇÃO: a etiqueta é um BLAKE2b com chave, usado aqui como substituto de um MAC
apenas para o protótipo. Em implantação, use uma construção auditada (ex.: AES-SIV,
RFC 5297) e revisão criptográfica, sobretudo se a etiqueta for truncada.
Os hashes por trecho não são secretos: a confidencialidade, quando exigida pelo
perfil de implantação, é responsabilidade da camada de cifra.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

CHUNK_KEY = b"assim-chunk-v1"  # separação de domínio, não é segredo


def chunk_ends(length: int, chunk_bytes: int) -> list[int]:
    """Posições (em bytes) onde cada trecho termina; o último pode ser menor."""
    if length <= 0:
        raise ValueError("mensagem vazia")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes deve ser positivo")
    ends = list(range(chunk_bytes, length, chunk_bytes))
    ends.append(length)
    return ends


@dataclass(frozen=True)
class HashScheme:
    """Bits de hash por trecho: `head_bits` nos trechos que tocam os primeiros
    `head_bytes` bytes e `tail_bits` nos demais.

    É convenção fixa da implantação: o receptor a conhece e ela não é transmitida.
    """

    head_bits: int
    head_bytes: int
    tail_bits: int

    def __post_init__(self) -> None:
        if self.head_bytes < 0:
            raise ValueError("head_bytes não pode ser negativo")
        for bits in (self.head_bits, self.tail_bits):
            if not 1 <= bits <= 64:
                raise ValueError("bits deve estar entre 1 e 64")

    @classmethod
    def uniform(cls, bits: int) -> HashScheme:
        return cls(bits, 0, bits)

    def bits_for(self, length: int, chunk_bytes: int) -> list[int]:
        """Bits de cada trecho de uma mensagem de `length` bytes."""
        ends = chunk_ends(length, chunk_bytes)
        starts = [0] + ends[:-1]
        return [self.head_bits if s < self.head_bytes else self.tail_bits for s in starts]


def chunk_hash(index: int, prefix: bytes, bits: int) -> int:
    """Hash truncado do prefixo inteiro até o fim do trecho `index`."""
    if not 1 <= bits <= 64:
        raise ValueError("bits deve estar entre 1 e 64")
    digest = hashlib.blake2b(
        index.to_bytes(2, "big") + prefix, key=CHUNK_KEY, digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") >> (64 - bits)


class PrefixHasher:
    """Fixa (índice, prefixo) e estende com sufixos curtos sem refazer o prefixo.

    `extended(s, b)` é igual a `chunk_hash(index, prefix + s, b)`: o índice vem antes do
    prefixo, então o estado após (índice, prefixo) é reaproveitável por qualquer sufixo.
    """

    def __init__(self, index: int, prefix: bytes):
        self._base = hashlib.blake2b(index.to_bytes(2, "big") + prefix, key=CHUNK_KEY, digest_size=8)

    def extended(self, suffix: bytes, bits: int) -> int:
        h = self._base.copy()
        h.update(suffix)
        return int.from_bytes(h.digest(), "big") >> (64 - bits)


def tag(message: bytes, key: bytes, bits: int = 32) -> int:
    """Etiqueta sobre a mensagem original: autentica e rejeita decodificações erradas."""
    if not key:
        raise ValueError("a chave da etiqueta não pode ser vazia")
    if not 32 <= bits <= 64 or bits % 8:
        raise ValueError("bits da etiqueta deve ser múltiplo de 8 entre 32 e 64")
    digest = hashlib.blake2b(message, key=key, digest_size=bits // 8).digest()
    return int.from_bytes(digest, "big")
