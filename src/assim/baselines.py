"""Referências de comparação. Todas recebem a mesma etiqueta de 32 bits no relatório."""
from __future__ import annotations

import zlib


def raw_bits(message: bytes) -> int:
    return 8 * len(message)


def deflate_bits(message: bytes) -> int:
    """DEFLATE cru (sem cabeçalho zlib), nível máximo."""
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    return 8 * len(comp.compress(message) + comp.flush())


class ZstdDict:
    """zstd com dicionário treinado: a referência justa para mensagens curtas."""

    def __init__(self, train: list[bytes], dict_size: int = 4096, level: int = 19):
        import zstandard as zstd

        self.dict = zstd.train_dictionary(dict_size, train)
        try:  # remove o número mágico do quadro, se a versão suportar
            params = zstd.ZstdCompressionParameters.from_level(
                level,
                format=zstd.FORMAT_ZSTD1_MAGICLESS,
                write_checksum=0,
                write_content_size=0,
                write_dict_id=0,
            )
            self.cctx = zstd.ZstdCompressor(dict_data=self.dict, compression_params=params)
        except (AttributeError, TypeError):
            self.cctx = zstd.ZstdCompressor(
                level=level,
                dict_data=self.dict,
                write_checksum=False,
                write_content_size=False,
                write_dict_id=False,
            )

    def bits(self, message: bytes) -> int:
        return 8 * len(self.cctx.compress(message))


def try_zstd_dict(train: list[bytes]) -> ZstdDict | None:
    try:
        return ZstdDict(train)
    except Exception as exc:  # corpus pequeno demais, pacote ausente etc.
        print(f"[aviso] dicionário zstd indisponível: {exc}")
        return None


def unishox_bits(message: bytes) -> int | None:
    """Opcional (pip install unishox2-py). API não verificada: confira antes de confiar."""
    try:
        import unishox2
    except ImportError:
        return None
    try:
        out = unishox2.compress(message.decode("utf-8"))
        data = out[0] if isinstance(out, tuple) else out
        return 8 * len(data)
    except Exception:
        return None
