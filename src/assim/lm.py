"""Modelos usados pelo receptor.

Interface mínima:
  token_bytes: list[bytes]                  bytes de cada token (b"" = inválido)
  next_topk(seqs, k) -> (valores, índices)  log-probabilidades do próximo token
  sequence_bits(data) -> float              custo ideal da versão simétrica (referência)
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np


def _gpt2_byte_decoder() -> dict[str, int]:
    """Inverso do mapeamento byte→caractere usado pelo BPE em nível de byte (GPT-2, Qwen)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class ByteBigramLM:
    """Modelo de brinquedo (bigrama de bytes) para testes, sem download."""

    def __init__(self, corpus: list[bytes], alpha: float = 0.1):
        self.start_id = 256
        self.token_bytes = [bytes([i]) for i in range(256)] + [b""]
        counts = np.full((257, 256), alpha, dtype=np.float64)
        for msg in corpus:
            prev = self.start_id
            for b in msg:
                counts[prev, b] += 1
                prev = b
        logp = np.log(counts / counts.sum(axis=1, keepdims=True))
        self._lp = np.full((257, 257), -np.inf)
        self._lp[:, :256] = logp

    def next_topk(self, seqs: list[list[int]], k: int):
        k = min(k, 256)
        rows = np.stack([self._lp[s[-1] if s else self.start_id] for s in seqs])
        idx = np.argpartition(-rows, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(rows, idx, axis=1)
        return vals, idx

    def sequence_bits(self, data: bytes) -> float:
        prev, total = self.start_id, 0.0
        for b in data:
            total -= self._lp[prev, b] / math.log(2)
            prev = b
        return total


def _pick_device(torch) -> str:
    """CUDA (NVIDIA), MPS (Apple Silicon) ou CPU, nessa ordem."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class _KVRunner:
    """Estado oculto da última posição de [início]+seq, reaproveitando o cache KV do pai.

    Cada sequência expandida guarda seu KV sob a chave `tuple(ids)`. Um filho (seq + 1 token)
    processa só o token novo sobre o KV do pai; se o pai foi despejado (orçamento de memória)
    ou nunca existiu, recalcula o prefixo inteiro. O resultado é o mesmo da passada completa,
    a menos de diferenças numéricas de ponto flutuante.
    """

    def __init__(self, torch, base, start_id: int, device: str, budget_bytes: int):
        self.torch, self.base, self.start_id, self.device = torch, base, start_id, device
        self.budget = budget_bytes
        self._kv: OrderedDict[tuple[int, ...], object] = OrderedDict()  # [camadas, 2, kv_heads, T, dim]
        self._bytes = 0
        self.hits = self.misses = 0

    @staticmethod
    def _size(kv) -> int:
        return kv.numel() * kv.element_size()

    def _put(self, key: tuple[int, ...], kv) -> None:
        old = self._kv.pop(key, None)
        if old is not None:
            self._bytes -= self._size(old)
        self._kv[key] = kv
        self._bytes += self._size(kv)
        while self._bytes > self.budget and len(self._kv) > 1:
            _, evicted = self._kv.popitem(last=False)
            self._bytes -= self._size(evicted)

    def _stack(self, cache):
        torch = self.torch
        return torch.stack([torch.stack([layer.keys, layer.values]) for layer in cache.layers])  # [L, 2, B, H, T, D]

    def _from_parents(self, seqs, parents):
        from transformers import DynamicCache

        torch = self.torch
        lens = [p.shape[3] for p in parents]
        n, past_len = len(seqs), max(lens)
        layers, _, heads, _, dim = parents[0].shape
        past = torch.zeros((n, layers, 2, heads, past_len, dim), dtype=parents[0].dtype, device=self.device)
        mask = torch.zeros((n, past_len + 1), dtype=torch.long)
        for j, p in enumerate(parents):
            past[j, :, :, :, : lens[j]] = p
            mask[j, : lens[j]] = 1
        mask[:, past_len] = 1
        ids = torch.tensor([[s[-1]] for s in seqs], dtype=torch.long, device=self.device)
        pos = torch.tensor([[t] for t in lens], dtype=torch.long, device=self.device)
        cache = DynamicCache(ddp_cache_data=[(past[:, i, 0], past[:, i, 1]) for i in range(layers)])
        hidden = self.base(
            input_ids=ids, attention_mask=mask.to(self.device), position_ids=pos, past_key_values=cache, use_cache=True
        ).last_hidden_state[:, 0]
        new = self._stack(cache)
        for j, s in enumerate(seqs):  # descarta o preenchimento do meio e copia (não guarda view do lote)
            self._put(tuple(s), torch.cat([new[:, :, j, :, : lens[j]], new[:, :, j, :, past_len : past_len + 1]], dim=-2))
        return hidden

    def _from_scratch(self, seqs):
        torch = self.torch
        full = [[self.start_id] + list(s) for s in seqs]
        length = max(len(s) for s in full)
        ids = torch.full((len(full), length), self.start_id, dtype=torch.long)
        mask = torch.zeros((len(full), length), dtype=torch.long)
        for i, s in enumerate(full):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        out = self.base(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), use_cache=True)
        kv = self._stack(out.past_key_values)
        for i, s in enumerate(seqs):
            self._put(tuple(s), kv[:, :, i, :, : len(full[i])].clone())
        return out.last_hidden_state[torch.arange(len(full), device=self.device), torch.tensor([len(s) - 1 for s in full], device=self.device)]

    def last_hidden(self, seqs: list[list[int]]):
        torch = self.torch
        hit, miss, parents = [], [], []
        for i, s in enumerate(seqs):
            parent = self._kv.get(tuple(s[:-1])) if s else None
            if parent is None:
                miss.append(i)
            else:
                self._kv.move_to_end(tuple(s[:-1]))
                hit.append(i)
                parents.append(parent)
        self.hits += len(hit)
        self.misses += len(miss)
        parts = []
        with torch.no_grad():
            if hit:
                parts.append((hit, self._from_parents([seqs[i] for i in hit], parents)))
            if miss:
                parts.append((miss, self._from_scratch([seqs[i] for i in miss])))
        first = parts[0][1]
        out = torch.empty((len(seqs), first.shape[-1]), dtype=first.dtype, device=self.device)
        for idx, h in parts:
            out[torch.tensor(idx, device=self.device)] = h
        return out


# Revisões fixadas no Hub: sem isso, `from_pretrained` segue o `main`, que pode mudar
# entre execuções. 060db649 é o único snapshot no cache local de 2026-10-26, então
# provavelmente o usado nas rodadas anteriores (DECISOES 5.52).
PINNED_REVISIONS = {
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
}


class HFLM:
    """Modelo causal do Hugging Face com BPE em nível de byte (GPT-2, Qwen2.5...)."""

    def __init__(self, name: str, device: str | None = None, kv_budget_bytes: int = 2 << 30,
                 revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or _pick_device(torch)
        revision = revision or PINNED_REVISIONS.get(name)
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        byte_decoder = _gpt2_byte_decoder()
        # fp16 é seguro aqui: só o receptor usa o modelo, sem exigência de determinismo
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        try:  # transformers recentes usam `dtype`; versões antigas, `torch_dtype`
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, revision=revision)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, revision=revision)
        model = model.to(self.device).eval()

        vocab = model.get_output_embeddings().weight.shape[0]
        excluded = set(tok.all_special_ids)
        excluded |= set(getattr(tok, "added_tokens_decoder", {}).keys())
        token_bytes = [b""] * vocab
        for tid in range(min(len(tok), vocab)):
            if tid in excluded:
                continue
            piece = tok.convert_ids_to_tokens(tid)
            if not piece:
                continue
            try:
                token_bytes[tid] = bytes(byte_decoder[ch] for ch in piece)
            except KeyError:
                pass

        start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        if start is None:
            raise ValueError("modelo sem token inicial (bos/eos)")

        self.tok, self.model, self.token_bytes, self.start_id = tok, model, token_bytes, start
        self.revision = revision or ""
        probe = "Ação, coração: 40% às 18h — ok!".encode("utf-8")
        probe_ids = tok.encode(probe.decode("utf-8"), add_special_tokens=False)
        if b"".join(token_bytes[i] for i in probe_ids) != probe:
            raise ValueError(
                "Tokenizador incompatível: use um modelo com BPE em nível de byte "
                "no padrão GPT-2 (ex.: gpt2, Qwen/Qwen2.5-0.5B)."
            )
        self._invalid = torch.tensor([len(b) == 0 for b in token_bytes], device=self.device)
        self._base = model.base_model
        self._head = model.get_output_embeddings()
        # orçamento 0 desliga o cache (passada completa a cada passo)
        self._runner = _KVRunner(torch, self._base, start, self.device, kv_budget_bytes) if kv_budget_bytes > 0 else None

    def _last_hidden(self, seqs: list[list[int]]):
        torch = self.torch
        full = [[self.start_id] + list(s) for s in seqs]
        length = max(len(s) for s in full)
        ids = torch.full((len(full), length), self.start_id, dtype=torch.long)
        mask = torch.zeros((len(full), length), dtype=torch.long)
        for i, s in enumerate(full):  # preenchimento à direita; a atenção causal preserva o resto
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            hidden = self._base(input_ids=ids, attention_mask=mask).last_hidden_state
        last = torch.tensor([len(s) - 1 for s in full], device=self.device)
        return hidden[torch.arange(len(full), device=self.device), last]

    def _logprobs(self, hidden):
        torch = self.torch
        with torch.no_grad():
            logits = self._head(hidden).float()
        logits = logits.masked_fill(self._invalid, float("-inf"))
        return torch.log_softmax(logits, dim=-1)

    def next_topk(self, seqs: list[list[int]], k: int):
        hidden = self._runner.last_hidden(seqs) if self._runner else self._last_hidden(seqs)
        logp = self._logprobs(hidden)
        if self.device == "mps":  # topk com k grande é limitado em algumas versões do MPS
            logp = logp.cpu()
        vals, idx = self.torch.topk(logp, k=min(k, logp.shape[-1]), dim=-1)
        return vals.cpu().numpy(), idx.cpu().numpy()

    def sequence_bits(self, data: bytes) -> float:
        """Custo ideal (−log2 p) da mensagem na versão simétrica, com o mesmo modelo."""
        torch = self.torch
        ids = self.tok.encode(data.decode("utf-8"), add_special_tokens=False)
        if b"".join(self.token_bytes[i] for i in ids) != data:
            raise ValueError("a tokenização não reproduz os bytes originais")
        x = torch.tensor([[self.start_id] + ids], device=self.device)
        with torch.no_grad():
            hidden = self._base(input_ids=x).last_hidden_state[0, :-1]
        logp = self._logprobs(hidden)
        target = torch.tensor(ids, device=self.device)
        nats = -logp[torch.arange(len(ids), device=self.device), target].sum().item()
        return nats / math.log(2)
