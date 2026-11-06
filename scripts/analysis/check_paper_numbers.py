#!/usr/bin/env python3
"""Check the numbers of artigo/arxiv/main.tex against the source docs and the result CSVs.

Not part of the audit gate (the gate reads docs, not the article). It lists every number of the
article body that does not appear in any source document, and recomputes from the CSVs the counts
the article states without a gate marker. Plot coordinates, label positions and LaTeX syntax show up
in the "not found" list by design; every other entry must be explained by hand.

  python -I scripts/analysis/check_paper_numbers.py
"""
import csv
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    "artigo/paper.md",
    "docs/RESULTADO_ARTIGO_UNICO.md",
    "docs/medidas/single_summary.txt",
    "docs/PREREGISTRO_ARTIGO_UNICO.md",
    "docs/DECISOES.md",
    "docs/medidas/sender_memory.csv",
]


def variants(tok: str) -> set[str]:
    v = {tok, tok.replace(",", ""), tok.replace(",", ".")}
    if "." in tok:
        v.add(tok.replace(".", ","))
    return v


def rows(name: str) -> list[dict]:
    with open(ROOT / "results" / name, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    tex = (ROOT / "artigo/arxiv/main.tex").read_text()
    body = tex.split(r"\begin{thebibliography}")[0]
    present = [p for p in SOURCES if (ROOT / p).exists()]
    print(f"source docs present: {present}")
    sources = "\n".join((ROOT / p).read_text() for p in present)
    nums = {n.replace("{,}", ",").strip(",") for n in re.findall(r"(?<![\w.])\d[\d,{}]*(?:\.\d+)?", body)}
    missing = sorted(
        n for n in nums
        if len(n.replace(",", "").replace(".", "")) >= 2 and not any(v in sources for v in variants(n))
    )
    print(f"numbers in article body: {len(nums)}; not found in sources: {missing}")

    prim, ngr, ret = rows("single_primary.csv"), rows("single_receiver_ngram.csv"), rows("single_retry.csv")
    ok = lambda r: r["sucesso"] == "True"
    pm = {r["msg_sha256"]: ok(r) for r in prim}
    nm = {r["msg_sha256"]: ok(r) for r in ngr}
    print(f"primary delivered {sum(pm.values())}/{len(prim)}; ngram delivered {sum(nm.values())}/{len(ngr)}, "
          f"mean {statistics.mean(float(r['segundos']) for r in ngr):.2f} s")
    print(f"both {sum(pm[k] and nm[k] for k in pm)}, llm only {sum(pm[k] and not nm[k] for k in pm)}, "
          f"ngram only {sum(nm[k] and not pm[k] for k in pm)}")
    for beam in ("256", "1024"):
        rb = [r for r in ret if r["beam"] == beam]
        s = [float(r["segundos"]) for r in rb]
        print(f"retry beam {beam}: {sum(map(ok, rb))}/{len(rb)}, mean {statistics.mean(s):.1f} s, max {max(s):.1f} s")
    print(f"retry max RSS {max(int(r['rss_max_bytes']) for r in ret) / 1e9:.2f} GB, "
          f"max MPS {max(int(r['mps_driver_bytes']) for r in ret) / 1e9:.2f} GB")
    win = [r for r in prim if 650 <= int(r["id"]) <= 728]
    rest = [r for r in prim if not 650 <= int(r["id"]) <= 728]
    print(f"messages 650-728: {sum(not ok(r) for r in win)}/{len(win)} failed, slowest "
          f"{max(float(r['segundos']) for r in win):.0f} s; elsewhere {sum(not ok(r) for r in rest)}/{len(rest)}")
    wrong = sum(r["correto"] == "False" and ok(r) for r in prim + ngr + ret)
    print(f"wrong messages accepted (all stages): {wrong}")

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    plain = re.sub(r"\s+", " ", abstract.replace("\\%", "%")).strip()
    print(f"abstract: {len(plain)} characters (arXiv limit 1,920)")
    print(f"non-ASCII characters: {sorted({c for c in tex if ord(c) > 127})}")


if __name__ == "__main__":
    main()
