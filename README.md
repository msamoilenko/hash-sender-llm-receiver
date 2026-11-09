# hash-sender-llm-receiver

Files needed to check the article *How far does a hash-only sender go? A pre-registered map of
sender memory versus bits for short text messages* (Marcelo Samoilenko, 2026).

Preprint: [arXiv:2611.09848](https://arxiv.org/abs/2611.09848) (cs.IT, cross-list cs.CL).

A sender that only computes hashes transmits short English messages; a receiver with a language
model (Qwen2.5-0.5B) reconstructs them by hash-pruned beam search, checked by a 32-bit tag. One
pre-registered run on 1,000 Taskmaster-1 user utterances compared it with in-domain PPM senders,
smaz2, Unishox2 and raw text. **The result does not favour the scheme:** to deliver every message
it spends 18.4 bits more per message than an in-domain PPM order-1 sender (95% bootstrap interval
10.9 to 26.3).

This is a snapshot, not the development repository. The development history (with the commits
cited in the article), the decision log and its errata are available from the author on request.
Most project documents are in Portuguese; the article is in English.

## Contents

| Path | What |
|---|---|
| `artigo/arxiv/main.tex` | Article source (pdflatex, single file) |
| `docs/PREREGISTRO_ARTIGO_UNICO.md` | Pre-registration (Portuguese): data, frozen parameters, endpoint, reading rule, forecast |
| `docs/RESULTADO_ARTIGO_UNICO.md` | Result report (Portuguese), including deviations of the run |
| `results/single_*.csv` | Per-message decoding results: primary run, n-gram receiver (A), wider search (B) |
| `docs/medidas/` | Competitor bits, sender memory, per-message analysis, summary, raw run logs |
| `data/` | Taskmaster-1 split IDs and the 1,000 test utterances (CC BY 4.0, attribution in `data/README_*.md`); also the two 30-utterance development lists, which the selection script needs to exclude the development conversations |
| `docs/medidas/frontier_dev.csv` | Competitor bits on development data, read by the frontier script |
| `src/assim/` | Sender, receiver and baselines |
| `scripts/analysis/` | Selection, run, frontier and analysis scripts |
| `scripts/sender_memory/` | C sender instrument used to measure sender memory |
| `tests/` | Test suite (no model download needed) |

## Checking the numbers

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # validated with Python 3.12; older versions not tested
.venv/bin/python -m pytest
.venv/bin/python -I scripts/analysis/single_analysis.py        # recomputes docs/medidas/single_analysis.csv and single_summary.txt
.venv/bin/python -I scripts/analysis/check_paper_numbers.py    # checks the article against the CSVs
```

The first run of the tests or of `single_analysis.py` clones BLAKE2, Unishox2 and smaz2 at the
commits pinned in `scripts/sender_memory/vendor.lock` into the git-ignored `vendor/`; the Unishox2
clone is about 190 MB and can take several minutes.

`check_paper_numbers.py` also lists article numbers it cannot find verbatim in the documents here:
plot coordinates and LaTeX syntax; constants of the method (255, 1024, 1200); values rounded from the
CSVs (11.9, 10.4, 402); memory increments computed from `docs/medidas/sender_memory.csv` (366, 2,536,
7,873, 17,020) and the BLAKE2b code size (7,720); and the development numbers of the article's
Section 3.7 (143.1, 177.5, 179.9, 264.3, 2,443), which come from the private decision log.

Re-running the decoder needs the raw Taskmaster-1 file
(`.venv/bin/python scripts/analysis/taskmaster_prereg.py --download`, SHA-256 checked) and the
receiver model at the revision pinned in the pre-registration; the decoder runs in fp16 on Apple
MPS, and bit-identical results on other hardware are not claimed.

## Citation

```bibtex
@misc{samoilenko2026hashonly,
  title         = {How far does a hash-only sender go? A pre-registered map of sender memory
                   versus bits for short text messages},
  author        = {Samoilenko, Marcelo},
  year          = {2026},
  eprint        = {2611.09848},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT}
}
```

## License

Code: MIT. Documents, article text and derived data: CC BY 4.0. Taskmaster-1 excerpts: CC BY 4.0,
Byrne et al. (EMNLP-IJCNLP 2019). Details in `LICENSE`.
