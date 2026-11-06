# Taskmaster-1 — attribution and selection

`taskmaster_en_30.txt` contains 30 user utterances from Taskmaster-1
(`TM-1-2019/self-dialogs.json`), selected mechanically for the pre-registered
test of 2026-10-26.

- Source: https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019
- Raw file SHA-256: `1e590ed0ccee279e40c2fb9e083d3b9417477c6bfe35ce5b2277167698dd858d`
- License: Creative Commons Attribution 4.0 (CC BY 4.0), as stated in
  `TM-1-2019/README.md`.
- Citation: Bill Byrne, Karthik Krishnamoorthi, Chinnadhurai Sankar, Arvind
  Neelakantan, Ben Goodrich, Daniel Duckworth, Semih Yavuz, Amit Dubey, Kyu-Young
  Kim and Andy Cedilnik. *Taskmaster-1: Toward a Realistic and Diverse Dialog
  Dataset.* EMNLP-IJCNLP 2019, pp. 4516–4525. (Author list corrected on
  2026-11-06 against the ACL Anthology; it used to omit three authors.)

Selection (fixed before decoding; full protocol in
`docs/PREREGISTRO_TASKMASTER_2026-10-26.md`):

- `speaker == "USER"`, `strip()`, 28–49 UTF-8 bytes, no CR/LF, no leading `#`;
- exact duplicates dropped, first occurrence kept;
- the 30 smallest `SHA-256("taskmaster-1-2019|seed=20261026|" + message)` digests,
  listed by digest.

The raw 63 MB JSON is not versioned. Download and verify it with:

```bash
.venv/bin/python scripts/analysis/taskmaster_prereg.py --download
```
