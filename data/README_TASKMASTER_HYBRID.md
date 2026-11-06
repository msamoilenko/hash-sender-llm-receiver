# Taskmaster-1 — hybrid test split attribution

This directory contains the official Taskmaster-1 split IDs used by the
receiver-side hybrid experiment of 2026-10-26:

- `taskmaster_train_ids.txt`: 6,168 conversation IDs from
  `TM-1-2019/train-dev-test/train.csv`, normalized without the trailing comma.
- `taskmaster_test_ids.txt`: 770 conversation IDs from
  `TM-1-2019/train-dev-test/test.csv`, normalized without the trailing comma.
- `taskmaster_hybrid_30.txt`: the 30 user utterances selected from the official
  test conversations: the 30 smallest SHA-256 digests (seed 20261027) among the
  2,443 eligible messages left after excluding the 30 development conversations
  and applying the fixed 28–49 byte rule. (Corrected on 2026-11-05, DECISOES 5.59:
  this line used to say "all eligible messages", which was false. One of the 30,
  "What is the telephone number?", also appears verbatim in the training split.)

Source: https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019

License: Creative Commons Attribution 4.0 (CC BY 4.0), as stated in
`TM-1-2019/README.md`.

Citation: Bill Byrne, Karthik Krishnamoorthi, Chinnadhurai Sankar, Arvind
Neelakantan, Ben Goodrich, Daniel Duckworth, Semih Yavuz, Amit Dubey, Kyu-Young
Kim and Andy Cedilnik. *Taskmaster-1: Toward a Realistic and Diverse Dialog
Dataset.* EMNLP-IJCNLP 2019, pp. 4516–4525. (Author list corrected on
2026-11-06 against the ACL Anthology; it used to omit three authors.)

The raw `self-dialogs.json` (SHA-256
`1e590ed0ccee279e40c2fb9e083d3b9417477c6bfe35ce5b2277167698dd858d`) is not
versioned; download it with:

```bash
.venv/bin/python scripts/analysis/taskmaster_prereg.py --download
```

The hybrid n-gram is trained only on the official train conversations (excluding
the 30 development conversations). The test conversations are disjoint from the
training conversations. See `docs/PREREGISTRO_HIBRIDO_2026-10-26.md`.
