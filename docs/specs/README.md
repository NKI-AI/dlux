# dlux — design specs

Precise, code-coupled specifications of the pipeline, one per subsystem. Each is written to a fixed frame: a blockquote stating what it covers, a **Source of truth** line naming the code it documents, then **Part 1** (what the code does today) and **Part 2** (why it looks that way).

These describe the current implementation for someone reading or changing it — they are not a user tutorial. A spec is kept true by re-verifying it against its source-of-truth files when that code changes, not by describing intent.

| spec                                 | covers                                                                         | source of truth                                 |
| ------------------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------------- |
| [CONTRACT_SPEC.md](CONTRACT_SPEC.md) | the cohort contract — what a prediction target is and what it admits           | `dlux/config/cohort.py`                     |
| [SHEET_SPEC.md](SHEET_SPEC.md)       | the rigid input sheets `build_db` ingests                                      | `dlux/data/sheets.py`, `build_db.py`        |
| [SPLITS_SPEC.md](SPLITS_SPEC.md)     | predefined CV splits — import a patient→fold assignment into `build_db`        | `dlux/data/splits.py`, `build_db.py`        |
| [CONSUMER_SPEC.md](CONSUMER_SPEC.md) | how a project consumes dlux (`config/`+`catalog/` search-path, `DLUX_PROJECT`) | each bin's primary config                       |
| [COMPARE_SPEC.md](COMPARE_SPEC.md)   | `compare` (stage 1) + `compare_resplits` (stage 2), the measured noise floor   | `dlux/eval/compare.py`, `resplit.py`        |
| [VIEWER_SPEC.md](VIEWER_SPEC.md)     | the cohort slide viewer (`serve_slides`) with mask + attention overlays        | `dlux/viewer/app.py`, `bin/serve_slides.py` |
