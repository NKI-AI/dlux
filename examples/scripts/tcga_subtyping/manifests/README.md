# Pinned GDC manifests

These files pin the exact TCGA slides the example uses, so everyone downloads the identical set.

- `<center>.txt` is the gdc-client manifest. Download with `gdc-client download -m <center>.txt -d <dir>`.
- `<center>_metadata.tsv` is the sidecar that `build_sheets.py` reads: file UUID, patient, project, center.

Both are public text and contain no pixels. They are committed as the source of truth. Pinning is what makes the download reproducible: everyone gets the same 30 slides per center.

## Regenerate, or pin a larger set

`make_manifest.py` produced these files by querying the public GDC API. Re-run it to pin a different set, for example a larger one.

```bash
python ../make_manifest.py --center christiana --n-per-class 25 --out-dir .
python ../make_manifest.py --center mskcc      --n-per-class 25 --out-dir .
```

`--n-per-class` sets how many BRCA and how many COAD slides to pin per center. Omit it to pin every available slide. The available diagnostic slides cap it either way: Christiana has 59 BRCA and 49 COAD, MSKCC has 47 and 37. The script writes `<center>.txt` and `<center>_metadata.tsv`. Commit the regenerated files to re-pin the set.

A larger set gives more patients per cross-validation fold. If you raise it a lot, also raise the split grid (`n_outer`, `n_inner`) in the study config and the training batch size in the experiment config.
