# Labelled simulator data

`dataset.yaml` defines Cube as class 0 and Tree as class 1. The image and label stems match. Each nonempty annotation line contains:

```text
class_id normalized_centre_x normalized_centre_y normalized_width normalized_height
```

An empty `.txt` file represents a background image with no target annotation. This layout follows the [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/).

Training uses runs 1–4. Validation uses run 5. There are 115 training file pairs and 23 validation file pairs. Training contains only 92 distinct image byte hashes, so 23 files are repeated image contents under different filenames. The published files preserve the inputs used in the saved experiments.

The [manifest](split_manifest.csv) records every pair, capture run, and image SHA-256. The [audit](audit.json) records label counts and exact content overlap. No exact hash appears in both splits. Similar frames or common scene structure may still occur; byte hashing does not measure visual independence.

Run `python scripts/verify_artifacts.py` from the repository root to recompute these counts and validate the annotations. This audit uses only Python's standard library. It checks files and records; it does not rerun model inference or training.
