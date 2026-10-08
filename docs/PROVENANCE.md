# Provenance and file guide

This is Jirath Thanabarameth's portfolio copy of a Lab 04 robotics coursework project. The original lab directory remains separate from this repository.

## Supplied material and work performed

The original Lab 04 material supplied Python scripts, CoppeliaSim scenes, and a workflow covering image collection, annotation, training, evaluation, and mapping. This repository does not claim that all of those scripts or scenes were authored from scratch by Jirath.

Jirath's work shown by the conversation and saved outputs includes collecting simulator views, labelling Cube and Tree in Label Studio, preparing the run-based split, executing and comparing the two training profiles, inspecting predictions, running the mapping controller, reading relative simulator positions, and recording the live detection demonstration.

The `--show` OpenCV preview, its stop/cleanup handling, and `--output-dir` option were added with coding assistance during the lab session. The README, portable repository copy, checksum records, dataset audit, and artifact verification script were prepared with coding assistance for this portfolio.

## What was copied and what was adapted

| Location | Contents and provenance |
| --- | --- |
| Root Python scripts | Lab scripts as present after the session, including the added live preview |
| `scenes/` | Supplied original scene and the saved `my_scene.ttt` variation |
| `data/images`, `data/labels` | The actual `dataset_by_run` inputs used in the recorded experiments |
| `models/` | Both original fine-tuned `best.pt` checkpoints, with SHA-256 records |
| `artifacts/training/` | Saved logs and figures for the two training runs |
| `artifacts/evaluation/` | Saved augmented-checkpoint metrics and all 23 comparison images |
| `artifacts/mapping/` | Original-scene custom and generic detector maps, landmarks, and odometry |
| `docs/images/` | Selected recorded output images and a still from the supplied demo video |
| `docs/demo.json` | Metadata and checksum of the original recording hosted on GitHub |
| `scripts/verify_artifacts.py` | Later verification of the published records, without retraining |

Only the portfolio copy receives the following portability changes:

- `main.py` defaults to `scenes/Demo_CV.ttt` and `output/demo`, rather than the session's current custom scene/output directory. The slower 0.5 rad/s setting is retained.
- `data/dataset.yaml` uses relative split paths instead of the original computer's absolute directory.
- Saved training argument files replace machine-specific data/project/output paths with repository-relative paths. Numerical settings are unchanged.
- Evaluation summary image paths are changed to repository-relative paths. Counts and comparison images are unchanged.

The dataset audit, split manifest, training comparison JSON, model checksum file, video metadata, and mapping comparison CSV are derived records created for publication. The map reference coordinates in the CSV were transcribed from the user's CoppeliaSim console screenshot. They are not newly measured during packaging.

## Evidence boundaries

The original experiment inputs and saved outputs are included so a reader can inspect what was done. The training set was not silently deduplicated. The validation set was used during training and selection. New-scene independence, physical-robot performance, and detector FPS have not been measured.

The live video is a separate run from the saved quantitative original-scene map. Its title and recorder frame rate do not establish end-to-end detector throughput.

Virtual environments, caches, the submission ZIP, full capture folders, the unneeded random split, lab handouts, and unrelated desktop screenshots are omitted. The original video is hosted separately to keep normal code clones smaller.

## Attribution and licences

The project uses [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), [CoppeliaSim and its ZeroMQ remote API](https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm), and [OpenCV](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html). Annotation was performed with [Label Studio](https://labelstud.io/guide/).

No blanket licence is assigned to the supplied coursework code or scenes because their redistribution licence was not established in the provided material. Upstream software and pretrained weights retain their own licence terms. Ultralytics describes its licensing options in its [licence reference](https://www.ultralytics.com/license).
