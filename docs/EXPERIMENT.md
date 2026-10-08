# Experiment details

This document explains the saved experiment. It does not claim an independent final test or a new simulator run during portfolio preparation.

## Capture and annotation

The lab workflow saves clean `rgb_*.png` files and annotated `frame_*.png` files. Only clean images are annotation inputs. `gather_images.py --step 4` keeps every fourth saved RGB image within each run. The labelled export contains Cube and Tree boxes; empty label files are retained for background images.

The run-based preparation command used the following patterns, including the leading wildcard because exported filenames contain a prefix:

```bash
python prepare_dataset.py --source export --output dataset_by_run --train-pattern "*run[1-4]_*" --val-pattern "*run5_*"
```

This is a historical command for an extracted Label Studio export. The repository already contains its prepared split under `data/`; the full export and capture folders are not required to train or evaluate the included data. `prepare_dataset.py` checks filename overlap, but does not compare image bytes. The later [audit](../data/audit.json) checks exact image hashes.

The recorded training data contains 23 repeated image files. Those files are preserved here to match the experiment. Removing them would change the training inputs and would require a new experiment.

## Training settings

The starting checkpoint is YOLO11n. Both recorded runs use 100 epochs, CPU, image size 320, batch size 16, and seed 0 in their saved argument files. Source: [baseline arguments](../artifacts/training/objects_no_aug/args.yaml) and [augmented arguments](../artifacts/training/objects_augmented/args.yaml).

The mild profile in [train.py](../train.py) is:

| Setting | Value |
| --- | ---: |
| `hsv_h`, `hsv_s`, `hsv_v` | 0.01, 0.25, 0.25 |
| `degrees`, `translate`, `scale` | 5, 0.1, 0.25 |
| `fliplr`, `flipud` | 0.5, 0 |
| `mosaic`, `mixup`, `copy_paste` | 0.3, 0, 0 |
| `shear`, `perspective` | 0, 0 |
| `close_mosaic` at 100 epochs | 10 |

The none profile zeros the configured transformations and suppresses the optional Albumentations hook. The best recorded validation mAP50–95 row occurs at epoch 54 for the baseline and epoch 81 for the augmented run. The 100-epoch training limit and the selected checkpoint epoch are different concepts.

## Validation metrics and preview counts

Ultralytics validation reports aggregate precision, recall and average precision. AP50 matches boxes at IoU 0.5; AP50–95 averages over stricter overlap thresholds. See the [Ultralytics validation reference](https://docs.ultralytics.com/modes/val/) for metric definitions.

The custom preview in [evaluate.py](../evaluate.py) instead predicts with confidence at least 0.5, then greedily matches each prediction to one same-class annotation at IoU at least 0.5. Its saved summary totals 27 matches, no false positives, and one missed annotation. This fixed operating point is distinct from aggregate validation confidence handling. The missed annotation is the Cube in `0c3d9d9d-run5_rgb_000140.png`.

## Position estimation

`main.py` uses measured wheel angle changes:

```text
dl = wheel_radius * delta_left_angle
dr = wheel_radius * delta_right_angle
distance = (dl + dr) / 2
heading_change = (dr - dl) / wheel_track
```

The odometry integration accounts for the turning arc. Its initial pose is `(0, 0, 0)`, so the map axes are the initial robot body axes, not automatically the simulator world axes. The supplied effective wheel track is 0.3 m. An independent calibration measurement is not included in this portfolio.

For a detected box, the controller uses the median valid axial depth from a 5-by-5 patch around the box centre. It computes a focal length from the sensor resolution and field of view, projects the pixel into sensor coordinates, transforms it into the robot body, and rotates/translates it using the estimated odometry pose. This sequence is implemented in `project_pixel` and `detect_and_map`. Detections with no valid centre depth are skipped for mapping.

The landmark merger averages observations of the same class within 0.35 m. It stores the maximum confidence and the observation count. This can merge two nearby objects of the same class, and a class error can create a wrong landmark.

## Reference positions for the saved original-scene map

The reference was read in CoppeliaSim's Lua console relative to the robot body, with the simulation stopped in the initial pose:

```lua
local p=sim.getObjectPosition(sim.getObject('/shape'),sim.getObject('/body')); print('Tree x,y:',p[1],p[2])
local p=sim.getObjectPosition(sim.getObject('/Cuboid'),sim.getObject('/body')); print('Cube x,y:',p[1],p[2])
```

The supplied screenshot of that console shows:

```text
Tree x,y: 0.374403, -1.08567
Cube x,y: 0.6, -0.525
```

This is a transcription of the user's simulator evidence, not a fresh API measurement performed during packaging. The API's relative-frame argument is documented in [sim.getObjectPosition](https://manual.coppeliarobotics.com/en/sim/simGetObjectPosition.htm).

The [comparison CSV](../artifacts/mapping/original_scene/comparison.csv) computes:

```text
error_m = sqrt((estimated_x - reference_x)^2 + (estimated_y - reference_y)^2)
```

The saved map has 30 Tree observations and 8 Cube observations. Its final encoder-estimated heading is about 359 degrees, according to [odometry.csv](../artifacts/mapping/original_scene/odometry.csv). The original saved scan used a rotation speed setting of 3.0 rad/s; the portfolio defaults are slower at 0.5 rad/s for viewing. A new scan's samples and merged positions can therefore differ from the saved artifacts.

Observed surface depth and simulator object origins are not the same physical reference. This may contribute to position error, but the contribution has not been isolated. No detector FPS, physical-robot accuracy, or six-object ground truth is established by these records.
