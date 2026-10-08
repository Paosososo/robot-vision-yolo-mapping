# Trained checkpoints

| Path | Recorded training profile |
| --- | --- |
| `objects_no_aug/best.pt` | No augmentation, 100 epochs, image size 320, CPU |
| `objects_augmented/best.pt` | Mild augmentation, 100 epochs, image size 320, CPU |

Both files are the original fine-tuned checkpoints copied from the lab runs. Their SHA-256 values are in [checksums.json](checksums.json). Training settings and logs are under [artifacts/training](../artifacts/training/).

They detect class 0 Cube and class 1 Tree. The generic pretrained `yolo11n.pt` is not bundled. See the [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) for the starting model and [Ultralytics licensing](https://www.ultralytics.com/license) for upstream terms.
