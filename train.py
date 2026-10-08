"""Train a custom detector from a labeled YOLO dataset."""
import argparse
from pathlib import Path


# Mild transformations for the simulated camera. YOLO adjusts boxes automatically.
MILD_AUGMENTATION = {
    'hsv_h': 0.01, 'hsv_s': 0.25, 'hsv_v': 0.25,
    'degrees': 5.0, 'translate': 0.1, 'scale': 0.25,
    'fliplr': 0.5, 'flipud': 0.0,
    'shear': 0.0, 'perspective': 0.0,
    'mosaic': 0.3, 'mixup': 0.0, 'copy_paste': 0.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('dataset.yaml'),
                        help='Dataset YAML; defaults to dataset.yaml beside this script')
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu', help='cpu or GPU index such as 0')
    parser.add_argument('--augmentation', choices=['mild', 'default', 'none'], default='mild',
                        help='mild, original library defaults, or none (disable training transformations)')
    parser.add_argument('--name', default='objects_augmented', help='Training run folder name')
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 1:
        parser.error('Epochs and image size must be positive.')
    if not args.data.is_file():
        parser.error(f'Dataset YAML does not exist: {args.data.resolve()}. Use --data /path/to/dataset.yaml')
    from ultralytics import YOLO
    model = YOLO(args.model)
    augmentation = MILD_AUGMENTATION.copy() if args.augmentation == 'mild' else {}
    if args.augmentation == 'none':
        augmentation = {key: 0.0 for key in MILD_AUGMENTATION}
        augmentation.update(close_mosaic=0, bgr=0.0)
        # Suppress optional Albumentations transforms in library versions that support this hook.
        from ultralytics.data import augment
        class NoAlbumentations:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, labels):
                return labels
        augment.Albumentations = NoAlbumentations
    if args.augmentation == 'mild':
        # Keep mosaic active in short pilots; turn it off near the end of longer runs.
        augmentation['close_mosaic'] = min(10, args.epochs // 5)
    print(f'Augmentation profile: {args.augmentation}')
    if augmentation:
        print(augmentation)
    result = model.train(data=str(args.data.resolve()), epochs=args.epochs,
                         imgsz=args.imgsz, device=args.device,
                         project=str(Path(__file__).parent / 'runs'), name=args.name, plots=True, **augmentation)
    print(f'Training artifacts: {result.save_dir}')
    print('Run main.py --weights /absolute/path/to/weights/best.pt.')


if __name__ == '__main__':
    main()
