"""Copy a flat Label Studio YOLO export into repeatable train/val splits.

Use independent photos only; related video frames should be split by scene.
Original export files are preserved. Existing split folders are never overwritten.
"""

import argparse
import fnmatch
import json
from pathlib import Path
import random
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Extracted export containing images/ and labels/')
    parser.add_argument('--output', type=Path,
                        help='Dataset directory; defaults to dataset/ beside this script')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-pattern', help='Filename glob for training images; requires --val-pattern')
    parser.add_argument('--val-pattern', help='Filename glob for validation images; requires --train-pattern')
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error('--val-ratio must be between 0 and 1')
    if bool(args.train_pattern) != bool(args.val_pattern):
        parser.error('--train-pattern and --val-pattern must be supplied together')
    source = args.source.resolve()
    output = (args.output.resolve() if args.output
              else Path(__file__).resolve().parent / 'dataset')
    if output.exists() and not output.is_dir():
        parser.error(f'Output is not a directory: {output}')
    for kind in ('images', 'labels'):
        parent = output / kind
        if parent.exists() and not parent.is_dir():
            parser.error(f'Expected a directory: {parent}')
        for split in ('train', 'val'):
            target = parent / split
            if target.exists():
                parser.error(f'Split already exists; choose a new output directory: {target}')
    images, labels = source / 'images', source / 'labels'
    if not images.is_dir() or not labels.is_dir():
        parser.error('Source must contain images/ and labels/ directories')
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
    files = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if args.train_pattern:
        train = [p for p in files if fnmatch.fnmatchcase(p.name, args.train_pattern)]
        val = [p for p in files if fnmatch.fnmatchcase(p.name, args.val_pattern)]
        if set(train) & set(val):
            parser.error('Training and validation patterns overlap')
        if not train or not val:
            parser.error('Each pattern must match at least one image')
        print(f'Excluded {len(files) - len(train) - len(val)} images not matching either pattern')
        files = train + val
    if len(files) < 2:
        parser.error('At least two images are needed for train/val splits')
    if len({p.stem for p in files}) != len(files):
        parser.error('Image filename stems must be unique')
    missing = [p.name for p in files if not (labels / (p.stem + '.txt')).is_file()]
    if missing:
        parser.error('Missing labels for: ' + ', '.join(missing)
                     + '. For reviewed background images, provide empty .txt files.')
    if args.train_pattern:
        splits = [('train', train), ('val', val)]
    else:
        random.Random(args.seed).shuffle(files)
        val_count = max(1, min(len(files) - 1, round(len(files) * args.val_ratio)))
        splits = [('val', files[:val_count]), ('train', files[val_count:])]
    classes_file = source / 'classes.txt'
    classes = classes_file.read_text().splitlines() if classes_file.is_file() else []
    if classes and any(not name.strip() for name in classes):
        parser.error('classes.txt contains an empty class name')
    config = output / 'dataset.yaml'
    if classes and config.exists():
        parser.error(f'Configuration already exists: {config}')
    for split, members in splits:
        for kind in ('images', 'labels'):
            (output / kind / split).mkdir(parents=True)
        for img in members:
            label = labels / (img.stem + '.txt')
            shutil.copy2(img, output / 'images' / split / img.name)
            shutil.copy2(label, output / 'labels' / split / label.name)
        print(f'{split}: {len(members)} image/label pairs')
    print(f'Dataset: {output}')
    if classes:
        # JSON is valid YAML; avoids a third-party YAML dependency.
        config.write_text(json.dumps({'path': str(output), 'train': 'images/train',
                                      'val': 'images/val', 'names': classes}, indent=2) + '\n')
        print(f'Configuration: {config}')
    print('Set dataset.yaml path to this directory; verify class IDs and class coverage.')
    print('Related frames must stay in one split; use scene-based splitting for videos.')


if __name__ == '__main__':
    main()
