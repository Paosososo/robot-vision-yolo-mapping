"""Recompute the published artifact checks using only Python's standard library.

This checks recorded evidence and file integrity. It does not train a model,
rerun inference, or establish independence of similar-looking images.
"""
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(relative):
    with (ROOT / relative).open(newline='') as stream:
        return [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(stream)]


def verify_data():
    config = read_json('data/dataset.yaml')  # This file is JSON-valid YAML.
    require(config['names'] == ['Cube', 'Tree'], 'Unexpected class order')
    require(config['train'] == 'images/train' and config['val'] == 'images/val',
            'Dataset split paths must remain portable')
    require('path' not in config, 'Dataset contains a machine-specific root')
    manifest = csv_rows('data/split_manifest.csv')
    recorded = read_json('data/audit.json')
    actual = {}
    hashes = {}
    for split, expected_runs in [('train', {'run1', 'run2', 'run3', 'run4'}),
                                 ('val', {'run5'})]:
        images = sorted((ROOT / 'data/images' / split).glob('*.png'))
        labels = sorted((ROOT / 'data/labels' / split).glob('*.txt'))
        require({p.stem for p in images} == {p.stem for p in labels},
                f'{split}: image/label pairing mismatch')
        entries = [row for row in manifest if row['split'] == split]
        require(len(entries) == len(images), f'{split}: manifest size mismatch')
        by_image = {row['image']: row for row in entries}
        require(len(by_image) == len(entries), f'{split}: duplicate manifest entry')
        contents = []
        boxes, runs = Counter(), Counter()
        background = 0
        for image in images:
            image_name = image.relative_to(ROOT).as_posix()
            require(image_name in by_image, f'Not in manifest: {image_name}')
            entry = by_image[image_name]
            match = re.search(r'(run[1-5])_', image.name)
            require(match is not None, f'Missing capture run: {image.name}')
            run = match.group(1)
            require(run in expected_runs and entry['capture_run'] == run,
                    f'{split}: incorrect capture group for {image.name}')
            label = ROOT / 'data/labels' / split / (image.stem + '.txt')
            require(entry['label'] == label.relative_to(ROOT).as_posix(),
                    f'Wrong label path: {image.name}')
            digest = sha256(image)
            require(digest == entry['image_sha256'], f'Image hash mismatch: {image.name}')
            contents.append(digest)
            runs[run] += 1
            lines = [line for line in label.read_text().splitlines() if line.strip()]
            background += not lines
            for number, line in enumerate(lines, 1):
                fields = line.split()
                require(len(fields) == 5, f'Invalid label line: {label.name}:{number}')
                require(fields[0] in ('0', '1'), f'Invalid class: {label.name}:{number}')
                x, y, width, height = map(float, fields[1:])
                require(all(math.isfinite(v) and 0 <= v <= 1
                            for v in (x, y, width, height)) and width > 0 and height > 0,
                        f'Invalid normalized box: {label.name}:{number}')
                boxes[config['names'][int(fields[0])]] += 1
        hashes[split] = set(contents)
        actual[split] = {
            'image_label_pairs': len(images),
            'distinct_image_contents': len(hashes[split]),
            'repeated_image_files': len(images) - len(hashes[split]),
            'object_labels': dict(boxes),
            'background_images': background,
            'capture_runs': dict(runs),
        }
    require(actual == recorded['splits'], 'Dataset counts differ from audit.json')
    overlap = len(hashes['train'] & hashes['val'])
    require(overlap == recorded['exact_image_hashes_shared_across_splits'] == 0,
            'Exact image contents cross the train/validation boundary')
    require(len(manifest) == sum(v['image_label_pairs'] for v in actual.values()),
            'Manifest contains an unknown split')
    return actual


def verify_results():
    comparison = read_json('artifacts/training/comparison.json')
    for run, expected in comparison['rows'].items():
        rows = csv_rows(f'artifacts/training/{run}/results.csv')
        require(len(rows) == 100, f'{run}: expected 100 epoch rows')
        best = max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))
        require(best == expected, f'{run}: best recorded row mismatch')
    previews = read_json('artifacts/evaluation/comparisons/summary.json')
    validation = {p.relative_to(ROOT).as_posix()
                  for p in (ROOT / 'data/images/val').glob('*.png')}
    require(len(previews) == len(validation) and
            {row['image'] for row in previews} == validation,
            'Evaluation previews do not cover exactly the validation split')
    for row in previews:
        require((ROOT / 'artifacts/evaluation/comparisons' / row['comparison']).is_file(),
                f'Missing comparison: {row["comparison"]}')
    totals = {key: sum(row[key] for row in previews)
              for key in ('correct', 'false_positives', 'missed')}
    require(totals == {'correct': 27, 'false_positives': 0, 'missed': 1},
            'Recorded preview counts have changed')
    landmarks = read_json('artifacts/mapping/original_scene/objects.json')
    positions = {obj['type']: obj for obj in landmarks}
    map_rows = csv_rows('artifacts/mapping/original_scene/comparison.csv')
    require({row['class'] for row in map_rows} == set(positions) == {'Cube', 'Tree'},
            'Unexpected mapped object classes')
    for row in map_rows:
        obj = positions[row['class']]
        x, y = float(row['estimated_x_m']), float(row['estimated_y_m'])
        require(x == obj['x'] and y == obj['y'] and
                int(row['observations']) == obj['observations'],
                f'Mapping record mismatch: {row["class"]}')
        error = math.hypot(x - float(row['true_x_m']), y - float(row['true_y_m']))
        require(math.isclose(error, float(row['error_m']), abs_tol=1e-12),
                f'Incorrect mapping error: {row["class"]}')
    return totals


def verify_files():
    checksums = read_json('models/checksums.json')
    for name, expected in checksums.items():
        require(sha256(ROOT / name) == expected, f'Model checksum mismatch: {name}')
    for source in ROOT.rglob('*.py'):
        if '.git' not in source.parts:
            ast.parse(source.read_text(), filename=str(source))
    # Check repository-local Markdown targets; external URLs are checked at publication.
    for document in ROOT.rglob('*.md'):
        for link in re.findall(r'\]\(([^\s)]+)\)', document.read_text()):
            if '://' in link or link.startswith('#'):
                continue
            target = (document.parent / link.split('#', 1)[0]).resolve()
            require(target.exists(), f'Broken local link in {document.name}: {link}')
        for link in re.findall(r'<img\s+src="([^"]+)"', document.read_text()):
            require((document.parent / link).is_file(),
                    f'Broken image in {document.name}: {link}')
    return len(checksums)


def main():
    data = verify_data()
    previews = verify_results()
    models = verify_files()
    print(json.dumps({'status': 'passed', 'dataset': data,
                      'fixed_threshold_previews': previews,
                      'model_checksums_verified': models}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f'Artifact verification failed: {error}', file=sys.stderr)
        sys.exit(1)
