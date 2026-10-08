"""Evaluate trained YOLO weights on labeled validation or test images."""
import argparse
import json
from pathlib import Path


def box_iou(a, b):
    """Intersection over union of two pixel-space xyxy rectangles."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_predictions(predictions, labels, threshold):
    """Match each prediction to at most one same-class label, by confidence."""
    matched = set()
    correct = set()
    for index in sorted(range(len(predictions)), key=lambda i: predictions[i]['confidence'], reverse=True):
        prediction = predictions[index]
        candidates = [(box_iou(prediction['box'], label['box']), j)
                      for j, label in enumerate(labels)
                      if j not in matched and prediction['class'] == label['class']]
        if candidates:
            overlap, j = max(candidates)
            if overlap >= threshold:
                matched.add(j)
                correct.add(index)
    return correct, matched


def save_previews(model, metrics, args):
    """Save one ground-truth/prediction comparison for every split image."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ultralytics.data.utils import check_det_dataset, img2label_paths

    # Use the same dataset path resolution rules as Ultralytics validation.
    dataset = check_det_dataset(str(args.data.resolve()))
    source = dataset[args.split]
    entries = source if isinstance(source, list) else [source]
    images = []
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            images.extend(p.resolve() for p in path.rglob('*') if p.suffix.lower() in extensions)
        elif path.suffix.lower() == '.txt':
            for line in path.read_text().splitlines():
                if line.strip():
                    item = Path(line.strip())
                    images.append((item if item.is_absolute() else path.parent / item).resolve())
        else:
            raise ValueError(f'Unsupported split source: {path}')
    output = Path(metrics.save_dir) / 'comparisons'
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for number, image_path in enumerate(sorted(set(images)), 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Cannot read image: {image_path}')
        height, width = image.shape[:2]
        label_path = Path(img2label_paths([str(image_path)])[0])
        # Missing labels may be legitimate negatives, but announce them for review.
        if not label_path.exists():
            print(f'Warning: no label file for {image_path.name}; treating as empty ground truth.')
        labels = []
        for line in label_path.read_text().splitlines() if label_path.exists() else []:
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f'Expected five-field detection labels: {label_path}')
            cls, x, y, w, h = map(float, values)
            labels.append({'class': int(cls), 'box': [(x-w/2)*width, (y-h/2)*height,
                                                     (x+w/2)*width, (y+h/2)*height]})
        result = model.predict(str(image_path), conf=args.preview_conf, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        predictions = [{'class': int(box.cls.item()), 'confidence': float(box.conf.item()),
                        'box': box.xyxy[0].cpu().tolist()} for box in result.boxes]
        correct, matched = match_predictions(predictions, labels, args.match_iou)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for ax in axes:
            ax.imshow(rgb)
            ax.axis('off')
        def draw(ax, box, text, color):
            from matplotlib.patches import Rectangle
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2))
            ax.text(x1, max(0, y1-4), text, color='white', fontsize=9,
                    bbox={'facecolor': color, 'alpha': 0.8, 'pad': 2})
        for i, label in enumerate(labels):
            status = 'matched' if i in matched else 'MISSED'
            draw(axes[0], label['box'], f"{dataset['names'][label['class']]} {status}",
                 'green' if i in matched else 'orange')
        for i, prediction in enumerate(predictions):
            status = 'CORRECT' if i in correct else 'FALSE POSITIVE'
            draw(axes[1], prediction['box'],
                 f"{result.names[prediction['class']]} {prediction['confidence']:.2f} {status}",
                 'green' if i in correct else 'red')
        tp, fp, fn = len(correct), len(predictions)-len(correct), len(labels)-len(matched)
        axes[0].set_title(f'Ground truth: {len(labels)} objects; missed={fn}')
        axes[1].set_title(f'Predictions: correct={tp}, false positives={fp}')
        fig.suptitle(f'{image_path.name} | confidence >= {args.preview_conf}, match IoU >= {args.match_iou}')
        fig.tight_layout()
        filename = f'{number:04d}_{image_path.stem}.png'
        fig.savefig(output / filename, dpi=130)
        plt.close(fig)
        summary.append({'image': str(image_path), 'comparison': filename, 'correct': tp,
                        'false_positives': fp, 'missed': fn})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Saved {len(summary)} image comparisons to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='Dataset YAML')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--preview', action='store_true', help='Save side-by-side labels and predictions for each image')
    parser.add_argument('--preview-conf', type=float, default=0.5, help='Preview confidence threshold, default matches robot runtime')
    parser.add_argument('--match-iou', type=float, default=0.5, help='Minimum overlap for a correct preview detection')
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    if not 0 <= args.preview_conf <= 1 or not 0 < args.match_iou <= 1 or args.imgsz <= 0:
        parser.error('Confidence must be 0..1, match IoU must be >0 and <=1, and image size must be positive.')
    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error(f'File does not exist: {path}')
    from ultralytics import YOLO
    model = YOLO(str(args.weights.resolve()))
    metrics = model.val(data=str(args.data.resolve()), split=args.split,
                        device=args.device, imgsz=args.imgsz, project=str(Path(__file__).parent / 'runs'),
                        name='evaluation', plots=True)
    report = {key: float(value) for key, value in metrics.results_dict.items()}
    (Path(metrics.save_dir) / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f'Evaluation artifacts: {metrics.save_dir}')
    if args.preview:
        save_previews(model, metrics, args)


if __name__ == '__main__':
    main()
