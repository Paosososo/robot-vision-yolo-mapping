"""Collect the clean camera pictures of several capture runs into ONE folder, with the run name in the file name.

main.py saves, for every run, rgb_000000.png ... (clean camera pictures: label THESE) and frame_000000.png ...
(the same pictures with the detector's boxes drawn: do not label these).

    python gather_images.py                 # captures/run1, captures/run2 ...  ->  images_all/run1_rgb_000000.png ...
    python gather_images.py --step 2        # keep every 2nd picture only (neighbouring pictures are almost identical)
"""
import argparse
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--captures", type=Path, default=Path(__file__).with_name("captures"))
p.add_argument("--out", type=Path, default=Path(__file__).with_name("images_all"))
p.add_argument("--step", type=int, default=1, help="keep every Nth picture of each run")
a = p.parse_args()

a.out.mkdir(exist_ok=True)
total = 0
for run in sorted(d for d in a.captures.iterdir() if d.is_dir()):
    pics = sorted(run.glob("rgb_*.png"))[:: max(1, a.step)]
    for f in pics:
        shutil.copy2(f, a.out / f"{run.name}_{f.name}")
    print(f"{run.name}: {len(pics)} pictures")
    total += len(pics)
print(f"Total {total} pictures in {a.out}")
