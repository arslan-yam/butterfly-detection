import argparse
import io
import json
import os

import pandas as pd
import requests
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--out", default="examples_png")
ap.add_argument("--max-side", type=int, default=900)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

for r in pd.read_csv(a.ref, dtype=str).itertuples():
    img = Image.open(io.BytesIO(requests.get(r.image, timeout=60).content)).convert("RGB")
    img.thumbnail((a.max_side, a.max_side))
    w, h = img.size
    draw = ImageDraw.Draw(img)
    boxes = json.loads(r.boxes) if isinstance(r.boxes, str) else []
    for b in boxes:
        x1, y1 = b["left"] * w, b["top"] * h
        x2, y2 = x1 + b["width"] * w, y1 + b["height"] * h
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=max(3, w // 200))
    path = os.path.join(a.out, f"{r.photo_id}.png")
    img.save(path)
    print(path)
