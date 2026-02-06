#!/usr/bin/env python3
"""
make_five_strips_yaml.py  –  build LR / tiny / small / big / HR 1×8 strips
                             using per-column crop specs from a YAML or JSON
                             file (keys col1 … col8).

Grid layout (identical to the earlier script)
--------------------------------------------
row 0 : LR              <- use for strip_lr
row 1 : model output    <- use for strip_tiny / _small / _big
row 2 : HR              <- use for strip_hr

Tile geometry : 256×256 px, 1-pixel black borders
"""

from pathlib import Path
from typing import Dict, Tuple, List
import argparse, json, sys

try:
    import yaml  # only needed if config is YAML
except ModuleNotFoundError:
    yaml = None

from PIL import Image

CELL, GAP = 256, 1
COLS      = 8
Coord     = Tuple[int, int, int, int]   # (cx, cy, cw, ch)


# ───────────────────────────── utilities ──────────────────────────────
def load_config(path: str | Path) -> Dict[int, Coord]:
    """
    Accept YAML or JSON:
        col1: [cx, cy, cw, ch]
        …
    Returns {1: (cx,cy,cw,ch), …}
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"[config] file not found: {p}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            sys.exit("PyYAML not installed → pip install pyyaml")
        data = yaml.safe_load(p.read_text())
    else:
        data = json.loads(p.read_text())

    out: Dict[int, Coord] = {}
    for k, v in data.items():
        col_idx = int(k.lstrip("col"))       # accepts "col3" or "3"
        out[col_idx] = tuple(map(int, v))    # type: ignore[arg-type]
    return out


def crop_tile_strip(img: Image.Image,
                    src_row: int,
                    col_rects: List[Coord]) -> Image.Image:
    """
    Build one horizontal strip by applying *different* crop rects
    to each of the 8 tiles in `src_row`.
    """
    # total output width = Σ crop widths
    out_w = sum(r[2] for r in col_rects)
    out_h = col_rects[0][3]                 # assume same height for all crops
    strip = Image.new("RGB", (out_w, out_h))

    x_dst = 0
    for c, rect in enumerate(col_rects):
        cx, cy, cw, ch = rect
        x_src = GAP + c * (CELL + GAP) + cx
        y_src = GAP + src_row * (CELL + GAP) + cy
        tile = img.crop((x_src, y_src, x_src + cw, y_src + ch))
        strip.paste(tile, (x_dst, 0))
        x_dst += cw
    return strip


def save_strip(tag: str, strip: Image.Image):
    out = Path(f"strip_{tag}.png")
    strip.save(out)
    print(f"✓ wrote {out}")


# ─────────────────────────────── main ─────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Create LR / tiny / small / big / HR strips with per-column crops"
    )
    ap.add_argument("--tiny",  required=True, help="tiny_grid.png")
    ap.add_argument("--small", required=True, help="small_grid.png")
    ap.add_argument("--big",   required=True, help="big_grid.png")
    ap.add_argument("--config", required=True,
                    help="YAML or JSON mapping col1…col8 → [cx,cy,cw,ch]")
    ap.add_argument("--default", nargs=4, metavar=("CX","CY","CW","CH"),
                    type=int, default=[64,64,128,128],
                    help="fallback crop if a column is missing in the config")
    args = ap.parse_args()

    default_rect: Coord = tuple(args.default)  # type: ignore[arg-type]
    col_cfg             = load_config(args.config)

    # Build list of 8 rects in order, falling back to default if needed
    col_rects: List[Coord] = [
        col_cfg.get(i, default_rect) for i in range(1, COLS + 1)
    ]

    # --- load images once ---
    tiny_img   = Image.open(args.tiny)
    small_img  = Image.open(args.small)
    big_img    = Image.open(args.big)

    # --- make & save 5 strips ---
    save_strip("lr",    crop_tile_strip(tiny_img, 0, col_rects))  # any grid works
    save_strip("tiny",  crop_tile_strip(tiny_img, 1, col_rects))
    save_strip("small", crop_tile_strip(small_img, 1, col_rects))
    save_strip("big",   crop_tile_strip(big_img, 1, col_rects))
    save_strip("hr",    crop_tile_strip(tiny_img, 2, col_rects))

if __name__ == "__main__":
    main()
