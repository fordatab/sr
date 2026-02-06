#!/usr/bin/env python3
"""
rowstrips.py  –  Build five 1×5 strips from a 5×5 SR grid, with a clean API.

Tile       : 256×256 px  |  Grid lines : 2-pixel black  |  Layout : 5 rows × 5 cols
Outputs    : row1_strip.png … row5_strip.png  (saved alongside the grid)

CLI EXAMPLES
------------
# Single default crop applied to every row
python rowstrips.py grid2.png --default 64 64 128 128

# Override specific rows (others use default)
python rowstrips.py grid2.png --default 64 64 128 128 \
                              --row 3 80 80 96 96 \
                              --row 5 0 0 200 200

# All row specs at once from JSON / YAML
python rowstrips.py grid2.png --config crops.yaml

PYTHON API EXAMPLE
------------------
from rowstrips import make_strips
make_strips(
    "grid2.png",
    row_specs={
        1: (64, 64, 128, 128),
        4: (80, 80, 96, 96)     # rows are 1-indexed
    },
    default=(32, 32, 192, 192)   # optional
)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Dict, Tuple, Sequence

try:
    import yaml  # optional, only if user provides YAML
except ModuleNotFoundError:
    yaml = None

from PIL import Image

# --- grid constants -------------------------------------------------------
CELL   = 256  # width & height of each tile
GAP    = 2    # grid line thickness
ROWS   = COLS = 5
# --------------------------------------------------------------------------

Coord  = Tuple[int, int, int, int]  # (cx, cy, cw, ch)


def _parse_number(tok: str, full: int) -> int:
    """
    Accept either an absolute pixel integer or a percentage string
    like '%0.25' (fraction of `full`). Returns an int pixel value.
    """
    if tok.startswith("%"):
        return int(float(tok[1:]) * full + 0.5)
    return int(tok)


def _parse_row_spec(tokens: Sequence[str]) -> Coord:
    if len(tokens) != 4:
        raise ValueError("Need 4 numbers per row: cx cy cw ch")
    cx = _parse_number(tokens[0], CELL)
    cy = _parse_number(tokens[1], CELL)
    cw = _parse_number(tokens[2], CELL)
    ch = _parse_number(tokens[3], CELL)
    return cx, cy, cw, ch


# --------------------------------------------------------------------------
#   CORE LOGIC — can be called from your own code
# --------------------------------------------------------------------------
def _crop_strip(grid: Image.Image, row_idx: int, spec: Coord) -> Image.Image:
    """Return one horizontal strip from row `row_idx` (0-based) using `spec`."""
    cx, cy, cw, ch = spec
    strip = Image.new("RGB", (COLS * cw, ch))

    for col in range(COLS):
        x0 = GAP + col * (CELL + GAP) + cx
        y0 = GAP + row_idx * (CELL + GAP) + cy
        tile = grid.crop((x0, y0, x0 + cw, y0 + ch))
        strip.paste(tile, (col * cw, 0))

    return strip


def make_strips(
        grid_path: str | Path,
        row_specs: Dict[int, Coord] | None = None,
        default: Coord = (64, 64, 128, 128),
        out_dir: str | Path | None = None,
) -> Dict[int, Path]:
    """
    Build five row-strips from `grid_path`.

    Parameters
    ----------
    grid_path  : path to the 5×5 grid PNG/JPG
    row_specs  : mapping {row_number (1-5): (cx,cy,cw,ch)}
    default    : fallback crop if a row is not specified
    out_dir    : where to save ( default: same folder as grid )

    Returns
    -------
    dict {row_number: Path_to_strip_png}
    """
    row_specs = row_specs or {}
    grid_img  = Image.open(grid_path)
    out_dir   = Path(out_dir or Path(grid_path).resolve().parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Path] = {}
    for row in range(1, ROWS + 1):
        spec = row_specs.get(row, default)
        strip = _crop_strip(grid_img, row - 1, spec)
        out_p = out_dir / f"row{row}_strip.png"
        strip.save(out_p)
        results[row] = out_p
    return results


# --------------------------------------------------------------------------
#                           COMMAND-LINE FRONT-END
# --------------------------------------------------------------------------
def _load_config(path: str | Path) -> Dict[int, Coord]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"Config file {p} not found")
    if p.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            sys.exit("PyYAML not installed -- cannot read YAML")
        data = yaml.safe_load(p.read_text())
    else:
        data = json.loads(p.read_text())

    # Expect mapping {"row1": [cx, cy, cw, ch], ...}
    specs: Dict[int, Coord] = {}
    for k, v in data.items():
        row = int(k.strip("row")).__int__()  # also accepts "1" or "row1"
        specs[row] = tuple(map(int, v))  # type: ignore[arg-type]
    return specs


def _main():
    ap = argparse.ArgumentParser(
        description="Create five 1×5 strips with per-row crops."
    )
    ap.add_argument("grid", help="source grid image (PNG/JPG)")
    ap.add_argument(
        "--default",
        nargs=4,
        metavar=("CX", "CY", "CW", "CH"),
        default=("64", "64", "128", "128"),
        help="fallback crop for rows without an explicit --row (px or %)",
    )
    ap.add_argument(
        "--row",
        action="append",
        nargs=5,
        metavar=("ROW", "CX", "CY", "CW", "CH"),
        help="override crop for a single row (1-5)",
    )
    ap.add_argument(
        "--config",
        help="JSON/YAML file with crops, e.g. {'row1':[cx,cy,cw,ch], ...}",
    )
    ap.add_argument("--out-dir", help="output folder (default: alongside grid)")
    args = ap.parse_args()

    # 1. start with global default
    default_spec = _parse_row_spec(args.default)

    # 2. load per-row overrides
    row_specs: Dict[int, Coord] = {}
    if args.config:
        row_specs.update(_load_config(args.config))

    if args.row:
        for row_tok, *crop_toks in args.row:
            row_idx = int(row_tok)
            row_specs[row_idx] = _parse_row_spec(crop_toks)

    paths = make_strips(
        grid_path=args.grid,
        row_specs=row_specs,
        default=default_spec,
        out_dir=args.out_dir,
    )
    for r, p in paths.items():
        print(f"✓ row {r}  →  {p.relative_to(Path.cwd())}")


if __name__ == "__main__":
    _main()
