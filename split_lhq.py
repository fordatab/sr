#!/usr/bin/env python
# split_lhq.py
import pathlib, random, shutil, math

SRC_DIR   = pathlib.Path("./lhq_256")          # source folder with 90k .png/.jpg
OUT_ROOT  = pathlib.Path("./lhq_split")        # will create train/val/test here
TRAIN_PCT = 0.80
VAL_PCT   = 0.10
SEED      = 42
COPY_MODE = "copy"   # "symlink" | "copy" – symlinks save disk space

random.seed(SEED)

# 1. gather all image paths
paths = sorted([p for p in SRC_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
n     = len(paths)
assert n > 0, f"no images found in {SRC_DIR}"

# 2. shuffle & compute split indices
random.shuffle(paths)
n_train = math.floor(n * TRAIN_PCT)
n_val   = math.floor(n * VAL_PCT)
splits  = {
    "train": paths[:n_train],
    "val"  : paths[n_train:n_train+n_val],
    "test" : paths[n_train+n_val:],
}

# 3. write split files & optionally make dirs
OUT_ROOT.mkdir(exist_ok=True)
for split, items in splits.items():
    # text file with relative paths (handy for custom loaders)
    (OUT_ROOT / f"{split}.txt").write_text("\n".join(str(p.relative_to(SRC_DIR)) for p in items))

    # dir-of-images layout (TorchVision datasets.ImageFolder-compatible)
    split_dir = OUT_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for src in items:
        dst = split_dir / src.name
        if COPY_MODE == "symlink":
            try:
                dst.symlink_to(src.resolve())
            except FileExistsError:
                pass
        else:  # COPY_MODE == "copy"
            shutil.copy2(src, dst)

print(f"Done.  Train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")
