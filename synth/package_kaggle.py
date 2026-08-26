"""Package the synthetic Hinglish set for Kaggle upload.

- Dedupes manifest.jsonl, assigns sentence-level splits (85/8/15 would exceed
  100 — actual: 77 train / 8 val / 15 test, hashed on sentence_id so no
  sentence leaks across splits and cut/tail variants follow their parent).
- Audits label balance; downsamples the "cut" kind in TRAIN only if the
  incomplete class exceeds 55%.
- Writes manifest.parquet and hinglish-synth.zip (FLAC already compressed,
  so the zip is stored, not deflated).

Run: python -m synth.package_kaggle
"""

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import polars as pl

OUT = Path(__file__).parent / "output"
SEED_SALT = "hinglish-v1"


def split_of(sentence_id: str) -> str:
    h = int(hashlib.md5((SEED_SALT + sentence_id).encode()).hexdigest(), 16) % 100
    if h < 15:
        return "test"
    if h < 23:
        return "val"
    return "train"


def main():
    rows, seen = [], set()
    for l in open(OUT / "manifest.jsonl", encoding="utf-8"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r["id"] in seen or not (OUT / r["path"]).exists():
            continue
        seen.add(r["id"])
        r["split"] = split_of(r["sentence_id"])
        rows.append(r)

    df = pl.DataFrame(rows)

    # balance train split: drop excess "cut" rows if incomplete > 55%
    train = df.filter(pl.col("split") == "train")
    n_inc = train.filter(pl.col("label") == 0).height
    if n_inc / max(train.height, 1) > 0.55:
        target_inc = int(train.filter(pl.col("label") == 1).height * 55 / 45)
        cuts = train.filter((pl.col("label") == 0) & (pl.col("kind") == "cut"))
        drop_n = min(cuts.height, n_inc - target_inc)
        if drop_n > 0:
            drop_ids = set(
                cuts.sort(pl.col("id").hash(seed=7)).head(drop_n)["id"].to_list()
            )
            df = df.filter(~pl.col("id").is_in(drop_ids))
            print(f"balanced train: dropped {drop_n} cut rows")

    df.write_parquet(OUT / "manifest.parquet")

    print(f"total clips: {df.height}, hours: {df['duration_s'].sum() / 3600:.2f}")
    for split in ("train", "val", "test"):
        d = df.filter(pl.col("split") == split)
        bal = d.filter(pl.col("label") == 1).height / max(d.height, 1)
        print(f"  {split:5s}: {d.height:5d} clips, complete={bal:.1%}, "
              f"kinds={dict(Counter(d['kind'].to_list()))}")
    print(f"  voices: {dict(Counter(df['voice'].to_list()))}")

    zpath = OUT / "hinglish-synth.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        z.write(OUT / "manifest.parquet", "manifest.parquet")
        for p in df["path"].to_list():
            z.write(OUT / p, p)
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
