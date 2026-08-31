"""Guards on the generated Kaggle notebooks.

Two failure modes that only surface once a Kaggle session is already running:
an empty `%%writefile` cell raises UsageError on the first run, and filtering
smart-turn v3.2 on long language names ("english") silently keeps zero rows:
v3.2 stores ISO-639-3 codes ("eng").
"""

import hashlib
import io
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import polars as pl
import pytest
import soundfile as sf

ROOT = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks" / "kaggle"


@pytest.fixture(scope="module")
def notebooks():
    subprocess.run([sys.executable, "-m", "tools.build_notebooks"],
                   cwd=ROOT, check=True, capture_output=True)
    return {name: nbformat.read(NB_DIR / f"{name}.ipynb", as_version=4)
            for name in ("01_data_prep", "02_train")}


def sources(nb, cell_type="code"):
    return [c.source for c in nb.cells if c.cell_type == cell_type]


def test_prep_filters_iso639_3_codes(notebooks):
    code = "\n".join(sources(notebooks["01_data_prep"]))
    assert '("eng", "hin")' in code
    assert '{"eng": "english", "hin": "hindi"}' in code
    # the long-name filter matched nothing and must not come back
    assert 'not in ("english", "hindi")' not in code


def test_prep_writes_normalised_language_names(notebooks):
    """The manifest must carry english/hindi: train.py slice_metrics and
    tools/aggregate_results.py both mask on the long names."""
    code = "\n".join(sources(notebooks["01_data_prep"]))
    assert 'LANG_MAP.get(code, code)' in code      # eng/hin -> long names
    assert '"language": lang,' in code


def test_prep_keeps_full_english_and_a_multilingual_tail(notebooks):
    """E6 prep: English effectively uncapped, other languages capped per
    (language, label) and train-only so val/test stay EN+HI."""
    code = "\n".join(sources(notebooks["01_data_prep"]))
    assert "EN_TRAIN_CAP_PER_LABEL = 33000" in code
    assert "OTHER_CAP_PER_LANG_LABEL = 850" in code
    # other languages keep their raw ISO code (LANG_MAP maps eng/hin only)
    assert 'lang = LANG_MAP.get(code, code)' in code
    # and never land in val: model selection must stay comparable
    assert 'if src == "test" and not core' in code


def test_prep_writes_audio_into_zip_shards(notebooks):
    """Loose files are the bug: >100k of them make Kaggle publish an empty
    _output_.zip and call the version a success."""
    code = "\n".join(sources(notebooks["01_data_prep"]))
    assert "SHARDS = 64" in code
    assert 'SHARD_DIR / f"shard_{i:02d}.zip"' in code
    assert 'mode="a"' in code and "ZIP_STORED" in code
    assert 'sf.write(buf, wav, SR, format="FLAC", subtype="PCM_16")' in code
    # the per-file write is what broke publishing
    assert "sub.mkdir" not in code
    assert "AUDIO_DIR" not in code
    # the manifest path and the arcname must be the same string, or notebook 02
    # reads the extracted tree at paths that do not exist
    assert 'arcname = f"audio/{rid[:2]}/{rid}.flac"' in code
    assert '"path": arcname,' in code


def test_prep_finalize_verifies_shards_against_the_manifest(notebooks):
    code = "\n".join(sources(notebooks["01_data_prep"]))
    assert 'glob("shard_*.zip")' in code
    assert "assert entries >= df.height" in code
    # opening + namelist, not a full CRC pass over ~15 GB
    assert "len(zf.namelist())" in code
    assert "zf.testzip()" not in code


def test_train_extracts_prep_shards_to_tmp(notebooks):
    code = "\n".join(sources(notebooks["02_train"]))
    assert 'shards = sorted(Path(PREP, "shards").glob("shard_*.zip"))' in code
    assert 'PREP_AUDIO = "/tmp/prep_audio"' in code
    assert "zf.extractall(PREP_AUDIO)" in code
    # manifest still comes from the mount, audio from the extraction root
    assert 'real = [(f"{PREP}/manifest.parquet", PREP_AUDIO)]' in code
    # legacy loose-file preps keep working
    assert "PREP_AUDIO = PREP" in code
    # resolve_mount still keys on the file that sits beside shards/
    assert 'marker: str = "manifest.parquet"' in code


def test_shard_roundtrip_feeds_the_dataset_unchanged(tmp_path):
    """End-to-end path fidelity: write FLAC into shard zips the way the prep
    notebook does, extract them, and read the result through the real
    load_manifests + TurnDataset. Guards arcname and the manifest `path`
    against drifting apart."""
    from tools.build_notebooks import PREP_MAIN
    from turn_detector.config import EXPERIMENTS
    from turn_detector.dataset import TurnDataset, load_manifests
    from turn_detector.features import N_SAMPLES

    SR, SHARDS = 16000, 64
    # the sharding rule under test is the notebook's, not a copy that drifted
    assert "int(hashlib.md5(rid.encode()).hexdigest()[:2], 16) % SHARDS" in PREP_MAIN

    prep = tmp_path / "prep"
    (prep / "shards").mkdir(parents=True)
    rng = np.random.default_rng(11)
    rows, zips = [], {}
    for i in range(3):
        rid = str(uuid.uuid4())
        wav = (0.2 * np.sin(np.linspace(0, 900, SR))).astype(np.float32)
        wav += 0.02 * rng.normal(0, 1, SR).astype(np.float32)
        arcname = f"audio/{rid[:2]}/{rid}.flac"
        idx = int(hashlib.md5(rid.encode()).hexdigest()[:2], 16) % SHARDS
        if idx not in zips:
            zips[idx] = zipfile.ZipFile(prep / "shards" / f"shard_{idx:02d}.zip",
                                        mode="a", compression=zipfile.ZIP_STORED)
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format="FLAC", subtype="PCM_16")
        zips[idx].writestr(arcname, buf.getvalue())
        rows.append({"id": rid, "path": arcname, "label": i % 2,
                     "language": "hindi", "split": "train"})
    for zf in zips.values():
        zf.close()
    pl.DataFrame(rows).write_parquet(prep / "manifest.parquet")

    # what notebook 02 does with the mount
    extracted = tmp_path / "prep_audio"
    shards = sorted((prep / "shards").glob("shard_*.zip"))
    assert shards, "no shard zips written"
    entries = 0
    for sp in shards:
        with zipfile.ZipFile(sp) as zf:
            zf.extractall(extracted)
            entries += len(zf.namelist())
    assert entries == len(rows)

    df = load_manifests([(str(prep / "manifest.parquet"), str(extracted))], "train")
    assert df.height == 3
    ds = TurnDataset(df, EXPERIMENTS["e1_baseline"], train=False)
    for i in range(len(ds)):
        wav, label, _ = ds[i]                     # resolves audio_root / path
        assert wav.shape == (N_SAMPLES,)
        assert label.item() in (0.0, 1.0)


def test_train_notebook_wires_the_distillation_teacher(notebooks):
    code = "\n".join(sources(notebooks["02_train"]))
    assert "TEACHER_FROM" in code
    assert "teacher_dir=teacher_dir" in code
    assert 'marker="ckpt_best.pt"' in code


def test_no_empty_writefile_cells(notebooks):
    for name, nb in notebooks.items():
        for i, src in enumerate(sources(nb)):
            if not src.lstrip().startswith("%%writefile"):
                continue
            body = src.split("\n", 1)[1] if "\n" in src else ""
            assert body.strip(), f"{name} cell {i}: empty %%writefile body"


def test_train_embeds_every_module(notebooks):
    from tools.build_notebooks import MODULES
    code = "\n".join(sources(notebooks["02_train"]))
    for mod in MODULES:
        assert f"%%writefile turn_detector/{mod}.py" in code


def test_train_has_time_budget_and_hard_resume(notebooks):
    code = "\n".join(sources(notebooks["02_train"]))
    assert "TIME_BUDGET_MIN = 630" in code
    assert "time_budget_minutes=TIME_BUDGET_MIN" in code
    assert "time_budget_reached" in code
    assert code.count("raise RuntimeError") >= 2      # missing ckpt + hash mismatch
    assert 'weights_only=False' in code


def test_kernelspec_present(notebooks):
    for name, nb in notebooks.items():
        ks = nb.metadata.get("kernelspec")
        assert ks and ks["name"] == "python3", f"{name} missing kernelspec"
        assert nb.metadata["language_info"]["name"] == "python"


def test_prep_pip_pins_datasets_below_4(notebooks):
    """datasets>=4 needs torchcodec to decode Audio; 3.x uses soundfile."""
    code = "\n".join(sources(notebooks["01_data_prep"]))
    assert '"datasets>=3.6,<4"' in code
