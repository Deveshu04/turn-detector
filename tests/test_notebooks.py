"""Guards on the generated Kaggle notebooks.

These two failure modes each cost a full Kaggle session before they show up:
an empty `%%writefile` cell raises UsageError on the first run, and filtering
smart-turn v3.2 on long language names ("english") silently keeps zero rows —
v3.2 stores ISO-639-3 codes ("eng").
"""

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

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
    # and never the old long-name filter, which matched nothing
    assert 'not in ("english", "hindi")' not in code


def test_prep_writes_normalised_language_names(notebooks):
    """The manifest must carry english/hindi: train.py slice_metrics and
    tools/aggregate_results.py both mask on the long names."""
    code = "\n".join(sources(notebooks["01_data_prep"]))
    assert 'LANG_MAP[row["language"]]' in code
    assert '"language": lang,' in code


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
