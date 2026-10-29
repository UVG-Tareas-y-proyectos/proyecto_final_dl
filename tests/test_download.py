import random
import runpy
from pathlib import Path

import pytest

downloader = runpy.run_path(str(Path(__file__).parents[1] / "scripts/download_pilot.py"))


def test_sampling_remains_reproducible_and_full_mode_keeps_unknown_metadata():
    entries = [dict(path=f"uuid___leaf{i}_{angle}.jpg", type="blob", sha="unused")
               for i in range(5) for angle in range(3)]
    entries.append(dict(path="unknown.jpg", type="blob", sha="unused"))
    metadata = {f"leaf{i}_{angle}": [f"label:::{i}"] for i in range(5) for angle in range(3)}
    select = downloader["select_images"]
    a = select(entries, metadata, "label", random.Random(42), 3, 2)
    b = select(list(reversed(entries)), metadata, "label", random.Random(42), 3, 2)
    assert a == b and len(a) == 6
    assert len(select(entries, metadata, "label", random.Random(42), all_leaves=True, all_images=True)) == 16


def test_corrupt_download_rejected_before_save():
    content = b"hello\n"
    sha = "ce013625030ba8dba906f756967f9e9ca394464a"
    downloader["verify_blob"](content, sha)
    with pytest.raises(ValueError, match="no coincide"):
        downloader["verify_blob"](b"truncated", sha)
