#!/usr/bin/env python3
"""Copy a run's tool_descriptions.json into any directory (overwrites; atomic).

    python copy_descriptions.py ../features-extractor-results/hi /data/hi
    python copy_descriptions.py ../features-extractor-results/hi/tool_descriptions.json /data/hi

`run`, `merge` and `describe` call this automatically for the folder they processed.
Standalone: needs only Python.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

NAME = "tool_descriptions.json"


def copy_descriptions(source, target_dir) -> Path:
    src = Path(source)
    if src.is_dir():
        src = src / NAME
    rows = json.loads(src.read_text(encoding="utf-8"))           # refuse to copy a broken file
    if not isinstance(rows, list) or not all({"audio_path", "tool_description"} <= set(r) for r in rows):
        raise ValueError(f"{src} is not a list of {{audio_path, tool_description}}")
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / (NAME + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(target / NAME)
    return target / NAME


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(copy_descriptions(sys.argv[1], sys.argv[2]))
