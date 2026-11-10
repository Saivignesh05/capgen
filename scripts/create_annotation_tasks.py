#!/usr/bin/env python3
"""Build one annotation folder per clip, so two tool descriptions can be judged side by side.

    python make_annotation_tasks.py --clips ../devset \
        --tool1 run-a/tool_descriptions.json \
        --tool2 run-b/tool_descriptions.json \
        --out annotation-tasks

Each description JSON is what the extractor writes: a list of
{"audio_path": "ta/clips/x.wav", "tool_description": "..."} (a dict with an "items",
"descriptions" or "clips" list is accepted too). Clips are matched between the two files by
audio_path, and by file name when the paths were written against a different base.

Output, one folder per clip:

    annotation-tasks/
      index.csv                        every task and whether it has been filled in
      README.md                        what the fields mean
      Ak2_sample17/
        Ak2_sample17.wav               a COPY of the clip
        tool1_description.txt
        tool2_description.txt
        results.json

Re-running is safe. Clips and tool text are rewritten (they are derived), but a results.json
that someone has already edited is never touched -- annotator work is not something a second
run may destroy. The run reports what it skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

AUDIO_KEYS = ("audio_path", "path", "audio", "file", "clip")
TEXT_KEYS = ("tool_description", "description", "text")
LIST_KEYS = ("items", "descriptions", "clips", "rows", "data")

# Exactly the fields asked for. `best_tool_description` starts empty on purpose: pre-filling
# "tool1" would mean every skipped field silently votes for tool1. Use --prefill-best to get
# the literal template instead.
TEMPLATE = {
    "tool1_errors": 0,
    "tool2_errors": 0,
    "best_tool_description": "",
    "annotator description": "",
    "final_description": "",
}

README = """# Annotation tasks

One folder per clip. In each:

| File | What it is |
| --- | --- |
| `<clip>.wav` | a copy of the clip, so the folder is self-contained |
| `tool1_description.txt` | what tool 1 said about it |
| `tool2_description.txt` | what tool 2 said about it |
| `results.json` | the only file you edit |

## results.json

| Field | What to put |
| --- | --- |
| `tool1_errors` | how many statements in tool 1's description are wrong against the audio |
| `tool2_errors` | the same for tool 2 |
| `best_tool_description` | `tool1` or `tool2` -- whichever you would rather hand to a voice model |
| `annotator description` | what you hear, in your own words, in English |
| `final_description` | the description to keep: usually the better tool's text with its errors fixed and your own additions folded in |

Count an error only where the description contradicts what you can actually hear. A missing
detail is not an error -- note it in `annotator description` instead. If a claim is about
something you cannot judge (a number in Hz, a time in milliseconds), leave it alone: those come
from measurement and are more accurate than listening.

`index.csv` lists every task. The `status` column reads `template` until a results.json is
edited, then `filled`, so progress is visible without opening folders.
"""


def load(path: Path) -> dict[str, str]:
    """-> {audio_path: description}. Accepts a list, or a dict wrapping one."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for k in LIST_KEYS:
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:                                   # {"ta/clips/x.wav": "text", ...}
            if all(isinstance(v, str) for v in data.values()):
                return {str(k): v for k, v in data.items()}
            raise SystemExit(f"{path}: expected a list of records or a path->text map")
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list of records")
    out = {}
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: record {i} is not an object")
        key = next((row[k] for k in AUDIO_KEYS if row.get(k)), None)
        text = next((row[k] for k in TEXT_KEYS if row.get(k)), None)
        if key is None:
            raise SystemExit(f"{path}: record {i} has none of {AUDIO_KEYS}")
        if text:
            out[str(key)] = str(text)
    return out


def index_clips(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Every wav under the clip folder, by relative posix path and by file name."""
    by_rel, by_name = {}, {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() == ".wav":
            by_rel[p.relative_to(root).as_posix()] = p
            by_name.setdefault(p.name, []).append(p)
    return by_rel, by_name


def find_clip(audio_path: str, root: Path, by_rel: dict, by_name: dict) -> Path | None:
    """The clip an audio_path refers to. Tries the path as given, then relative to the clip
    folder and its parent, then the file name -- the extractor writes audio_path relative to
    the folder's PARENT ("ta/clips/x.wav"), which is not where the script is pointed."""
    p = Path(audio_path)
    if p.is_absolute() and p.is_file():
        return p
    rel = p.as_posix()
    if rel in by_rel:
        return by_rel[rel]
    for base in (root, root.parent):
        cand = base / p
        if cand.is_file():
            return cand
    parts = rel.split("/")
    for i in range(1, len(parts)):               # drop leading components one at a time
        tail = "/".join(parts[i:])
        if tail in by_rel:
            return by_rel[tail]
    hits = by_name.get(p.name, [])
    return hits[0] if len(hits) == 1 else None   # ambiguous name: refuse to guess


def folder_names(keys: list[str]) -> dict[str, str]:
    """A folder per clip, named by stem. Two clips with the same stem in different language
    folders would collide, so those get their parent directories folded into the name."""
    names, stems = {}, {}
    for k in keys:
        stems.setdefault(Path(k).stem, []).append(k)
    for stem, ks in stems.items():
        if len(ks) == 1:
            names[ks[0]] = stem
            continue
        for k in ks:                             # "ta/clips/x.wav" -> "ta__clips__x"
            parts = Path(k).with_suffix("").parts
            names[k] = "__".join(parts[-3:]) if len(parts) > 1 else stem
    return names


def is_filled(path: Path) -> bool:
    """True once someone has actually written something, so the file is left alone."""
    try:
        cur = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True                              # unreadable or hand-edited: do not touch
    if not isinstance(cur, dict):
        return True
    for k, blank in TEMPLATE.items():
        if cur.get(k, blank) != blank:
            return True
    return any(k not in TEMPLATE for k in cur)   # extra fields = someone edited it


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, type=Path, help="folder holding the clips")
    ap.add_argument("--tool1", required=True, type=Path)
    ap.add_argument("--tool2", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output folder (created)")
    ap.add_argument("--only-both", action="store_true",
                    help="skip clips that only one tool described")
    ap.add_argument("--prefill-best", action="store_true",
                    help='start best_tool_description at "tool1" instead of empty')
    ap.add_argument("--link", action="store_true", help="symlink clips instead of copying")
    a = ap.parse_args()

    if not a.clips.is_dir():
        raise SystemExit(f"--clips {a.clips} is not a directory")
    for f in (a.tool1, a.tool2):
        if not f.is_file():
            raise SystemExit(f"{f} not found")

    t1, t2 = load(a.tool1), load(a.tool2)
    by_rel, by_name = index_clips(a.clips)
    if not by_rel:
        raise SystemExit(f"no .wav files under {a.clips}")

    keys = sorted(set(t1) | set(t2))
    if a.only_both:
        keys = [k for k in keys if k in t1 and k in t2]
    if not keys:
        raise SystemExit("no clips in common between the two description files")

    template = dict(TEMPLATE)
    if a.prefill_best:
        template["best_tool_description"] = "tool1"

    names = folder_names(keys)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "README.md").write_text(README, encoding="utf-8")

    rows, made, kept, missing, one_sided = [], 0, 0, [], 0
    for k in keys:
        clip = find_clip(k, a.clips, by_rel, by_name)
        if clip is None:
            missing.append(k)
            continue
        if (k in t1) != (k in t2):
            one_sided += 1
        d = a.out / names[k]
        d.mkdir(exist_ok=True)

        dest = d / clip.name
        if not dest.exists() or dest.stat().st_size != clip.stat().st_size:
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            if a.link:
                dest.symlink_to(clip.resolve())
            else:
                shutil.copy2(clip, dest)

        for n, table in ((1, t1), (2, t2)):
            text = table.get(k, f"(tool{n} produced no description for this clip)")
            (d / f"tool{n}_description.txt").write_text(text.rstrip() + "\n", encoding="utf-8")

        res = d / "results.json"
        if res.exists() and is_filled(res):
            kept += 1
            status = "filled"
        else:
            res.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            made += 1
            status = "template"

        rows.append({"folder": names[k], "audio_path": k, "clip": str(clip),
                     "tool1": "yes" if k in t1 else "no",
                     "tool2": "yes" if k in t2 else "no", "status": status})

    with open(a.out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "audio_path", "clip", "tool1", "tool2",
                                          "status"])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} task folders in {a.out}")
    print(f"  tool1 described {len(t1)}, tool2 described {len(t2)}, "
          f"{len(set(t1) & set(t2))} in both")
    if one_sided:
        print(f"  {one_sided} clip(s) described by only one tool "
              f"(the other file says so; --only-both skips them)")
    print(f"  results.json: {made} written, {kept} already filled in and left untouched")
    if missing:
        print(f"  {len(missing)} description(s) had no matching .wav under {a.clips}:",
              file=sys.stderr)
        for k in missing[:10]:
            print(f"    {k}", file=sys.stderr)
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())