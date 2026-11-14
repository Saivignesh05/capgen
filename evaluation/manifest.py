"""The run manifest: what exactly produced a set of scores (docs/evaluation_plan.md, ground rule 4).

Every script that writes scores also writes a manifest next to them: the code commit, the SHA-256
of the settings file, of every input file and of every checkpoint used, and when it ran.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

CHUNK_BYTES = 1024 * 1024


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    script: str                         # which script wrote the scores
    started_utc: str
    code_commit: str                    # "<commit>" or "<commit>+uncommitted-changes"; "unknown" outside git
    settings_sha256: str
    input_sha256: dict[str, str]        # {path: SHA-256} of every input file (rubric, run records)
    checkpoint_sha256: dict[str, str]   # {model name: SHA-256 of its folder}; empty if no model was used
    notes: dict[str, str]               # anything else that defines the run, e.g. the grader implementation


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def folder_sha256(folder: str) -> str:
    """One SHA-256 for a model folder: over the relative path and the SHA-256 of every file in it,
    in sorted order. Folders named .cache, __pycache__ or .git are skipped: they are not part of the model."""
    digest = hashlib.sha256()
    for root, directories, files in os.walk(folder):
        directories[:] = sorted(d for d in directories if d not in (".cache", "__pycache__", ".git"))
        for name in sorted(files):
            path = os.path.join(root, name)
            relative = os.path.relpath(path, folder).replace(os.sep, "/")
            digest.update(f"{relative}\n{file_sha256(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def code_commit(repo_dir: str) -> str:
    """The commit of the code, marked if there are uncommitted changes."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    if dirty:
        return commit + "+uncommitted-changes"
    return commit


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(manifest: RunManifest, path: str) -> None:
    with open(path + ".partial", "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(path + ".partial", path)
