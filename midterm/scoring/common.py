"""Shared helpers: logging, JSON Lines files, reading a round 0 run, and the bootstrap.

Every stage logs the same way (see setup_logging): to the console and to
<scores>/logs/<stage>.log, with timestamps, and with the full traceback of every failure. Per-item
failures are also written to <scores>/errors.jsonl, so they can be counted and grepped later.
"""
import datetime
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import traceback
from dataclasses import dataclass

LANGUAGES = ("hi", "ta")


# ---------------------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------------------

def setup_logging(scores_dir: str, stage: str) -> logging.Logger:
    """Log to the console and to <scores>/logs/<stage>.log. Uncaught exceptions are logged too."""
    os.makedirs(os.path.join(scores_dir, "logs"), exist_ok=True)
    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(os.path.join(scores_dir, "logs", f"{stage}.log"), encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def log_uncaught(kind, value, tb):
        logger.critical("Uncaught exception, the stage stops:\n%s", "".join(traceback.format_exception(kind, value, tb)))
    sys.excepthook = log_uncaught
    return logger


def log_environment(logger: logging.Logger) -> None:
    """Where and with what this process runs: the first thing to check when the HPC misbehaves."""
    logger.info("host=%s python=%s platform=%s slurm_job=%s cwd=%s", socket.gethostname(), platform.python_version(),
                platform.platform(), os.environ.get("SLURM_JOB_ID", "-"), os.getcwd())
    logger.info("CUDA_VISIBLE_DEVICES=%s HF_HUB_CACHE=%s HF_HUB_OFFLINE=%s TORCH_HOME=%s",
                os.environ.get("CUDA_VISIBLE_DEVICES", "-"), os.environ.get("HF_HUB_CACHE", "-"),
                os.environ.get("HF_HUB_OFFLINE", "-"), os.environ.get("TORCH_HOME", "-"))
    try:
        import torch
        logger.info("torch=%s cuda_available=%s cuda=%s", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            logger.info("gpu %d: %s, %.1f GB", index, properties.name, properties.total_memory / 1e9)
    except ImportError:
        logger.warning("torch is not installed in this environment")


def record_error(scores_dir: str, stage: str, item: str, error: BaseException, logger: logging.Logger) -> None:
    """Log one item's failure with its traceback, and append it to errors.jsonl. The stage goes on."""
    logger.error("%s FAILED: %r", item, error, exc_info=error)
    append_jsonl(os.path.join(scores_dir, "errors.jsonl"), {
        "time": now(), "stage": stage, "item": item, "error": repr(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    })


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------

def read_jsonl(path: str) -> list[dict]:
    """Every object in a JSON Lines file, or [] if it does not exist. A broken line raises with its number."""
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}, line {number}: not valid JSON ({error})") from error
    return rows


def append_jsonl(path: str, row: dict) -> None:
    """Append one line and flush, so a killed job loses at most the item in progress."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: str, data) -> None:
    with open(path + ".partial", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(path + ".partial", path)


def read_json_or_jsonl(path: str) -> list[dict]:
    """A .jsonl file (or a .jsonl that actually holds one JSON list), or a .json file holding a list.
    Empty or missing file -> []. Invalid JSON raises ValueError naming the file and the position."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    if path.endswith(".jsonl"):
        try:
            return read_jsonl(path)
        except ValueError as line_error:
            try:        # a hand-written file is often a pretty-printed list instead of one object per line
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                raise line_error from None
            if not isinstance(data, list):
                raise line_error from None
            return data
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error})") from error
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of objects")
    return data


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# A round 0 run, as the scoring stages see it
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    sample: int
    path: str           # absolute path of the generated wav


@dataclass(frozen=True)
class Item:
    """One devset clip and everything round 0 made from it."""
    clip_id: str
    language: str       # from the devset folder (hi/ta), not from the ASR
    asr_language: str | None
    clip_path: str
    transcript: str | None      # the text the generator was asked to say
    description: str | None     # the round 0 caption, also the generator's instruction
    samples: tuple[Sample, ...]


def folder_language(path: str) -> str | None:
    """hi or ta, from a folder named hi or ta somewhere in the path."""
    parts = os.path.normpath(path).replace("\\", "/").split("/")
    for part in reversed(parts[:-1]):
        if part in LANGUAGES:
            return part
    return None


def load_run(run_folder: str, logger: logging.Logger) -> list[Item]:
    """Read a round 0 run folder (round0/code/scripts/run.py) into Items, logging anything missing."""
    if not os.path.isfile(os.path.join(run_folder, "clips.jsonl")):
        raise FileNotFoundError(f"{run_folder} is not a round 0 run folder (no clips.jsonl)")
    clips = read_jsonl(os.path.join(run_folder, "clips.jsonl"))
    transcripts = {r["clip_id"]: r for r in read_jsonl(os.path.join(run_folder, "transcripts.jsonl"))}
    descriptions = {r["clip_id"]: r["description"] for r in read_jsonl(os.path.join(run_folder, "descriptions.jsonl"))}
    samples: dict[str, list[Sample]] = {}
    for record in read_jsonl(os.path.join(run_folder, "audio.jsonl")):
        path = os.path.join(run_folder, record["file"])
        samples.setdefault(record["clip_id"], []).append(Sample(record["sample"], path))

    items = []
    for clip in clips:
        clip_id = clip["clip_id"]
        transcript = transcripts.get(clip_id)
        language = folder_language(clip["path"]) or (transcript or {}).get("language")
        if language not in LANGUAGES:
            logger.warning("%s: language unknown (path %s, ASR %s); skipped", clip_id, clip["path"], (transcript or {}).get("language"))
            continue
        items.append(Item(
            clip_id=clip_id, language=language,
            asr_language=transcript["language"] if transcript else None,
            clip_path=clip["path"],
            transcript=transcript["transcript"] if transcript else None,
            description=descriptions.get(clip_id),
            samples=tuple(sorted(samples.get(clip_id, []), key=lambda s: s.sample))))

    missing = {
        "transcript": sum(i.transcript is None for i in items),
        "description": sum(i.description is None for i in items),
        "audio": sum(not i.samples for i in items),
    }
    logger.info("run %s: %d clips (%s), missing: %s", run_folder, len(items),
                ", ".join(f"{lang}={sum(i.language == lang for i in items)}" for lang in LANGUAGES), missing)
    lid_errors = [i.clip_id for i in items if i.asr_language and i.asr_language != i.language]
    if lid_errors:
        logger.warning("ASR language differs from the devset folder for %d clips: %s", len(lid_errors), lid_errors)
    return items


# ---------------------------------------------------------------------------------------------
# Bootstrap (Rime, Appendix A): replicate r, draw j picks position SHA-256("seed|bootstrap|measure|r|j") mod n
# ---------------------------------------------------------------------------------------------

def bootstrap_ci(by_item: dict[str, float], seed: int, measure: str, replicates: int,
                 statistic=lambda xs: sum(xs) / len(xs)) -> tuple[float, float] | None:
    """95% CI of `statistic` over items, with Rime's hash-based draws. None for fewer than 2 items.
    Items are put in the order of their sorted ids, as Rime does. With 1000 replicates the endpoints
    are the sorted values at positions 24 and 974."""
    values = [by_item[key] for key in sorted(by_item)]
    n = len(values)
    if n < 2:
        return None
    stats = []
    for r in range(1, replicates + 1):
        draw = []
        for j in range(n):
            digest = hashlib.sha256(f"{seed}|bootstrap|{measure}|{r}|{j}".encode("utf-8")).hexdigest()
            draw.append(values[int(digest, 16) % n])
        stats.append(statistic(draw))
    stats.sort()
    low = round(0.025 * replicates) - 1
    high = round(0.975 * replicates) - 1
    return stats[max(low, 0)], stats[min(high, replicates - 1)]
