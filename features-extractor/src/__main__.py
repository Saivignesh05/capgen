"""python -m src <command>        (from the features-extractor folder; no install needed)

  fetch-models          download the frozen models into models/ (needs internet)
  run DIR               every .wav under DIR, as a NEW run (nothing is overwritten):
                          ../features-extractor-results/<DIR name>/<run>/features/<clip>.json
                          ../features-extractor-results/<DIR name>/<run>/tool_descriptions.json
                          ../features-extractor-results/<DIR name>/<run>/tool_descriptions.rejected.json
                          ../features-extractor-results/<DIR name>/<run>/run.json   (when, config, status)
                        <run> = start time, e.g. 2026-11-08_23-20-05 (or --run-id NAME).
                        Then copies the run's tool_descriptions.json into DIR (copy_descriptions.py).
                        audio_path is relative to DIR's parent ("hi/clip_001.wav").
                        --resume RUN: continue that run (reuses its finished clips, redoes the last one).
                        --shard I/N: every N-th file from I (SLURM arrays, all with the same --run-id).
                        --results PATH: another base folder instead of ../features-extractor-results/<DIR name>.
  runs DIR              list the runs of DIR
  merge DIR --run RUN   after shards: build the run's tool_descriptions.json, copy into DIR
  describe DIR --run RUN  same rebuild, e.g. after editing describe.py (no models needed)
  clip FILE             print one clip's features
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent                 # features-extractor/
DEFAULT_CONFIG = ROOT / "config.yaml"
OUT = "tool_descriptions"


def results_dir(folder, results=None) -> Path:
    """Base folder holding the runs of `folder`. Default: features-extractor-results/<folder name>,
    next to the project folder."""
    return Path(results).resolve() if results else ROOT.parent / "features-extractor-results" / Path(folder).resolve().name


def list_runs(base: Path) -> list[dict]:
    """Runs under base, oldest first: [{"run", "started", "status", "clips", ...}]."""
    out = []
    for d in sorted(p for p in base.glob("*") if p.is_dir()):
        parts = sorted(d.glob("run.part-*-of-*.json"))       # a sharded run not merged yet
        info = read_json(d / "run.json") or (read_json(parts[0]) if parts else None) or {}
        out.append({"run": d.name, **info})
    return sorted(out, key=lambda r: (r.get("started", ""), r["run"]))


def find_run(base: Path, run_id: str) -> Path:
    runs = [r["run"] for r in list_runs(base)] if base.exists() else []
    if run_id == "latest" and runs:
        run_id = runs[-1]
    if run_id not in runs:
        raise SystemExit(f"run {run_id!r} not found in {base}; runs: {', '.join(runs) or 'none'}")
    return base / run_id


def wavs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")


def write_json(path: Path, obj) -> None:
    """Temp file + rename: a killed job never leaves a half-written file. Indented."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_outputs(res: Path, feats: list[dict], suffix: str = "") -> int:
    from .describe import describe
    feats = sorted(feats, key=lambda f: f["clip"]["path"])
    rows = [{"audio_path": f["clip"]["path"], "tool_description": describe(f)} for f in feats if f["status"] == "ok"]
    write_json(res / f"{OUT}{suffix}.json", rows)
    write_json(res / f"{OUT}.rejected{suffix}.json",
               [{"audio_path": f["clip"]["path"], "checks": f["checks"]} for f in feats if f["status"] != "ok"])
    return len(rows)


def copy_to(res: Path, folder: Path) -> Path:
    """Runs copy_descriptions.py (the same script you can run by hand later)."""
    spec = importlib.util.spec_from_file_location("copy_descriptions", ROOT / "copy_descriptions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.copy_descriptions(res / f"{OUT}.json", folder)


def fmt(sec: float) -> str:
    sec = max(0.0, sec)
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(round(sec)), 60)
    return f"{m}m {s:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"


def run(config, folder, results=None, shard=(0, 1), resume: str | None = None, run_id: str | None = None,
        log=print) -> dict:
    from . import __version__
    from .pipeline import Extractor

    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    base = results_dir(folder, results)
    if resume:
        res = find_run(base, resume)
    else:
        name = run_id or time.strftime("%Y-%m-%d_%H-%M-%S")
        res, k = base / name, 2
        while run_id is None and res.exists():           # two runs in the same second
            res, k = base / f"{name}-{k}", k + 1
    i, n = shard
    suffix = f".part-{i}-of-{n}" if n > 1 else ""
    files = wavs(folder)[i::n]
    t_load = time.perf_counter()
    ex = Extractor(config)
    load_s = time.perf_counter() - t_load

    def fpath(wav: Path) -> Path:
        return res / "features" / wav.relative_to(folder).with_suffix(".json")

    done: dict[int, dict] = {}
    if resume:
        for k, wav in enumerate(files):
            f = read_json(fpath(wav))
            if f and f.get("run", {}).get("config_sha") == ex.config_sha and f["run"].get("version") == __version__:
                done[k] = f
        if done:
            last = max(done)
            done.pop(last)                                  # redo the last finished clip, then continue
            log(f"resuming: {len(done)} clips reused; redoing {files[last].name} and continuing")
    todo = len(files) - len(done)
    log(f"run {res.name}: {todo} clips to process -> {res}  (device {ex.device}, models loaded in {fmt(load_s)})")
    meta_path = res / (f"run{suffix}.json")
    meta = {**(read_json(meta_path) or {}), "run": res.name, "folder": str(folder), "config": str(config)
            if isinstance(config, (str, Path)) else "dict", "config_sha": ex.config_sha, "version": __version__,
            "device": ex.device, "shard": f"{i}/{n}", "status": "running"}
    meta.setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(meta_path, meta)
    status = "stopped"

    results_, steps, t0, computed, audio_s = dict(done), defaultdict(float), time.perf_counter(), 0, 0.0
    try:
        for k, wav in enumerate(files):
            if k in done:
                continue
            rel = wav.relative_to(folder.parent).as_posix()
            try:
                f = ex.extract(wav)
            except Exception as exc:        # one unreadable or odd clip must not lose the whole run
                log(f"[{k + 1}/{len(files)}] {rel} FAILED {type(exc).__name__}: {exc}")
                f = {"clip": {"path": rel}, "status": "failed", "global": {}, "local": {},
                     "checks": [{"check": "extract", "message": f"{type(exc).__name__}: {exc}", "reject": True}],
                     "run": {"version": __version__, "config_sha": ex.config_sha, "device": ex.device,
                             "models": ex.models_used, "seconds": {}}}
                write_json(fpath(wav), f)
                results_[k] = f
                continue
            f["clip"]["path"] = rel
            write_json(fpath(wav), f)
            results_[k], computed = f, computed + 1
            audio_s += f["clip"].get("duration_s") or 0.0
            for s, sec in f["run"]["seconds"].items():
                steps[s] += sec
            elapsed = time.perf_counter() - t0
            log(f"[{k + 1}/{len(files)}] {rel} {f['status']} {f['run']['seconds'].get('total', 0):.2f}s"
                f"  | elapsed {fmt(elapsed)}, left ~{fmt(elapsed / computed * (todo - computed))}")
        status = "finished"
    finally:                                                # also on Ctrl-C / SLURM time limit
        written = write_outputs(res, list(results_.values()), suffix)
        copied = copy_to(res, folder) if n == 1 else None
        wall = time.perf_counter() - t0
        log(f"\n{written} descriptions, {len(results_) - written} rejected ({computed} computed, {len(done)} reused)")
        log(f"time: {fmt(wall)} for {computed} clips ({fmt(audio_s)} of audio) + {fmt(load_s)} model loading")
        if computed:
            spent = steps.pop("total", 0.0) or 1e-9        # the clips that finished, not the wall clock
            log(f"      {spent / computed:.2f} s per clip, {audio_s / spent:.1f}x real time")
            for s, sec in sorted(steps.items(), key=lambda kv: -kv[1]):
                log(f"        {s:<10}{sec / computed:7.3f} s/clip  {100 * sec / spent:5.1f}%")
        write_json(meta_path, {**meta, "status": status, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "clips": len(files), "descriptions": written, "rejected": len(results_) - written})
        log(f"copied to {copied}" if copied else
            f"shard done; after all shards: python -m src merge {folder} --run {res.name}")
        if status == "stopped":
            log(f"continue with: python -m src run {folder} --resume {res.name}")
    return {"run": res.name, "written": written, "rejected": len(results_) - written, "computed": computed,
            "reused": len(done), "results": str(res), "copied_to": str(copied) if copied else None}


def rebuild(folder, run_id: str, results=None) -> int:
    """A run's tool_descriptions.json from its saved features (clips whose .wav still exists);
    removes shard parts; copies into the folder."""
    folder = Path(folder).resolve()
    res = find_run(results_dir(folder, results), run_id)
    present = {w.relative_to(folder.parent).as_posix() for w in wavs(folder)}
    feats = [f for f in (read_json(p) for p in sorted((res / "features").rglob("*.json")))
             if f and f.get("clip", {}).get("path") in present]
    written = write_outputs(res, feats)
    for p in res.glob(f"{OUT}*.part-*-of-*.json"):
        p.unlink()
    parts = sorted(res.glob("run.part-*-of-*.json"))
    if parts:                                               # one run.json for the whole sharded run
        metas = [read_json(p) or {} for p in parts]
        write_json(res / "run.json", {**metas[0], "shard": f"{len(parts)} shards merged",
                                      "status": "finished" if all(m.get("status") == "finished" for m in metas)
                                      else "incomplete", "started": min(m.get("started", "") for m in metas),
                                      "finished": max(m.get("finished", "") for m in metas), "descriptions": written})
        for p in parts:
            p.unlink()
    copy_to(res, folder)
    return written


def _stop(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-models").add_argument("--models-dir", default=str(ROOT / "models"))
    r = sub.add_parser("run")
    r.add_argument("folder")
    r.add_argument("--config", default=str(DEFAULT_CONFIG))
    r.add_argument("--results")
    r.add_argument("--shard", default="0/1", help="I/N")
    r.add_argument("--resume", metavar="RUN", help="continue this run ('latest' = the newest)")
    r.add_argument("--run-id", help="name for a new run (default: start time); shards of one job share it")
    for name in ("merge", "describe"):
        p = sub.add_parser(name)
        p.add_argument("folder")
        p.add_argument("--run", required=True, help="run name ('latest' = the newest)")
        p.add_argument("--results")
    p = sub.add_parser("runs")
    p.add_argument("folder")
    p.add_argument("--results")
    c = sub.add_parser("clip")
    c.add_argument("file")
    c.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = ap.parse_args(argv)

    if a.cmd == "fetch-models":
        from .models import fetch
        lock = fetch(a.models_dir)
        print(json.dumps({k: v.get("revision") or v.get("url") for k, v in lock.items()}, indent=2))
    elif a.cmd == "run":
        i, n = (int(v) for v in a.shard.split("/"))
        if not 0 <= i < n:
            ap.error("--shard I/N needs 0 <= I < N")
        signal.signal(signal.SIGTERM, _stop)                # SLURM sends SIGTERM at the time limit
        try:
            s = run(a.config, a.folder, a.results, (i, n), a.resume, a.run_id,
                    log=lambda m: print(m, file=sys.stderr, flush=True))
        except KeyboardInterrupt:
            print("stopped; outputs written for the clips done so far", file=sys.stderr)
            return 130
        print(json.dumps(s))
    elif a.cmd in ("merge", "describe"):
        print(f"{rebuild(a.folder, a.run, a.results)} descriptions -> {Path(a.folder).resolve() / (OUT + '.json')}")
    elif a.cmd == "runs":
        base = results_dir(a.folder, a.results)
        for r in list_runs(base) if base.exists() else []:
            print(f"{r['run']:<24} {r.get('status', '?'):<9} started {r.get('started', '?')}  "
                  f"{r.get('descriptions', '?')} descriptions, {r.get('rejected', '?')} rejected")
    elif a.cmd == "clip":
        from .pipeline import Extractor
        print(json.dumps(Extractor(a.config).extract(a.file), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
