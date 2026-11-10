"""Round 0 captioner: Qwen3-Omni-30B-A3B-Captioner, frozen, no fine-tuning.

One .wav (<= 30 s) in, one English description out. The model takes audio only:
no text prompt, no tool descriptions, no transcript.

Usage (from the repo root, on a GPU node):
    python round0/code/captioner.py --audio_dir devset --out_dir round0/outputs/devset
    python round0/code/captioner.py --audio_dir devset --out_dir /tmp/test --limit 2   # quick test

Writes, in --out_dir:
    captions_hi.jsonl, captions_ta.jsonl   one line per clip
    run_info.json                          settings, versions, GPU, GPU-hours, commit
Re-running skips clips already in the output, so a crashed job can be resubmitted.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import librosa
import soundfile as sf
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

SAMPLE_RATE = 16000  # the captioner's audio encoder expects 16 kHz mono

# ---- settings: read from round0/settings.py (CAPTIONER dict) if present -------------
DEFAULTS = {
    "model_id": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    "model_revision": "a2bd106cbf527db5676e79662674da22b0545ec0",
    "model_dir": "/scratch/b112301031-akkoorath/qwen3-omni-captioner",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "do_sample": False,          # greedy: same input -> same description, every run
    "max_new_tokens": 1024,
    "seed": 0,
    "max_clip_seconds": 30.0,
    "word_limit": 300,           # Rime cuts descriptions at 300 words before grading
}
ROUND0_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROUND0_DIR)
try:
    from settings import CAPTIONER as _USER
except Exception:
    _USER = {}
SETTINGS = {**DEFAULTS, **_USER}

LANGS = ("hi", "ta")


def language_of(path):
    """Language = the 'hi' or 'ta' folder somewhere in the path (devset/hi/clips/x.wav)."""
    parts = os.path.normpath(path).split(os.sep)
    for p in reversed(parts):
        if p in LANGS:
            return p
    return "unknown"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROUND0_DIR,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


class Captioner:
    def __init__(self, s):
        self.s = s
        torch.manual_seed(s["seed"])
        # Processor first: it loads in seconds, so a setup error shows before the 4-minute model load.
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(s["model_dir"])
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            s["model_dir"],
            dtype=getattr(torch, s["dtype"]),
            device_map="auto",
            attn_implementation=s["attn_implementation"],
        ).eval()

    @torch.no_grad()
    def describe(self, wav_path):
        conversation = [{"role": "user", "content": [{"type": "audio", "audio": wav_path}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        inputs = self.processor(text=text, audio=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        out = self.model.generate(
            **inputs,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=self.s["max_new_tokens"],
            thinker_do_sample=self.s["do_sample"],
        )
        ids = out[0] if isinstance(out, tuple) else out
        seqs = ids.sequences if hasattr(ids, "sequences") else ids
        return self.processor.batch_decode(
            seqs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="searched recursively for .wav; language from hi/ or ta/ in the path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (testing)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True))
    if args.limit:
        clips = clips[: args.limit]
    print(f"{len(clips)} clips found under {args.audio_dir}")

    out_paths = {lang: os.path.join(args.out_dir, f"captions_{lang}.jsonl") for lang in LANGS + ("unknown",)}
    done = set()
    for p in out_paths.values():
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                done |= {json.loads(l)["path"] for l in f if l.strip()}
    todo = [c for c in clips if os.path.abspath(c) not in done]
    print(f"{len(clips) - len(todo)} already done, {len(todo)} to caption")

    t_load = time.time()
    cap = Captioner(SETTINGS)
    load_s = time.time() - t_load
    print(f"model loaded in {load_s:.0f}s")

    t_run = time.time()
    for i, path in enumerate(todo, 1):
        lang = language_of(path)
        dur = sf.info(path).duration
        if dur > SETTINGS["max_clip_seconds"]:
            print(f"WARNING: {path} is {dur:.1f}s (> {SETTINGS['max_clip_seconds']}s)")
        t0 = time.time()
        desc = cap.describe(path)
        secs = time.time() - t0
        words = desc.split()
        rec = {
            "clip_id": os.path.splitext(os.path.basename(path))[0],
            "language": lang,
            "path": os.path.abspath(path),
            "duration_s": round(dur, 2),
            "description": desc,
            "description_300": " ".join(words[: SETTINGS["word_limit"]]),
            "n_words": len(words),
            "over_word_limit": len(words) > SETTINGS["word_limit"],
            "seconds": round(secs, 2),
        }
        with open(out_paths[lang], "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {lang}/{rec['clip_id']}  {secs:.1f}s  {len(words)} words")

    run_s = time.time() - t_run
    n_gpu = torch.cuda.device_count()
    info = {
        "settings": SETTINGS,
        "code_commit": git_commit(),
        "audio_dir": os.path.abspath(args.audio_dir),
        "n_clips_this_run": len(todo),
        "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpu)],
        "model_load_seconds": round(load_s, 1),
        "caption_seconds": round(run_s, 1),
        "gpu_hours_this_run": round((load_s + run_s) * n_gpu / 3600, 4),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        },
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
