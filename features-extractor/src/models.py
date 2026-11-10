"""Neural models, frozen.

`fetch()` (on a machine with internet) downloads each model at one fixed revision into
models/<name>/ and writes models/models.lock.json: source, exact revision and the sha256 of
every file. The first fetch pins the then-current commit; every later fetch downloads that
same commit. At run time `local_model()` re-hashes the files against the lock and refuses to
run on anything else, and loading never touches the network. Copy models/ (with the lock)
to the cluster.

Components (pick with `impl:` in config.yaml; add a class + one dict entry to extend):
    vad       silero (Silero VAD v5.1.2, TorchScript, CPU) | energy (no model)
    language  whisper (openai/whisper-small language head) | folder (hi/ ta/ in the path)
    events    ast (MIT AST fine-tuned on AudioSet)          | none (omit the section)
    syllables phonemes (wav2vec2 XLSR-53 phoneme recogniser: vowels = syllables) | envelope (no model)
    speakers  wavlm (WavLM-SV x-vectors per phrase, clustered) | none (omit the section)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOCK = "models.lock.json"
SOURCES = {
    "silero_vad": {"url": "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.jit"},
    "whisper-small": {"repo": "openai/whisper-small"},
    "ast-audioset": {"repo": "MIT/ast-finetuned-audioset-10-10-0.4593"},
    "phoneme-xlsr": {"repo": "facebook/wav2vec2-xlsr-53-espeak-cv-ft"},
    "speaker-xvector": {"repo": "microsoft/wavlm-base-plus-sv"},
    "whisper-large-v3": {"repo": "openai/whisper-large-v3"},     # transcription, ~3 GB
}
# Licences, all permissive and none gated: Silero VAD MIT, Whisper MIT, AST BSD-3-Clause,
# wav2vec2-XLSR phonemes Apache-2.0, WavLM-SV MIT (Microsoft UniSpeech).


class ModelError(Exception):
    pass


# ------------------------------------------------------------------ freezing


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hash_dir(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): sha256(p) for p in sorted(d.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(d).parts}


def fetch(models_dir, names=None, log=print) -> dict:
    """Download (or re-download) models at their locked revision; create the lock entry
    the first time. Raises ModelError if a re-download differs from the lock."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    lock_path = models_dir / LOCK
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    for name in names or SOURCES:
        src, old, dest = SOURCES[name], lock.get(name), models_dir / name
        dest.mkdir(exist_ok=True)
        if "url" in src:
            target = dest / Path(src["url"]).name
            if not target.exists():
                log(f"{name}: downloading {src['url']}")
                urllib.request.urlretrieve(src["url"], target)
            entry = {"url": src["url"]}
        else:
            from huggingface_hub import HfApi, snapshot_download
            api = HfApi()
            rev = old["revision"] if old else api.model_info(src["repo"]).sha
            files = [s.rfilename for s in api.model_info(src["repo"], revision=rev).siblings]
            weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
            want = [f for f in files if "/" not in f and (f == weights or f.endswith((".json", ".txt")))]
            log(f"{name}: {src['repo']}@{rev[:10]} ({len(want)} files)")
            snapshot_download(src["repo"], revision=rev, local_dir=dest, allow_patterns=want)
            entry = {"repo": src["repo"], "revision": rev}
        entry["files"] = _hash_dir(dest)
        if old and old["files"] != entry["files"]:
            raise ModelError(f"{name}: downloaded files differ from {lock_path}; refusing to change a frozen model")
        lock[name] = entry
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def local_model(models_dir, name: str, verify: bool = True) -> Path:
    """Path of a frozen model, after checking every file against the lock."""
    models_dir = Path(models_dir)
    lock_path, dest = models_dir / LOCK, models_dir / name
    hint = "run `python -m src fetch-models` where there is internet and copy models/ here"
    if not lock_path.exists():
        raise ModelError(f"{lock_path} not found: {hint}")
    entry = json.loads(lock_path.read_text()).get(name)
    if entry is None:
        raise ModelError(f"{name} is not in {lock_path}: {hint}")
    if verify:
        for rel, digest in entry["files"].items():
            p = dest / rel
            if not p.is_file():
                raise ModelError(f"{name}: missing {p}: {hint}")
            if sha256(p) != digest:
                raise ModelError(f"{name}: {p} does not match the lock (changed or corrupted)")
    return dest


def lock_info(models_dir) -> dict:
    p = Path(models_dir) / LOCK
    if not p.exists():
        return {}
    return {k: v.get("revision") or v.get("url") for k, v in json.loads(p.read_text()).items()}


def _hf_offline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def fix_weight_norm(model, path) -> int:
    """Copy weight-norm tensors that `from_pretrained` leaves randomly initialised.

    PyTorch 2.1 renamed weight-norm parameters: `<conv>.weight_g` / `.weight_v` became
    `<conv>.parametrizations.weight.original0` / `original1`. Checkpoints saved before that —
    wav2vec2 and WavLM among them — then load with those tensors random, which transformers
    reports as "were not used" plus "newly initialized ... You should probably TRAIN this
    model" and which quietly ruins the output. Copy them across by name. Returns how many
    tensors were copied; a no-op (same values) on a transformers version that already maps them."""
    import torch
    sd, f = {}, path / "model.safetensors"
    if f.is_file():
        from safetensors.torch import load_file
        sd = load_file(str(f))
    elif (f := path / "pytorch_model.bin").is_file():
        sd = torch.load(str(f), map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    n = 0
    for key, tensor in sd.items():
        for old, new in ((".weight_g", ".parametrizations.weight.original0"),
                         (".weight_v", ".parametrizations.weight.original1")):
            if key.endswith(old) and (target := params.get(key.replace(old, new))) is not None:
                if tuple(target.shape) == tuple(tensor.shape):
                    with torch.no_grad():
                        target.copy_(tensor)
                    n += 1
    return n


@contextlib.contextmanager
def quiet_load():
    """Hide transformers' load warnings about the renamed weight-norm keys (fix_weight_norm
    handles them) and torch's mismatched-mask notice from WavLM's attention."""
    import warnings
    from transformers import logging as hf_logging
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*mismatched key_padding_mask and attn_mask.*")
        warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")
        try:
            yield
        finally:
            hf_logging.set_verbosity(level)


# ------------------------------------------------------------------ voice activity


@dataclass
class Segmentation:
    threshold: float = 0.50          # enter speech at p >= threshold
    neg_threshold: float = 0.35      # leave speech at p < neg_threshold
    min_speech_s: float = 0.15
    min_silence_s: float = 0.20      # join phrases separated by less
    pad_s: float = 0.05

    def phrases(self, t: np.ndarray, p: np.ndarray, duration: float) -> list[tuple[float, float]]:
        raw, on, start = [], False, 0.0
        for ti, pi in zip(t, p):
            if not on and pi >= self.threshold:
                on, start = True, max(0.0, ti - self.pad_s)
            elif on and pi < self.neg_threshold:
                on = False
                raw.append([start, min(duration, ti + self.pad_s)])
        if on:
            raw.append([start, duration])
        out = []
        for a, b in raw:
            if out and a - out[-1][1] < self.min_silence_s:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [(float(a), float(b)) for a, b in out if b - a >= self.min_speech_s]


class SileroVAD:
    name = "silero"

    def __init__(self, ctx, threads: int | None = None):
        import torch
        # `torch.set_num_threads` is process-wide, so setting it here (it used to default to 1,
        # as Silero's own single-stream example does) also throttled Whisper, AST, the phoneme
        # model and WavLM to one core for the whole run. Left alone unless asked for.
        if threads:
            torch.set_num_threads(threads)
        path = local_model(ctx["models_dir"], "silero_vad", ctx["verify"]) / "silero_vad.jit"
        self.torch, self.model = torch, torch.jit.load(str(path), map_location="cpu").eval()

    def probabilities(self, clip):
        if clip.sr != 16000:
            raise ValueError("Silero VAD expects 16 kHz analysis audio")
        x, w = clip.audio.astype(np.float32), 512
        starts = range(0, len(x) - w + 1, w)
        self.model.reset_states()
        with self.torch.no_grad():
            p = [float(self.model(self.torch.from_numpy(x[i:i + w]).unsqueeze(0), 16000)[0][0]) for i in starts]
        return np.arange(len(p)) * w / 16000, np.array(p)


class EnergyVAD:
    """Frame level through a sigmoid centred 35 % of the way from the noise floor (15th
    percentile) to the speech ceiling (85th). No model; weaker than Silero."""
    name = "energy"

    def __init__(self, ctx):
        pass

    def probabilities(self, clip):
        n, hop = int(clip.sr * 0.03), int(clip.sr * 0.01)
        from .dsp import db, frame
        d = db(np.sqrt(np.mean(frame(clip.audio, n, hop, center=False) ** 2, axis=1)), 1e-12)
        t = np.arange(len(d)) * hop / clip.sr
        lo, hi = np.percentile(d, 15), np.percentile(d, 85)
        if hi - lo < 6:                                  # flat level: silence -> no speech, else undecided
            return t, np.full(len(d), 0.0 if hi < -60 else 0.5)
        return t, 1 / (1 + np.exp(-(d - (lo + 0.35 * (hi - lo))) / 4))


# ------------------------------------------------------------------ language


class WhisperLanguage:
    """Whisper's language head: one decoder step after <|startoftranscript|>, softmax over
    the allowed language tokens only, on the VAD speech (first 30 s)."""
    name = "whisper-small"

    def __init__(self, ctx, allowed=("hi", "ta")):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        _hf_offline()
        path = local_model(ctx["models_dir"], "whisper-small", ctx["verify"])
        self.torch, self.device, self.allowed = torch, ctx["device"], tuple(allowed)
        with quiet_load():
            self.proc = WhisperProcessor.from_pretrained(path, local_files_only=True)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                path, local_files_only=True).to(self.device).eval()
        tok = self.proc.tokenizer
        self.ids = [tok.convert_tokens_to_ids(f"<|{c}|>") for c in self.allowed]
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

    def identify(self, clip, phrases) -> dict:
        x = clip.audio
        if phrases:
            x = np.concatenate([x[int(a * clip.sr):int(b * clip.sr)] for a, b in phrases])
        enc = self.proc(x[:30 * clip.sr], sampling_rate=clip.sr, return_tensors="pt",
                        return_attention_mask=True)
        feats = enc.input_features.to(self.device)
        mask = getattr(enc, "attention_mask", None)
        kw = {"attention_mask": mask.to(self.device)} if mask is not None else {}
        with self.torch.no_grad():
            logits = self.model(input_features=feats, use_cache=False,
                                decoder_input_ids=self.torch.tensor([[self.sot]], device=self.device),
                                **kw).logits[0, -1]
        p = self.torch.softmax(logits[self.ids].float(), -1).cpu().numpy()
        scores = {c: float(v) for c, v in zip(self.allowed, p)}
        best = max(scores, key=scores.get)
        return {"code": best, "confidence": scores[best], "scores": scores, "source": self.name}


class WhisperTranscribe:
    """Native-script transcription, so the word-level measures stop being optional.

    The task and language are BOTH forced: left to detect, Whisper on Indic audio often
    translates to English or romanises instead of transcribing, which is why published Indic
    benchmarks sometimes cannot score it at all. The language comes from the LID step that has
    already run on this clip, so the two always agree.

    Clips at or under the model's 30 s window go in one pass. Longer ones are split at phrase
    boundaries into groups of at most `window_s`, never mid-phrase, and the pieces joined.

    `model` picks the checkpoint: whisper-large-v3 by default (far better on Hindi and Tamil),
    or whisper-small to reuse the one already frozen for language ID at a real cost in accuracy.
    """
    LANGUAGES = {"hi": "hindi", "ta": "tamil"}

    def __init__(self, ctx, model="whisper-large-v3", beams=1, window_s=30.0,
                 max_new_tokens=440):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        _hf_offline()
        path = local_model(ctx["models_dir"], model, ctx["verify"])
        self.torch, self.device, self.name = torch, ctx["device"], model
        self.beams, self.window_s, self.max_new_tokens = beams, window_s, max_new_tokens
        with quiet_load():
            self.proc = WhisperProcessor.from_pretrained(path, local_files_only=True)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                path, local_files_only=True).to(self.device).eval()

    def _pass(self, x, sr, language) -> str:
        # return_attention_mask: Whisper's pad token IS its eos token, so without a mask
        # generate() cannot tell padding from a real end-of-text and warns. One clip at a
        # time is never padded, so the mask the feature extractor returns is all ones -- this
        # is the correct value, not a way of silencing the warning.
        enc = self.proc(x, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
        feats = enc.input_features.to(self.device)
        mask = getattr(enc, "attention_mask", None)
        kw = {"attention_mask": mask.to(self.device)} if mask is not None else {}
        with self.torch.no_grad():
            # `language=` / `task=` rather than forced_decoder_ids: the latter is deprecated and
            # gone from newer transformers, while these two have been supported since 4.28.
            ids = self.model.generate(feats, language=self.LANGUAGES[language], task="transcribe",
                                      num_beams=self.beams, do_sample=False,
                                      max_new_tokens=self.max_new_tokens, **kw)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def transcribe(self, clip, phrases, language) -> dict | None:
        if language not in self.LANGUAGES or not phrases:
            return None                        # only the two languages this tool is built for
        sr, win = clip.sr, self.window_s
        groups, cur = [], []
        for a, b in phrases:                   # split between phrases, never inside one
            if cur and b - cur[0][0] > win:
                groups.append(cur)
                cur = []
            cur.append((a, b))
        if cur:
            groups.append(cur)
        parts = []
        for grp in groups:
            a, b = grp[0][0], grp[-1][1]
            seg = clip.audio[int(a * sr):int(b * sr)]
            if len(seg) < int(0.1 * sr):
                continue
            t = self._pass(seg, sr, language)
            if t:
                parts.append(t)
        text = " ".join(parts).strip()
        return {"text": text, "source": f"asr:{self.name}", "passes": len(groups)} if text else None


class FolderLanguage:
    """Language from a folder name in the path (…/hi/clip.wav, …/tamil/x/clip.wav)."""
    name = "folder"

    def __init__(self, ctx, allowed=("hi", "ta"), names=None):
        names = names or {"hi": "hi", "hindi": "hi", "ta": "ta", "tamil": "ta"}
        self.names = {k.lower(): v for k, v in names.items() if v in allowed}

    def identify(self, clip, phrases) -> dict:
        for part in reversed(Path(clip.path or "").parent.parts):
            if part.lower() in self.names:
                code = self.names[part.lower()]
                return {"code": code, "confidence": 1.0, "scores": {code: 1.0}, "source": self.name}
        return {"code": "und", "confidence": 0.0, "scores": {}, "source": self.name}


# ------------------------------------------------------------------ sound events


class ASTEvents:
    """Audio Spectrogram Transformer (AudioSet, 527 classes), two passes to save compute:
    1. one pass per 10.24 s chunk of the clip (the model's input length);
    2. only for chunks where some label scores >= gate x its threshold: 0.96 s windows every
       hop_s, each scored on its own (sigmoid, multi-label). A label's score is the max over its
       AudioSet classes (case-insensitive substring match). Runs of windows above the label's
       threshold become events, edges snapped to where the energy is >= 25 % of its max,
       same-label events closer than merge_gap_s joined.
    "background": one pass over the audio outside speech (joined, up to 10.24 s)."""
    name = "ast-audioset"
    chunk_s = 10.24

    def __init__(self, ctx, labels: dict, window_s: float = 0.96, hop_s: float = 0.48, batch_size: int = 32,
                 min_event_s: float = 0.20, merge_gap_s: float = 0.25, gate: float = 0.5):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        _hf_offline()
        path = local_model(ctx["models_dir"], "ast-audioset", ctx["verify"])
        self.torch, self.device = torch, ctx["device"]
        with quiet_load():
            self.fe = AutoFeatureExtractor.from_pretrained(path, local_files_only=True)
            self.model = AutoModelForAudioClassification.from_pretrained(
                path, local_files_only=True).to(self.device).eval()
        self.id2label = self.model.config.id2label
        self._setup(labels, window_s, hop_s, batch_size, min_event_s, merge_gap_s, gate)

    def _setup(self, labels, window_s, hop_s, batch_size, min_event_s, merge_gap_s, gate):
        self.labels, self.window_s, self.hop_s, self.batch = labels, window_s, hop_s, batch_size
        self.min_event_s, self.merge_gap_s, self.gate = min_event_s, merge_gap_s, gate
        self.index = class_index(self.id2label, {k: v["classes"] for k, v in labels.items()})

    def _probs(self, chunks: list) -> np.ndarray:
        """(len(chunks), 527) AudioSet probabilities."""
        out = []
        with self.torch.no_grad():
            for b in range(0, len(chunks), self.batch):
                inp = self.fe(chunks[b:b + self.batch], sampling_rate=16000, return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                out.append(self.torch.sigmoid(self.model(**inp).logits.float()).cpu().numpy())
        return np.concatenate(out)

    def windows(self, x: np.ndarray, offset_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Window start times (clip time) and probabilities for one stretch of audio."""
        w, h = int(16000 * self.window_s), int(16000 * self.hop_s)
        n = max(1, 1 + (len(x) - w) // h)
        return offset_s + np.arange(n) * self.hop_s, self._probs([x[i * h:i * h + w] for i in range(n)])

    def scores(self, clip) -> tuple[np.ndarray, np.ndarray]:
        """(window start times, (windows, labels) scores) for the whole clip (no gating)."""
        t, p = self.windows(clip.audio.astype(np.float32), 0.0)
        return t, label_scores(p, self.index, self.labels)

    def detect(self, clip, phrases=()) -> dict:
        """{"events": timed label events, "background": top AudioSet classes outside speech}."""
        if clip.sr != 16000:
            raise ValueError("AST expects 16 kHz analysis audio")
        x = clip.audio.astype(np.float32)
        L = int(16000 * self.chunk_s)
        starts = list(range(0, max(1, len(x)), L))
        coarse = label_scores(self._probs([x[a:a + L] for a in starts]), self.index, self.labels)
        thr = np.array([cfg["threshold"] for cfg in self.labels.values()])
        ts, ps = [], []
        for i, a in enumerate(starts):
            if (coarse[i] >= self.gate * thr).any():                 # something may be there: look closely
                t, p = self.windows(x[a:a + L], a / 16000)
                ts.append(t)
                ps.append(p)
        events = []
        if ts:
            t, p = np.concatenate(ts), np.concatenate(ps)
            events = events_from_scores(clip, t, label_scores(p, self.index, self.labels), self.labels,
                                        self.window_s, self.min_event_s, self.merge_gap_s)
        from .dsp import gaps
        outside = [x[int(a * 16000):int(b * 16000)] for a, b in gaps(list(phrases), clip.duration, 0.2)]
        background = None
        if outside and sum(map(len, outside)) >= 16000:
            bg = np.concatenate(outside)[:L]
            background = background_from_probs(self._probs([bg])[0], self.id2label)
        return {"events": events, "background": background}


def label_scores(p: np.ndarray, index: dict, labels: dict) -> np.ndarray:
    s = np.zeros((len(p), len(labels)))
    for j, lab in enumerate(labels):
        if index[lab]:
            s[:, j] = p[:, index[lab]].max(axis=1)
    return s


SPEECH_LIKE = ("speech", "speaking", "conversation", "narration", "monologue", "babbling", "silence")


def background_from_probs(p: np.ndarray, id2label: dict, min_p: float = 0.15, top: int = 3) -> list[dict]:
    """What else the model hears in the audio outside speech: AudioSet classes from one pass,
    speech-like classes and silence left out; the top ones with probability >= min_p."""
    names = {int(i): str(n) for i, n in id2label.items()}
    order = [i for i in np.argsort(-p) if not any(k in names[int(i)].lower() for k in SPEECH_LIKE)]
    return [{"class": names[int(i)], "p": float(p[i])} for i in order[:top] if p[i] >= min_p]


def class_index(id2label: dict, classes: dict) -> dict:
    names = {int(i): str(n).lower() for i, n in id2label.items()}
    return {lab: [i for i, n in names.items() if any(w.lower() in n for w in words)] for lab, words in classes.items()}


def events_from_scores(clip, t, s, labels: dict, window_s: float, min_event_s: float, merge_gap_s: float) -> list[dict]:
    from .dsp import runs
    ev = []
    for j, (lab, cfg) in enumerate(labels.items()):
        for a, b in runs(s[:, j] >= cfg["threshold"]):
            start, end = float(t[a]), min(clip.duration, float(t[b]) + window_s)
            if end - start < min_event_s:
                continue
            start, end = snap(clip, start, end)
            ev.append({"start": start, "end": end, "label": lab, "p": float(s[a:b + 1, j].mean()),
                       "p_max": float(s[a:b + 1, j].max())})
    ev.sort(key=lambda e: e["start"])
    out = []
    for e in ev:                                   # merge same-label events closer than the gap
        prev = next((o for o in reversed(out) if o["label"] == e["label"]), None)
        if prev and e["start"] - prev["end"] <= merge_gap_s:
            prev.update(end=max(prev["end"], e["end"]), p=(prev["p"] + e["p"]) / 2, p_max=max(prev["p_max"], e["p_max"]))
        else:
            out.append(e)
    return out


def snap(clip, start: float, end: float, pad: float = 0.10) -> tuple[float, float]:
    x, sr, hop = clip.audio, clip.sr, int(clip.sr * 0.01)
    a, b = max(0, int((start - pad) * sr)), min(len(x), int((end + pad) * sr))
    n = (b - a) // hop
    if n < 2:
        return start, end
    rms = np.sqrt(np.mean(x[a:a + n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    on = np.where(rms >= 0.25 * rms.max())[0]
    return max(0.0, (a + on[0] * hop) / sr), min(clip.duration, (a + (on[-1] + 1) * hop) / sr)


# ------------------------------------------------------------------ syllables


VOWELS = set("aeiouyɑɐɒæɛɜəɘɵɤɨɪʊʉɯɔœøʌʏɞɶɚɝ")


def is_vowel(token: str) -> bool:
    """An IPA phoneme token is a syllable nucleus if its first letter is a vowel
    (diphthongs like 'aɪ' and long vowels like 'eː' count once)."""
    return bool(token) and token[0] in VOWELS


def vowel_times(ids, id2tok: dict, blank: int, frame_s: float, offset_s: float = 0.0) -> list[float]:
    """Greedy CTC decoding: a token starts where the frame's id differs from the previous
    frame's and is not blank; vowel tokens -> their start times."""
    out, prev = [], blank
    for f, i in enumerate(ids):
        i = int(i)
        if i != blank and i != prev and is_vowel(id2tok.get(i, "")):
            out.append(offset_s + (f + 0.5) * frame_s)
        prev = i
    return out


class PhonemeSyllables:
    """Syllable nuclei from a multilingual phoneme recogniser (wav2vec2 XLSR-53 fine-tuned on
    Common Voice phonemes): every recognised vowel is one syllable, timed by its CTC frame
    (20 ms). No words, no transcript. Runs in 20 s pieces."""
    name = "xlsr-phonemes"

    def __init__(self, ctx, piece_s: float = 20.0):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        _hf_offline()
        path = local_model(ctx["models_dir"], "phoneme-xlsr", ctx["verify"])
        self.torch, self.device, self.piece_s = torch, ctx["device"], piece_s
        with quiet_load():
            self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True)
        self.fixed_weight_norm = fix_weight_norm(model, path)
        self.model = model.to(self.device).eval()
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        self.id2tok = {int(i): t for t, i in vocab.items()}
        self.blank = int(vocab.get("<pad>", self.model.config.pad_token_id or 0))

    def nuclei(self, clip) -> np.ndarray:
        x, n = clip.audio.astype(np.float32), int(self.piece_s * clip.sr)
        out = []
        with self.torch.no_grad():
            for a in range(0, len(x), n):
                seg = x[a:a + n]
                if len(seg) < clip.sr // 10:
                    continue
                inp = self.fe(seg, sampling_rate=clip.sr, return_tensors="pt").input_values.to(self.device)
                ids = self.model(inp).logits[0].argmax(-1).cpu().numpy()
                out += vowel_times(ids, self.id2tok, self.blank, len(seg) / clip.sr / len(ids), a / clip.sr)
        return np.array(out)


class EnvelopeSyllables:
    """No model: peaks of the 300-3000 Hz envelope (dsp.syllable_nuclei). Undercounts fast speech."""
    name = "envelope"

    def __init__(self, ctx):
        pass

    def nuclei(self, clip) -> np.ndarray:
        from .dsp import syllable_nuclei
        return syllable_nuclei(clip.audio, clip.sr)


# ------------------------------------------------------------------ speakers


class WavLMSpeakers:
    """How many people speak, and where the voice changes.

    One x-vector per phrase (WavLM-base-plus-sv, MIT), joined into speakers by average-link
    clustering on cosine similarity. One expressive speaker moves a lot in x-vector space
    (on three single-speaker test clips, the model card's verification threshold 0.86 split
    one voice into 2-3), so the claim is two-sided:
      * phrases merge at `threshold` (0.5): clearly different voices stay apart;
      * "confident": "multi" when even that merge leaves > 1 voice,
        "single" when every pair is >= `same_threshold` (0.8) similar,
        None in between - and the description then says nothing.
    Phrases shorter than min_phrase_s get the nearest speaker's label but never create one.
    States how many and where the voice changes, never who they are."""
    name = "wavlm-sv"

    def __init__(self, ctx, threshold: float = 0.5, same_threshold: float = 0.8,
                 min_phrase_s: float = 1.0, max_seconds: float = 10.0, min_embed_s: float = 0.5):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        _hf_offline()
        path = local_model(ctx["models_dir"], "speaker-xvector", ctx["verify"])
        self.torch, self.device = torch, ctx["device"]
        with quiet_load():
            self.fe = AutoFeatureExtractor.from_pretrained(path, local_files_only=True)
            model = WavLMForXVector.from_pretrained(path, local_files_only=True)
        self.fixed_weight_norm = fix_weight_norm(model, path)
        self.model = model.to(self.device).eval()
        self.threshold, self.same_threshold = threshold, same_threshold
        self.min_phrase_s, self.max_seconds, self.min_embed_s = min_phrase_s, max_seconds, min_embed_s

    def embed(self, clip, phrases) -> np.ndarray:
        import warnings
        x, out = clip.audio.astype(np.float32), []
        # torch's own deprecation notice from WavLM's attention; nothing to act on, and it
        # would otherwise print once per clip.
        with warnings.catch_warnings(), self.torch.no_grad():
            warnings.filterwarnings("ignore", message=".*mismatched key_padding_mask and attn_mask.*")
            for a, b in phrases:
                seg = embed_segment(x, clip.sr, a, b, self.min_embed_s, self.max_seconds)
                inp = self.fe(seg, sampling_rate=clip.sr, return_tensors="pt", padding=True)
                inp = {k: v.to(self.device) for k, v in inp.items()}
                e = self.model(**inp).embeddings[0].float().cpu().numpy()
                out.append(e / (np.linalg.norm(e) + 1e-9))
        return np.array(out)

    def detect(self, clip, phrases) -> dict | None:
        phrases = list(phrases)
        if not phrases:
            return None
        emb = self.embed(clip, phrases)
        long = [i for i, (a, b) in enumerate(phrases) if b - a >= self.min_phrase_s]
        if not long:
            return None
        labels = cluster_cosine(emb[long], self.threshold)
        strict = cluster_cosine(emb[long], self.same_threshold)
        count = len(set(labels))
        confident = "multi" if count > 1 else ("single" if len(set(strict)) == 1 else None)
        per = {}
        for i, lab in zip(long, labels):
            per[i] = int(lab)
        for i in range(len(phrases)):                       # short phrases: nearest long phrase
            if i not in per:
                j = max(long, key=lambda k: float(emb[i] @ emb[k]))
                per[i] = per[j]
        changes = [phrases[i][0] for i in range(1, len(phrases)) if per[i] != per[i - 1]]
        return {"count": count, "confident": confident, "per_phrase": [per[i] for i in range(len(phrases))],
                "change_times": [round(float(t), 2) for t in changes],
                "thresholds": {"different_below": self.threshold, "same_above": self.same_threshold},
                "source": self.name}


def embed_segment(audio, sr, a: float, b: float, min_s: float, max_s: float) -> np.ndarray:
    """Audio for one phrase, long enough for WavLM's speaker head.

    Its TDNN layers are dilated convolutions that need about 15 of the encoder's 20 ms frames
    (~0.3 s); a shorter phrase makes conv1d fail with "Kernel size can't be greater than actual
    input size", and `split_at_silences` can emit phrases as short as 0.15 s. A short phrase is
    therefore widened symmetrically into the surrounding audio (real context, so the embedding
    stays meaningful) and only zero-padded when the whole clip is shorter than min_s."""
    x = np.asarray(audio, np.float32)
    need = max(1, int(min_s * sr))
    i, j = int(a * sr), int(min(b, a + max_s) * sr)
    if j - i < need:
        mid = (i + j) // 2
        i = max(0, min(mid - need // 2, len(x) - need))
        j = i + need
    seg = x[max(0, i):j]
    return np.pad(seg, (0, need - len(seg))) if len(seg) < need else seg


def cluster_cosine(emb: np.ndarray, threshold: float) -> list[int]:
    """Agglomerative clustering (average link) on unit vectors: merge the closest pair while
    its mean cosine similarity is >= threshold."""
    groups = [[i] for i in range(len(emb))]
    sim = emb @ emb.T
    while len(groups) > 1:
        best, pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                s = float(np.mean(sim[np.ix_(groups[a], groups[b])]))
                if s > best:
                    best, pair = s, (a, b)
        if best < threshold:
            break
        a, b = pair
        groups[a] += groups.pop(b)
    out = [0] * len(emb)
    for k, g in enumerate(sorted(groups, key=min)):
        for i in g:
            out[i] = k
    return out


# ------------------------------------------------------------------ registry

VAD = {"silero": SileroVAD, "energy": EnergyVAD}
LANGUAGE = {"whisper": WhisperLanguage, "folder": FolderLanguage}
EVENTS = {"ast": ASTEvents}
SYLLABLES = {"phonemes": PhonemeSyllables, "envelope": EnvelopeSyllables}
SPEAKERS = {"wavlm": WavLMSpeakers}
TRANSCRIBE = {"whisper": WhisperTranscribe}


def build(registry: dict, spec: dict | None, ctx: dict, what: str):
    if spec is None:
        return None
    spec = dict(spec)
    impl = spec.pop("impl", None)
    if impl not in registry:
        raise ValueError(f"{what}.impl must be one of {sorted(registry)}, got {impl!r}")
    return registry[impl](ctx, **spec)
