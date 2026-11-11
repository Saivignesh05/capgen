"""
naturalness_score.py - naturalness scores for generated speech, meant to be called from a pipeline.

Metrics
  utmos   per clip   MOS predictor (UTMOS22-strong via SpeechMOS).          higher = better
  sigmos  per clip   Microsoft SIGMOS (col, disc, loud, noise, reverb, ...)  higher = better
  fad     corpus     Frechet distance between XLS-R embeddings of your clips
                     and native clips.                                       LOWER  = better
  psd     corpus     Frechet distance between prosody features of your clips
                     and the same native clips.                              LOWER  = better

Per-clip metrics give one number per clip. Corpus metrics give one number for a whole set of
clips (use 100+ clips, and the same number of clips whenever you compare two sets).

Audio input: every function takes `audio` and `sample_rate`.
  audio = path to a wav/flac/ogg file  -> `sample_rate` is ignored, the file's own rate is used
  audio = 1-D mono numpy float array   -> `sample_rate` must be the rate of that array

Requirements (all imported at the top, so they must be installed or on PYTHONPATH):
  torch, transformers, librosa, numpy
  sigmos.py + its .onnx model (from microsoft/DNS-Challenge): put the folder on PYTHONPATH
  the psp-eval repo (folder containing evaluation/fad.py): put it on PYTHONPATH
Do not name any of your own files evaluation.py or sigmos.py.

Typical use
  scorer = NaturalnessScorer(device, utmos_repo, sigmos_model_dir, xlsr_model, xlsr_layer)
  ref_emb, ref_prosody = scorer.build_references(native_clips)       # once per language
  result_a = scorer.score_clips(clips_a, ref_emb, ref_prosody)       # clips = {clip_id: (audio, sr)}
  result_b = scorer.score_clips(clips_b, ref_emb, ref_prosody)
  rows = compare_results(result_a, result_b, tol_percent, n_boot, seed)

Use the same clip_ids in both sets (same text/description -> same id) so the comparison is paired.
"""
import math

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from evaluation.fad import compute_fad
from evaluation.psd import compute_psd, extract_prosodic_features
from sigmos import SigMOS

LOWER_IS_BETTER = {"fad", "psd"}  # every other metric: higher is better


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def load_audio(audio, sample_rate, target_rate):
    """File path or mono waveform -> mono float32 numpy array at `target_rate`."""
    if isinstance(audio, (str, bytes)) or hasattr(audio, "__fspath__"):
        wave, _ = librosa.load(audio, sr=target_rate, mono=True)
    else:
        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim != 1:
            raise ValueError("waveform must be 1-D mono")
        if sample_rate != target_rate:
            wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=target_rate)
    return np.ascontiguousarray(wave, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Scorer: loads the models once, then scores any number of clips
# --------------------------------------------------------------------------- #
class NaturalnessScorer:
    def __init__(self, device, utmos_repo, sigmos_model_dir, xlsr_model, xlsr_layer):
        """
        device            "cuda" or "cpu"
        utmos_repo        torch.hub repo of SpeechMOS, e.g. "tarepan/SpeechMOS:v1.2.0"
        sigmos_model_dir  folder holding the SIGMOS .onnx model
        xlsr_model        Hugging Face id of the embedding model for FAD, e.g. "facebook/wav2vec2-xls-r-300m"
        xlsr_layer        hidden layer of that model used for FAD, e.g. 9
        """
        self.device = device
        self.xlsr_layer = xlsr_layer
        self.utmos = torch.hub.load(utmos_repo, "utmos22_strong", trust_repo=True).to(device).eval()
        self.sigmos = SigMOS(model_dir=sigmos_model_dir)
        self.xlsr_processor = Wav2Vec2FeatureExtractor.from_pretrained(xlsr_model)
        self.xlsr = Wav2Vec2Model.from_pretrained(xlsr_model).to(device).eval()

    def utmos_score(self, audio, sample_rate):
        """One UTMOS value for one clip (about 1 to 5, higher = more natural)."""
        wave = load_audio(audio, sample_rate, 16000)
        x = torch.from_numpy(wave).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.utmos(x, 16000).reshape(-1)[0])

    def sigmos_scores(self, audio, sample_rate):
        """Dict of SIGMOS scores for one clip, keys like "sigmos.col", "sigmos.ovrl". SIGMOS expects 48 kHz."""
        wave = load_audio(audio, sample_rate, 48000)
        raw = self.sigmos.run(wave, sr=48000)
        return {"sigmos." + k.lower().replace("mos_", ""): float(v)
                for k, v in raw.items() if isinstance(v, (int, float, np.floating))}

    def embedding(self, audio, sample_rate):
        """XLS-R embedding of one clip: mean over time of one hidden layer (vector used by FAD)."""
        wave = load_audio(audio, sample_rate, 16000)
        x = self.xlsr_processor(wave, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            hidden = self.xlsr(x, output_hidden_states=True).hidden_states[self.xlsr_layer]
        return hidden.mean(dim=1).squeeze(0).float().cpu().numpy()

    def prosody(self, audio, sample_rate):
        """Prosody feature vector of one clip (pitch and rhythm statistics, used by PSD)."""
        wave = load_audio(audio, sample_rate, 16000)
        return np.asarray(extract_prosodic_features(wave), dtype=np.float32)

    def build_references(self, native_clips):
        """
        Native reference data for FAD and PSD. Run once per language.
        native_clips: {clip_id: (audio, sample_rate)} of native-speaker clips that you will NOT also score.
        Returns (ref_embeddings, ref_prosody) as 2-D arrays, one row per clip; pass them to score_clips.
        """
        ref_embeddings = np.stack([self.embedding(a, sr) for a, sr in native_clips.values()])
        ref_prosody = np.stack([self.prosody(a, sr) for a, sr in native_clips.values()])
        return ref_embeddings, ref_prosody

    def score_clips(self, clips, ref_embeddings, ref_prosody, metrics):
        """
        Score a set of clips.
        clips:   {clip_id: (audio, sample_rate)}
        metrics: list from ["utmos", "sigmos", "fad", "psd"]
        Returns {"clips": {clip_id: {metric: value}}, "corpus": {"fad": value, "psd": value}}
        (corpus only holds the corpus metrics you asked for).
        """
        per_clip = {clip_id: {} for clip_id in clips}
        for clip_id, (audio, sample_rate) in clips.items():
            if "utmos" in metrics:
                per_clip[clip_id]["utmos"] = self.utmos_score(audio, sample_rate)
            if "sigmos" in metrics:
                per_clip[clip_id].update(self.sigmos_scores(audio, sample_rate))

        corpus = {}
        if "fad" in metrics:
            embeddings = np.stack([self.embedding(a, sr) for a, sr in clips.values()])
            corpus["fad"] = float(compute_fad(embeddings, ref_embeddings)[0])
        if "psd" in metrics:
            features = np.stack([self.prosody(a, sr) for a, sr in clips.values()])
            corpus["psd"] = float(compute_psd(features, ref_prosody)[0])  # compute_psd drops NaN rows itself
        return {"clips": per_clip, "corpus": corpus}


# --------------------------------------------------------------------------- #
# Comparison of two results
# --------------------------------------------------------------------------- #
def bootstrap_ci(differences, n_boot, rng):
    """95% bootstrap confidence interval of the mean of `differences` (1-D array)."""
    draws = differences[rng.integers(0, len(differences), (n_boot, len(differences)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def clip_values(result, metric):
    """{clip_id: value} for one per-clip metric, skipping missing or non-finite values."""
    values = {}
    for clip_id, scores in result["clips"].items():
        value = scores.get(metric)
        if value is not None and math.isfinite(value):
            values[clip_id] = value
    return values


def compare_results(result_a, result_b, tol_percent, n_boot, seed):
    """
    Compare result A (baseline) with result B for every metric present in both.
    Inputs are dicts returned by score_clips. Returns a list of dicts, one per metric:
      metric, a, b, delta, rel_pct, ci (95% CI of delta or None), paired (bool), verdict.

    Per-clip metrics: if most clip_ids exist in both results the bootstrap is PAIRED (on per-clip
    differences, more sensitive); otherwise it is UNPAIRED (two independent sets of clips).
    Corpus metrics (fad, psd) are one number per result, so there is no CI: a change smaller than
    `tol_percent` is reported as noise instead of better/worse.
    """
    rng = np.random.default_rng(seed)
    clip_metrics = {m for r in (result_a, result_b) for s in r["clips"].values() for m in s}
    corpus_metrics = set(result_a["corpus"]) & set(result_b["corpus"])
    rows = []

    for metric in sorted(clip_metrics | corpus_metrics):
        row = {"metric": metric, "ci": None, "paired": False}
        values_a = clip_values(result_a, metric)
        values_b = clip_values(result_b, metric)

        if values_a and values_b:
            common = sorted(set(values_a) & set(values_b))
            if len(common) >= max(5, 0.5 * min(len(values_a), len(values_b))):
                diffs = np.array([values_b[c] - values_a[c] for c in common])
                row.update(paired=True,
                           a=float(np.mean([values_a[c] for c in common])),
                           b=float(np.mean([values_b[c] for c in common])))
                row["ci"] = bootstrap_ci(diffs, n_boot, rng)
            else:
                xa, xb = np.array(list(values_a.values())), np.array(list(values_b.values()))
                diffs = (xb[rng.integers(0, len(xb), (n_boot, len(xb)))].mean(axis=1)
                         - xa[rng.integers(0, len(xa), (n_boot, len(xa)))].mean(axis=1))
                row.update(a=float(xa.mean()), b=float(xb.mean()))
                row["ci"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
        elif metric in corpus_metrics:
            row.update(a=result_a["corpus"][metric], b=result_b["corpus"][metric])
        else:
            continue

        row["delta"] = row["b"] - row["a"]
        row["rel_pct"] = row["delta"] / abs(row["a"]) * 100 if row["a"] else None
        lower_better = metric.split(".")[-1] in LOWER_IS_BETTER

        if row["ci"] and row["ci"][0] <= 0 <= row["ci"][1]:
            row["verdict"] = "no clear change"
        elif row["delta"] == 0:
            row["verdict"] = "no change"
        elif row["ci"] is None and row["rel_pct"] is not None and abs(row["rel_pct"]) < tol_percent:
            row["verdict"] = "within " + str(tol_percent) + "% (no CI)"
        else:
            better = (row["delta"] < 0) == lower_better
            row["verdict"] = ("better" if better else "WORSE") + ("" if row["ci"] else " (no CI)")
        rows.append(row)
    return rows
