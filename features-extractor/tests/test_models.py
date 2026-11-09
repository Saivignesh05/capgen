"""Frozen-model lock, VAD segmentation, events post-processing, language sources.
The real networks are tested only with FX_MODELS=1 (they need models/ from fetch-models)."""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import ROOT, SR, harmonic, needs_models, noise, place, sine, utterance, wav_bytes

from src import audio, models


def clip_of(x, path=None):
    c = audio.load(wav_bytes(x, SR, np.float32))
    c.path = path
    return c


# ------------------------------------------------------------------ freezing


@pytest.fixture
def fake_download(monkeypatch):
    payload = {"bytes": b"model-v1"}

    def urlretrieve(url, target):
        Path(target).write_bytes(payload["bytes"])
    monkeypatch.setattr(models.urllib.request, "urlretrieve", urlretrieve)
    return payload


def test_fetch_writes_lock_and_verification_passes(tmp_path, fake_download):
    lock = models.fetch(tmp_path, ["silero_vad"], log=lambda *_: None)
    entry = lock["silero_vad"]
    assert entry["url"].endswith("v5.1.2/src/silero_vad/data/silero_vad.jit")
    assert entry["files"] == {"silero_vad.jit": models.sha256(tmp_path / "silero_vad" / "silero_vad.jit")}
    assert json.loads((tmp_path / models.LOCK).read_text()) == lock
    assert models.local_model(tmp_path, "silero_vad") == tmp_path / "silero_vad"


def test_changed_model_file_is_refused(tmp_path, fake_download):
    models.fetch(tmp_path, ["silero_vad"], log=lambda *_: None)
    (tmp_path / "silero_vad" / "silero_vad.jit").write_bytes(b"something else")
    with pytest.raises(models.ModelError, match="does not match the lock"):
        models.local_model(tmp_path, "silero_vad")
    models.local_model(tmp_path, "silero_vad", verify=False)          # explicit opt-out still loads


def test_refetch_that_differs_from_lock_is_refused(tmp_path, fake_download):
    models.fetch(tmp_path, ["silero_vad"], log=lambda *_: None)
    (tmp_path / "silero_vad" / "silero_vad.jit").unlink()
    fake_download["bytes"] = b"model-v2"
    with pytest.raises(models.ModelError, match="refusing to change a frozen model"):
        models.fetch(tmp_path, ["silero_vad"], log=lambda *_: None)


def test_missing_lock_or_model_explains_what_to_do(tmp_path):
    with pytest.raises(models.ModelError, match="fetch-models"):
        models.local_model(tmp_path, "silero_vad")
    (tmp_path / models.LOCK).write_text("{}")
    with pytest.raises(models.ModelError, match="not in"):
        models.local_model(tmp_path, "whisper-small")


def test_build_rejects_unknown_impl():
    with pytest.raises(ValueError, match="vad.impl must be one of"):
        models.build(models.VAD, {"impl": "webrtc"}, {}, "vad")
    assert models.build(models.EVENTS, None, {}, "events") is None


# ------------------------------------------------------------------ VAD


def test_segmentation_hysteresis_padding_and_joining():
    t = np.arange(0, 6, 0.01)
    p = np.where(((t >= 1) & (t < 2)) | ((t >= 2.1) & (t < 3)) | ((t >= 4) & (t < 4.1)), 0.9, 0.1)
    ph = models.Segmentation(min_speech_s=0.2).phrases(t, p, 6.0)
    assert len(ph) == 1                                          # 0.1 s gap joined, 0.1 s blip dropped
    assert ph[0][0] == pytest.approx(0.95) and ph[0][1] == pytest.approx(3.05)


def test_energy_vad_finds_the_spoken_spans():
    x = place([(1.0, 2.0), (3.0, 4.5)], 5.0, lambda d: harmonic(150, d, 0.3), floor=noise(5.0, -60))
    c = clip_of(x)
    t, p = models.EnergyVAD({}).probabilities(c)
    ph = models.Segmentation().phrases(t, p, c.duration)
    assert len(ph) == 2
    for (a, b), (ta, tb) in zip(ph, [(1.0, 2.0), (3.0, 4.5)]):
        assert abs(a - ta) < 0.1 and abs(b - tb) < 0.1


# ------------------------------------------------------------------ sound events (shared post-processing)


def test_class_index_matches_audioset_names():
    idx = models.class_index({0: "Laughter", 1: "Baby laughter", 2: "Speech", 3: "Dog"},
                             {"laughter": ["laughter"], "animal": ["Dog", "Cat"], "x": ["Nope"]})
    assert idx == {"laughter": [0, 1], "animal": [3], "x": []}


def test_events_from_window_scores_threshold_snap_and_merge():
    x = place([(1.0, 1.6), (1.8, 2.4), (3.5, 3.6)], 5.0, lambda d: sine(400, d, 0.3), floor=noise(5.0, -70))
    c = clip_of(x)
    t = np.arange(0, 4.6, 0.48)
    loud = np.array([any(a < b2 and a + 0.96 > a2 for a2, b2 in [(1.0, 1.6), (1.8, 2.4)]) for a in t])
    s = np.stack([np.where(loud, 0.8, 0.1), np.full(len(t), 0.1)], axis=1)
    labels = {"laughter": {"threshold": 0.3}, "music": {"threshold": 0.5}}
    ev = models.events_from_scores(c, t, s, labels, 0.96, 0.2, 0.25)
    assert [e["label"] for e in ev] == ["laughter"]
    assert ev[0]["start"] == pytest.approx(1.0, abs=0.05) and ev[0]["end"] == pytest.approx(2.4, abs=0.05)
    assert ev[0]["p"] == pytest.approx(0.8) and ev[0]["p_max"] == pytest.approx(0.8)


# ------------------------------------------------------------------ language


def test_folder_language():
    lid = models.FolderLanguage({})
    assert lid.identify(clip_of(sine(200, 0.5), "/data/ta/x/clip.wav"), [])["code"] == "ta"
    assert lid.identify(clip_of(sine(200, 0.5), "/data/Hindi/clip.wav"), [])["code"] == "hi"
    assert lid.identify(clip_of(sine(200, 0.5), "/data/misc/clip.wav"), [])["code"] == "und"
    only_hi = models.FolderLanguage({}, allowed=["hi"])
    assert only_hi.identify(clip_of(sine(200, 0.5), "/data/ta/clip.wav"), [])["code"] == "und"


# ------------------------------------------------------------------ real frozen models (FX_MODELS=1)


def ctx():
    return {"models_dir": ROOT / "models", "device": "cpu", "verify": True}


@needs_models
def test_silero_separates_voice_from_silence():
    c = clip_of(place([(1.0, 3.0)], 4.0, lambda d: harmonic(150, d, 0.3), floor=noise(4.0, -70)))
    t, p = models.SileroVAD(ctx()).probabilities(c)
    assert len(t) == len(p) > 0 and p[t < 0.8].mean() < 0.3


@needs_models
def test_whisper_language_returns_allowed_scores():
    r = models.WhisperLanguage(ctx()).identify(clip_of(utterance()), [])
    assert set(r["scores"]) == {"hi", "ta"} and abs(sum(r["scores"].values()) - 1) < 1e-4


@needs_models
def test_ast_scores_shape():
    labels = {"music": {"threshold": 0.5, "classes": ["Music"]}}
    t, s = models.ASTEvents(ctx(), labels).scores(clip_of(sine(440, 3.0)))
    assert s.shape == (len(t), 1) and ((0 <= s) & (s <= 1)).all()


def test_background_from_one_pass_leaves_out_speech_and_silence():
    id2label = {0: "Speech", 1: "Vehicle", 2: "Silence", 3: "Fan", 4: "Male speech, man speaking", 5: "Dog"}
    bg = models.background_from_probs(np.array([0.9, 0.6, 0.9, 0.2, 0.9, 0.05]), id2label)
    assert [b["class"] for b in bg] == ["Vehicle", "Fan"] and bg[0]["p"] == pytest.approx(0.6)


# ------------------------------------------------------------------ AST two passes (fake network)


class FakeAST(models.ASTEvents):
    """Real ASTEvents logic, fake network: class 0 'Laughter' fires on loud chunks."""

    def __init__(self, labels):
        self.id2label = {0: "Laughter", 1: "Music", 2: "Speech", 3: "Fan"}
        self.calls = []
        self._setup(labels, 0.96, 0.48, 32, 0.2, 0.25, 0.5)

    def _probs(self, chunks):
        self.calls.append(len(chunks))
        p = np.zeros((len(chunks), 4))
        for i, c in enumerate(chunks):
            p[i, 0] = 0.9 if np.sqrt(np.mean(np.asarray(c) ** 2)) > 0.05 else 0.01
            p[i, 3] = 0.3
        return p


def test_ast_skips_the_window_pass_when_nothing_is_heard():
    ast = FakeAST({"laughter": {"threshold": 0.4, "classes": ["Laughter"]}})
    quiet = clip_of(noise(20.0, -60))
    out = ast.detect(quiet, [(0.0, 19.6)])                                         # < 1 s outside speech
    assert out["events"] == [] and ast.calls == [2] and out["background"] is None  # 2 chunks, nothing more


def test_ast_looks_closely_only_where_something_is():
    ast = FakeAST({"laughter": {"threshold": 0.4, "classes": ["Laughter"]}})
    x = noise(20.0, -60)
    x[int(12.0 * SR):int(13.0 * SR)] += sine(400, 1.0, 0.3)                        # second chunk only
    out = ast.detect(clip_of(x), [(0.5, 2.0)])
    (e,) = out["events"]
    assert e["label"] == "laughter" and 11.5 < e["start"] < 12.2 and 12.8 < e["end"] < 13.6
    assert ast.calls[:2] == [2, 19]            # coarse pass (2 chunks), then windows over the 2nd chunk (9.76 s) only
    assert out["background"] == [{"class": "Fan", "p": 0.3}]                      # one pass over the pauses


# ------------------------------------------------------------------ syllables from phonemes


def test_vowel_tokens_are_syllables_with_times():
    id2tok = {0: "<pad>", 1: "k", 2: "a", 3: "aɪ", 4: "t", 5: "eː", 6: "n"}
    ids = [0, 1, 2, 2, 0, 4, 3, 3, 0, 0, 5, 0, 5, 6]       # k a | t aɪ | eː _ eː n -> vowels at frames 2, 6, 10, 12
    t = models.vowel_times(ids, id2tok, 0, 0.02)
    assert t == pytest.approx([0.05, 0.13, 0.21, 0.25])
    assert models.is_vowel("ɛ") and models.is_vowel("eː") and not models.is_vowel("ʃ") and not models.is_vowel("")


@needs_models
def test_phoneme_model_counts_syllables_of_a_syllable_train():
    from conftest import syllables
    n = len(models.PhonemeSyllables(ctx()).nuclei(clip_of(syllables(4, 3.0))))
    assert 6 <= n <= 18                                    # a buzz, not speech: only checks it runs and counts


# ------------------------------------------------------------------ speakers (clustering, fake embeddings)


def test_cluster_cosine_groups_similar_vectors():
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    emb = np.array([a, a * 0.99 + b * 0.01, b, b * 0.98 + a * 0.02, a])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = models.cluster_cosine(emb, 0.86)
    assert labels == [0, 0, 1, 1, 0]
    assert models.cluster_cosine(emb, 0.0) == [0] * 5                     # everything merges
    assert len(set(models.cluster_cosine(emb, 0.999))) == 2               # only exact duplicates merge


class FakeSpeakers(models.WavLMSpeakers):
    """Real clustering, confidence and labelling; fake embeddings with a chosen similarity:
    phrase k is voice k % 2, and the two voices' vectors have cosine similarity `sim`."""

    def __init__(self, sim=0.0, **kw):
        self.threshold, self.same_threshold = kw.get("threshold", 0.5), kw.get("same_threshold", 0.8)
        self.min_phrase_s, self.max_seconds = kw.get("min_phrase_s", 1.0), 10.0
        a = np.array([1.0, 0.0])
        b = np.array([sim, np.sqrt(max(0.0, 1 - sim ** 2))])
        self.voices = [a, b]

    def embed(self, clip, phrases):
        return np.array([self.voices[i % 2] for i in range(len(phrases))])


def test_clearly_different_voices_are_a_confident_multi():
    phrases = [(0.0, 1.0), (1.2, 2.2), (2.4, 3.4)]
    out = FakeSpeakers(sim=0.1).detect(clip_of(noise(4.0, -30)), phrases)
    assert out["count"] == 2 and out["confident"] == "multi" and out["per_phrase"] == [0, 1, 0]
    assert out["change_times"] == [1.2, 2.4] and out["source"] == "wavlm-sv"
    assert FakeSpeakers().detect(clip_of(noise(1.0, -30)), []) is None


def test_one_expressive_voice_is_not_claimed_either_way():
    """Similarity between the clips' own phrases often sits between the two thresholds
    (a real single speaker was split into 3 "voices" at 0.86): then nothing is claimed."""
    phrases = [(0.0, 1.0), (1.2, 2.2), (2.4, 3.4)]
    out = FakeSpeakers(sim=0.65).detect(clip_of(noise(4.0, -30)), phrases)
    assert out["count"] == 1 and out["confident"] is None                 # merged, but not "single"
    near = FakeSpeakers(sim=0.9).detect(clip_of(noise(4.0, -30)), phrases)
    assert near["count"] == 1 and near["confident"] == "single"


def test_short_phrases_follow_the_nearest_voice_instead_of_adding_one():
    phrases = [(0.0, 1.2), (1.4, 2.6), (2.8, 3.0)]                        # last phrase is 200 ms
    out = FakeSpeakers(sim=0.1).detect(clip_of(noise(3.2, -30)), phrases)
    assert out["count"] == 2 and out["per_phrase"] == [0, 1, 0]           # labelled, but no third voice


@needs_models
def test_wavlm_embeddings_are_unit_vectors():
    sp = models.WavLMSpeakers(ctx())
    e = sp.embed(clip_of(utterance()), [(0.5, 2.0), (2.8, 4.5)])
    assert e.shape[0] == 2 and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)


# ------------------------------------------------------------------ weight-norm checkpoints


def test_fix_weight_norm_copies_the_renamed_tensors(tmp_path):
    """PyTorch >= 2.1 stores weight_norm as parametrizations; an older checkpoint's weight_g /
    weight_v then load as random numbers unless they are copied across by name."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file
    from torch import nn
    conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(4, 4, 3), name="weight", dim=2)
    model = nn.Sequential(conv)
    names = dict(model.named_parameters())
    assert "0.parametrizations.weight.original0" in names                  # new-style parameter names
    g, v = (torch.full_like(names["0.parametrizations.weight.original0"], 0.5),
            torch.full_like(names["0.parametrizations.weight.original1"], 0.25))
    save_file({"0.weight_g": g, "0.weight_v": v, "0.bias": names["0.bias"].detach()},
              str(tmp_path / "model.safetensors"))
    assert models.fix_weight_norm(model, tmp_path) == 2
    assert torch.allclose(names["0.parametrizations.weight.original0"], g)
    assert torch.allclose(names["0.parametrizations.weight.original1"], v)
    assert models.fix_weight_norm(model, tmp_path / "empty") == 0          # no checkpoint: nothing to do


@needs_models
@pytest.mark.parametrize("cls", ["PhonemeSyllables", "WavLMSpeakers"])
def test_frozen_models_load_their_weight_norm_tensors(cls):
    m = getattr(models, cls)(ctx())
    assert m.fixed_weight_norm > 0           # 0 would mean part of the network is random


def test_short_phrases_are_widened_so_wavlm_can_embed_them():
    """WavLM's TDNN head needs ~0.3 s; split_at_silences can emit 0.15 s phrases, which made
    conv1d fail with "Kernel size can't be greater than actual input size"."""
    x = np.arange(16000 * 4, dtype=np.float32)                     # 4 s, each sample = its index
    seg = models.embed_segment(x, SR, 1.0, 1.15, 0.5, 10.0)        # 150 ms phrase
    assert len(seg) == 8000                                        # widened to 0.5 s
    assert seg[0] < 1.075 * SR < seg[-1]                           # centred on the phrase
    assert np.array_equal(models.embed_segment(x, SR, 1.0, 2.0, 0.5, 10.0), x[SR:2 * SR])   # long enough: as is
    assert len(models.embed_segment(x, SR, 0.0, 0.1, 0.5, 10.0)) == 8000                    # at the start
    assert len(models.embed_segment(x, SR, 3.95, 4.0, 0.5, 10.0)) == 8000                   # at the end
    assert len(models.embed_segment(x, SR, 0.5, 20.0, 0.5, 10.0)) == int(3.5 * SR)          # to the clip's end
    long = np.zeros(15 * SR, dtype=np.float32)
    assert len(models.embed_segment(long, SR, 0.0, 15.0, 0.5, 10.0)) == 10 * SR             # capped at max_s
    tiny = np.ones(1000, dtype=np.float32)                         # clip shorter than min_s: zero-padded
    out = models.embed_segment(tiny, SR, 0.0, 0.06, 0.5, 10.0)
    assert len(out) == 8000 and out[-1] == 0.0


def test_silero_does_not_throttle_the_whole_process_by_default():
    """torch.set_num_threads is process-wide: Silero defaulting to 1 thread also made Whisper,
    AST, the phoneme model and WavLM run single-threaded (CPU runs were ~4x slower)."""
    import inspect
    assert inspect.signature(models.SileroVAD.__init__).parameters["threads"].default is None
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    try:
        models.SileroVAD.__init__(object.__new__(models.SileroVAD), {"models_dir": ROOT / "models",
                                                                     "verify": False})
    except Exception:
        pass                                              # no model files here: the thread check is the point
    assert torch.get_num_threads() == before
