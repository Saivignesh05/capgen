"""Intelligibility (generator): CER of IndicConformer on each generated sample, against the text the
generator was asked to say. Rime: CER with AI4Bharat IndicConformer against the native-script text,
averaged over descriptions.

Output intelligibility.jsonl, one line per sample:
    clip_id, sample, language, reference, hypothesis, edits, n (reference units), cer
"""
import os

from scoring.common import append_jsonl, read_jsonl, record_error
from scoring.stage import main

NAME = "intelligibility"


def run(settings, items, scores_dir, run_folder, logger):
    from normalize_hi_ta import normalize_text      # repo root, on PYTHONPATH (scoring.run)
    from score_hi_ta import score_pair
    from scoring.models import IndicConformer

    output = os.path.join(scores_dir, f"{NAME}.jsonl")
    done_keys = {(r["clip_id"], r["sample"]) for r in read_jsonl(output)}
    options = settings.intelligibility
    todo = []
    for item in items:
        if not item.transcript:
            logger.warning("%s: no transcript (no speech, or ASR failed); not scored", item.clip_id)
            continue
        try:        # CAREFUL I1
            normalize_text(item.transcript, item.language, hindi_lenient=options.hindi_lenient, on_foreign="raise")
        except ValueError as error:
            logger.warning("%s: reference is not clean %s script, not scored: %s", item.clip_id, item.language, error)
            continue
        todo += [(item, s) for s in item.samples if (item.clip_id, s.sample) not in done_keys]
    logger.info("%d samples to transcribe (%d already done)", len(todo), len(done_keys))
    if not todo:
        return 0, 0

    asr = IndicConformer(settings, logger)
    done = failed = 0
    for number, (item, sample) in enumerate(todo, 1):
        key = f"{item.clip_id}#{sample.sample}"
        try:
            hypothesis = asr.transcribe(sample.path, item.language)
            scored = score_pair(item.transcript, hypothesis, item.language, options.hindi_lenient)
            counts = scored[options.cer_metric]
            if counts["N"] == 0:
                raise ValueError("reference is empty after normalisation")
            append_jsonl(output, {
                "clip_id": item.clip_id, "sample": sample.sample, "language": item.language,
                "reference": item.transcript, "hypothesis": hypothesis,
                "edits": counts["edits"], "n": counts["N"], "cer": counts["edits"] / counts["N"]})
            done += 1
            logger.info("[%d/%d] %s cer=%.3f", number, len(todo), key, counts["edits"] / counts["N"])
        except Exception as error:
            failed += 1
            record_error(scores_dir, NAME, key, error, logger)
    return done, failed


if __name__ == "__main__":
    main(NAME, run)

# CAREFUL I1  score_hi_ta refuses a reference with Latin letters (code-mixed speech), so such clips are
#             logged and skipped here before any transcription. The count shows up in the log.
