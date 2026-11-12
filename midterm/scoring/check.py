"""Smoke test of every scoring model on one devset clip per language. Run it once on the HPC GPU node
before the real job, so a missing model or package shows up in a minute, not after hours:

    python -m scoring.check --settings settings.yaml [--skip judge]

Prints PASS/FAIL per model with the reason, logs to <model_dir>/check-logs/check.log, and exits
non-zero if anything failed.
"""
import argparse
import glob
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--skip", default="", help="comma-separated checks to skip: asr,speaker,naturalness,judge")
    args = parser.parse_args()

    from scoring.run import stage_environment
    from scoring.settings import load_settings
    settings = load_settings(args.settings)
    environment = stage_environment(settings)
    os.environ.update({k: v for k, v in environment.items() if k != "PYTHONPATH"})       # offline, caches
    sys.path[:0] = [p for p in environment["PYTHONPATH"].split(os.pathsep) if p not in sys.path]

    from scoring.common import log_environment, setup_logging
    logger = setup_logging(os.path.join(settings.model_dir, "check-logs"), "check")
    log_environment(logger)
    skip = set(filter(None, args.skip.split(",")))

    clips = {}
    for language in ("hi", "ta"):
        found = sorted(glob.glob(os.path.join(settings.devset, language, "clips", "*.wav")))
        if not found:
            logger.critical("no clips in %s", os.path.join(settings.devset, language, "clips"))
            sys.exit(2)
        clips[language] = found[0]

    results = {}

    def check(name, function):
        if name in skip:
            logger.info("SKIP %s", name)
            return
        started = time.time()
        try:
            detail = function()
            results[name] = True
            logger.info("PASS %s (%.0f s): %s", name, time.time() - started, detail)
        except Exception as error:
            results[name] = False
            logger.error("FAIL %s: %r", name, error, exc_info=error)

    def asr():
        from scoring.models import IndicConformer
        model = IndicConformer(settings, logger)
        return {language: model.transcribe(path, language)[:80] for language, path in clips.items()}

    def speaker():
        import numpy as np
        from scoring.models import SpeakerEmbedder
        model = SpeakerEmbedder(settings, logger)
        a, b = model.embed(clips["hi"]), model.embed(clips["ta"])
        return {"same_clip_similarity": float(np.dot(a, a)), "hi_vs_ta_clip_similarity": float(np.dot(a, b)),
                "threshold": settings.speaker_diversity.threshold}

    def naturalness():
        from scoring.stages.naturalness import stub_psp_eval
        if settings.naturalness.psp_eval_dir is None:
            stub_psp_eval(logger)
        import nat
        from scoring.models import snapshot_folder
        scorer = nat.NaturalnessScorer(settings.device, settings.speechmos_dir, settings.sigmos_dir,
                                       snapshot_folder(settings, settings.models.xlsr), settings.naturalness.xlsr_layer)
        return {"utmos": scorer.utmos_score(clips["hi"], None),
                "sigmos.ovrl": scorer.sigmos_scores(clips["hi"], None).get("sigmos.ovrl"),
                "xlsr_embedding_dim": int(scorer.embedding(clips["hi"], None).shape[0])}

    def judge():
        from scoring.llm import TextJudge
        model = TextJudge(settings, logger)
        statements = model.extract("A young woman speaks Hindi calmly at a moderate pace. She laughs at 2.1-2.6 s.")
        labels, missing = model.grade_rubric(statements, "A woman speaks Hindi in a calm voice and laughs around 2.3 s.")
        return {"statements": [s["statement"] for s in statements], "labels": labels, "missing": missing}

    for name, function in (("asr", asr), ("speaker", speaker), ("naturalness", naturalness), ("judge", judge)):
        check(name, function)

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.error("checks failed: %s (details above)", failed)
        sys.exit(1)
    logger.info("all checks passed: %s", sorted(results))


if __name__ == "__main__":
    main()
