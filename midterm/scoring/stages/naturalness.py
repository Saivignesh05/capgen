"""Naturalness (generator), automatic stand-ins for Rime's blind CMOS, computed with nat.py:

    utmos, sigmos.*   per sample (higher is better)
    fad, psd          per language, generated samples vs the native devset clips (lower is better)

Rime's naturalness is a listening test against other teams, which cannot be run here.
The native devset clips are scored with UTMOS and SIGMOS too, as the level real speech gets.

Outputs (rewritten on every attempt, as this stage is fast):
    naturalness.jsonl          one line per sample: clip_id, sample, language, utmos, sigmos.*
    naturalness_native.jsonl   one line per devset clip: clip_id, language, utmos, sigmos.*
    naturalness_corpus.json    {language: {"fad", "psd", "n_generated", "n_native"}}
"""
import os
import sys
import types

from scoring.common import append_jsonl, record_error, write_json
from scoring.stage import main

NAME = "naturalness"


def stub_psp_eval(logger) -> None:
    """nat.py imports psp-eval's evaluation.fad / evaluation.psd at the top. Without psp-eval, put
    stand-ins in their place so UTMOS and SIGMOS still run; FAD and PSD are then skipped."""
    def missing(*_args, **_kwargs):
        raise RuntimeError("psp-eval is not installed: set naturalness.psp_eval_dir in midterm/settings.yaml")
    package = types.ModuleType("evaluation")
    for name, functions in (("fad", ["compute_fad"]), ("psd", ["compute_psd", "extract_prosodic_features"])):
        module = types.ModuleType(f"evaluation.{name}")
        for function in functions:
            setattr(module, function, missing)
        setattr(package, name, module)
        sys.modules[f"evaluation.{name}"] = module
    sys.modules["evaluation"] = package
    logger.warning("naturalness.psp_eval_dir is not set: FAD and PSD are skipped, UTMOS and SIGMOS still run")


def run(settings, items, scores_dir, run_folder, logger):
    options = settings.naturalness
    metrics = list(options.metrics)
    if options.psp_eval_dir is None:
        stub_psp_eval(logger)
        metrics = [m for m in metrics if m not in ("fad", "psd")]
    elif not os.path.isfile(os.path.join(options.psp_eval_dir, "evaluation", "fad.py")):
        raise FileNotFoundError(f"naturalness.psp_eval_dir={options.psp_eval_dir} has no evaluation/fad.py")
    if not os.path.isfile(os.path.join(settings.sigmos_dir, "sigmos.py")):
        raise FileNotFoundError(f"{settings.sigmos_dir}/sigmos.py is missing: run scoring/download.py")
    logger.info("metrics: %s", metrics)

    import nat      # repo root; imports sigmos and psp-eval at the top
    from scoring.models import fix_weight_norm, snapshot_folder

    xlsr_folder = snapshot_folder(settings, settings.models.xlsr)
    scorer = nat.NaturalnessScorer(settings.device, settings.speechmos_dir, settings.sigmos_dir,
                                   xlsr_folder, options.xlsr_layer)
    logger.info("nat.py scorer loaded (XLS-R weight-norm tensors re-copied: %d)", fix_weight_norm(scorer.xlsr, xlsr_folder))

    per_sample = os.path.join(scores_dir, f"{NAME}.jsonl")
    native = os.path.join(scores_dir, f"{NAME}_native.jsonl")
    for path in (per_sample, native):
        if os.path.exists(path):
            os.remove(path)

    done = failed = 0
    corpus = {}
    for language in sorted({i.language for i in items}):
        language_items = [i for i in items if i.language == language]
        generated = {f"{i.clip_id}#{s.sample}": (s.path, None) for i in language_items for s in i.samples}
        natives = {i.clip_id: (i.clip_path, None) for i in language_items}
        logger.info("%s: %d generated samples, %d native clips", language, len(generated), len(natives))
        if not generated:
            continue
        try:
            references = (None, None)
            if "fad" in metrics or "psd" in metrics:
                references = scorer.build_references(natives)
            result = scorer.score_clips(generated, references[0], references[1], metrics)
            per_clip_metrics = [m for m in metrics if m in ("utmos", "sigmos")]
            native_result = scorer.score_clips(natives, None, None, per_clip_metrics) if per_clip_metrics else {"clips": {}}
        except Exception as error:
            failed += len(generated)
            record_error(scores_dir, NAME, language, error, logger)
            continue
        for key, scores in result["clips"].items():
            clip_id, sample = key.rsplit("#", 1)
            append_jsonl(per_sample, {"clip_id": clip_id, "sample": int(sample), "language": language, **scores})
            done += 1
        for clip_id, scores in native_result["clips"].items():
            append_jsonl(native, {"clip_id": clip_id, "language": language, **scores})
        corpus[language] = {**result["corpus"], "n_generated": len(generated), "n_native": len(natives)}
        logger.info("%s: corpus %s", language, corpus[language])
        if len(generated) < 100:
            logger.warning("%s: FAD/PSD on fewer than 100 clips are noisy (nat.py recommends 100+)", language)
    write_json(os.path.join(scores_dir, f"{NAME}_corpus.json"), corpus)
    return done, failed


if __name__ == "__main__":
    main(NAME, run)
