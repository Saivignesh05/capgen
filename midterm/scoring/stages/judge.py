"""The text-LLM judge (Qwen3-8B), for the two measures that Rime grades with people or rubrics:

rubric score (captioner), once per reference source in settings.references:
    the reference description of a clip is split into rubric items (all true for the clip), and the
    round 0 caption of that clip, cut at 300 words, is graded item by item:
    score = (stated - contradicted) / items, floored at 0 (Rime's formula, equal weights).

instruction relevance (generator):
    the generator's instruction (the round 0 caption it was given) is split into checks, and each
    check is judged met / not met / unverifiable from the caption of a generated sample
    (caption_generated stage). relevance = met / checks.

Outputs:
    extractions.jsonl   cache: text sha -> extracted statements (reused across attempts and measures)
    rubric.jsonl        one line per (reference source, clip)
    relevance.jsonl     one line per generated sample
"""
import os

from scoring.common import append_jsonl, read_json_or_jsonl, read_jsonl, record_error, text_sha
from scoring.llm import TextJudge, rubric_score, truncate_words
from scoring.stage import main

NAME = "judge"


def load_references(reference, logger) -> dict[str, str]:
    """clip_id -> reference text, from every file of one reference source."""
    texts = {}
    for path in reference.files:
        if not os.path.isfile(path):
            logger.warning("reference %s: %s does not exist", reference.name, path)
            continue
        try:
            rows = read_json_or_jsonl(path)
        except ValueError as error:
            logger.error("reference %s: %s cannot be read, its clips are skipped: %s", reference.name, path, error)
            continue
        if not rows:
            logger.warning("reference %s: %s is empty", reference.name, path)
        without_text = 0
        for row in rows:
            text = str(row.get(reference.text_field) or "").strip()
            without_text += not text
            audio_path = row.get("audio_path") or row.get("clip_id") or ""
            if text and audio_path:
                texts[os.path.splitext(os.path.basename(audio_path))[0]] = text
        if without_text:
            logger.warning("reference %s: %d entries in %s have no '%s' field (check text_field in settings.yaml)",
                           reference.name, without_text, path, reference.text_field)
    logger.info("reference %s: %d clips with a reference", reference.name, len(texts))
    return texts


class Extractions:
    """Statements extracted from a text, cached on disk by the text's SHA-256."""

    def __init__(self, path: str, judge: TextJudge):
        self.path, self.judge = path, judge
        self.cache = {r["sha"]: r["statements"] for r in read_jsonl(path)}

    def get(self, text: str, source: str) -> list[dict]:
        sha = text_sha(text)
        if sha not in self.cache:
            self.cache[sha] = self.judge.extract(text)
            append_jsonl(self.path, {"sha": sha, "source": source, "text": text, "statements": self.cache[sha]})
        return self.cache[sha]


def run(settings, items, scores_dir, run_folder, logger):
    rubric_path = os.path.join(scores_dir, "rubric.jsonl")
    relevance_path = os.path.join(scores_dir, "relevance.jsonl")
    rubric_done = {(r["reference"], r["clip_id"]) for r in read_jsonl(rubric_path)}
    relevance_done = {(r["clip_id"], r["sample"]) for r in read_jsonl(relevance_path)}
    max_words = settings.judge.max_description_words
    by_id = {i.clip_id: i for i in items}

    # ---- plan everything first, so nothing loads the judge if there is nothing to do ----
    rubric_todo = []
    for reference in settings.references:
        texts = load_references(reference, logger)
        if not texts:
            logger.warning("no references for %s: measure %s is skipped", reference.name, reference.measure)
        for item in items:
            if (reference.name, item.clip_id) in rubric_done:
                continue
            if item.clip_id in texts and item.description:
                rubric_todo.append((reference, item, texts[item.clip_id]))
            elif item.clip_id in texts:
                logger.warning("%s: has a %s reference but no round 0 description", item.clip_id, reference.name)
    captions = read_jsonl(os.path.join(scores_dir, "caption_generated.jsonl"))
    if not captions:
        logger.warning("no captions of generated audio (caption_generated stage): instruction relevance is skipped")
    relevance_todo = [c for c in captions if (c["clip_id"], c["sample"]) not in relevance_done
                      and by_id.get(c["clip_id"]) and by_id[c["clip_id"]].description]
    logger.info("to do: %d rubric gradings, %d relevance gradings (done before: %d, %d)",
                len(rubric_todo), len(relevance_todo), len(rubric_done), len(relevance_done))
    if not rubric_todo and not relevance_todo:
        return 0, 0

    judge = TextJudge(settings, logger)
    extractions = Extractions(os.path.join(scores_dir, "extractions.jsonl"), judge)
    done = failed = 0

    for number, (reference, item, text) in enumerate(rubric_todo, 1):
        key = f"rubric/{reference.name}/{item.clip_id}"
        try:
            statements = extractions.get(text, f"reference:{reference.name}")
            candidate = truncate_words(item.description, max_words)
            labels, missing = judge.grade_rubric(statements, candidate)
            if missing:
                logger.warning("%s: judge gave no valid label for %d of %d items (counted as not stated)", key, missing, len(labels))
            score = rubric_score(labels)
            append_jsonl(rubric_path, {
                "reference": reference.name, "measure": reference.measure, "clip_id": item.clip_id,
                "language": item.language, "n_items": len(labels), "missing_labels": missing,
                "stated": labels.count("stated"), "contradicted": labels.count("contradicted"),
                "not_stated": len(labels) - labels.count("stated") - labels.count("contradicted"),
                "score": score, "labels": labels})
            done += 1
            logger.info("[rubric %d/%d] %s score=%.3f (%d items)", number, len(rubric_todo), key, score, len(labels))
        except Exception as error:
            failed += 1
            record_error(scores_dir, NAME, key, error, logger)

    for number, caption in enumerate(relevance_todo, 1):
        item = by_id[caption["clip_id"]]
        key = f"relevance/{item.clip_id}#{caption['sample']}"
        try:
            checks = extractions.get(item.description, "instruction")
            labels, missing = judge.grade_relevance(checks, caption["caption"])
            if missing:
                logger.warning("%s: judge gave no valid label for %d of %d checks (counted as not met)", key, missing, len(labels))
            met = labels.count("met")
            append_jsonl(relevance_path, {
                "clip_id": item.clip_id, "sample": caption["sample"], "language": item.language,
                "n_checks": len(labels), "missing_labels": missing, "met": met,
                "not_met": labels.count("not_met"), "unverifiable": labels.count("unverifiable"),
                "share": met / len(labels), "labels": labels})
            done += 1
            logger.info("[relevance %d/%d] %s met %d/%d", number, len(relevance_todo), key, met, len(labels))
        except Exception as error:
            failed += 1
            record_error(scores_dir, NAME, key, error, logger)
    return done, failed


if __name__ == "__main__":
    main(NAME, run)
