"""Speaker diversity (generator): distinct speakers among the samples of each item, by a speaker
verification model at its same-speaker threshold. Rime counts among the samples that pass the
relevance checks; here every sample is counted (relevance is only judged on a few samples).

Output speaker_diversity.jsonl, one line per item:
    clip_id, language, n_samples, n_speakers, min_similarity, max_similarity
"""
import os

import numpy as np

from scoring.common import append_jsonl, read_jsonl, record_error
from scoring.stage import main

NAME = "speaker_diversity"


def run(settings, items, scores_dir, run_folder, logger):
    from scoring.models import SpeakerEmbedder, count_speakers

    output = os.path.join(scores_dir, f"{NAME}.jsonl")
    done_ids = {r["clip_id"] for r in read_jsonl(output)}
    todo = [i for i in items if i.samples and i.clip_id not in done_ids]
    sizes = sorted({len(i.samples) for i in todo})
    logger.info("%d items to score (%d already done); samples per item: %s", len(todo), len(done_ids), sizes)
    if any(size < 10 for size in sizes):
        logger.warning("Rime counts speakers among 10 samples per description; this run has fewer for some "
                       "items, so the count is not comparable (run round 0 with settings_eval.yaml)")
    if not todo:
        return 0, 0

    embedder = SpeakerEmbedder(settings, logger)
    threshold = settings.speaker_diversity.threshold
    done = failed = 0
    for number, item in enumerate(todo, 1):
        try:
            embeddings = [embedder.embed(s.path) for s in item.samples]
            n_speakers = count_speakers(embeddings, threshold)
            similarity = np.array([[float(np.dot(a, b)) for b in embeddings] for a in embeddings])
            off_diagonal = similarity[~np.eye(len(embeddings), dtype=bool)]
            append_jsonl(output, {
                "clip_id": item.clip_id, "language": item.language, "n_samples": len(embeddings),
                "n_speakers": n_speakers,
                "min_similarity": float(off_diagonal.min()) if off_diagonal.size else None,
                "max_similarity": float(off_diagonal.max()) if off_diagonal.size else None})
            done += 1
            logger.info("[%d/%d] %s: %d speakers in %d samples", number, len(todo), item.clip_id, n_speakers, len(embeddings))
        except Exception as error:
            failed += 1
            record_error(scores_dir, NAME, item.clip_id, error, logger)
    return done, failed


if __name__ == "__main__":
    main(NAME, run)
