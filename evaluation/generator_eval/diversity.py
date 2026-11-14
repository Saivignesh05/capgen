"""Speaker diversity (problem statement) and speaker similarity (doubt session), both from the
embeddings of a speaker-verification model (docs/evaluation_plan.md, sections 4.4 and 4.5).

    diversity   distinct speakers among the samples of one description, divided by the number
                of samples the protocol asks for (10)
    similarity  cosine similarity between a generated sample and the target recording
"""
import math

from evaluation.generator_eval.interfaces import SpeakerEmbedderInterface


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError(f"embeddings of different sizes: {len(first)} and {len(second)}")
    dot = sum(a * b for a, b in zip(first, second))
    norm_first = math.sqrt(sum(a * a for a in first))
    norm_second = math.sqrt(sum(b * b for b in second))
    if norm_first == 0.0 or norm_second == 0.0:
        raise ValueError("an embedding is all zeros")
    return dot / (norm_first * norm_second)


def count_distinct_speakers(embeddings: list[list[float]], same_speaker_threshold: float) -> int:
    """Distinct speakers among samples, counted greedily in sample order. CAREFUL D1

    The first sample is a speaker. Each later sample is a new speaker if its similarity to every
    speaker kept so far is below the threshold; otherwise it is one of them and is not kept.
    """
    representatives: list[list[float]] = []
    for embedding in embeddings:
        is_new = True
        for representative in representatives:
            if cosine_similarity(embedding, representative) >= same_speaker_threshold:
                is_new = False
                break
        if is_new:
            representatives.append(embedding)
    return len(representatives)


def speaker_diversity(embeddings: list[list[float]], same_speaker_threshold: float, samples_per_description: int) -> float:
    """Distinct speakers divided by the number of samples the protocol asks for. Fewer samples than
    that is an error: the count would not be comparable."""
    if len(embeddings) != samples_per_description:
        raise ValueError(f"speaker diversity needs {samples_per_description} samples, got {len(embeddings)}")
    return count_distinct_speakers(embeddings, same_speaker_threshold) / samples_per_description


def similarity_to_target(embedder: SpeakerEmbedderInterface, generated_path: str, target_path: str) -> float:
    """Speaker similarity of one generated sample to the recording it should sound like."""
    return cosine_similarity(embedder.embed(generated_path), embedder.embed(target_path))


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# D1   The official clustering rule is not published. The evaluation plan asks for this greedy
#      count, which depends on the order of the samples (they are always given in sample-number
#      order, so the result is still fixed). The midterm scoring used complete-linkage clustering
#      instead; the two can differ by a speaker on borderline samples. Open choice, listed in
#      docs/EXPLANATION.md. The plan counts only samples that pass the relevance check; relevance
#      is not built in Phase 0, so every sample is counted (the plan's "unconditioned count").
# ---------------------------------------------------------------------------------------------
