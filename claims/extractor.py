"""Claim extractors: turn an existing caption into a list of claims.

ClaimExtractorInterface is what the rest of the code uses. RuleBasedExtractor is the first
implementation: it needs no model, so ladder rung 2 measures the composer and nothing else. A
model-based extractor (rungs 3 and later) is another implementation of the same interface.
"""
import re
from abc import ABC, abstractmethod

from claims.schema import Claim
from claims.settings import ExtractorOptions
from evaluation.rubric.schema import ItemType

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")       # after a full stop, not inside "3.2 s"
LIST_MARKER = re.compile(r"^[\s\-*#•>]+")           # markdown bullets and headings at the start of a line
MARKDOWN_EMPHASIS = re.compile(r"[*_`]+")


class ClaimExtractorInterface(ABC):
    """Interface for claim extractors."""

    @abstractmethod
    def extract(self, caption: str) -> list[Claim]:
        """Return the claims a caption makes, in the order it makes them."""


def split_sentences(caption: str) -> list[str]:
    """The sentences of a caption, with markdown list markers and emphasis removed."""
    sentences = []
    for line in caption.splitlines():
        line = LIST_MARKER.sub("", line)
        line = MARKDOWN_EMPHASIS.sub("", line)
        for sentence in SENTENCE_BREAK.split(line):
            sentence = " ".join(sentence.split())
            if sentence != "":
                sentences.append(sentence)
    return sentences


def contains_keyword(sentence: str, keyword: str) -> bool:
    """True if `keyword` is in `sentence` as a whole word or phrase, ignoring case."""
    return re.search(r"\b" + re.escape(keyword.lower()) + r"\b", sentence.lower()) is not None


class RuleBasedExtractor(ClaimExtractorInterface):
    """One claim per sentence. The item type is the first type in the settings whose keyword the
    sentence contains; every claim gets the same probability. CAREFUL X1"""

    def __init__(self, options: ExtractorOptions):
        self.options = options

    def item_type_of(self, sentence: str) -> ItemType:
        for item_type, keywords in self.options.type_keywords.items():
            for keyword in keywords:
                if contains_keyword(sentence, keyword):
                    return item_type
        return self.options.fallback_type

    def extract(self, caption: str) -> list[Claim]:
        claims = []
        for sentence in split_sentences(caption):
            if len(sentence) < self.options.min_chars:
                continue        # a heading or a stray list number, not a claim
            claims.append(Claim(item_type=self.item_type_of(sentence), statement=sentence,
                                start_s=None, end_s=None,       # the sentence keeps its own wording of any time
                                source="model", probability=self.options.probability))
        return claims


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# X1   Typing by keyword is rough: a sentence about several things gets the first matching type,
#      and one sentence is one claim even if it states three facts. That is acceptable for rung 2,
#      where all priors and probabilities are equal and the type does not change the ranking. It is
#      not good enough once priors differ: use a model-based extractor then.
# ---------------------------------------------------------------------------------------------
