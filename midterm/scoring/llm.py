"""The text-LLM judge: splits descriptions into checkable statements and grades them.

Used two ways, both approximations of what Rime's people do:

  rubric score (captioner)        reference description -> rubric items; each item is stated,
                                  contradicted or not stated by the captioner's description
  instruction relevance (generator)   the generator's instruction -> checks; each check is met,
                                  not met or unverifiable, judged from a caption of the generated audio

The parsing and scoring functions are pure (no model), so they are unit tested.
"""
import json
import logging
import re

RUBRIC_LABELS = ("stated", "contradicted", "not_stated")
RELEVANCE_LABELS = ("met", "not_met", "unverifiable")

EXTRACT_SYSTEM = """You turn a description of a speech recording into a rubric: a list of atomic statements that a careful listener could each check by ear.
Rules:
- One fact per statement. Split "a calm, low-pitched male voice" into three statements.
- Keep every number, unit and time exactly as written. Do not add facts that are not in the description.
- If a statement is about a specific moment or span, give its start and end in seconds; otherwise use null for both.
- Skip statements that are not about the audio (for example notes about the method).
Answer with JSON only: a list of objects {"statement": string, "start": number or null, "end": number or null}."""

RUBRIC_SYSTEM = """You grade a candidate description of a speech recording against rubric items that are TRUE for the recording.
For each item give one label:
- "stated": the candidate clearly commits to the same fact. A guess between alternatives ("male or female") does not count. For an item with a time span, the candidate must place it inside the span widened by 0.25 s on each side.
- "contradicted": the candidate asserts something incompatible with the item.
- "not_stated": the candidate does not commit either way.
Answer with JSON only: a list of objects {"id": number, "label": "stated" | "contradicted" | "not_stated"}, one per item."""

RELEVANCE_SYSTEM = """You check whether generated speech followed an instruction. You cannot hear the audio; you get a caption of it written by an audio captioner.
For each check give one label:
- "met": the caption clearly says the audio has this property.
- "not_met": the caption clearly says something incompatible with the check.
- "unverifiable": the caption does not say either way.
Answer with JSON only: a list of objects {"id": number, "label": "met" | "not_met" | "unverifiable"}, one per check."""


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------

def truncate_words(text: str, max_words: int) -> str:
    """Rime cuts every description at 300 words before grading."""
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def parse_json(reply: str):
    """The first JSON list or object in a model reply, ignoring <think> blocks and code fences."""
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in reply: {reply[:300]!r}")


def clean_statements(value) -> list[dict]:
    """Validate the extractor's output into [{"statement", "start", "end"}]."""
    if isinstance(value, dict):
        value = next((v for v in value.values() if isinstance(v, list)), [])
    statements = []
    for entry in value if isinstance(value, list) else []:
        if isinstance(entry, str):
            entry = {"statement": entry}
        if not isinstance(entry, dict) or not str(entry.get("statement", "")).strip():
            continue
        def seconds(x):
            try:
                return float(x) if x is not None else None
            except (TypeError, ValueError):
                return None
        statements.append({"statement": str(entry["statement"]).strip(),
                           "start": seconds(entry.get("start")), "end": seconds(entry.get("end"))})
    return statements


def collect_labels(value, n: int, allowed: tuple[str, ...]) -> tuple[list[str | None], int]:
    """Labels for ids 1..n from the judge's reply. Returns (labels, number of ids missing or invalid);
    a missing label is None and the caller decides what it counts as."""
    if isinstance(value, dict):
        value = next((v for v in value.values() if isinstance(v, list)), [])
    labels: list[str | None] = [None] * n
    for entry in value if isinstance(value, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("id")) - 1
        except (TypeError, ValueError):
            continue
        label = str(entry.get("label", "")).strip().lower().replace(" ", "_")
        if 0 <= index < n and label in allowed:
            labels[index] = label
    return labels, sum(label is None for label in labels)


def rubric_score(labels: list[str | None]) -> float:
    """Rime's clip score with equal weights: (stated - contradicted) / items, floored at 0.
    A missing label counts as not stated."""
    if not labels:
        raise ValueError("no rubric items")
    stated = sum(label == "stated" for label in labels)
    contradicted = sum(label == "contradicted" for label in labels)
    return max(0.0, (stated - contradicted) / len(labels))


def format_items(statements: list[dict]) -> str:
    lines = []
    for number, s in enumerate(statements, 1):
        span = f" [{s['start']:.2f}-{s['end']:.2f} s]" if s.get("start") is not None and s.get("end") is not None else ""
        lines.append(f"{number}. {s['statement']}{span}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------

class TextJudge:
    """Qwen3 (or any chat model) with greedy decoding and thinking switched off."""

    def __init__(self, settings, logger: logging.Logger):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo = settings.models.judge
        from scoring.models import snapshot_folder
        folder = snapshot_folder(settings, repo)
        self.torch = torch
        self.logger = logger
        self.max_new_tokens = settings.judge.max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.model = AutoModelForCausalLM.from_pretrained(folder, dtype=settings.judge.dtype, device_map="auto").eval()
        logger.info("judge loaded: %s@%s (%s)", repo.repo_id, repo.revision, settings.judge.dtype)

    def ask(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         temperature=None, top_p=None, top_k=None)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        if len(new_tokens) >= self.max_new_tokens:
            self.logger.warning("judge reply hit max_new_tokens=%d; it may be cut off", self.max_new_tokens)
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def ask_json(self, system: str, user: str):
        """Ask, parse JSON; on a parse failure ask once more with a reminder. Raises if both fail."""
        reply = self.ask(system, user)
        try:
            return parse_json(reply)
        except ValueError:
            self.logger.warning("judge reply was not JSON, asking again: %r", reply[:300])
            reply = self.ask(system, user + "\n\nAnswer with the JSON only, nothing else.")
            return parse_json(reply)

    def extract(self, description: str) -> list[dict]:
        statements = clean_statements(self.ask_json(EXTRACT_SYSTEM, f"Description:\n{description}"))
        if not statements:
            raise ValueError("the judge found no checkable statements")
        return statements

    def grade_rubric(self, statements: list[dict], candidate: str) -> tuple[list[str | None], int]:
        user = f"Rubric items (true for the recording):\n{format_items(statements)}\n\nCandidate description:\n{candidate}"
        return collect_labels(self.ask_json(RUBRIC_SYSTEM, user), len(statements), RUBRIC_LABELS)

    def grade_relevance(self, checks: list[dict], caption: str) -> tuple[list[str | None], int]:
        user = f"Checks (from the instruction):\n{format_items(checks)}\n\nCaption of the generated audio:\n{caption}"
        return collect_labels(self.ask_json(RELEVANCE_SYSTEM, user), len(checks), RELEVANCE_LABELS)
