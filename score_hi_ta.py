"""
score_hi_ta.py - intelligibility scoring (CER / WER) for Hindi and Tamil ONLY.

Input: a CSV of transcript pairs. Required columns:
    reference   - the text that SHOULD have been said (target text)
    hypothesis  - the ASR transcript of the generated audio
    lang        - "hi" or "ta" (also accepts "hindi" / "tamil")
Optional columns:
    id, speaker

Usage:
    python score_hi_ta.py --input pairs.csv --out results
    python score_hi_ta.py --input pairs.csv --out results --worst 10 --hindi-lenient

Writes:
    results_per_clip.csv   one row per clip: raw + normalized text, every metric, S/D/I counts
    results_summary.json   overall / per-language / per-speaker numbers with CIs

Metrics (all computed on text normalized by normalize_hi_ta.py):
    cer_ns  character error rate, spaces removed   <-- PRIMARY (headline) metric
    wer     word error rate                        <-- reported next to it for comparability
    cer     character error rate, spaces included  (diagnostic)
    gcer    grapheme-cluster (akshara) error rate, spaces removed (diagnostic)

Group-level rates are total edits / total reference units (NOT a mean of
per-clip rates), so short clips do not skew the result. Metrics are NOT blended
into one weighted score: they count the same mistakes at different granularities.

Your original texts are never modified. Normalization (punctuation stripped, spaces
removed for cer_ns/gcer) happens on copies inside this script. Feed the ORIGINAL
text to your TTS, not the normalized one.
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict

from normalize_hi_ta import NORMALIZER_VERSION, graphemes, normalize_text

METRICS = ("cer_ns", "wer", "cer", "gcer")
PRIMARY = "cer_ns"
LABELS = {
    "cer_ns": "CER (no space)",
    "wer": "WER",
    "cer": "CER",
    "gcer": "grapheme CER",
}
LANG_ALIASES = {"hi": "hi", "hindi": "hi", "ta": "ta", "tamil": "ta"}


# --------------------------------------------------------------------------- #
# Edit distance with substitution / deletion / insertion counts
# --------------------------------------------------------------------------- #
def align_counts(ref, hyp):
    """Levenshtein alignment between two sequences. Returns (S, D, I)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)

    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def _units(text):
    """The four views of one normalized text."""
    no_space = text.replace(" ", "")
    return {
        "cer": list(text),
        "cer_ns": list(no_space),
        "gcer": graphemes(no_space),
        "wer": text.split(),
    }


def score_pair(reference, hypothesis, lang, hindi_lenient=False):
    """Score one clip. Reference must be clean Hindi/Tamil (raises ValueError otherwise).
    Foreign characters in the hypothesis are kept, so they count as errors."""
    ref_n = normalize_text(reference, lang, hindi_lenient=hindi_lenient, on_foreign="raise")
    hyp_n = normalize_text(hypothesis, lang, hindi_lenient=hindi_lenient, on_foreign="keep")
    ref_u, hyp_u = _units(ref_n), _units(hyp_n)

    out = {"ref_norm": ref_n, "hyp_norm": hyp_n}
    for m in METRICS:
        s, d, i = align_counts(ref_u[m], hyp_u[m])
        n = len(ref_u[m])
        out[m] = {"S": s, "D": d, "I": i, "N": n, "edits": s + d + i}
    return out


# --------------------------------------------------------------------------- #
# Aggregation and bootstrap
# --------------------------------------------------------------------------- #
def pooled_rate(pairs):
    """pairs = [(edits, N), ...] -> total edits / total N."""
    n = sum(p[1] for p in pairs)
    return sum(p[0] for p in pairs) / n if n else float("nan")


def bootstrap_ci(pairs, iters=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI over clips (resampling clips with replacement)."""
    if len(pairs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(pairs)
    rates = []
    for _ in range(iters):
        sample = [pairs[rng.randrange(k)] for _ in range(k)]
        rates.append(pooled_rate(sample))
    rates.sort()
    lo = rates[int((alpha / 2) * iters)]
    hi = rates[min(iters - 1, int((1 - alpha / 2) * iters))]
    return lo, hi


def summarize(rows, iters, seed):
    """rows = list of per-clip dicts. Returns {metric: {rate, ci_low, ci_high, ...}}."""
    summary = {"n_clips": len(rows)}
    for m in METRICS:
        pairs = [(r[m]["edits"], r[m]["N"]) for r in rows]
        lo, hi = bootstrap_ci(pairs, iters=iters, seed=seed)
        tot = {k: sum(r[m][k] for r in rows) for k in ("S", "D", "I", "N")}
        summary[m] = {"rate": pooled_rate(pairs), "ci_low": lo, "ci_high": hi, **tot}
    return summary


def fmt(ms):
    r, lo, hi = ms["rate"], ms["ci_low"], ms["ci_high"]
    if r != r:  # NaN
        return "n/a"
    if lo != lo:
        return f"{100 * r:5.1f}%"
    return f"{100 * r:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with reference,hypothesis,lang[,id,speaker]")
    ap.add_argument("--out", default="results", help="output prefix")
    ap.add_argument("--worst", type=int, default=0, help="print the K clips with the highest primary-metric error")
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--hindi-lenient",
        action="store_true",
        help="Hindi only, lossy: delete nukta and map chandrabindu to anusvara",
    )
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for needed in ("reference", "hypothesis", "lang"):
            if needed not in cols:
                sys.exit(f"Input CSV is missing required column '{needed}'. Found: {cols}")
        raw = list(reader)

    rows, skipped_empty, skipped_bad = [], 0, 0
    for idx, r in enumerate(raw):
        lang = LANG_ALIASES.get((r.get("lang") or "").strip().lower())
        if lang is None:
            sys.exit(f"Row {idx}: lang must be hi/ta (or hindi/tamil), got {r.get('lang')!r}. Only Hindi and Tamil are supported.")
        try:
            res = score_pair(r["reference"], r["hypothesis"], lang, args.hindi_lenient)
        except ValueError as e:
            skipped_bad += 1
            print(f"warning: row {idx} skipped, reference is not clean {lang}: {e}", file=sys.stderr)
            continue
        if res["cer"]["N"] == 0:
            skipped_empty += 1
            print(f"warning: row {idx} has an empty reference after normalization; skipped", file=sys.stderr)
            continue
        res["id"] = r.get("id") or str(idx)
        res["lang"] = lang
        res["speaker"] = (r.get("speaker") or "").strip()
        res["ref_raw"] = r["reference"]
        res["hyp_raw"] = r["hypothesis"]
        rows.append(res)

    if not rows:
        sys.exit("No scorable rows.")

    # ---- per-clip CSV ----
    clip_path = f"{args.out}_per_clip.csv"
    text_cols = ["id", "lang", "speaker", "ref_raw", "hyp_raw", "ref_norm", "hyp_norm"]
    fields = list(text_cols)
    for m in METRICS:
        fields += [f"{m}_rate", f"{m}_S", f"{m}_D", f"{m}_I", f"{m}_N"]
    with open(clip_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            line = {k: r[k] for k in text_cols}
            for m in METRICS:
                c = r[m]
                line[f"{m}_rate"] = round(c["edits"] / c["N"], 4) if c["N"] else ""
                line[f"{m}_S"], line[f"{m}_D"], line[f"{m}_I"], line[f"{m}_N"] = c["S"], c["D"], c["I"], c["N"]
            w.writerow(line)

    # ---- grouped summaries ----
    by_lang, by_speaker = defaultdict(list), defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
        if r["speaker"]:
            by_speaker[(r["lang"], r["speaker"])].append(r)

    per_language = {k: summarize(v, args.bootstrap_iters, args.seed) for k, v in sorted(by_lang.items())}
    # Macro average: each language counts equally, however many clips it has.
    macro = {m: sum(s[m]["rate"] for s in per_language.values()) / len(per_language) for m in METRICS}

    summary = {
        "normalizer_version": NORMALIZER_VERSION,
        "hindi_lenient": args.hindi_lenient,
        "primary_metric": PRIMARY,
        "skipped_empty_reference": skipped_empty,
        "skipped_non_hi_ta_reference": skipped_bad,
        "overall_pooled": summarize(rows, args.bootstrap_iters, args.seed),
        "macro_average_over_languages": macro,
        "per_language": per_language,
    }
    if by_speaker:
        summary["per_speaker"] = {
            f"{lang}|{spk}": summarize(v, args.bootstrap_iters, args.seed)
            for (lang, spk), v in sorted(by_speaker.items())
        }

    with open(f"{args.out}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- console report ----
    print(
        f"normalizer v{NORMALIZER_VERSION}  hindi_lenient={args.hindi_lenient}  clips={len(rows)}  "
        f"skipped(empty ref)={skipped_empty}  skipped(non hi/ta ref)={skipped_bad}"
    )
    print(f"Primary metric: {LABELS[PRIMARY]}\n")
    header = f"{'group':<10}{'clips':>6}  " + "".join(f"{LABELS[m]:<24}" for m in METRICS)
    print(header)
    print("-" * len(header))

    def line(name, s):
        print(f"{name:<10}{s['n_clips']:>6}  " + "".join(f"{fmt(s[m]):<24}" for m in METRICS))

    for lang, s in per_language.items():
        line(lang, s)
    line("POOLED", summary["overall_pooled"])
    print(f"{'MACRO':<10}{'':>6}  " + "".join(f"{100 * macro[m]:5.1f}%{'':<18}" for m in METRICS))

    if args.worst:
        print(f"\nWorst {args.worst} clips by {LABELS[PRIMARY]}:")
        worst = sorted(rows, key=lambda r: r[PRIMARY]["edits"] / r[PRIMARY]["N"], reverse=True)[: args.worst]
        for r in worst:
            c = r[PRIMARY]
            print(f"  [{r['lang']}] id={r['id']}  {LABELS[PRIMARY]}={100 * c['edits'] / c['N']:.1f}%  S={c['S']} D={c['D']} I={c['I']}")
            print(f"      ref: {r['ref_norm']}")
            print(f"      hyp: {r['hyp_norm']}")

    print(f"\nWrote {clip_path} and {args.out}_summary.json")


if __name__ == "__main__":
    main()
