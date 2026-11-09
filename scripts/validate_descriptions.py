import argparse
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


EXPECTED_HEADERS = ["description_id", "language", "text", "description"]
EXPECTED_DESCRIPTIONS = 30
MAX_DESCRIPTION_WORDS = 300
LANGUAGE_CODES = {
    "ta": "ta",
    "tamil": "ta",
    "hi": "hi",
    "hindi": "hi",
}


@dataclass
class Violation:
    condition: str
    location: str
    error: str


def resolve_expected_language(csv_path: Path) -> Optional[str]:
    for parent in csv_path.resolve().parents:
        code = LANGUAGE_CODES.get(parent.name.lower())
        if code:
            return code
    return None


def column_index(header: List[str], name: str) -> Optional[int]:
    return header.index(name) if name in header else None


def row_location(path: Path, header: List[str], line_no: int, row: List[str]) -> str:
    index = column_index(header, "description_id")
    desc_id = row[index] if index is not None and index < len(row) else "?"
    return f"{path.name} line {line_no} (description_id={desc_id!r})"


def value(header: List[str], row: List[str], name: str) -> Optional[str]:
    index = column_index(header, name)
    if index is None or index >= len(row):
        return None
    return row[index]


def check_csv_file(path: Path) -> Tuple[List[Violation], Optional[List[Tuple[int, List[str]]]]]:
    condition = "File is a valid UTF-8 CSV"
    violations = []

    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [Violation(condition, str(path), f"Not valid UTF-8: {exc}")], None
    except OSError as exc:
        return [Violation(condition, str(path), f"{type(exc).__name__}: {exc}")], None

    if content.startswith("\ufeff"):
        violations.append(Violation(condition, str(path), "File starts with a UTF-8 BOM."))
        content = content[1:]

    rows = []
    try:
        reader = csv.reader(io.StringIO(content, newline=""), strict=True)
        for row in reader:
            if row:
                rows.append((reader.line_num, row))
    except csv.Error as exc:
        violations.append(Violation(condition, str(path), f"Could not parse CSV: {exc}"))
        return violations, None

    if not rows:
        violations.append(Violation(condition, str(path), "File is empty."))
        return violations, None

    return violations, rows


def check_headers(path: Path, header: List[str]) -> List[Violation]:
    condition = f"Headers are exactly {', '.join(EXPECTED_HEADERS)}"
    if header == EXPECTED_HEADERS:
        return []

    missing = [name for name in EXPECTED_HEADERS if name not in header]
    extra = [name for name in header if name not in EXPECTED_HEADERS]

    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if extra:
            problems.append(f"unexpected {extra}")
    else:
        problems = [f"wrong order or duplicate columns: found {header}"]

    return [Violation(condition, str(path), "; ".join(problems))]


def check_row_shape(path: Path, header: List[str], data) -> List[Violation]:
    condition = "Every row has the same number of fields as the header"
    violations = []

    for line_no, row in data:
        if len(row) != len(header):
            violations.append(Violation(
                condition,
                f"{path.name} line {line_no}",
                f"Has {len(row)} fields; expected {len(header)}."
            ))

    return violations


def check_description_count(path: Path, data) -> List[Violation]:
    count = len(data)
    if count == EXPECTED_DESCRIPTIONS:
        return []

    return [Violation(
        f"Exactly {EXPECTED_DESCRIPTIONS} descriptions",
        str(path),
        f"Found {count} data rows; expected exactly {EXPECTED_DESCRIPTIONS}."
    )]


def check_language(path: Path, header: List[str], data, expected: Optional[str]) -> List[Violation]:
    condition = f"language column equals '{expected}'"
    if expected is None:
        return [Violation(
            "Language folder found",
            str(path),
            "No parent folder named ta, tamil, hi, or hindi was found."
        )]

    violations = []
    for line_no, row in data:
        language = value(header, row, "language")
        if language != expected:
            violations.append(Violation(
                condition,
                row_location(path, header, line_no, row),
                f"language is {language!r}; expected {expected!r}."
            ))

    return violations


def check_description_length(path: Path, header: List[str], data) -> List[Violation]:
    condition = f"Description has at most {MAX_DESCRIPTION_WORDS} words"
    violations = []

    for line_no, row in data:
        description = value(header, row, "description") or ""
        word_count = len(description.split())
        if word_count > MAX_DESCRIPTION_WORDS:
            violations.append(Violation(
                condition,
                row_location(path, header, line_no, row),
                f"Description has {word_count} words; maximum is {MAX_DESCRIPTION_WORDS}."
            ))

    return violations


def check_no_empty_fields(path: Path, header: List[str], data) -> List[Violation]:
    condition = "No empty fields"
    violations = []

    for line_no, row in data:
        empty = [
            name for name in EXPECTED_HEADERS
            if not (value(header, row, name) or "").strip()
        ]
        if empty:
            violations.append(Violation(
                condition,
                row_location(path, header, line_no, row),
                f"Empty fields: {empty}."
            ))

    return violations


def check_unique_ids(path: Path, header: List[str], data) -> List[Violation]:
    condition = "description_id values are unique"
    seen = {}
    violations = []

    for line_no, row in data:
        desc_id = value(header, row, "description_id")
        if desc_id in seen:
            violations.append(Violation(
                condition,
                row_location(path, header, line_no, row),
                f"Duplicate of line {seen[desc_id]}."
            ))
        else:
            seen[desc_id] = line_no

    return violations


def check_native_script(path: Path, header: List[str], data) -> List[Violation]:
    condition = "text contains Tamil script"
    violations = []

    for line_no, row in data:
        text = value(header, row, "text") or ""
        if not any("\u0b80" <= char <= "\u0bff" for char in text):
            violations.append(Violation(
                condition,
                row_location(path, header, line_no, row),
                "text has no Tamil characters."
            ))

    return violations


def print_report(condition: str, total: int, violations: List[Violation]) -> None:
    passed = total - len(violations)
    status = "✗" if violations else "✓"
    print(f"\n{status} {condition}: {passed}/{total} passed")

    for violation in violations:
        print(f"{violation.location} | {violation.error}")


def validate_file(path: Path) -> List[Violation]:
    print(f"\nChecking: {path}")

    expected_language = resolve_expected_language(path)
    print(f"Expected language: {expected_language or 'unresolved'}")

    all_violations = []
    csv_violations, rows = check_csv_file(path)
    print_report("File is a valid UTF-8 CSV", 1, csv_violations)
    all_violations.extend(csv_violations)

    if rows is None:
        print("Remaining checks skipped because the CSV could not be read.")
        return all_violations

    header = rows[0][1]
    data = rows[1:]
    print(f"Found {len(data)} data row(s).")

    checks = [
        (f"Headers are exactly {', '.join(EXPECTED_HEADERS)}",
         1, check_headers(path, header)),
        ("Every row has the same number of fields as the header",
         len(data), check_row_shape(path, header, data)),
        (f"Exactly {EXPECTED_DESCRIPTIONS} descriptions",
         1, check_description_count(path, data)),
    ]

    for condition, total, violations in checks:
        print_report(condition, total, violations)
        all_violations.extend(violations)

    missing = [name for name in EXPECTED_HEADERS if name not in header]

    if "language" in missing:
        print("\nLanguage check skipped: no 'language' column.")
    else:
        violations = check_language(path, header, data, expected_language)
        print_report(
            f"language column equals '{expected_language}'",
            len(data),
            violations
        )
        all_violations.extend(violations)

    if "description" in missing:
        print("\nDescription length check skipped: no 'description' column.")
    else:
        violations = check_description_length(path, header, data)
        print_report(
            f"Description has at most {MAX_DESCRIPTION_WORDS} words",
            len(data),
            violations
        )
        all_violations.extend(violations)

    if not missing:
        extra_checks = [
            ("No empty fields", check_no_empty_fields(path, header, data)),
            ("description_id values are unique", check_unique_ids(path, header, data)),
        ]

        if expected_language == "ta":
            extra_checks.append((
                "text contains Tamil script",
                check_native_script(path, header, data)
            ))

        for condition, violations in extra_checks:
            print_report(condition, len(data), violations)
            all_violations.extend(violations)

    print(f"\nTotal violation objects: {len(all_violations)}")
    return all_violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate descriptions.csv files in a directory tree."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Root directory containing descriptions.csv files"
    )
    args = parser.parse_args()

    root = args.dir.expanduser()
    if not root.is_dir():
        parser.error(f"'{root}' is not a valid directory.")

    csv_files = sorted(
        path for path in root.rglob("descriptions.csv") if path.is_file()
    )

    if not csv_files:
        print(f"No descriptions.csv files found under: {root}")
        return 1

    print(f"Found {len(csv_files)} descriptions.csv file(s).")
    all_violations = []

    for path in csv_files:
        all_violations.extend(validate_file(path))

    print(f"\nValidation complete: {len(csv_files)} file(s), "
          f"{len(all_violations)} violation(s).")
    return 1 if all_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())