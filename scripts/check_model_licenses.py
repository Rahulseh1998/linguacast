#!/usr/bin/env python3
"""Verify every model in MODELS.md has a permissive license.

Parses the single Markdown table under the "## Models" heading in
MODELS.md and fails (exit 1) if any row's license is not on the LinguaCast
permissive allowlist. Mirrors the rules enforced by ``cargo-deny`` for Rust
deps and ``pip-licenses`` for Python deps.

Lifted from the OPE-4 track (workspaces/7ab06d1c-e658-4d1d-bcd7-e86497acb404)
per the intra-company lift approved in the OPE-19 CTO ack (2026-05-19).
Pre-wires the OPE-17 license-CI gate.

Usage:
    python scripts/check_model_licenses.py [path/to/MODELS.md]

Exits 0 on success, 1 on any allowlist violation or parse error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "MIT-0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "Unicode-3.0",
        "CC0-1.0",
        "MPL-2.0",
    }
)

REQUIRED_HEADERS: tuple[str, ...] = (
    "Role",
    "Model",
    "HF Repo / Source",
    "License",
    "License URL",
)


def parse_models_table(text: str) -> list[dict[str, str]]:
    """Return one dict per row of the Models table, keyed by column name."""
    section = _extract_section(text, "Models")
    if section is None:
        raise ValueError("MODELS.md is missing a '## Models' section")

    rows = _table_rows(section)
    if not rows:
        raise ValueError("'## Models' section has no Markdown table")

    header = rows[0]
    if tuple(header) != REQUIRED_HEADERS:
        raise ValueError(
            "Models table header mismatch.\n"
            f"  expected: {REQUIRED_HEADERS}\n"
            f"  got:      {tuple(header)}"
        )

    data_rows = rows[1:]
    out: list[dict[str, str]] = []
    for idx, row in enumerate(data_rows, start=1):
        if len(row) != len(REQUIRED_HEADERS):
            raise ValueError(
                f"Row {idx} has {len(row)} columns, expected {len(REQUIRED_HEADERS)}: {row!r}"
            )
        out.append(dict(zip(REQUIRED_HEADERS, row)))
    return out


def _extract_section(text: str, heading: str) -> str | None:
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1) if m else None


def _table_rows(section: str) -> list[list[str]]:
    rows: list[list[str]] = []
    in_table = False
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            if in_table:
                break
            continue
        if _is_alignment_row(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
        in_table = True
    return rows


def _is_alignment_row(line: str) -> bool:
    cells = [c.strip() for c in line.strip("|").split("|")]
    return all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c)


def check(models: list[dict[str, str]]) -> list[str]:
    violations: list[str] = []
    seen_models: set[str] = set()
    for row in models:
        model = row["Model"]
        license_ = row["License"]
        url = row["License URL"]

        if model in seen_models:
            violations.append(f"duplicate model row: {model!r}")
        seen_models.add(model)

        if license_ not in ALLOWED_LICENSES:
            violations.append(
                f"{model}: license {license_!r} is not on the allowlist"
            )

        if not url.lower().startswith(("http://", "https://")):
            violations.append(
                f"{model}: license URL {url!r} is not an absolute http(s) link"
            )
    return violations


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    target = Path(argv[1]) if len(argv) > 1 else repo_root / "MODELS.md"
    if not target.is_file():
        print(f"error: {target} not found", file=sys.stderr)
        return 1

    text = target.read_text(encoding="utf-8")
    try:
        models = parse_models_table(text)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not models:
        print(f"error: {target} has no model rows", file=sys.stderr)
        return 1

    violations = check(models)
    if violations:
        print(f"Model license check FAILED ({len(violations)} issue(s)):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "\nAllowed licenses: " + ", ".join(sorted(ALLOWED_LICENSES)),
            file=sys.stderr,
        )
        return 1

    print(f"Model license check PASSED — {len(models)} model(s):")
    for row in models:
        print(f"  - {row['Model']}: {row['License']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
