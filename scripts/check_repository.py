"""Check public repository links, tracked artifacts, and reference digests."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from fibertypeqc.evidence_registry import (
    validate_dataset_evidence_inventory,
    validate_dataset_split_ledger,
    validate_model_registry,
)
from scripts.validate_reference_outputs import validate_reference_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_TARGET_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
TEXT_SUFFIXES = frozenset((".csv", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"))
ALLOWED_TRACKED_MICROSCOPY_FIXTURES = frozenset(
    (
        "examples/reference/synthetic_reference.tif",
        "examples/reference/synthetic_reference_labels.tif",
    )
)


def tracked_files(repo_root: Path = REPO_ROOT) -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
    )
    return {entry.decode() for entry in output.split(b"\0") if entry}


def _markdown_lines_outside_fences(path: Path):
    inside_fence = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("```"):
            inside_fence = not inside_fence
            continue
        if not inside_fence:
            yield line_number, line


def broken_documentation_links(repo_root: Path, tracked: set[str]) -> list[str]:
    broken: list[str] = []
    for relative_path in sorted(path for path in tracked if path.endswith(".md")):
        source_path = repo_root / relative_path
        for line_number, line in _markdown_lines_outside_fences(source_path):
            for match in MARKDOWN_LINK_RE.finditer(line):
                target = match.group(1).strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1].strip()
                if not target or target.startswith("#") or EXTERNAL_TARGET_RE.match(target):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                resolved = (source_path.parent / target).resolve()
                try:
                    resolved_relative = resolved.relative_to(repo_root.resolve()).as_posix()
                except ValueError:
                    broken.append(
                        f"{relative_path}:{line_number}: link escapes repository: {target}"
                    )
                    continue
                tracked_target = resolved_relative in tracked
                directory_prefix = f"{resolved_relative.rstrip('/')}/"
                tracked_directory = any(
                    candidate.startswith(directory_prefix) for candidate in tracked
                )
                if not tracked_target and not tracked_directory:
                    broken.append(f"{relative_path}:{line_number}: target is not tracked: {target}")
    return broken


def forbidden_tracked_artifacts(tracked: set[str]) -> list[str]:
    forbidden: list[str] = []
    for relative_path in sorted(tracked):
        path = Path(relative_path)
        suffix = path.suffix.lower()
        if relative_path.startswith(("outputs/", "data/runs/", "data/labels/", "test_inputs/")):
            forbidden.append(f"forbidden tracked output/private path: {relative_path}")
        if suffix in {".czi", ".tif", ".tiff"} and (
            relative_path not in ALLOWED_TRACKED_MICROSCOPY_FIXTURES
        ):
            forbidden.append(f"forbidden tracked microscopy file: {relative_path}")
    return forbidden


def private_absolute_paths(repo_root: Path, tracked: set[str]) -> list[str]:
    findings: list[str] = []
    patterns = (
        re.compile("/" + r"Users/[^\s`\"']+"),
        re.compile("/" + r"Volumes/[^\s`\"']+"),
        re.compile("/" + r"home/[^\s`\"']+"),
        re.compile("/" + r"temp_work/[^\s`\"']+"),
    )
    for relative_path in sorted(tracked):
        path = repo_root / relative_path
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern in patterns:
                if pattern.search(line):
                    findings.append(f"{relative_path}:{line_number}: private absolute path")
    return findings


def check_repository(repo_root: Path = REPO_ROOT) -> None:
    tracked = tracked_files(repo_root)
    problems = [
        *broken_documentation_links(repo_root, tracked),
        *forbidden_tracked_artifacts(tracked),
        *private_absolute_paths(repo_root, tracked),
    ]
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"Repository checks failed:\n{details}")
    validate_reference_inputs(repo_root / "examples/reference/reference_contract.json")
    validate_model_registry(repo_root / "manifests/model_registry.v1.yaml", repo_root=repo_root)
    validate_dataset_split_ledger(
        repo_root / "examples/reference/dataset_split_ledger.example.yaml"
    )
    validate_dataset_evidence_inventory(
        repo_root / "examples/reference/dataset_evidence_inventory.example.yaml"
    )


def main() -> None:
    check_repository()
    print("repository checks passed")


if __name__ == "__main__":
    main()
