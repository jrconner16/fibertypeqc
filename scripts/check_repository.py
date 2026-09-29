"""Check public repository links, tracked artifacts, and reference digests."""

from __future__ import annotations

import os
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
        "examples/reference_four_class/synthetic_four_class.tif",
        "examples/reference_four_class/synthetic_four_class_labels.tif",
    )
)
MODEL_ARTIFACT_SUFFIXES = frozenset(
    (".joblib", ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".npy", ".npz", ".h5")
)
# Model artifacts explicitly approved for public distribution. Models trained on private or
# unpublished biological data must stay out of Git and be resolved from a private model root.
APPROVED_PUBLIC_MODEL_ARTIFACTS = frozenset(
    (
        "data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib",
        # Fitted only on generated synthetic fibers by scripts.generate_four_class_reference.
        "examples/reference_four_class/synthetic_four_class_rf.joblib",
    )
)
# Per-fiber tables are only allowed as synthetic/demo fixtures.
FIBER_TABLE_ALLOWED_PREFIXES = ("examples/", "tests/")
FIBER_TABLE_HEADER_RE = re.compile(r"(^|,)fiber_id(,|$)")
PRIVATE_DENYLIST_ENV = "FIBERTYPEQC_PRIVATE_DENYLIST"
DEFAULT_PRIVATE_DENYLIST = Path.home() / ".config" / "fibertypeqc" / "private_denylist.txt"


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
        if suffix in MODEL_ARTIFACT_SUFFIXES and (
            relative_path not in APPROVED_PUBLIC_MODEL_ARTIFACTS
        ):
            forbidden.append(f"model artifact not approved for public release: {relative_path}")
    return forbidden


def unapproved_fiber_tables(repo_root: Path, tracked: set[str]) -> list[str]:
    findings: list[str] = []
    for relative_path in sorted(tracked):
        if not relative_path.lower().endswith(".csv"):
            continue
        if relative_path.startswith(FIBER_TABLE_ALLOWED_PREFIXES):
            continue
        with (repo_root / relative_path).open(encoding="utf-8") as handle:
            header = handle.readline().strip()
        if FIBER_TABLE_HEADER_RE.search(header):
            findings.append(f"per-fiber table outside examples/ or tests/: {relative_path}")
    return findings


def private_absolute_paths(repo_root: Path, tracked: set[str]) -> list[str]:
    findings: list[str] = []
    patterns = (
        re.compile("/" + r"Users/[^\s`\"']+"),
        re.compile("/" + r"Volumes/[^\s`\"']+"),
        re.compile("/" + r"home/[^\s`\"']+"),
        re.compile("/" + r"temp_work/[^\s`\"']+"),
        re.compile(r"Google" + r" ?Drive|My" + r" Drive|Cloud" + r"Storage/", re.IGNORECASE),
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


def load_private_denylist(path: Path | None = None) -> list[re.Pattern[str]] | None:
    """Load private identifier patterns kept outside the repository.

    Returns ``None`` when no denylist is available (for example in public CI), so callers can
    report that the private identifier scan was skipped rather than silently passing.
    """
    if path is None:
        configured = os.environ.get(PRIVATE_DENYLIST_ENV)
        path = Path(configured).expanduser() if configured else DEFAULT_PRIVATE_DENYLIST
    if not path.is_file():
        return None
    patterns: list[re.Pattern[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            patterns.append(re.compile(entry))
    return patterns


def private_identifier_matches(
    repo_root: Path, tracked: set[str], patterns: list[re.Pattern[str]]
) -> list[str]:
    """Report tracked paths or text lines matching private patterns without echoing the match."""
    findings: list[str] = []
    for relative_path in sorted(tracked):
        if any(pattern.search(relative_path) for pattern in patterns):
            findings.append(f"{relative_path}: path matches private denylist")
        path = repo_root / relative_path
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                findings.append(f"{relative_path}:{line_number}: matches private denylist")
    return findings


def check_repository(repo_root: Path = REPO_ROOT) -> bool:
    """Run repository checks; return whether the private identifier scan ran."""
    tracked = tracked_files(repo_root)
    denylist = load_private_denylist()
    problems = [
        *broken_documentation_links(repo_root, tracked),
        *forbidden_tracked_artifacts(tracked),
        *unapproved_fiber_tables(repo_root, tracked),
        *private_absolute_paths(repo_root, tracked),
        *(private_identifier_matches(repo_root, tracked, denylist) if denylist else []),
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
    return denylist is not None


def main() -> None:
    denylist_scanned = check_repository()
    if not denylist_scanned:
        print(
            "private identifier scan skipped: no denylist "
            f"(set {PRIVATE_DENYLIST_ENV} or create {DEFAULT_PRIVATE_DENYLIST.name} "
            "under ~/.config/fibertypeqc/)"
        )
    print("repository checks passed")


if __name__ == "__main__":
    main()
