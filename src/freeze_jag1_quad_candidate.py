"""Create a private, immutable lock record for the Jag1 Quad final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str | int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def freeze_candidate(
    *,
    model: Path,
    metadata: Path,
    manifest: Path,
    training_script: Path,
    review_sources: list[Path],
    output_dir: Path,
) -> Path:
    """Copy the candidate model and write a no-final-test-input lock record."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to alter an existing lock directory: {output_dir}")
    if not review_sources:
        raise ValueError("At least one current development review source is required")
    artifacts = {
        "model": _artifact(model),
        "baseline_metadata": _artifact(metadata),
        "canonical_manifest": _artifact(manifest),
        "training_script": _artifact(training_script),
        "development_review_sources": [_artifact(path) for path in review_sources],
    }
    output_dir.mkdir(parents=True)
    locked_model = output_dir / "jag1_quad_four_class_random_forest_v2.joblib"
    locked_metadata = output_dir / "baseline_metadata.json"
    shutil.copy2(model, locked_model)
    shutil.copy2(metadata, locked_metadata)
    lock = {
        "lock_version": "jag1_quad_final_evaluation_lock.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit_at_freeze": _git_commit(),
        "candidate_policy": {
            "classes": ["i", "iia", "iib", "iix"],
            "iix_status": "exploratory; not optimized after this lock",
            "legacy_supervision_used": False,
            "excluded_development_mouse": "351545_R",
        },
        "artifacts_before_copy": artifacts,
        "locked_model": _artifact(locked_model),
        "locked_metadata": _artifact(locked_metadata),
        "final_test_status": "not read or used by this freeze operation",
    }
    path = output_dir / "final_evaluation_lock.json"
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the Jag1 Quad candidate before final test")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--review-source", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lock = freeze_candidate(
        model=args.model.resolve(),
        metadata=args.baseline_metadata.resolve(),
        manifest=args.manifest.resolve(),
        training_script=args.training_script.resolve(),
        review_sources=[path.resolve() for path in args.review_source],
        output_dir=args.output_dir.resolve(),
    )
    print(f"frozen candidate lock: {lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
