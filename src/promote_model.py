"""Switch a project to a candidate model, explicitly and reversibly.

Promotion never happens as a side effect of building candidates. It archives the current typing
outputs and review state, re-types every section with the candidate on the existing masks (no
segmentation, same fiber IDs), carries review decisions forward, and records the change in
``models/model_history.csv``. ``--rollback`` restores the most recent archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from fibertypeqc.model_resolution import default_model_id, resolve_model_argument
from src.fiber_type_labels import normalize_review_label
from src.review.finalization import image_input_fingerprints
from src.review.project import Project, load_project
from src.review.schemas import ObjectReviewStatus
from src.review.session import ReviewSession
from src.review.storage import load_session, save_session

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "fibertypeqc_project.yaml"
HISTORY_COLUMNS = (
    "changed_at_utc",
    "action",
    "from_model",
    "to_model",
    "to_artifact_sha256",
    "recommended_by_improver",
    "archive",
    "reviewed_fibers",
    "agree_before",
    "agree_after",
    "flagged_before",
    "flagged_after",
)
MASK_SUFFIX = "_cellpose_labels.tif"


class PromotionError(ValueError):
    """The requested model change cannot be made safely."""


def find_candidate(root: Path, candidate: str) -> Path:
    """A candidate given as a model file, or by name from the latest improver run."""
    path = Path(candidate).expanduser()
    if path.is_file():
        return path.resolve()
    runs = sorted((root / "models").glob("improve_*"))
    for run in reversed(runs):
        found = run / "candidates" / candidate / f"{candidate}.joblib"
        if found.is_file():
            return found.resolve()
    raise PromotionError(
        f"No candidate named {candidate!r} in {root / 'models'}. Run 'improve' first, or pass "
        "the candidate's .joblib file."
    )


def improver_recommended(candidate: Path) -> bool | None:
    """Whether the improver run that built this candidate recommended it (None if unknown)."""
    summary = candidate.parents[2] / "improver_summary.json"
    if not summary.is_file():
        return None
    return json.loads(summary.read_text())["decision"].get("recommended") == candidate.stem


def carry_review_forward(
    session: ReviewSession, new_calls: dict[str, dict[int, str]], new_model_id: str
) -> dict[str, int]:
    """Keep every human judgement and restate it against the new model's calls.

    The reviewer's label is unchanged. Only the recorded model call, and whether the reviewer's
    label counts as keeping or correcting it, are updated.
    """
    counts = {"reviewed_fibers": 0, "agree_before": 0, "agree_after": 0}
    updated = []
    for decision in session.object_decisions:
        old_call = normalize_review_label(decision.model_fiber_type)
        new_call = normalize_review_label(new_calls[decision.image_id][decision.fiber_id])
        status = decision.review_status
        if status in (ObjectReviewStatus.ACCEPTED, ObjectReviewStatus.CORRECTED):
            human = (
                old_call
                if status is ObjectReviewStatus.ACCEPTED
                else normalize_review_label(decision.reviewed_fiber_type)
            )
            counts["reviewed_fibers"] += 1
            counts["agree_before"] += int(human == old_call)
            counts["agree_after"] += int(human == new_call)
            status = (
                ObjectReviewStatus.ACCEPTED if human == new_call else ObjectReviewStatus.CORRECTED
            )
            updated.append(
                replace(
                    decision,
                    model_fiber_type=new_call,
                    reviewed_fiber_type=human,
                    review_status=status,
                )
            )
        else:
            updated.append(replace(decision, model_fiber_type=new_call))
    session.object_decisions = updated
    session.model_version = new_model_id
    session.touch()
    return counts


def _typing_outputs(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and not p.name.endswith(MASK_SUFFIX)]


def _flagged(project: Project) -> int:
    total = 0
    for image in project.images:
        table = pd.read_csv(
            image.outputs["fiber_table"], usecols=lambda name: name == "needs_review"
        )
        if "needs_review" in table:
            total += int(table["needs_review"].fillna(False).astype(bool).sum())
    return total


def _calls(project: Project) -> dict[str, dict[int, str]]:
    calls = {}
    for image in project.images:
        table = pd.read_csv(image.outputs["fiber_table"], low_memory=False)
        key = "fiber_id" if "fiber_id" in table.columns else "label"
        calls[image.image_id] = dict(
            zip(table[key].astype(int), table["fiber_type"].astype(str), strict=True)
        )
    return calls


def _archive(root: Path, project: Project, name: str) -> Path:
    archive = root / "models" / "history" / name
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / CONFIG_NAME, archive / CONFIG_NAME)
    shutil.copy2(project.manifest_path, archive / "review_project.yaml")
    for folder in (project.review_directory, project.root / "qc", root / "final", root / "results"):
        if folder.is_dir():
            shutil.copytree(folder, archive / folder.relative_to(root))
    for image in project.images:
        target = archive / "batch" / image.image_id
        target.mkdir(parents=True)
        for path in _typing_outputs(image.prediction_directory):
            shutil.copy2(path, target / path.name)
    return archive


def _restore(root: Path, project: Project, archive: Path) -> None:
    shutil.copy2(archive / CONFIG_NAME, root / CONFIG_NAME)
    shutil.copy2(archive / "review_project.yaml", project.manifest_path)
    for folder in (project.review_directory, project.root / "qc", root / "final", root / "results"):
        saved = archive / folder.relative_to(root)
        if folder.is_dir():
            shutil.rmtree(folder)
        if saved.is_dir():
            shutil.copytree(saved, folder)
    for image in project.images:
        for path in _typing_outputs(image.prediction_directory):
            path.unlink()
        for path in sorted((archive / "batch" / image.image_id).iterdir()):
            shutil.copy2(path, image.prediction_directory / path.name)


def _retype(project: Project, model: Path, panel: Path) -> None:
    """Re-type every section with ``model`` on its existing mask; masks are left untouched."""
    with tempfile.TemporaryDirectory(dir=project.root) as scratch:
        staged = []
        for index, image in enumerate(project.images, start=1):
            print(f"re-typing {index}/{len(project.images)}: {image.image_id}", flush=True)
            out = Path(scratch) / image.image_id
            command = [
                sys.executable, "-m", "scripts.run_pipeline",
                "--input", str(image.raw_image_path),
                "--labels-path", str(image.outputs["fiber_labels"]),
                "--output-dir", str(out),
                "--image-id", image.image_id,
                "--panel-config", str(panel),
                "--model", str(model),
                "--no-crop-auto",
            ]  # fmt: skip
            done = subprocess.run(
                command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
            )
            new_table = out / image.outputs["fiber_table"].name
            if done.returncode != 0 or not new_table.is_file():
                tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["unknown error"]
                raise PromotionError(f"{image.image_id}: re-typing failed: {tail[0]}")
            old = pd.read_csv(image.outputs["fiber_table"], usecols=[0])
            new = pd.read_csv(new_table, usecols=[0])
            if not old.iloc[:, 0].equals(new.iloc[:, 0]):
                raise PromotionError(f"{image.image_id}: fiber IDs changed during re-typing.")
            staged.append((image, out))
        # Nothing is replaced until every section has re-typed cleanly.
        for image, out in staged:
            for path in _typing_outputs(image.prediction_directory):
                path.unlink()
            for path in _typing_outputs(out):
                shutil.copy2(path, image.prediction_directory / path.name)


def _append_history(root: Path, row: dict[str, object]) -> None:
    path = root / "models" / "model_history.csv"
    table = (
        pd.read_csv(path, dtype=str, keep_default_na=False)
        if path.is_file()
        else pd.DataFrame(columns=list(HISTORY_COLUMNS))
    )
    table = pd.concat([table, pd.DataFrame([{k: str(v) for k, v in row.items()}])])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, columns=list(HISTORY_COLUMNS))


def _set_review_model(project: Project, model_id: str) -> None:
    raw = yaml.safe_load(project.manifest_path.read_text(encoding="utf-8"))
    raw["model_version"] = model_id
    project.manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def promote(root: Path, candidate: str, panel: Path, *, override: bool = False) -> dict:
    config = yaml.safe_load((root / CONFIG_NAME).read_text(encoding="utf-8"))
    project = load_project(root / "review" / "project.yaml")
    artifact = find_candidate(root, candidate)
    resolved = resolve_model_argument(str(artifact))
    recommended = improver_recommended(artifact)
    if recommended is not True and not override:
        raise PromotionError(
            f"The improver did not recommend {artifact.stem}"
            + (" (no improver summary found)" if recommended is None else "")
            + ". To switch anyway, repeat the command with --override; this is recorded."
        )
    old_model = project.model_version or str(config.get("model") or default_model_id())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    flagged_before = _flagged(project)
    archive = _archive(root, project, f"{stamp}__{old_model}")
    try:
        _retype(project, artifact, panel)
        session_path = project.review_state_path
        counts = {"reviewed_fibers": 0, "agree_before": 0, "agree_after": 0}
        if session_path.is_file():
            session = load_session(session_path, expected_project_id=project.project_id)
            counts = carry_review_forward(session, _calls(project), resolved.model_id)
            for image in project.images:
                if image.image_id in session.input_fingerprints:
                    session.input_fingerprints[image.image_id] = image_input_fingerprints(image)
            save_session(session_path, session)
        _set_review_model(project, resolved.model_id)
        try:
            config["model"] = str(artifact.relative_to(root))
        except ValueError:
            config["model"] = str(artifact)
        (root / CONFIG_NAME).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        for stale in (root / "final", root / "results"):
            if stale.is_dir():
                shutil.rmtree(stale)  # archived above; they describe the previous model
    except Exception:
        _restore(root, project, archive)
        raise
    row = {
        "changed_at_utc": datetime.now(UTC).isoformat(),
        "action": "promote",
        "from_model": old_model,
        "to_model": resolved.model_id,
        "to_artifact_sha256": yaml.safe_load(resolved.manifest_path.read_text())["artifact_sha256"],
        "recommended_by_improver": recommended,
        "archive": str(archive.relative_to(root)),
        **counts,
        "flagged_before": flagged_before,
        "flagged_after": _flagged(project),
    }
    _append_history(root, row)
    return row


def rollback(root: Path) -> dict:
    history_path = root / "models" / "model_history.csv"
    if not history_path.is_file():
        raise PromotionError("This project has no model changes to roll back.")
    history = pd.read_csv(history_path, dtype=str, keep_default_na=False)
    undone = set(history.loc[history["action"] == "rollback", "archive"])
    pending = history[(history["action"] == "promote") & ~history["archive"].isin(undone)]
    if pending.empty:
        raise PromotionError("Every model change has already been rolled back.")
    last = pending.iloc[-1]
    if last.name != history[history["action"] == "promote"].index[-1]:
        raise PromotionError("Only the most recent model change can be rolled back.")
    project = load_project(root / "review" / "project.yaml")
    _restore(root, project, root / last["archive"])
    row = {
        "changed_at_utc": datetime.now(UTC).isoformat(),
        "action": "rollback",
        "from_model": last["to_model"],
        "to_model": last["from_model"],
        "to_artifact_sha256": "",
        "recommended_by_improver": "",
        "archive": last["archive"],
        "reviewed_fibers": "",
        "agree_before": "",
        "agree_after": "",
        "flagged_before": "",
        "flagged_after": "",
    }
    _append_history(root, row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True, help="Project folder")
    parser.add_argument("--panel-config", type=Path, required=True)
    parser.add_argument("--candidate", default=None, help="Candidate name or model file")
    parser.add_argument(
        "--override",
        action="store_true",
        help="Switch even though the improver did not recommend this candidate (recorded).",
    )
    parser.add_argument("--rollback", action="store_true", help="Undo the most recent change")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    try:
        if args.rollback:
            row = rollback(root)
            print(
                f"Rolled back to {row['to_model']}; its outputs, QC, and review state are restored."
            )
            return 0
        if not args.candidate:
            raise PromotionError("Name the candidate to switch to, for example: promote logistic")
        row = promote(root, args.candidate, args.panel_config, override=args.override)
    except (PromotionError, ValueError, OSError) as exc:
        print(f"Model change refused: {exc}", file=sys.stderr)
        return 1
    print(f"\nProject now uses {row['to_model']} (was {row['from_model']}).")
    if row["reviewed_fibers"]:
        print(
            f"Reviewed fibers: {row['agree_after']}/{row['reviewed_fibers']} agree with the new "
            f"model's call (was {row['agree_before']}/{row['reviewed_fibers']}). Your decisions "
            "were kept. The candidate was trained on these fibers, so this is not a measure of "
            "accuracy; judge it on sections reviewed after the switch."
        )
    print(f"Flagged for review: {row['flagged_after']} (was {row['flagged_before']}).")
    print(f"Previous outputs archived in {row['archive']}; undo with: promote --rollback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
