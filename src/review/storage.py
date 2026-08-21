"""Crash-resistant flat-file storage for review sessions and masks."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.review.schemas import RegionAnnotation, ReviewEvent
from src.review.session import ReviewSession

EVENT_COLUMNS = [
    "schema_version",
    "event_id",
    "image_id",
    "scope",
    "domain",
    "subdomain",
    "target_id",
    "action",
    "reason_code",
    "old_value",
    "new_value",
    "reviewer",
    "timestamp",
    "model_version",
    "qc_version",
]


def _atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_dataframe(path: Path | str, table: pd.DataFrame) -> None:
    """Write a CSV through atomic replacement."""
    output_path = Path(path)
    _atomic_replace_text(output_path, table.to_csv(index=False, lineterminator="\n"))


def save_session(path: Path | str, session: ReviewSession) -> None:
    state_path = Path(path)
    session.touch()
    text = json.dumps(session.to_dict(), indent=2, sort_keys=True) + "\n"
    _atomic_replace_text(state_path, text)


def load_session(
    path: Path | str,
    *,
    expected_project_id: str | None = None,
) -> ReviewSession:
    state_path = Path(path)
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Review state does not exist: {state_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Review state is not valid JSON: {state_path}: {exc}") from exc
    session = ReviewSession.from_dict(raw)
    if expected_project_id is not None and session.project_id != expected_project_id:
        raise ValueError(
            f"Review state project_id {session.project_id!r} does not match {expected_project_id!r}"
        )
    return session


def append_review_event(path: Path | str, event: ReviewEvent) -> None:
    """Append logically while atomically replacing the on-disk CSV."""
    event_path = Path(path)
    rows: list[dict[str, str]] = []
    if event_path.exists():
        try:
            with event_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != EVENT_COLUMNS:
                    raise ValueError(f"Review event CSV has incompatible columns: {event_path}")
                rows.extend(reader)
        except csv.Error as exc:
            raise ValueError(f"Review event CSV is corrupt: {event_path}: {exc}") from exc
    rows.append(event.to_csv_row())

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        _atomic_replace_text(event_path, handle.read())


def save_regions_geojson(path: Path | str, regions: list[RegionAnnotation]) -> None:
    """Atomically materialize the current region annotations as GeoJSON."""
    features = [
        {
            "type": "Feature",
            "id": region.region_id,
            "geometry": region.geometry,
            "properties": {
                key: value
                for key, value in region.to_dict().items()
                if key not in {"region_id", "geometry"}
            },
        }
        for region in regions
    ]
    _atomic_replace_text(
        Path(path),
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2, sort_keys=True)
        + "\n",
    )


def materialize_reviewed_mask(
    predicted_path: Path | str,
    reviewed_path: Path | str,
) -> bool:
    """Copy a predicted mask on first edit without ever modifying the prediction.

    Returns ``True`` when the reviewed file was created and ``False`` when a
    reviewed copy already existed.
    """
    predicted = Path(predicted_path).expanduser().resolve()
    reviewed = Path(reviewed_path).expanduser().resolve()
    if predicted == reviewed:
        raise ValueError("Reviewed mask path must differ from predicted mask path")
    if not predicted.is_file():
        raise FileNotFoundError(f"Predicted mask does not exist: {predicted}")
    if reviewed.exists():
        if not reviewed.is_file():
            raise ValueError(f"Reviewed mask path is not a file: {reviewed}")
        return False

    reviewed.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=reviewed.parent,
            prefix=f".{reviewed.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        shutil.copyfile(predicted, temporary_path)
        os.replace(temporary_path, reviewed)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return True
