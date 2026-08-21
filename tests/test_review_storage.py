import csv
from pathlib import Path

import numpy as np
import tifffile

from src.review.schemas import Domain, ReviewEvent, Scope
from src.review.storage import append_review_event, materialize_reviewed_mask


def test_reviewed_mask_is_copy_on_write_and_prediction_is_never_overwritten(
    tmp_path: Path,
) -> None:
    predicted = tmp_path / "predictions" / "labels.tif"
    predicted.parent.mkdir()
    original = np.array([[0, 1], [2, 2]], dtype=np.int32)
    tifffile.imwrite(predicted, original)
    reviewed = tmp_path / "review" / "reviewed_fiber_labels" / "labels.tif"

    assert materialize_reviewed_mask(predicted, reviewed) is True
    changed = tifffile.imread(reviewed)
    changed[0, 0] = 3
    tifffile.imwrite(reviewed, changed)
    assert materialize_reviewed_mask(predicted, reviewed) is False

    np.testing.assert_array_equal(tifffile.imread(predicted), original)
    assert tifffile.imread(reviewed)[0, 0] == 3


def test_reviewed_mask_must_not_use_prediction_path(tmp_path: Path) -> None:
    predicted = tmp_path / "labels.tif"
    predicted.write_bytes(b"prediction")

    try:
        materialize_reviewed_mask(predicted, predicted)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("Expected same-path protection")


def test_review_events_append_with_model_and_qc_provenance(tmp_path: Path) -> None:
    path = tmp_path / "review" / "review_events.csv"
    first = ReviewEvent(
        image_id="image_1",
        scope=Scope.OBJECT,
        domain=Domain.FIBER_TYPING,
        target_id="12",
        action="change_type",
        old_value={"model_fiber_type": "iib"},
        new_value={"reviewed_fiber_type": "iia"},
        reviewer="reviewer",
        model_version="model.v1",
        qc_version="qc.v1",
    )
    second = ReviewEvent(
        image_id="image_1",
        scope=Scope.IMAGE,
        domain=Domain.NUCLEI,
        target_id="image_1",
        action="needs_review",
    )

    append_review_event(path, first)
    append_review_event(path, second)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["event_id"] == first.event_id
    assert rows[0]["model_version"] == "model.v1"
    assert rows[0]["qc_version"] == "qc.v1"
    assert rows[1]["domain"] == "nuclei"
