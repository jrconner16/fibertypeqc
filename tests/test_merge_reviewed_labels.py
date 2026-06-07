from __future__ import annotations

import pandas as pd

from src.merge_reviewed_labels import merge_reviewed_labels


def test_merge_reviewed_labels_smoke_workflow():
    fibers = pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5],
            "fiber_type": ["type1", "type2", "unknown", "type1", "type2"],
        }
    )
    review = pd.DataFrame(
        {
            "fiber_id": [2, 3, 4, 5],
            "corrected_type": ["iib", "iix", "", ""],
            "is_uncertain": [False, False, True, False],
            "is_hybrid": [False, False, False, True],
            "is_excluded": [False, False, False, False],
            "label_source": ["manual_gold", "manual_gold", "manual_gold", "manual_gold"],
        }
    )

    merged = merge_reviewed_labels(fibers, review).sort_values("fiber_id").reset_index(drop=True)

    assert merged["fiber_id"].tolist() == [1, 2, 3, 4, 5]
    assert merged["predicted_type"].tolist() == ["type1", "type2", "unknown", "type1", "type2"]
    assert merged["predicted_biological_type"].tolist() == [
        "iib",
        "iia",
        "iix_candidate",
        "iib",
        "iia",
    ]
    assert merged["final_type"].tolist() == ["iib", "iib", "iix", "uncertain", "hybrid"]
