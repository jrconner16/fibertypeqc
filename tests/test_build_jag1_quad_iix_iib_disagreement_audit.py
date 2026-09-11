from __future__ import annotations

import pandas as pd

from src.build_jag1_quad_iix_iib_disagreement_audit import STRATUM, build_queue


def test_disagreement_queue_is_blinded_and_development_only() -> None:
    manifest = pd.DataFrame(
        [
            {
                "mouse_id": "351545_L",
                "image_id": "351545_L_section_01",
                "section_id": "section-01",
                "split": "development",
                "cre_status": "cre_positive_mdxJAG",
                "raw_image_path": "/raw.czi",
                "fiber_labels_path": "/labels.tif",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "mouse_id": "351545_L",
                "image_id": "351545_L_section_01",
                "fiber_id": 1,
                "label": "iix",
                "lomo_prediction": "iib",
            },
            {
                "mouse_id": "351545_L",
                "image_id": "351545_L_section_01",
                "fiber_id": 2,
                "label": "iib",
                "lomo_prediction": "iix",
            },
        ]
    )
    queue = build_queue(manifest, predictions)
    assert queue[["image_id", "fiber_id"]].to_dict("records") == [
        {"image_id": "351545_L_section_01", "fiber_id": 1}
    ]
    assert set(queue.sampling_stratum) == {STRATUM}
    assert "label" in queue and queue["label"].eq("").all()
    assert "lomo_prediction" not in queue
