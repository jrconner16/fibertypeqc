from __future__ import annotations

import pandas as pd
import pytest

from src.run_locked_protected_evaluation import prepare_protected_table


def _contract() -> dict:
    return {
        "target": {"classes": ["iia", "iix", "iib"]},
        "evidence_roles": {"protected_evaluation": {"groups": 1}},
    }


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["image_a"],
            "group_id": ["mouse_a"],
            "pipeline_label_id": [1],
            "target_label": ["IIa"],
            "label_authority": ["reviewed_myosight"],
            "evidence_role": ["protected_final_evaluation"],
            "supervision_role": ["protected_final_evaluation"],
            "eligible_for_model_fitting": [False],
        }
    )


def test_prepare_protected_table_preserves_sealed_nonfitting_rows():
    result = prepare_protected_table(_table(), _contract())

    assert result["target_label"].tolist() == ["iia"]


def test_prepare_protected_table_rejects_fitting_eligibility():
    table = _table()
    table.loc[0, "eligible_for_model_fitting"] = True

    with pytest.raises(ValueError, match="never be eligible"):
        prepare_protected_table(table, _contract())
