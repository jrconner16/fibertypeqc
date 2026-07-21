import pandas as pd

from src.train_panel_baselines import _manual_table, semantic_feature_columns


def test_manual_table_uses_only_manual_gold_allowed_classes():
    reviewed = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "final_type": ["i", "iia", "exclude"],
            "label_source": ["manual_gold", "manual_gold", "manual_gold"],
        }
    )
    diagnostics = pd.DataFrame({"label": [1, 2, 3], "type_i.mean": [1.0, 2.0, 3.0]})

    table = _manual_table(reviewed, diagnostics, ("i", "iia"))

    assert table["label"].tolist() == [1, 2]


def test_semantic_feature_columns_intersects_panel_features():
    train = pd.DataFrame({"type_i.mean": [1.0], "type_iia.p90": [2.0], "type_iib.mean": [3.0]})
    heldout = pd.DataFrame({"type_i.mean": [4.0], "type_iia.p90": [5.0]})

    assert semantic_feature_columns(train, heldout) == ["type_i.mean", "type_iia.p90"]
