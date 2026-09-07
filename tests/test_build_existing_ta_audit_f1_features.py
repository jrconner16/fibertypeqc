import pandas as pd

from src.build_existing_ta_audit_f1_features import F1_COLUMNS, build_existing_ta_features


def test_build_existing_ta_features_uses_only_weighted_manual_target_rows(tmp_path):
    diagnostics_root = tmp_path / "diagnostics"
    image_dir = diagnostics_root / "image_a"
    image_dir.mkdir(parents=True)
    pd.DataFrame({"label": [1, 2], **{column: [0.1, 0.2] for column in F1_COLUMNS}}).to_csv(
        image_dir / "image_a_feature_diagnostics.csv", index=False
    )
    audit = pd.DataFrame(
        {
            "image_id": ["image_a", "image_a", "image_a"],
            "label": [1, 2, 1],
            "audit_final_label": ["iib", "exclude", "iia"],
            "manual_supervision_weight": [1.0, 1.0, 0.0],
        }
    )

    result = build_existing_ta_features(audit, diagnostics_root)

    assert result["target_label"].tolist() == ["iib"]
    assert result["group_id"].tolist() == ["image_a"]
    assert result["label_authority"].tolist() == ["manual_audit"]
    assert result.loc[0, "type_iia.center_mean"] == 0.1
