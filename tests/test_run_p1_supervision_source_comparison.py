import pandas as pd

from src.build_existing_ta_audit_f1_features import F1_COLUMNS
from src.run_p1_supervision_source_comparison import run_comparison


def _table(prefix: str, authority: str, groups: list[str]) -> pd.DataFrame:
    rows = []
    for group_index, group in enumerate(groups):
        for class_index, label in enumerate(["iia", "iix", "iib"]):
            row = {
                "image_id": f"{prefix}_{group}",
                "group_id": group,
                "pipeline_label_id": group_index * 10 + class_index,
                "target_label": label,
                "label_authority": authority,
            }
            row.update({column: float(group_index + class_index) for column in F1_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_run_comparison_emits_grouped_cross_cohort_and_reporting_scopes():
    existing = _table("existing", "manual_audit", ["e1", "e2"])
    notch = _table("notch", "reviewed_myosight", ["n1", "n2"])
    protected = _table("protected", "reviewed_myosight", ["p1"])

    predictions, metrics = run_comparison(existing, notch, protected)

    assert set(predictions["condition"]) == {
        "existing_only",
        "notch3_only",
        "pooled",
        "existing_to_notch3",
        "notch3_to_existing",
    }
    assert "protected_reporting_only" in set(predictions["evaluation_scope"])
    assert len(metrics.loc[metrics["scope_type"].eq("overall")]) == 9
