from __future__ import annotations

import pandas as pd

from src.run_quad_type_i_smoke import (
    CLASSIFIER_FEATURES,
    GATE_FEATURES,
    _manual_training_table,
    cross_validated_smoke,
)


def test_smoke_cross_validation_applies_iia_gate_before_non_iia_classifier():
    rows = []
    for label, kind, _base in (
        (1, "i", 1.0),
        (2, "i", 1.1),
        (3, "iia", 2.0),
        (4, "iia", 2.1),
        (5, "iib", 3.0),
        (6, "iib", 3.1),
    ):
        rows.append({"label": label, "final_type": kind, "label_source": "manual_gold"})
    reviewed = pd.DataFrame(rows)
    diagnostics_rows = []
    for row in rows:
        values = {"label": row["label"]}
        base = float(row["label"])
        for index, column in enumerate((*GATE_FEATURES, *CLASSIFIER_FEATURES)):
            values[column] = base + index / 100.0
        if row["final_type"] == "iia":
            values["type_iia.snr_mean"] = 10.0
            values["type_iia.coverage_high"] = 0.9
            values["type_iia.p90"] = 100.0
        else:
            values["type_iia.snr_mean"] = 0.1
            values["type_iia.coverage_high"] = 0.01
            values["type_iia.p90"] = 1.0
        diagnostics_rows.append(values)
    table = _manual_training_table(
        reviewed,
        pd.DataFrame(diagnostics_rows),
        include_provisional_iix=False,
    )

    predictions = cross_validated_smoke(table, gate_quantile=0.1)

    assert len(predictions) == 6
    assert predictions.loc[predictions["final_type"].eq("iia"), "iia_gate_passed"].all()
    assert not predictions.loc[~predictions["final_type"].eq("iia"), "iia_gate_passed"].any()
