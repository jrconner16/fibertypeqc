from __future__ import annotations

import pandas as pd

from src.plot_three_way_biological_story import build_long_table


def test_build_long_table_contains_three_methods() -> None:
    frozen = pd.DataFrame(
        [
            {
                "image_id": "001_mouse1_mdx_TA_1month",
                "myosight_total_fibers": 100,
                "pipeline_total_fibers": 110,
                "myosight_iib_n": 30,
                "pipeline_iib_n": 32,
                "myosight_iia_n": 20,
                "pipeline_iia_n": 18,
                "myosight_iix_n": 50,
                "pipeline_iix_n": 60,
                "myosight_iib_pct": 0.30,
                "pipeline_iib_pct": 0.29,
                "myosight_iia_pct": 0.20,
                "pipeline_iia_pct": 0.16,
                "myosight_iix_pct": 0.50,
                "pipeline_iix_pct": 0.55,
                "myosight_area_median": 1000,
                "pipeline_area_median": 1100,
            }
        ]
    )
    candidate = pd.DataFrame(
        [
            {
                "image_id": "001_mouse1_mdx_TA_1month",
                "myosight_total_fibers": 100,
                "pipeline_total_fibers": 105,
                "myosight_iib_n": 30,
                "pipeline_iib_n": 31,
                "myosight_iia_n": 20,
                "pipeline_iia_n": 21,
                "myosight_iix_n": 50,
                "pipeline_iix_n": 53,
                "myosight_iib_pct": 0.30,
                "pipeline_iib_pct": 0.295,
                "myosight_iia_pct": 0.20,
                "pipeline_iia_pct": 0.20,
                "myosight_iix_pct": 0.50,
                "pipeline_iix_pct": 0.505,
                "myosight_area_median": 1000,
                "pipeline_area_median": 1050,
            }
        ]
    )
    out = build_long_table(frozen, candidate)
    assert set(out["method"]) == {"MyoSight", "Frozen", "Candidate"}
