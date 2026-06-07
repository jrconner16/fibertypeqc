from __future__ import annotations

import pandas as pd

from src.build_hybrid_candidate_summary import main


def test_placeholder() -> None:
    assert callable(main)


def test_merge_logic_smoke(tmp_path, monkeypatch) -> None:
    base = pd.DataFrame(
        [
            {"image_id": "a", "value": 1},
            {"image_id": "b", "value": 2},
        ]
    )
    repl = pd.DataFrame(
        [
            {"image_id": "b", "value": 20},
            {"image_id": "c", "value": 30},
        ]
    )
    base_path = tmp_path / "base.csv"
    repl_path = tmp_path / "repl.csv"
    out_path = tmp_path / "out.csv"
    base.to_csv(base_path, index=False)
    repl.to_csv(repl_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_hybrid_candidate_summary",
            "--base-summary",
            str(base_path),
            "--replacement-summary",
            str(repl_path),
            "--output",
            str(out_path),
        ],
    )
    main()

    out = pd.read_csv(out_path)
    assert set(out["image_id"]) == {"a", "b", "c"}
    assert int(out.loc[out["image_id"].eq("b"), "value"].iloc[0]) == 20
