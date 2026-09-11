from __future__ import annotations

from pathlib import Path

from src.plot_jag1_quad_locked_composition import eligible_mouse_composition, plot_composition


def test_plot_uses_only_eligible_mouse_rows(tmp_path: Path) -> None:
    source = tmp_path / "mouse.csv"
    source.write_text(
        "mouse_id,cre_status,analysis_eligible,pct_i,pct_iia,pct_iib,pct_iix\n"
        "a,cre_negative_mdx,True,0.01,0.02,0.90,0.07\n"
        "b,cre_positive_mdxJAG,True,0.02,0.03,0.80,0.15\n"
        "excluded,cre_negative_mdx,False,0.10,0.10,0.70,0.10\n"
    )
    data = eligible_mouse_composition(source)
    output = tmp_path / "composition.png"

    plot_composition(data, output)

    assert data["mouse_id"].tolist() == ["a", "b"]
    assert output.is_file()
