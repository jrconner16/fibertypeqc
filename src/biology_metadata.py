from __future__ import annotations

import pandas as pd

AGE_LABELS = {1: "1 month", 4: "4 months", 12: "12 months"}
AGE_ORDER = ["1 month", "4 months", "12 months"]
GENOTYPE_ORDER = ["mdx5cv", "mdx5cv-JAG1"]
GENOTYPE_COLORS = {
    "mdx5cv": "#666666",
    "mdx5cv-JAG1": "#2f6f9f",
}
TYPE_COLORS = {
    "IIb": "#8b4a35",
    "IIa": "#179c52",
    "IIx": "#327c8a",
}


def infer_age_month(image_id: str) -> int:
    name = image_id.lower()
    if "12mo" in name or "1yo" in name:
        return 12
    if "4mo" in name:
        return 4
    if name.startswith("section001_"):
        return 4
    if "1month" in name or "1mo" in name:
        return 1
    raise ValueError(f"Could not infer age from image_id={image_id!r}")


def infer_genotype(image_id: str) -> str:
    if "jag" in image_id.lower():
        return "mdx5cv-JAG1"
    return "mdx5cv"


def add_biology_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_month"] = df["image_id"].map(infer_age_month)
    df["age"] = pd.Categorical(
        df["age_month"].map(AGE_LABELS),
        categories=AGE_ORDER,
        ordered=True,
    )
    df["genotype"] = pd.Categorical(
        df["image_id"].map(infer_genotype),
        categories=GENOTYPE_ORDER,
        ordered=True,
    )
    return df


def remove_unused_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(["category"]).columns:
        df[col] = df[col].cat.remove_unused_categories()
    return df
