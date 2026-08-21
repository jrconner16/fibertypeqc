"""Headless cohort-dashboard model built from Phase 2A QC tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.review.project import Project
from src.review.qc import (
    FIBER_QC_SCHEMA_VERSION,
    IMAGE_QC_SCHEMA_VERSION,
    NUCLEUS_QC_SCHEMA_VERSION,
)
from src.review.schemas import Domain, DomainStatus
from src.review.section_selection import (
    SECTION_SELECTION_SCHEMA_VERSION,
    SelectionStrategy,
    select_sections,
)
from src.review.session import ReviewSession


@dataclass(frozen=True)
class DashboardTables:
    image_qc: pd.DataFrame
    fiber_qc: pd.DataFrame | None
    nucleus_qc: pd.DataFrame | None
    stored_selection: pd.DataFrame | None


@dataclass(frozen=True)
class DashboardSummary:
    mouse_count: int
    section_count: int
    complete_mouse_count: int
    targeted_review_mouse_count: int
    no_acceptable_mouse_count: int
    applicable_domain_rows: int
    reviewed_domain_rows: int
    review_progress_fraction: float
    object_decision_count: int
    region_count: int
    reviewed_mask_count: int


@dataclass(frozen=True)
class DashboardModel:
    project: Project
    tables: DashboardTables
    selection: pd.DataFrame
    section_rows: pd.DataFrame
    domain_counts: pd.DataFrame
    mouse_domain_rows: pd.DataFrame
    mouse_rows: pd.DataFrame
    summary: DashboardSummary
    strategy: SelectionStrategy


def _read_csv(path: Path, label: str, *, required: bool) -> pd.DataFrame | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required dashboard {label} does not exist: {path}")
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Dashboard {label} is unreadable: {path}: {exc}") from exc


def _parse_bool_series(series: pd.Series, field_name: str) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        values = sorted(set(series[invalid].astype(str)))
        raise ValueError(f"{field_name} contains invalid boolean values: {values}")
    return normalized.map(mapping).astype(bool)


def _validate_provenance(
    table: pd.DataFrame,
    *,
    label: str,
    expected_schema: str,
    project: Project,
    required_columns: set[str],
) -> pd.DataFrame:
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    schemas = set(table["schema_version"].dropna().astype(str))
    if schemas != {expected_schema}:
        raise ValueError(
            f"{label} has unsupported schema_version values {sorted(schemas)}; "
            f"expected {expected_schema!r}"
        )
    project_ids = set(table["project_id"].dropna().astype(str))
    if project_ids != {project.project_id}:
        raise ValueError(
            f"{label} project_id values {sorted(project_ids)} do not match "
            f"{project.project_id!r}"
        )
    unknown_images = sorted(
        set(table["image_id"].dropna().astype(str))
        - {image.image_id for image in project.images}
    )
    if unknown_images:
        raise ValueError(f"{label} contains image IDs absent from the manifest: {unknown_images}")
    for field in ("qc_version", "rules_version", "model_version"):
        values = set(table[field].dropna().astype(str))
        if len(values) > 1:
            raise ValueError(f"{label} mixes {field} values: {sorted(values)}")
    model_versions = set(table["model_version"].dropna().astype(str))
    if model_versions and model_versions != {project.model_version}:
        raise ValueError(
            f"{label} model_version values {sorted(model_versions)} do not match "
            f"{project.model_version!r}"
        )
    return table.copy()


def _validate_image_qc(table: pd.DataFrame, project: Project) -> pd.DataFrame:
    required = {
        "schema_version",
        "qc_version",
        "rules_version",
        "model_version",
        "computed_at",
        "project_id",
        "image_id",
        "mouse_id",
        "section_id",
        "domain",
        "applicable",
        "status",
        "hard_fail",
        "technical_quality_score",
        "review_priority",
        "reason_codes",
    }
    out = _validate_provenance(
        table,
        label="image_qc.csv",
        expected_schema=IMAGE_QC_SCHEMA_VERSION,
        project=project,
        required_columns=required,
    )
    out["applicable"] = _parse_bool_series(out["applicable"], "image_qc.applicable")
    out["hard_fail"] = _parse_bool_series(out["hard_fail"], "image_qc.hard_fail")
    allowed_domains = {domain.value for domain in Domain}
    unknown_domains = sorted(set(out["domain"].astype(str)) - allowed_domains)
    if unknown_domains:
        raise ValueError(f"image_qc.csv contains unknown domains: {unknown_domains}")
    allowed_statuses = {
        DomainStatus.PASS.value,
        DomainStatus.REVIEW.value,
        DomainStatus.FAIL.value,
        DomainStatus.NOT_APPLICABLE.value,
    }
    unknown_statuses = sorted(set(out["status"].astype(str)) - allowed_statuses)
    if unknown_statuses:
        raise ValueError(f"image_qc.csv contains unsupported statuses: {unknown_statuses}")
    duplicates = out.duplicated(["image_id", "domain"], keep=False)
    if duplicates.any():
        keys = sorted(
            {
                f"{row.image_id}/{row.domain}"
                for row in out.loc[duplicates, ["image_id", "domain"]].itertuples(index=False)
            }
        )
        raise ValueError(f"image_qc.csv contains duplicate image/domain rows: {keys}")
    expected_keys = {
        (image.image_id, domain.value) for image in project.images for domain in Domain
    }
    actual_keys = set(zip(out["image_id"].astype(str), out["domain"].astype(str), strict=True))
    missing_keys = sorted(expected_keys - actual_keys)
    if missing_keys:
        raise ValueError(f"image_qc.csv is missing image/domain rows: {missing_keys}")
    manifest_metadata = {
        image.image_id: (image.mouse_id, image.section_id) for image in project.images
    }
    for row in out.itertuples(index=False):
        expected_mouse, expected_section = manifest_metadata[str(row.image_id)]
        if str(row.mouse_id) != expected_mouse or str(row.section_id) != expected_section:
            raise ValueError(
                f"image_qc.csv metadata for {row.image_id!r} does not match project.yaml"
            )
    return out


def _validate_object_qc(
    table: pd.DataFrame | None,
    *,
    label: str,
    schema: str,
    object_id_column: str,
    project: Project,
) -> pd.DataFrame | None:
    if table is None:
        return None
    required_columns = {
        "schema_version",
        "qc_version",
        "rules_version",
        "model_version",
        "project_id",
        "image_id",
        object_id_column,
        "review_priority",
        "technical_reason_codes",
    }
    if table.empty:
        missing = sorted(required_columns - set(table.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")
        return table.copy()
    return _validate_provenance(
        table,
        label=label,
        expected_schema=schema,
        project=project,
        required_columns=required_columns,
    )


def _validate_selection(
    table: pd.DataFrame | None,
    project: Project,
) -> pd.DataFrame | None:
    if table is None:
        return None
    required = {
        "schema_version",
        "qc_version",
        "rules_version",
        "model_version",
        "computed_at",
        "project_id",
        "mouse_id",
        "domain",
        "strategy",
        "selected_image_ids",
        "eligible_image_ids",
        "requires_manual_review",
        "reason_code",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"section_selection.csv is missing required columns: {missing}")
    schemas = set(table["schema_version"].dropna().astype(str))
    if schemas != {SECTION_SELECTION_SCHEMA_VERSION}:
        raise ValueError(
            "section_selection.csv has unsupported schema_version values "
            f"{sorted(schemas)}"
        )
    project_ids = set(table["project_id"].dropna().astype(str))
    if project_ids != {project.project_id}:
        raise ValueError("section_selection.csv project_id does not match project.yaml")
    out = table.copy()
    for field in ("qc_version", "rules_version", "model_version"):
        values = set(out[field].dropna().astype(str))
        if len(values) > 1:
            raise ValueError(
                f"section_selection.csv mixes {field} values: {sorted(values)}"
            )
    model_versions = set(out["model_version"].dropna().astype(str))
    if model_versions and model_versions != {project.model_version}:
        raise ValueError("section_selection.csv model_version does not match project.yaml")
    unknown_mice = sorted(
        set(out["mouse_id"].dropna().astype(str))
        - {image.mouse_id for image in project.images}
    )
    if unknown_mice:
        raise ValueError(
            f"section_selection.csv contains mice absent from the manifest: {unknown_mice}"
        )
    unknown_domains = sorted(
        set(out["domain"].dropna().astype(str)) - {domain.value for domain in Domain}
    )
    if unknown_domains:
        raise ValueError(
            f"section_selection.csv contains unknown domains: {unknown_domains}"
        )
    unknown_strategies = sorted(
        set(out["strategy"].dropna().astype(str))
        - {strategy.value for strategy in SelectionStrategy}
    )
    if unknown_strategies:
        raise ValueError(
            f"section_selection.csv contains unknown strategies: {unknown_strategies}"
        )
    duplicates = out.duplicated(["mouse_id", "domain"], keep=False)
    if duplicates.any():
        raise ValueError("section_selection.csv contains duplicate mouse/domain rows")
    out["requires_manual_review"] = _parse_bool_series(
        out["requires_manual_review"],
        "section_selection.requires_manual_review",
    )
    return out


def _single_value(table: pd.DataFrame, field: str) -> str | None:
    if table.empty or field not in table.columns:
        return None
    values = set(table[field].dropna().astype(str))
    return next(iter(values)) if len(values) == 1 else None


def _validate_cross_table_provenance(
    image_qc: pd.DataFrame,
    tables: dict[str, pd.DataFrame | None],
) -> None:
    for field in ("qc_version", "rules_version", "model_version"):
        expected = _single_value(image_qc, field)
        for label, table in tables.items():
            if table is None or table.empty:
                continue
            observed = _single_value(table, field)
            if observed != expected:
                raise ValueError(
                    f"{label} {field} {observed!r} does not match "
                    f"image_qc.csv {expected!r}"
                )


def load_dashboard_tables(
    project: Project,
    qc_directory: Path | str | None = None,
) -> DashboardTables:
    """Load and validate Phase 2A outputs without importing GUI dependencies."""
    directory = (
        Path(qc_directory).expanduser().resolve()
        if qc_directory is not None
        else (project.root / "qc").resolve()
    )
    image_qc = _validate_image_qc(
        _read_csv(directory / "image_qc.csv", "image_qc.csv", required=True),
        project,
    )
    fiber_qc = _validate_object_qc(
        _read_csv(directory / "fiber_qc.csv", "fiber_qc.csv", required=False),
        label="fiber_qc.csv",
        schema=FIBER_QC_SCHEMA_VERSION,
        object_id_column="fiber_id",
        project=project,
    )
    nucleus_qc = _validate_object_qc(
        _read_csv(directory / "nucleus_qc.csv", "nucleus_qc.csv", required=False),
        label="nucleus_qc.csv",
        schema=NUCLEUS_QC_SCHEMA_VERSION,
        object_id_column="nucleus_id",
        project=project,
    )
    stored_selection = _validate_selection(
        _read_csv(
            directory / "section_selection.csv",
            "section_selection.csv",
            required=False,
        ),
        project,
    )
    _validate_cross_table_provenance(
        image_qc,
        {
            "fiber_qc.csv": fiber_qc,
            "nucleus_qc.csv": nucleus_qc,
            "section_selection.csv": stored_selection,
        },
    )
    return DashboardTables(
        image_qc=image_qc,
        fiber_qc=fiber_qc,
        nucleus_qc=nucleus_qc,
        stored_selection=stored_selection,
    )


def _selected_ids(selection: pd.DataFrame, mouse_id: str, domain: Domain) -> list[str]:
    rows = selection[
        selection["mouse_id"].eq(mouse_id) & selection["domain"].eq(domain.value)
    ]
    if rows.empty:
        return []
    value = rows.iloc[0]["selected_image_ids"]
    if pd.isna(value) or not str(value).strip():
        return []
    return [item for item in str(value).split("|") if item]


def _correction_counts(session: ReviewSession | None) -> tuple[int, int, int]:
    if session is None:
        return 0, 0, 0
    mask_count = sum(len(paths) for paths in session.reviewed_mask_paths.values())
    return len(session.object_decisions), len(session.regions), mask_count


def build_dashboard_model(
    project: Project,
    tables: DashboardTables,
    *,
    strategy: SelectionStrategy | str = SelectionStrategy.ALL_PASSING,
    manual_selections: dict[str, dict[str, list[str]]] | None = None,
    session: ReviewSession | None = None,
) -> DashboardModel:
    """Build dashboard summaries and rows from validated Phase 2A outputs."""
    parsed_strategy = SelectionStrategy(strategy)
    selection = select_sections(
        project,
        tables.image_qc,
        strategy=parsed_strategy,
        manual_selections=manual_selections,
    )
    section_rows = tables.image_qc.copy()
    section_rows["selected"] = False
    for image in project.images:
        for domain in Domain:
            selected = _selected_ids(selection, image.mouse_id, domain)
            mask = section_rows["image_id"].eq(image.image_id) & section_rows["domain"].eq(
                domain.value
            )
            section_rows.loc[mask, "selected"] = image.image_id in selected
    manifest_order = {image.image_id: index for index, image in enumerate(project.images)}
    domain_order = {domain.value: index for index, domain in enumerate(Domain)}
    section_rows["_image_order"] = section_rows["image_id"].map(manifest_order)
    section_rows["_domain_order"] = section_rows["domain"].map(domain_order)
    section_rows = section_rows.sort_values(
        ["_image_order", "_domain_order"], kind="stable"
    ).drop(columns=["_image_order", "_domain_order"])

    domain_rows = []
    for domain in Domain:
        rows = section_rows[section_rows["domain"].eq(domain.value)]
        domain_rows.append(
            {
                "domain": domain.value,
                "pass_count": int(rows["status"].eq("pass").sum()),
                "review_count": int(rows["status"].eq("review").sum()),
                "fail_count": int(rows["status"].eq("fail").sum()),
                "not_applicable_count": int(rows["status"].eq("not_applicable").sum()),
            }
        )
    domain_counts = pd.DataFrame(domain_rows)

    mouse_domain_rows: list[dict[str, Any]] = []
    mouse_order = list(dict.fromkeys(image.mouse_id for image in project.images))
    for mouse_id in mouse_order:
        for domain in Domain:
            rows = section_rows[
                section_rows["mouse_id"].eq(mouse_id)
                & section_rows["domain"].eq(domain.value)
            ]
            applicable = rows[rows["applicable"]]
            acceptable = applicable[~applicable["hard_fail"]]
            selected_ids = _selected_ids(selection, mouse_id, domain)
            selection_row = selection[
                selection["mouse_id"].eq(mouse_id)
                & selection["domain"].eq(domain.value)
            ].iloc[0]
            if applicable.empty:
                readiness = "not_applicable"
            elif acceptable.empty:
                readiness = "no_acceptable_section"
            else:
                selected_rows = acceptable[acceptable["image_id"].isin(selected_ids)]
                targeted = (
                    bool(selection_row["requires_manual_review"])
                    or not applicable["status"].eq("pass").any()
                    or selected_rows["status"].eq("review").any()
                )
                readiness = "targeted_review" if targeted else "complete"
            mouse_domain_rows.append(
                {
                    "mouse_id": mouse_id,
                    "domain": domain.value,
                    "readiness": readiness,
                    "section_count": int(len(rows)),
                    "applicable_section_count": int(len(applicable)),
                    "pass_count": int(applicable["status"].eq("pass").sum()),
                    "review_count": int(applicable["status"].eq("review").sum()),
                    "fail_count": int(applicable["status"].eq("fail").sum()),
                    "acceptable_image_ids": "|".join(
                        acceptable["image_id"].astype(str).tolist()
                    ),
                    "selected_image_ids": "|".join(selected_ids),
                    "requires_manual_review": bool(
                        selection_row["requires_manual_review"]
                    ),
                    "selection_reason_code": str(selection_row["reason_code"]),
                }
            )
    mouse_domain = pd.DataFrame(mouse_domain_rows)

    mouse_rows: list[dict[str, Any]] = []
    for mouse_id in mouse_order:
        rows = mouse_domain[
            mouse_domain["mouse_id"].eq(mouse_id)
            & ~mouse_domain["readiness"].eq("not_applicable")
        ]
        if rows["readiness"].eq("no_acceptable_section").any():
            readiness = "no_acceptable_section"
        elif rows["readiness"].eq("targeted_review").any():
            readiness = "targeted_review"
        else:
            readiness = "complete"
        mouse_rows.append({"mouse_id": mouse_id, "readiness": readiness})
    mouse_table = pd.DataFrame(mouse_rows)

    applicable_rows = section_rows[section_rows["applicable"]]
    reviewed_rows = 0
    if session is not None:
        for row in applicable_rows.itertuples(index=False):
            if (
                session.get_status(str(row.image_id), str(row.domain))
                is not DomainStatus.NOT_REVIEWED
            ):
                reviewed_rows += 1
    object_count, region_count, mask_count = _correction_counts(session)
    applicable_count = int(len(applicable_rows))
    summary = DashboardSummary(
        mouse_count=len(mouse_order),
        section_count=len(project.images),
        complete_mouse_count=int(mouse_table["readiness"].eq("complete").sum()),
        targeted_review_mouse_count=int(
            mouse_table["readiness"].eq("targeted_review").sum()
        ),
        no_acceptable_mouse_count=int(
            mouse_table["readiness"].eq("no_acceptable_section").sum()
        ),
        applicable_domain_rows=applicable_count,
        reviewed_domain_rows=reviewed_rows,
        review_progress_fraction=(
            reviewed_rows / applicable_count if applicable_count else 0.0
        ),
        object_decision_count=object_count,
        region_count=region_count,
        reviewed_mask_count=mask_count,
    )
    return DashboardModel(
        project=project,
        tables=tables,
        selection=selection,
        section_rows=section_rows,
        domain_counts=domain_counts,
        mouse_domain_rows=mouse_domain,
        mouse_rows=mouse_table,
        summary=summary,
        strategy=parsed_strategy,
    )
