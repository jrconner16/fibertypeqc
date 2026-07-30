"""Versioned, human-readable QC rule configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from src.review.schemas import Domain, parse_enum

DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "review_qc_rules.v1.yaml"
)


class RuleSeverity(StrEnum):
    INFORMATIONAL = "informational"
    REVIEW = "review"
    HARD_FAIL = "hard_fail"


class RuleOperator(StrEnum):
    EQ = "eq"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"


@dataclass(frozen=True)
class QCRule:
    reason_code: str
    domain: Domain
    metric: str
    operator: RuleOperator
    threshold: Any
    severity: RuleSeverity
    enabled: bool
    description: str

    def __post_init__(self) -> None:
        if not self.reason_code.strip():
            raise ValueError("QC rule reason_code must not be empty")
        if not self.metric.strip():
            raise ValueError(f"QC rule {self.reason_code!r} metric must not be empty")
        object.__setattr__(self, "domain", parse_enum(Domain, self.domain, "QC rule domain"))
        try:
            object.__setattr__(self, "operator", RuleOperator(self.operator))
        except ValueError as exc:
            raise ValueError(
                f"QC rule {self.reason_code!r} has invalid operator {self.operator!r}"
            ) from exc
        try:
            object.__setattr__(self, "severity", RuleSeverity(self.severity))
        except ValueError as exc:
            raise ValueError(
                f"QC rule {self.reason_code!r} has invalid severity {self.severity!r}"
            ) from exc
        if self.enabled and self.threshold is None:
            raise ValueError(f"Enabled QC rule {self.reason_code!r} requires a threshold")

    def matches(self, metrics: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        observed = metrics.get(self.metric)
        if observed is None:
            return False
        operations = {
            RuleOperator.EQ: lambda left, right: left == right,
            RuleOperator.GT: lambda left, right: left > right,
            RuleOperator.GE: lambda left, right: left >= right,
            RuleOperator.LT: lambda left, right: left < right,
            RuleOperator.LE: lambda left, right: left <= right,
        }
        try:
            return bool(operations[self.operator](observed, self.threshold))
        except TypeError as exc:
            raise ValueError(
                f"QC rule {self.reason_code!r} cannot compare metric "
                f"{self.metric!r} value {observed!r} to {self.threshold!r}"
            ) from exc


@dataclass(frozen=True)
class QCRuleConfig:
    schema_version: str
    rules_version: str
    qc_version: str
    rules: tuple[QCRule, ...]

    def rules_for(self, domain: Domain | str) -> tuple[QCRule, ...]:
        parsed = parse_enum(Domain, domain, "QC rules domain")
        return tuple(rule for rule in self.rules if rule.domain is parsed)


def load_rule_config(path: Path | str = DEFAULT_RULES_PATH) -> QCRuleConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"QC rule configuration does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"QC rule configuration is not valid YAML: {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"QC rule configuration must contain a mapping: {config_path}")
    schema_version = data.get("schema_version")
    if schema_version != "review_qc_rules.v1":
        raise ValueError(
            f"Unsupported QC rule schema_version {schema_version!r}; "
            "expected 'review_qc_rules.v1'"
        )
    rules_version = data.get("rules_version")
    qc_version = data.get("qc_version")
    if not isinstance(rules_version, str) or not rules_version.strip():
        raise ValueError("QC rules_version must be a non-empty string")
    if not isinstance(qc_version, str) or not qc_version.strip():
        raise ValueError("QC qc_version must be a non-empty string")
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError("QC rules must be a list")
    rules: list[QCRule] = []
    seen_codes: set[str] = set()
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            raise ValueError(f"QC rules[{index}] must be a mapping")
        reason_code = str(raw.get("reason_code", "")).strip()
        if reason_code in seen_codes:
            raise ValueError(f"Duplicate QC reason_code: {reason_code}")
        rule = QCRule(
            reason_code=reason_code,
            domain=parse_enum(Domain, raw.get("domain", ""), f"QC rules[{index}].domain"),
            metric=str(raw.get("metric", "")).strip(),
            operator=raw.get("operator", ""),
            threshold=raw.get("threshold"),
            severity=raw.get("severity", ""),
            enabled=bool(raw.get("enabled", False)),
            description=str(raw.get("description", "")).strip(),
        )
        rules.append(rule)
        seen_codes.add(reason_code)
    return QCRuleConfig(
        schema_version=schema_version,
        rules_version=rules_version,
        qc_version=qc_version,
        rules=tuple(rules),
    )


def evaluate_rules(
    config: QCRuleConfig,
    domain: Domain,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    triggered: list[dict[str, Any]] = []
    for rule in config.rules_for(domain):
        if rule.matches(metrics):
            triggered.append(
                {
                    "reason_code": rule.reason_code,
                    "severity": rule.severity.value,
                    "metric": rule.metric,
                    "observed": metrics.get(rule.metric),
                    "operator": rule.operator.value,
                    "threshold": rule.threshold,
                    "description": rule.description,
                }
            )
    return triggered
