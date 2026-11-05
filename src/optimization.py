"""Transparent, single-objective molecular optimization for CHEM-WHATIF.

The optimizer deliberately ranks calculated descriptor changes, not claims
about synthesis, safety, or biological activity.  Every generated structure is
kept in the returned audit trail, including candidates that do not pass a
quality check or a user-selected requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import eq, ge, gt, le, lt
from typing import Callable, Iterable

from rdkit import Chem

from .candidate_quality import QualityResult, screen_candidate
from .chemistry import calculate_descriptors, descriptor_label, descriptor_value_text
from .functional_groups import group_count
from .modifications import Candidate, CandidateGeneration, generate_candidates
from .reasoning import candidate_reasoning, objective_explanation


@dataclass(frozen=True)
class NumericalConstraint:
    """One optional numerical requirement applied after quality screening."""

    property_key: str
    operator: str
    value: float


@dataclass(frozen=True)
class ConstraintResult:
    label: str
    passed: bool
    explanation: str


@dataclass(frozen=True)
class RetentionResult:
    label: str
    passed: bool
    explanation: str


@dataclass(frozen=True)
class ScreenedCandidate:
    """A candidate together with every decision made during optimization."""

    candidate: Candidate
    quality: QualityResult
    constraints: tuple[ConstraintResult, ...]
    retention: tuple[RetentionResult, ...]
    classification: str
    rejection_reasons: tuple[str, ...]
    objective_delta: float | None
    objective_score: float | None
    objective_message: str
    chemical_reasoning: str


@dataclass(frozen=True)
class StructuralFamily:
    """A tied family of positional isomers for one selected descriptor."""

    label: str
    transformation: str
    objective_value: float
    members: tuple[ScreenedCandidate, ...]
    rank: int


@dataclass(frozen=True)
class OptimizationResult:
    """Complete, user-facing optimization result and audit counts."""

    property_key: str
    direction: str
    generation: CandidateGeneration
    evaluated_count: int
    feasible_count: int
    improving_count: int
    rejected_count: int
    best_candidate: ScreenedCandidate | None
    improving_candidates: tuple[ScreenedCandidate, ...]
    feasible_not_preferred: tuple[ScreenedCandidate, ...]
    rejected_candidates: tuple[ScreenedCandidate, ...]
    all_screened: tuple[ScreenedCandidate, ...]
    positional_isomer_families: tuple[StructuralFamily, ...]


OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<": lt,
    "<=": le,
    ">": gt,
    ">=": ge,
    "==": eq,
}


def _constraint_result(candidate: Candidate, constraint: NumericalConstraint) -> ConstraintResult:
    if constraint.operator not in OPERATORS:
        return ConstraintResult(
            descriptor_label(constraint.property_key),
            False,
            f"Unsupported numerical condition: {constraint.operator}.",
        )
    actual = float(candidate.descriptors[constraint.property_key])
    passed = OPERATORS[constraint.operator](actual, constraint.value)
    label = descriptor_label(constraint.property_key)
    return ConstraintResult(
        label,
        passed,
        f"Calculated {label} is {descriptor_value_text(constraint.property_key, actual)}; "
        f"requirement: {constraint.operator} {descriptor_value_text(constraint.property_key, constraint.value)} "
        f"({'passed' if passed else 'failed'}).",
    )


def _retention_result(candidate: Candidate, key: str, label: str) -> RetentionResult:
    count = group_count(candidate.mol, key)
    passed = count > 0
    return RetentionResult(
        label,
        passed,
        f"{label} {'is retained' if passed else 'is not retained'} in the candidate"
        + (f" ({count} occurrence{'s' if count != 1 else ''})." if passed else "."),
    )


def objective_delta(
    property_key: str, direction: str, original: dict[str, object], candidate: dict[str, object]
) -> tuple[float, float, bool]:
    """Return descriptor change, direction-aware score, and improvement flag."""
    delta = float(candidate[property_key]) - float(original[property_key])
    score = delta if direction == "increase" else -delta
    return delta, score, score > 0.005


def _sort_improving(items: Iterable[ScreenedCandidate], direction: str) -> tuple[ScreenedCandidate, ...]:
    """Order only by the requested calculated improvement, then identity."""
    reverse = direction == "increase"
    return tuple(
        sorted(
            items,
            key=lambda item: (item.objective_score if item.objective_score is not None else float("-inf"), item.candidate.canonical_smiles),
            reverse=reverse,
        )
    )


def group_positional_isomers(
    items: Iterable[ScreenedCandidate], property_key: str, direction: str
) -> tuple[StructuralFamily, ...]:
    """Group tied aromatic positional isomers; never invent a rank difference."""
    buckets: dict[tuple[str, float], list[ScreenedCandidate]] = {}
    for item in items:
        if item.candidate.structural_family != "Aromatic substitution":
            continue
        if item.objective_score is None:
            continue
        value = round(float(item.candidate.descriptors[property_key]), 6)
        buckets.setdefault((item.candidate.transformation, value), []).append(item)

    families: list[StructuralFamily] = []
    eligible = [(key, members) for key, members in buckets.items() if len(members) > 1]
    eligible.sort(key=lambda pair: (pair[0][1], pair[0][0]), reverse=direction == "increase")
    for rank, ((transformation, value), members) in enumerate(eligible, start=1):
        ordered_members = tuple(sorted(members, key=lambda item: item.candidate.canonical_smiles))
        families.append(
            StructuralFamily(
                label="Positional isomers",
                transformation=transformation,
                objective_value=value,
                members=ordered_members,
                rank=rank,
            )
        )
    return tuple(families)


def optimize(
    starting_mol: Chem.Mol,
    property_key: str,
    direction: str,
    numerical_constraints: Iterable[NumericalConstraint] = (),
    retained_groups: Iterable[tuple[str, str]] = (),
    generation: CandidateGeneration | None = None,
) -> OptimizationResult:
    """Run the explicit candidate-to-classification pipeline for one objective."""
    if direction not in {"increase", "decrease"}:
        raise ValueError("Direction must be 'increase' or 'decrease'.")
    original = calculate_descriptors(starting_mol)
    if property_key not in original:
        raise ValueError(f"Unsupported optimization property: {property_key}")

    current_generation = generation or generate_candidates(starting_mol)
    constraints = tuple(numerical_constraints)
    retain = tuple(retained_groups)
    screened: list[ScreenedCandidate] = []

    for candidate in current_generation.candidates:
        quality = screen_candidate(candidate.mol, starting_mol)
        constraint_checks = tuple(_constraint_result(candidate, constraint) for constraint in constraints)
        retention_checks = tuple(_retention_result(candidate, key, label) for key, label in retain)
        failures = list(quality.rejection_reasons)
        failures.extend(check.explanation for check in constraint_checks if not check.passed)
        failures.extend(f"Functional-group retention failed: {check.explanation}" for check in retention_checks if not check.passed)
        delta, _, improved = objective_delta(property_key, direction, original, candidate.descriptors)
        score = float(candidate.descriptors[property_key])
        message = objective_explanation(property_key, direction, original, candidate.descriptors)
        reasoning = candidate_reasoning(candidate.reason, original, candidate.descriptors)
        if failures:
            classification = "rejected"
        elif improved:
            classification = "improving"
        else:
            classification = "feasible"
        screened.append(
            ScreenedCandidate(
                candidate=candidate,
                quality=quality,
                constraints=constraint_checks,
                retention=retention_checks,
                classification=classification,
                rejection_reasons=tuple(failures),
                objective_delta=delta,
                objective_score=score,
                objective_message=message,
                chemical_reasoning=reasoning,
            )
        )

    improving = _sort_improving((item for item in screened if item.classification == "improving"), direction)
    feasible = tuple(
        sorted(
            (item for item in screened if item.classification == "feasible"),
            key=lambda item: (abs(item.objective_delta or 0.0), item.candidate.canonical_smiles),
        )
    )
    rejected = tuple(
        sorted(
            (item for item in screened if item.classification == "rejected"),
            key=lambda item: item.candidate.canonical_smiles,
        )
    )
    return OptimizationResult(
        property_key=property_key,
        direction=direction,
        generation=current_generation,
        evaluated_count=len(screened),
        feasible_count=len(improving) + len(feasible),
        improving_count=len(improving),
        rejected_count=len(rejected),
        best_candidate=improving[0] if improving else None,
        improving_candidates=improving,
        feasible_not_preferred=feasible,
        rejected_candidates=rejected,
        all_screened=tuple(screened),
        positional_isomer_families=group_positional_isomers((*improving, *feasible), property_key, direction),
    )
