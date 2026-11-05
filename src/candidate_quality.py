"""Small, transparent structural sanity checks for generated candidates."""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from .chemistry import calculate_descriptors, canonical_smiles, validate_molecule


# This lightweight scope is intentionally clear: it covers typical first-year
# organic examples without suggesting the screen handles all of chemistry.
SUPPORTED_ATOMIC_NUMBERS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}


@dataclass(frozen=True)
class CheckResult:
    label: str
    passed: bool
    explanation: str


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    score: int
    checks: tuple[CheckResult, ...]
    rejection_reasons: tuple[str, ...]


def screen_candidate(candidate_mol: Chem.Mol, starting_mol: Chem.Mol) -> QualityResult:
    """Apply explainable quality checks; this is not a synthesis or safety screen."""
    checks: list[CheckResult] = []
    try:
        candidate = validate_molecule(candidate_mol)
        checks.append(CheckResult("RDKit chemical validity", True, "The candidate passed RDKit sanitization."))
    except Exception:
        checks.append(CheckResult("RDKit chemical validity", False, "The candidate could not be sanitized by RDKit."))
        return QualityResult(False, 0, tuple(checks), ("The candidate is not a valid RDKit chemical structure.",))

    descriptor = calculate_descriptors(candidate)
    unsupported = sorted({atom.GetSymbol() for atom in candidate.GetAtoms() if atom.GetAtomicNum() not in SUPPORTED_ATOMIC_NUMBERS})
    checks.append(
        CheckResult(
            "Supported element scope",
            not unsupported,
            "All elements are within the prototype's supported organic-chemistry scope."
            if not unsupported
            else f"Unsupported element(s) for this prototype: {', '.join(unsupported)}.",
        )
    )
    weight = float(descriptor["molecular_weight"])
    checks.append(
        CheckResult(
            "Reasonable molecular weight",
            12 <= weight <= 700,
            f"Calculated molecular weight is {weight:.2f} g/mol; the educational screen accepts 12–700 g/mol."
            if 12 <= weight <= 700
            else f"Calculated molecular weight is {weight:.2f} g/mol, outside the 12–700 g/mol screen.",
        )
    )
    heavy_atoms = candidate.GetNumHeavyAtoms()
    checks.append(
        CheckResult(
            "Reasonable structural size",
            1 <= heavy_atoms <= 80,
            f"The candidate has {heavy_atoms} heavy atoms, within the prototype's 1–80 range."
            if 1 <= heavy_atoms <= 80
            else f"The candidate has {heavy_atoms} heavy atoms, outside the prototype's 1–80 range.",
        )
    )
    different = canonical_smiles(candidate) != canonical_smiles(starting_mol)
    checks.append(
        CheckResult(
            "Different from starting molecule",
            different,
            "The canonical molecular structure differs from the starting molecule."
            if different
            else "The canonical molecular structure is identical to the starting molecule.",
        )
    )
    failed = tuple(check.explanation for check in checks if not check.passed)
    passed_count = sum(check.passed for check in checks)
    score = round(100 * passed_count / len(checks))
    return QualityResult(not failed, score, tuple(checks), failed)
