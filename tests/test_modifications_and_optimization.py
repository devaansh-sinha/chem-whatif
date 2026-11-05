from __future__ import annotations

import pytest

from src.candidate_quality import screen_candidate
from src.chemistry import calculate_descriptors, parse_smiles
from src.modifications import generate_candidates
from src.optimization import NumericalConstraint, objective_delta, optimize


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",  # caffeine
        "Oc1ccccc1",  # phenol
        "CCO",  # ethanol
        "CC(=O)C",  # acetone
        "O=Cc1ccccc1",  # benzaldehyde
        "Clc1ccccc1",  # chlorobenzene
        "O=[N+]([O-])c1ccccc1",  # nitrobenzene
        "C=Cc1ccccc1",  # styrene
    ],
)
def test_candidate_generation_is_valid_and_deduplicated(smiles):
    starting = parse_smiles(smiles).mol
    generation = generate_candidates(starting)
    canonical = [candidate.canonical_smiles for candidate in generation.candidates]
    assert len(canonical) == len(set(canonical))
    assert generation.unique_count == len(canonical)
    assert generation.generated_count >= generation.unique_count
    for candidate in generation.candidates:
        assert calculate_descriptors(candidate.mol)["formula"]


@pytest.mark.parametrize(
    ("smiles", "transformation"),
    [
        ("CCO", "Alcohol → methyl ether"),
        ("Oc1ccccc1", "Phenol → methyl ether"),
        ("Oc1ccccc1", "Phenol → fluorine"),
        ("CC(=O)Oc1ccccc1C(=O)O", "Carboxylic acid → methyl ester"),
        ("CC(=O)N", "Amide → N-methyl amide"),
        ("c1ccccc1", "Aromatic C–H → fluorine"),
        ("c1ccccc1", "Aromatic C–H → methyl"),
        ("c1ccccc1", "Aromatic C–H → hydroxyl"),
        ("c1ccccc1", "Aromatic C–H → methoxy"),
        ("c1ccccc1", "Aromatic C–H → amino"),
        ("Clc1ccccc1", "Aryl chloride/bromide → fluorine"),
        ("Fc1ccccc1", "Aryl fluorine → hydrogen"),
        ("C=C", "Alkene → alkane"),
        ("C#C", "Alkyne → alkene"),
        ("CC(=O)C", "Ketone → alcohol"),
        ("O=CC", "Aldehyde → alcohol"),
        ("O=[N+]([O-])c1ccccc1", "Nitro → amine"),
    ],
)
def test_each_supported_transformation_is_generated_when_its_site_exists(smiles, transformation):
    candidates = generate_candidates(parse_smiles(smiles).mol).candidates
    assert transformation in {candidate.transformation for candidate in candidates}


def test_no_supported_change_is_safe_for_methane():
    generation = generate_candidates(parse_smiles("C").mol)
    assert generation.candidates == ()
    result = optimize(parse_smiles("C").mol, "logp", "increase", generation=generation)
    assert result.evaluated_count == result.rejected_count == result.improving_count == 0


def test_quality_screen_rejects_starting_structure_as_not_different():
    ethanol = parse_smiles("CCO").mol
    result = screen_candidate(ethanol, ethanol)
    assert not result.passed
    assert any("identical" in reason.lower() for reason in result.rejection_reasons)


def test_numerical_constraint_and_rejection_reporting():
    ethanol = parse_smiles("CCO").mol
    generation = generate_candidates(ethanol)
    result = optimize(
        ethanol,
        "molecular_weight",
        "increase",
        numerical_constraints=(NumericalConstraint("molecular_weight", "<=", 50),),
        generation=generation,
    )
    assert result.rejected_count == 1
    rejected = result.rejected_candidates[0]
    assert rejected.candidate.mol.GetNumAtoms() > 0
    assert any("failed" in reason.lower() for reason in rejected.rejection_reasons)
    assert rejected.constraints and not rejected.constraints[0].passed


def test_functional_group_retention_rejects_methyl_ester_of_aspirin():
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").mol
    generation = generate_candidates(aspirin)
    result = optimize(
        aspirin,
        "logp",
        "decrease",
        retained_groups=(("carboxylic_acid", "Carboxylic acid"),),
        generation=generation,
    )
    methyl_ester = next(item for item in result.rejected_candidates if item.candidate.transformation == "Carboxylic acid → methyl ester")
    assert any("retention failed" in reason.lower() for reason in methyl_ester.rejection_reasons)


def test_objective_scoring_and_ketone_reduction_improves_hbd():
    acetone = parse_smiles("CC(=O)C").mol
    base = calculate_descriptors(acetone)
    generation = generate_candidates(acetone)
    product = next(candidate for candidate in generation.candidates if candidate.transformation == "Ketone → alcohol")
    delta, score, improved = objective_delta("hbd", "increase", base, product.descriptors)
    assert delta == score == 1
    assert improved
    result = optimize(acetone, "hbd", "increase", generation=generation)
    assert result.best_candidate is not None
    assert result.best_candidate.candidate.transformation == "Ketone → alcohol"


def test_positional_isomers_are_grouped_when_objective_is_tied():
    phenol = parse_smiles("Oc1ccccc1").mol
    result = optimize(phenol, "logp", "decrease")
    fluorination = next(family for family in result.positional_isomer_families if family.transformation == "Aromatic C–H → fluorine")
    assert fluorination.label == "Positional isomers"
    assert len(fluorination.members) == 3
    assert len({member.candidate.descriptors["logp"] for member in fluorination.members}) == 1
    assert len({member.candidate.canonical_smiles for member in fluorination.members}) == 3
