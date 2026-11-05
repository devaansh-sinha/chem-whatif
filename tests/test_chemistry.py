from __future__ import annotations

import pytest
from rdkit import Chem

from src import chemistry
from src.chemistry import MoleculeError, calculate_descriptors, parse_smiles, resolve_molecule


@pytest.mark.parametrize(
    ("name", "smiles", "formula"),
    [
        ("Ethanol", "CCO", "C2H6O"),
        ("Acetone", "CC(=O)C", "C3H6O"),
        ("Phenol", "Oc1ccccc1", "C6H6O"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4"),
        ("Caffeine", "Cn1cnc2n(C)c(=O)n(C)c(=O)c12", "C8H10N4O2"),
    ],
)
def test_parse_and_calculate_real_structure_descriptors(name, smiles, formula):
    record = parse_smiles(smiles, name=name)
    descriptors = calculate_descriptors(record.mol)
    assert record.canonical_smiles == Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    assert descriptors["formula"] == formula
    assert descriptors["molecular_weight"] > 0
    assert descriptors["atom_count"] == record.mol.GetNumAtoms()
    assert descriptors["bond_count"] == record.mol.GetNumBonds()


def test_invalid_smiles_has_friendly_error():
    with pytest.raises(MoleculeError):
        parse_smiles("this is not smiles")


def test_name_resolution_uses_pubchem_result(monkeypatch):
    monkeypatch.setattr(chemistry, "fetch_pubchem_smiles", lambda name: "CCO")
    record = resolve_molecule("ethanol")
    assert record.name == "Ethanol"
    assert record.source.startswith("PubChem PUG REST")
    assert calculate_descriptors(record.mol)["formula"] == "C2H6O"


def test_isomeric_smiles_and_stereochemistry_are_preserved():
    record = parse_smiles("F[C@H](Cl)Br", name="Chiral test")
    assert "@" in record.canonical_smiles
    assert chemistry.stereochemistry_summary(record.mol)["defined_centers"] == 1


def test_pubchem_prefers_isomeric_smiles_and_network_errors_are_friendly(monkeypatch):
    class FakeResponse:
        def read(self):
            return b'{"PropertyTable":{"Properties":[{"CanonicalSMILES":"FCCCl","IsomericSMILES":"F[C@H](Cl)C"}]}}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(chemistry, "urlopen", lambda *args, **kwargs: FakeResponse())
    assert chemistry.fetch_pubchem_smiles("chiral test") == "F[C@H](Cl)C"
    monkeypatch.setattr(chemistry, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(MoleculeError, match="could not be resolved"):
        chemistry.fetch_pubchem_smiles("unavailable")
