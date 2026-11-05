from __future__ import annotations

from rdkit.Chem import AllChem

from src.chemistry import parse_smiles
from src.functional_groups import detect_functional_groups, group_count
from src.visualization import generate_conformer, interactive_3d_html, molecule_png


def keys(smiles: str) -> set[str]:
    return {group.key for group in detect_functional_groups(parse_smiles(smiles).mol)}


def test_supported_functional_groups_across_multiple_molecules():
    assert {"alcohol"} <= keys("CCO")
    assert {"phenol", "aromatic_ring"} <= keys("Oc1ccccc1")
    assert {"carboxylic_acid", "ester", "carbonyl", "aromatic_ring"} <= keys("CC(=O)Oc1ccccc1C(=O)O")
    assert {"carbonyl", "ketone"} <= keys("CC(=O)C")
    assert {"carbonyl", "aldehyde"} <= keys("O=Cc1ccccc1")
    assert {"halogen", "aromatic_ring"} <= keys("Clc1ccccc1")
    assert {"nitro", "aromatic_ring"} <= keys("O=[N+]([O-])c1ccccc1")
    assert {"alkene", "aromatic_ring"} <= keys("C=Cc1ccccc1")
    assert {"carbonyl", "heteroaromatic_ring"} <= keys("Cn1cnc2n(C)c(=O)n(C)c(=O)c12")


def test_ester_oxygen_is_not_also_reported_as_ether():
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").mol
    assert group_count(aspirin, "ester") == 1
    assert group_count(aspirin, "ether") == 0


def test_2d_png_and_3d_viewer_controls_are_generated():
    phenol = parse_smiles("Oc1ccccc1").mol
    png = molecule_png(phenol)
    highlighted_png = molecule_png(phenol, highlight_atoms=(0,))
    assert png.startswith(b"\x89PNG")
    assert highlighted_png.startswith(b"\x89PNG")
    assert png != highlighted_png
    conformer = generate_conformer(phenol)
    assert "V2000" in conformer.molblock or "V3000" in conformer.molblock
    html = interactive_3d_html(conformer, theme="dark")
    assert all(control in html for control in ("Zoom Out", "Reset", "Zoom In"))
    assert "zoom(" in html
    assert all(token in html for token in ('data-zoom="out"', 'data-zoom="reset"', 'data-zoom="in"'))


def test_3d_embedding_uses_visible_random_coordinate_fallback(monkeypatch):
    phenol = parse_smiles("Oc1ccccc1").mol
    real_embed = AllChem.EmbedMolecule
    calls = {"count": 0}

    def fail_once(mol, params):
        calls["count"] += 1
        return -1 if calls["count"] == 1 else real_embed(mol, params)

    monkeypatch.setattr(AllChem, "EmbedMolecule", fail_once)
    conformer = generate_conformer(phenol)
    assert calls["count"] == 2
    assert "random-coordinate embedding fallback" in conformer.optimization_status
