"""Molecule resolution, validation, and RDKit descriptor calculations.

All reported properties in this module are calculated descriptors, not
experimental measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen
import json

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES,IsomericSMILES/JSON"


class MoleculeError(ValueError):
    """A friendly error raised for a molecule that cannot be used."""


@dataclass(frozen=True)
class MoleculeRecord:
    """A resolved molecular structure and the provenance of its input."""

    name: str
    mol: Chem.Mol
    canonical_smiles: str
    source: str


DESCRIPTOR_META: dict[str, dict[str, str]] = {
    "molecular_weight": {"label": "Molecular weight", "unit": "g/mol"},
    "logp": {"label": "Calculated LogP", "unit": ""},
    "tpsa": {"label": "TPSA", "unit": "Å²"},
    "hbd": {"label": "H-bond donors", "unit": ""},
    "hba": {"label": "H-bond acceptors", "unit": ""},
    "rotatable_bonds": {"label": "Rotatable bonds", "unit": ""},
    "atom_count": {"label": "Atom count", "unit": ""},
    "bond_count": {"label": "Bond count", "unit": ""},
    "formula": {"label": "Molecular formula", "unit": ""},
}


def validate_molecule(mol: Chem.Mol | None) -> Chem.Mol:
    """Copy, sanitize, and return an RDKit molecule or raise a friendly error."""
    if mol is None:
        raise MoleculeError("No chemical structure could be read from that input.")
    checked = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(checked)
    except Exception as exc:  # RDKit exception types vary by version.
        raise MoleculeError("The chemical structure could not be validated.") from exc
    if checked.GetNumAtoms() == 0:
        raise MoleculeError("The chemical structure does not contain any atoms.")
    return checked


def parse_smiles(smiles: str, name: str = "Entered structure", source: str = "SMILES") -> MoleculeRecord:
    """Parse a technical SMILES input for the optional advanced workflow."""
    if not smiles or not smiles.strip():
        raise MoleculeError("Please provide a chemical structure.")
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
        mol = validate_molecule(mol)
    except MoleculeError:
        raise
    except Exception as exc:
        raise MoleculeError("That structure could not be interpreted as a valid molecule.") from exc
    return MoleculeRecord(name=name, mol=mol, canonical_smiles=canonical_smiles(mol), source=source)


def canonical_smiles(mol: Chem.Mol) -> str:
    """Return a stereochemistry-preserving canonical identity without atom-map labels."""
    copy = Chem.Mol(validate_molecule(mol))
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def fetch_pubchem_smiles(name: str, timeout: int = 12) -> str:
    """Resolve a human-readable compound name with PubChem PUG REST."""
    clean_name = name.strip()
    if not clean_name:
        raise MoleculeError("Please enter a molecule name.")
    request = Request(
        PUBCHEM_URL.format(name=quote(clean_name, safe="")),
        headers={"User-Agent": "CHEM-WHATIF educational computational chemistry project"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS endpoint
            payload = json.loads(response.read().decode("utf-8"))
        properties = payload["PropertyTable"]["Properties"]
        if not properties:
            raise KeyError("empty property list")
        item = properties[0]
        # Prefer isomeric SMILES. Canonical/connectivity SMILES can omit
        # stereochemistry, which would make a structure-property exercise less
        # faithful for chiral or alkene-stereochemical compounds.
        return str(item.get("IsomericSMILES") or item.get("SMILES") or item.get("CanonicalSMILES") or item.get("ConnectivitySMILES"))
    except (KeyError, TypeError, ValueError) as exc:
        raise MoleculeError(f'PubChem did not find a usable structure for "{clean_name}".') from exc
    except Exception as exc:
        raise MoleculeError(
            "The molecule name could not be resolved right now. Check the spelling or try again when connected to the internet."
        ) from exc


def resolve_molecule(name: str) -> MoleculeRecord:
    """Resolve a normal user-facing molecule name through PubChem and validate it."""
    smiles = fetch_pubchem_smiles(name)
    return parse_smiles(smiles, name=name.strip().title(), source="PubChem PUG REST (isomeric structure when available)")


def calculate_descriptors(mol: Chem.Mol) -> dict[str, Any]:
    """Calculate the project's fixed panel of structural RDKit descriptors."""
    checked = validate_molecule(mol)
    return {
        "formula": rdMolDescriptors.CalcMolFormula(checked),
        "molecular_weight": round(Descriptors.MolWt(checked), 2),
        "logp": round(Crippen.MolLogP(checked), 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(checked), 2),
        "hbd": int(Lipinski.NumHDonors(checked)),
        "hba": int(Lipinski.NumHAcceptors(checked)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(checked)),
        "atom_count": int(checked.GetNumAtoms()),
        "bond_count": int(checked.GetNumBonds()),
    }


def stereochemistry_summary(mol: Chem.Mol) -> dict[str, object]:
    """Summarize stereochemical definition without exposing SMILES to normal users."""
    checked = validate_molecule(mol)
    centers = Chem.FindMolChiralCenters(checked, includeUnassigned=True, includeCIP=True)
    defined_centers = sum(1 for _, label in centers if label not in {"?", ""})
    unspecified_centers = sum(1 for _, label in centers if label in {"?", ""})
    stereo_bonds = sum(
        1
        for bond in checked.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    return {
        "defined_centers": defined_centers,
        "unspecified_centers": unspecified_centers,
        "defined_double_bonds": stereo_bonds,
        "has_defined_stereochemistry": bool(defined_centers or stereo_bonds),
    }


def descriptor_label(key: str) -> str:
    """Return a professor-facing label for an internal descriptor key."""
    return DESCRIPTOR_META.get(key, {"label": key.replace("_", " ").title()})["label"]


def descriptor_value_text(key: str, value: Any) -> str:
    """Format a descriptor without exposing implementation-oriented key names."""
    meta = DESCRIPTOR_META.get(key, {"unit": ""})
    if key == "logp":
        text = f"{float(value):.2f}"
    elif isinstance(value, float):
        text = f"{value:.2f}"
    else:
        text = str(value)
    return f"{text} {meta.get('unit', '')}".strip()
