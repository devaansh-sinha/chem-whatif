"""Structure-aware, explainable What-If transformations.

The transformations deliberately operate on RDKit atoms and bonds rather than
editing SMILES text. Each product is sanitized, descriptor-calculated, and
deduplicated by canonical SMILES before it reaches the interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Callable, Iterable

from rdkit import Chem

from .chemistry import calculate_descriptors, canonical_smiles, validate_molecule
from .functional_groups import find_matches


@dataclass(frozen=True)
class Candidate:
    """One valid molecular modification that can be compared or optimized."""

    identifier: str
    mol: Chem.Mol
    canonical_smiles: str
    transformation: str
    description: str
    reason: str
    structural_family: str
    site_atoms: tuple[int, ...]
    starting_highlight_atoms: tuple[int, ...]
    modified_highlight_atoms: tuple[int, ...]
    descriptors: dict[str, object]


@dataclass(frozen=True)
class CandidateGeneration:
    """A transparent record of structure-aware generation and deduplication."""

    candidates: tuple[Candidate, ...]
    generated_count: int
    unique_count: int
    invalid_or_unchanged_count: int


_SMARTS_CACHE: dict[str, Chem.Mol] = {}


def _smarts(query: str) -> Chem.Mol:
    if query not in _SMARTS_CACHE:
        pattern = Chem.MolFromSmarts(query)
        if pattern is None:
            raise RuntimeError(f"Invalid transformation SMARTS: {query}")
        _SMARTS_CACHE[query] = pattern
    return _SMARTS_CACHE[query]


def _add_atom(rw: Chem.RWMol, atomic_number: int, aromatic: bool = False) -> int:
    atom = Chem.Atom(atomic_number)
    atom.SetIsAromatic(aromatic)
    return rw.AddAtom(atom)


def _make_candidate(
    starting_mol: Chem.Mol,
    transformed: Chem.Mol | None,
    transformation: str,
    description: str,
    reason: str,
    structural_family: str,
    site_atoms: tuple[int, ...],
) -> Candidate | None:
    """Sanitize one product and turn it into a UI-ready candidate, if distinct."""
    if transformed is None:
        return None
    try:
        mol = validate_molecule(transformed)
        canonical = canonical_smiles(mol)
        if canonical == canonical_smiles(starting_mol):
            return None
        site_map_numbers = {starting_mol.GetAtomWithIdx(index).GetAtomMapNum() for index in site_atoms}
        # Original atoms keep a map number during RWMol edits. Newly added
        # atoms are unmapped, allowing the two structure panels to show the
        # actual location and material change behind this candidate.
        modified_highlights = tuple(
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomMapNum() in site_map_numbers or atom.GetAtomMapNum() == 0
        )
        return Candidate(
            identifier=sha1(canonical.encode("utf-8")).hexdigest()[:12],
            mol=mol,
            canonical_smiles=canonical,
            transformation=transformation,
            description=description,
            reason=reason,
            structural_family=structural_family,
            site_atoms=site_atoms,
            starting_highlight_atoms=site_atoms,
            modified_highlight_atoms=modified_highlights,
            descriptors=calculate_descriptors(mol),
        )
    except Exception:
        # A transformation product that cannot be sanitized is not a valid
        # computational candidate. Its failed construction is counted in the audit.
        return None


def _change_bond(mol: Chem.Mol, atom_a: int, atom_b: int, bond_type: Chem.BondType) -> Chem.Mol | None:
    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(atom_a, atom_b) is None:
        return None
    rw.RemoveBond(atom_a, atom_b)
    rw.AddBond(atom_a, atom_b, bond_type)
    return rw.GetMol()


def _add_substituent(mol: Chem.Mol, anchor: int, atomic_number: int) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    added = _add_atom(rw, atomic_number)
    rw.AddBond(anchor, added, Chem.BondType.SINGLE)
    return rw.GetMol()


def _add_methoxy(mol: Chem.Mol, anchor: int) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    oxygen = _add_atom(rw, 8)
    carbon = _add_atom(rw, 6)
    rw.AddBond(anchor, oxygen, Chem.BondType.SINGLE)
    rw.AddBond(oxygen, carbon, Chem.BondType.SINGLE)
    return rw.GetMol()


def _replace_neighbor_with(mol: Chem.Mol, anchor: int, leaving: int, atomic_number: int) -> Chem.Mol | None:
    if anchor == leaving:
        return None
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(leaving)
    # The anchor is always lower-indexed for every rule that calls this helper;
    # keep this guard for future transformation additions.
    adjusted_anchor = anchor - 1 if leaving < anchor else anchor
    added = _add_atom(rw, atomic_number)
    rw.AddBond(adjusted_anchor, added, Chem.BondType.SINGLE)
    return rw.GetMol()


def _remove_atom(mol: Chem.Mol, atom_index: int) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(atom_index)
    return rw.GetMol()


def _reduce_nitro_to_amine(mol: Chem.Mol, nitrogen: int, oxygens: tuple[int, int]) -> Chem.Mol | None:
    rw = Chem.RWMol(mol)
    # Remove in descending index order. Track the nitrogen after earlier removals.
    current_nitrogen = nitrogen
    for oxygen in sorted(oxygens, reverse=True):
        rw.RemoveAtom(oxygen)
        if oxygen < current_nitrogen:
            current_nitrogen -= 1
    atom = rw.GetAtomWithIdx(current_nitrogen)
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(0)
    return rw.GetMol()


def generate_candidates(mol: Chem.Mol) -> CandidateGeneration:
    """Generate all supported valid transformations at actual sites in *mol*.

    This function intentionally does not select a "best" molecule. It creates
    the full, deduplicated candidate set used both by What-If comparison and by
    the optimization laboratory.
    """
    starting = validate_molecule(mol)
    # Atom maps are internal provenance labels only. canonical_smiles() strips
    # them before identity comparisons and before anything is shown to users.
    for index, atom in enumerate(starting.GetAtoms(), start=1):
        atom.SetAtomMapNum(index)
    raw: list[Candidate] = []
    failed = 0

    def add(
        product: Chem.Mol | None,
        transformation: str,
        description: str,
        reason: str,
        family: str,
        site: tuple[int, ...],
    ) -> None:
        nonlocal failed
        candidate = _make_candidate(starting, product, transformation, description, reason, family, site)
        if candidate is None:
            failed += 1
        else:
            raw.append(candidate)

    # Alcohol / phenol O-methylation. O has an implicit H in the starting molecule;
    # adding a carbon replaces that hydrogen after RDKit sanitization.
    for match in find_matches(starting, "alcohol"):
        oxygen = match[1]
        add(
            _add_substituent(starting, oxygen, 6),
            "Alcohol → methyl ether",
            "The alcohol oxygen is converted to a methyl ether.",
            "Masking a hydroxyl hydrogen changes the calculated hydrogen-bonding and polarity profile.",
            "O-methylation",
            match,
        )
    for match in find_matches(starting, "phenol"):
        ring_carbon, oxygen = match
        add(
            _add_substituent(starting, oxygen, 6),
            "Phenol → methyl ether",
            "The phenolic oxygen is converted to a methoxy group.",
            "This retains an oxygen substituent while removing a hydrogen-bond donor.",
            "O-methylation",
            match,
        )
        add(
            _replace_neighbor_with(starting, ring_carbon, oxygen, 9),
            "Phenol → fluorine",
            "The phenolic hydroxyl substituent is replaced by fluorine.",
            "This tests removal of a polar hydroxyl group while keeping a small ring substituent.",
            "Aromatic substituent replacement",
            match,
        )

    # Carboxylic acid esterification.
    for match in find_matches(starting, "carboxylic_acid"):
        acid_oxygen = match[2]
        add(
            _add_substituent(starting, acid_oxygen, 6),
            "Carboxylic acid → methyl ester",
            "The acidic hydroxyl oxygen is converted to a methyl ester.",
            "Esterification removes an acidic donor and changes the calculated polarity profile.",
            "Carboxylic-acid derivatization",
            match,
        )

    # N-methylation is limited to amide nitrogens that still carry an H.
    for match in starting.GetSubstructMatches(_smarts("[CX3](=[OX1])[NX3;H1,H2]"), uniquify=True):
        nitrogen = match[2]
        add(
            _add_substituent(starting, nitrogen, 6),
            "Amide → N-methyl amide",
            "A hydrogen on the amide nitrogen is replaced by a methyl group.",
            "N-methylation tests a modest size increase and altered donor capacity.",
            "Amide derivatization",
            match,
        )

    # Aromatic C–H substitutions create genuine positional-isomer candidates.
    for (ring_carbon,) in starting.GetSubstructMatches(_smarts("[cH]"), uniquify=True):
        for atomic_number, name, description, reason in (
            (9, "Aromatic C–H → fluorine", "An aromatic hydrogen is replaced by fluorine.", "Fluorination changes ring substitution with a small electronegative atom."),
            (6, "Aromatic C–H → methyl", "An aromatic hydrogen is replaced by a methyl group.", "Methyl substitution tests a modest hydrophobic and size change."),
            (8, "Aromatic C–H → hydroxyl", "An aromatic hydrogen is replaced by a hydroxyl group.", "A new hydroxyl group can increase calculated polarity and donor capacity."),
            (7, "Aromatic C–H → amino", "An aromatic hydrogen is replaced by an amino group.", "An amino substituent adds a basic, hydrogen-bonding functionality in this structural model."),
        ):
            add(
                _add_substituent(starting, ring_carbon, atomic_number),
                name,
                description,
                reason,
                "Aromatic substitution",
                (ring_carbon,),
            )
        add(
            _add_methoxy(starting, ring_carbon),
            "Aromatic C–H → methoxy",
            "An aromatic hydrogen is replaced by a methoxy group.",
            "Methoxy substitution changes size and adds an ether oxygen without a donor hydrogen.",
            "Aromatic substitution",
            (ring_carbon,),
        )

    # Aryl-halogen interconversion.
    for ring_carbon, halogen in starting.GetSubstructMatches(_smarts("[c][Cl,Br]"), uniquify=True):
        add(
            _replace_neighbor_with(starting, ring_carbon, halogen, 9),
            "Aryl chloride/bromide → fluorine",
            "An aromatic chlorine or bromine is replaced by fluorine.",
            "This tests a smaller halogen substituent while retaining an aryl halide.",
            "Aryl-halogen replacement",
            (ring_carbon, halogen),
        )
    for ring_carbon, fluorine in starting.GetSubstructMatches(_smarts("[c][F]"), uniquify=True):
        add(
            _remove_atom(starting, fluorine),
            "Aryl fluorine → hydrogen",
            "An aromatic fluorine is removed, restoring an aromatic C–H position.",
            "This tests the calculated contribution of the fluorine substituent.",
            "Aryl-halogen replacement",
            (ring_carbon, fluorine),
        )

    for carbon_a, carbon_b in starting.GetSubstructMatches(_smarts("[CX3]=[CX3]"), uniquify=True):
        add(
            _change_bond(starting, carbon_a, carbon_b, Chem.BondType.SINGLE),
            "Alkene → alkane",
            "A carbon–carbon double bond is reduced to a single bond.",
            "Changing unsaturation alters flexibility and the calculated structural descriptor profile.",
            "Unsaturation reduction",
            (carbon_a, carbon_b),
        )
    for carbon_a, carbon_b in starting.GetSubstructMatches(_smarts("[CX2]#[CX2]"), uniquify=True):
        add(
            _change_bond(starting, carbon_a, carbon_b, Chem.BondType.DOUBLE),
            "Alkyne → alkene",
            "A carbon–carbon triple bond is reduced to a double bond.",
            "Partial reduction changes the unsaturation class while retaining a carbon–carbon multiple bond.",
            "Unsaturation reduction",
            (carbon_a, carbon_b),
        )

    for match in starting.GetSubstructMatches(_smarts("[#6][CX3](=[OX1])[#6]"), uniquify=True):
        carbonyl_carbon, oxygen = match[1], match[2]
        add(
            _change_bond(starting, carbonyl_carbon, oxygen, Chem.BondType.SINGLE),
            "Ketone → alcohol",
            "The ketone carbonyl is reduced to an alcohol-like C–O bond.",
            "Carbonyl reduction adds donor capacity in the neutral structural model.",
            "Carbonyl reduction",
            match,
        )
    for match in starting.GetSubstructMatches(_smarts("[CX3H1](=[OX1])[#6]"), uniquify=True):
        carbonyl_carbon, oxygen = match[0], match[1]
        add(
            _change_bond(starting, carbonyl_carbon, oxygen, Chem.BondType.SINGLE),
            "Aldehyde → alcohol",
            "The aldehyde carbonyl is reduced to a primary alcohol-like C–O bond.",
            "Carbonyl reduction introduces a hydroxyl group in the neutral structural model.",
            "Carbonyl reduction",
            match,
        )
    for nitrogen, oxygen_a, oxygen_b in starting.GetSubstructMatches(_smarts("[N+](=[OX1])[O-]"), uniquify=True):
        add(
            _reduce_nitro_to_amine(starting, nitrogen, (oxygen_a, oxygen_b)),
            "Nitro → amine",
            "The nitro group is converted to an amine-like nitrogen.",
            "This changes an oxygen-rich substituent into a hydrogen-bonding nitrogen functionality.",
            "Nitro reduction",
            (nitrogen, oxygen_a, oxygen_b),
        )

    unique: dict[str, Candidate] = {}
    for candidate in raw:
        unique.setdefault(candidate.canonical_smiles, candidate)
    candidates = tuple(sorted(unique.values(), key=lambda item: (item.transformation, item.canonical_smiles)))
    return CandidateGeneration(
        candidates=candidates,
        generated_count=len(raw),
        unique_count=len(candidates),
        invalid_or_unchanged_count=failed,
    )
