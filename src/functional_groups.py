"""Transparent RDKit SMARTS functional-group detection."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem


@dataclass(frozen=True)
class FunctionalGroup:
    """A human-readable functional-group result."""

    key: str
    label: str
    count: int
    explanation: str


@dataclass(frozen=True)
class GroupRule:
    key: str
    label: str
    smarts: str
    explanation: str


RULES = (
    GroupRule("alcohol", "Alcohol", "[CX4;!$(C=O)][OX2H1]", "A hydroxyl group attached to a non-aromatic carbon."),
    GroupRule("phenol", "Phenol", "[c][OX2H1]", "A hydroxyl group attached directly to an aromatic ring."),
    GroupRule("carboxylic_acid", "Carboxylic acid", "[CX3](=[OX1])[OX2H1]", "A carbonyl and hydroxyl group on the same carbon."),
    GroupRule("ester", "Ester", "[CX3](=[OX1])[OX2][#6]", "A carbonyl bonded to an alkoxy oxygen."),
    GroupRule("amide", "Amide", "[CX3](=[OX1])[NX3]", "A carbonyl directly bonded to nitrogen."),
    GroupRule("carbonyl", "Carbonyl", "[$([CX3]=[OX1]),$([c]=[OX1])]", "A carbon–oxygen double bond."),
    GroupRule("ketone", "Ketone", "[#6][CX3](=[OX1])[#6]", "A carbonyl carbon bonded to two carbon atoms."),
    GroupRule("aldehyde", "Aldehyde", "[CX3H1](=[OX1])[#6]", "A carbonyl group with a hydrogen on the carbonyl carbon."),
    GroupRule("amine", "Amine", "[NX3;!$(N[C,S]=O);!$(N=*)]", "A neutral amine-like nitrogen not directly in an amide."),
    GroupRule("ether", "Ether", "[OD2;!$([O]-[CX3](=O))]([#6])[#6]", "An oxygen atom joining two carbon-containing groups."),
    GroupRule("alkene", "Alkene", "[CX3]=[CX3]", "A carbon–carbon double bond."),
    GroupRule("alkyne", "Alkyne", "[CX2]#[CX2]", "A carbon–carbon triple bond."),
    GroupRule("halogen", "Halogen", "[F,Cl,Br,I]", "A fluorine, chlorine, bromine, or iodine substituent."),
    GroupRule("nitro", "Nitro", "[N+](=[OX1])[O-]", "A nitro nitrogen bearing two oxygens."),
)


@lru_cache(maxsize=64)
def _pattern(smarts: str) -> Chem.Mol:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:  # Fail early during development if a rule is malformed.
        raise RuntimeError(f"Invalid SMARTS rule: {smarts}")
    return pattern


def find_matches(mol: Chem.Mol, key: str) -> tuple[tuple[int, ...], ...]:
    """Return atom matches for a named rule; this is used by transformations too."""
    rule = next((rule for rule in RULES if rule.key == key), None)
    if rule is None:
        return ()
    return mol.GetSubstructMatches(_pattern(rule.smarts), uniquify=True)


def _aromatic_ring_count(mol: Chem.Mol) -> int:
    return sum(1 for ring in Chem.GetSymmSSSR(mol) if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))


def _heteroaromatic_ring_count(mol: Chem.Mol) -> int:
    return sum(
        1
        for ring in Chem.GetSymmSSSR(mol)
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        and any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (6, 1) for i in ring)
    )


def detect_functional_groups(mol: Chem.Mol) -> list[FunctionalGroup]:
    """Identify supported functional groups and ring classes from molecular structure."""
    results: list[FunctionalGroup] = []
    for rule in RULES:
        count = len(find_matches(mol, rule.key))
        if count:
            results.append(FunctionalGroup(rule.key, rule.label, count, rule.explanation))
    aromatic = _aromatic_ring_count(mol)
    if aromatic:
        results.append(FunctionalGroup("aromatic_ring", "Aromatic ring", aromatic, "A conjugated aromatic ring system."))
    heteroaromatic = _heteroaromatic_ring_count(mol)
    if heteroaromatic:
        results.append(
            FunctionalGroup("heteroaromatic_ring", "Heteroaromatic ring", heteroaromatic, "An aromatic ring containing a heteroatom."))
    return results


def group_count(mol: Chem.Mol, key: str) -> int:
    """Count a named group, including the two ring-derived groups."""
    return next((item.count for item in detect_functional_groups(mol) if item.key == key), 0)


def available_group_labels(mol: Chem.Mol) -> list[tuple[str, str]]:
    """Return stable (key, label) choices that genuinely occur in a molecule."""
    return [(group.key, group.label) for group in detect_functional_groups(mol)]
