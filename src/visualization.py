"""Reliable 2D and interactive 3D molecular visualization helpers."""
from __future__ import annotations

from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from .chemistry import MoleculeError, validate_molecule

# One explicit palette drives both the rendered atoms and their legend.
ELEMENT_COLORS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094",
    "S": "#FFFF30", "P": "#FF8000", "B": "#FFB5B5", "Si": "#F0C8A0",
}
ELEMENT_COLOR_NAMES = {
    "H": "White", "C": "Grey", "N": "Blue", "O": "Red", "F": "Light green",
    "Cl": "Green", "Br": "Dark red", "I": "Purple", "S": "Yellow",
    "P": "Orange", "B": "Pink", "Si": "Beige",
}


def atom_colour_key(mol: Chem.Mol) -> tuple[tuple[str, str, str, str], ...]:
    """Return element, name, colour and colour name for the displayed atoms.

    The conformer has explicit hydrogens, so include them in this key even
    when the input molecule uses implicit hydrogens.
    """
    explicit = Chem.AddHs(validate_molecule(mol))
    table = Chem.GetPeriodicTable()
    elements = sorted({atom.GetSymbol() for atom in explicit.GetAtoms()}, key=table.GetAtomicNumber)
    return tuple(
        (symbol, table.GetElementName(table.GetAtomicNumber(symbol)),
         ELEMENT_COLORS.get(symbol, "#DA70D6"), ELEMENT_COLOR_NAMES.get(symbol, "Orchid"))
        for symbol in elements
    )


@dataclass(frozen=True)
class ConformerResult:
    molblock: str
    force_field: str
    optimization_status: str


def molecule_png(
    mol: Chem.Mol,
    width: int = 620,
    height: int = 420,
    highlight_atoms: tuple[int, ...] = (),
) -> bytes:
    """Generate a centered 2D image, optionally highlighting an edited site."""
    try:
        copy = Chem.Mol(validate_molecule(mol))
        # Candidate construction uses atom-map numbers internally to track the
        # edited site. They are provenance metadata, not chemical labels, so
        # remove them from the temporary drawing molecule. Atom indices remain
        # unchanged and the requested highlight atoms still point to the right
        # place in the structure.
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
        rdDepictor.Compute2DCoords(copy, canonOrient=True)
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        options = drawer.drawOptions()
        options.padding = 0.08
        options.bondLineWidth = 2.2
        options.clearBackground = True
        drawer.SetColour((0.08, 0.12, 0.18))
        valid_highlights = tuple(index for index in highlight_atoms if 0 <= index < copy.GetNumAtoms())
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            copy,
            highlightAtoms=valid_highlights,
            highlightAtomColors={index: (0.12, 0.62, 0.56) for index in valid_highlights},
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        raise MoleculeError("A 2D structure image could not be generated for this molecule.") from exc


def generate_conformer(mol: Chem.Mol, seed: int = 0xC0FFEE) -> ConformerResult:
    """Create a deterministic ETKDG conformer and optimize it with MMFF/UFF.

    This is a convenient computational conformer—not a unique experimentally
    measured molecular geometry.
    """
    try:
        working = Chem.AddHs(validate_molecule(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useSmallRingTorsions = True
        embedded = AllChem.EmbedMolecule(working, params)
        embedding_note = ""
        if embedded != 0:
            # A second deterministic attempt with random initial coordinates
            # salvages some constrained or unusual small molecules while still
            # making the fallback visible to the user.
            fallback_params = AllChem.ETKDGv3()
            fallback_params.randomSeed = seed
            fallback_params.useRandomCoords = True
            embedded = AllChem.EmbedMolecule(working, fallback_params)
            embedding_note = " using random-coordinate embedding fallback"
        if embedded != 0:
            raise ValueError("RDKit embedding returned a non-zero status")
        properties = AllChem.MMFFGetMoleculeProperties(working, mmffVariant="MMFF94s")
        if properties is not None:
            status = AllChem.MMFFOptimizeMolecule(working, mmffVariant="MMFF94s", maxIters=500)
            force_field = "MMFF94s"
        else:
            status = AllChem.UFFOptimizeMolecule(working, maxIters=500)
            force_field = "UFF fallback"
        status_text = ("Converged" if status == 0 else "Generated; optimization reached the iteration limit") + embedding_note
        return ConformerResult(Chem.MolToMolBlock(working), force_field, status_text)
    except Exception as exc:
        raise MoleculeError("A computational 3D conformer could not be generated for this molecule.") from exc


def interactive_3d_html(conformer: ConformerResult, theme: str = "light", height: int = 430) -> str:
    """Return a py3Dmol viewer with working in-view zoom/reset controls."""
    try:
        import py3Dmol

        dark = theme == "dark"
        background = "#101827" if dark else "#f7fafc"
        viewer = py3Dmol.view(width="100%", height=height)
        viewer.addModel(conformer.molblock, "mol")
        structure = Chem.MolFromMolBlock(conformer.molblock, removeHs=False)
        if structure is None:
            raise ValueError("The conformer could not be read for element colouring.")
        elements = sorted({atom.GetSymbol() for atom in structure.GetAtoms()}, key=lambda symbol: Chem.GetPeriodicTable().GetAtomicNumber(symbol))
        for symbol in elements:
            color = ELEMENT_COLORS.get(symbol, "#DA70D6")
            viewer.setStyle({"elem": symbol}, {
                "stick": {"radius": 0.17, "color": color},
                "sphere": {"scale": 0.28, "color": color},
            })
        viewer.setBackgroundColor(background)
        viewer.zoomTo()
        viewer.render()
        rendered = viewer._make_html()
        viewer_variable = f"viewer_{viewer.uniqueid}"
        text_color = "#eaf1ff" if dark else "#162235"
        border = "#43526c" if dark else "#cbd5e1"
        button_bg = "#1f304a" if dark else "#ffffff"
        # These buttons live in the same iframe/document as py3Dmol. The calls
        # operate directly on its viewer object, so they are genuine 3D controls.
        controls = f"""
        <div style="display:flex;gap:8px;justify-content:center;margin:8px 0 0;font-family:Arial,sans-serif;">
          <button data-zoom="out" onclick="{viewer_variable}.zoom(0.8);{viewer_variable}.render();" style="cursor:pointer;border:1px solid {border};background:{button_bg};color:{text_color};border-radius:7px;padding:7px 13px;">Zoom Out</button>
          <button data-zoom="reset" onclick="{viewer_variable}.zoomTo();{viewer_variable}.render();" style="cursor:pointer;border:1px solid {border};background:{button_bg};color:{text_color};border-radius:7px;padding:7px 13px;">Reset</button>
          <button data-zoom="in" onclick="{viewer_variable}.zoom(1.25);{viewer_variable}.render();" style="cursor:pointer;border:1px solid {border};background:{button_bg};color:{text_color};border-radius:7px;padding:7px 13px;">Zoom In</button>
        </div>
        """
        return f"<div style='background:{background};padding:2px;border-radius:10px'>{rendered}{controls}</div>"
    except Exception as exc:
        raise MoleculeError("The interactive 3D viewer could not be prepared in this environment.") from exc
