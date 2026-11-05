# CHEM-WHATIF: A Computational Laboratory for Exploring Molecular Structure–Property Relationships

CHEM-WHATIF is a polished, local-first first-year BTech computational chemistry project. It connects **Chemistry** (molecular structure and functional groups), **Information Technology** (Python, APIs, modular software, transparent screening), and **Society** (careful communication of what computational descriptors can and cannot claim).

It is an **RDKit-based molecular structure–property exploration and optimization prototype**. It is not a clinical prediction system, a quantum-chemistry package, molecular dynamics workflow, docking platform, synthesis planner, or machine-learning drug-discovery system.

## Problem statement

Students can learn individual molecular descriptors without seeing how a real structural change affects several descriptors at once. CHEM-WHATIF starts with a molecule name, resolves its chemical structure, calculates a consistent descriptor panel, generates explainable structure-aware modifications, and transparently screens those candidates for a user-selected objective.

## Objectives

- Resolve a professor-friendly molecule name to a real chemical structure through PubChem PUG REST.
- Validate and analyze structures with RDKit rather than using prewritten molecule values.
- Visualize an RDKit 2D depiction and an interactive computational 3D conformer.
- Detect functional groups with transparent SMARTS patterns.
- Generate valid, explainable What-If transformations at actual sites in each molecule.
- Compare original and modified molecules using calculated descriptors.
- Optimize a selected descriptor while retaining an audit trail of accepted and rejected candidates.
- Explain results scientifically without overclaiming experimental, safety, efficacy, or clinical meaning.

## Project structure

```text
Chem-WhatIf/
├── app/
│   └── main.py                 # Streamlit laboratory dashboard
├── src/
│   ├── chemistry.py            # PubChem resolution, validation, RDKit descriptors
│   ├── functional_groups.py    # SMARTS group detection
│   ├── modifications.py        # Atom/bond-level What-If transformations
│   ├── visualization.py        # RDKit 2D, ETKDG/MMFF/UFF 3D, py3Dmol controls
│   ├── candidate_quality.py    # Transparent structural sanity checks
│   ├── optimization.py         # Auditable screening, constraints, rankings
│   ├── reasoning.py            # Descriptor-grounded explanations
├── tests/                      # Lightweight automated chemistry and pipeline tests
├── data/                       # Reserved for optional local teaching datasets
├── results/                    # Reserved for user-exported results
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

## Technology stack

| Component | Role |
|---|---|
| Python | Application and scientific workflow language |
| Streamlit | Professor-friendly local web interface |
| RDKit | Molecule parsing, sanitization, descriptors, SMARTS, drawings, conformers |
| PubChem PUG REST | Human-readable name to structure resolution |
| pandas | Clear comparison and audit tables |
| py3Dmol | Interactive in-browser molecular 3D viewer |

## Molecular descriptors

For every analyzed structure, CHEM-WHATIF calculates the following directly with RDKit:

- molecular formula and molecular weight;
- Wildman–Crippen calculated LogP;
- topological polar surface area (TPSA);
- hydrogen-bond donors and acceptors;
- rotatable bonds;
- atom count and bond count.

These are **computationally calculated molecular descriptors, not experimental measurements**. Changes in calculated LogP or TPSA should not be interpreted as guarantees about solubility, permeability, toxicity, biological activity, efficacy, or safety.

## Functional-group detection

The SMARTS rule layer detects alcohol, phenol, carboxylic acid, ester, amide, carbonyl, ketone, aldehyde, amine, ether, alkene, alkyne, halogen, aromatic rings, heteroaromatic rings, and nitro groups. Results are shown in plain language, with occurrence counts and short explanations.

## What-If transformation engine

Transformations are made with RDKit atoms and bonds—not by random string editing—and products are sanitized and canonical-SMILES deduplicated. Supported transformations include, where valid sites exist:

- alcohol/phenol to methyl ether; phenol to fluorine;
- carboxylic acid to methyl ester; amide to N-methyl amide;
- aromatic C–H to fluorine, methyl, hydroxyl, methoxy, or amino;
- aryl chloride/bromide to fluorine; aryl fluorine to hydrogen;
- alkene to alkane; alkyne to alkene;
- ketone/aldehyde to alcohol; nitro to amine.

Each generated candidate carries its transformation name, explanation, chemistry rationale, molecular object, and newly calculated descriptors. The products are computational candidate structures; they are **not experimentally synthesized compounds**.

### Traceable structural edits

CHEM-WHATIF preserves internal RDKit atom provenance during each edit. The comparison view highlights the edited site in the starting and modified 2D structure, lists functional groups added/lost/retained, and includes signed calculated descriptor-change bars. Atom maps are stripped before chemical identity, deduplication, and display labels, so they never alter the candidate chemistry.

## 3D visualization

RDKit generates a deterministic ETKDG conformer (fixed random seed), then attempts MMFF94s optimization with UFF fallback. The py3Dmol panel supports mouse/trackpad rotation and contains working **Zoom Out**, **Reset**, and **Zoom In** controls inside the 3D viewer. The conformer is an accessible computational model, not a unique experimentally measured geometry.

If the first ETKDG embedding fails, CHEM-WHATIF attempts a second deterministic random-coordinate embedding and explicitly reports that fallback. A failed 3D calculation remains a friendly, non-fatal interface message.

## Optimization methodology

The Molecular Optimization Laboratory runs this explicit pipeline:

```text
Starting Molecule
  → Candidate Generation
  → Structural Quality Screening
  → Constraint Screening
  → Property Calculation
  → Objective Evaluation
  → Structural Family Grouping
  → Improving / Feasible but Not Preferred / Rejected
```

Choose Molecular Weight, calculated LogP, TPSA, H-bond donors, H-bond acceptors, or rotatable bonds, then choose increase or decrease. Optional numerical constraints support `<`, `<=`, `>`, `>=`, and `==`. Starting functional groups can also be required to remain present; for example, a methyl ester candidate fails a carboxylic-acid retention requirement.

The quality screen is deliberately simple and explainable: RDKit validity, supported element scope, reasonable 12–700 g/mol molecular weight, 1–80 heavy atoms, and structural difference from the starting molecule. It is a computational sanity screen only—it does not prove synthesizability, stability, safety, efficacy, or clinical usefulness.

All evaluated candidates remain visible. The app displays a 2D structure and a concrete explanation for every rejected candidate, rather than silently removing it.

### Positional-isomer integrity

Aromatic substitutions at different ring positions can create positional isomers. If the selected calculated objective is identical for them, CHEM-WHATIF groups them as a **Positional isomers** structural family. It shows every distinct 2D structure and labels their objective as tied. The software ranks families, not individual tied isomers, so it never fabricates descriptor differences.

## Installation

CHEM-WHATIF is designed for a normal modern laptop; no GPU, database server, Docker installation, or external AI model is required.

```bash
cd Chem-WhatIf
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your platform does not offer an RDKit wheel through pip, use a Conda environment and install `rdkit` from conda-forge, then install the remaining requirements with pip:

```bash
conda create -n chem-whatif -c conda-forge python=3.11 rdkit
conda activate chem-whatif
pip install streamlit pandas py3Dmol pytest
```

## Run the laboratory

```bash
streamlit run app/main.py
```

The browser opens locally. Enter a molecule name such as **Aspirin**, **Caffeine**, **Paracetamol**, **Ibuprofen**, **Ethanol**, **Acetone**, **Phenol**, or **Benzaldehyde**. Name resolution uses PubChem, so it needs internet access for a previously unseen name. The optional technical SMILES input is hidden under *Technical input* and is not part of the normal professor-facing workflow.

The visible Light mode / Dark mode control persists in the current Streamlit session and is implemented with a full interface palette, including readable cards, controls, tables, structure containers, and 3D backgrounds.

Use the sidebar or bottom **Next / Back** buttons to move between three pages: **Molecule Analysis**, **What-If Modifications**, and **Optimization**. Analysis includes the original 2D and 3D structures; What-If includes the comparison and modified 3D structure. The starting molecule, generated modifications, and optimization results are shared across pages. Optimization can prepare candidates automatically. A visible **Atom colour key** above each 3D viewer shows the elements present, with a colour swatch and written colour name, including explicit hydrogens. The key and atom rendering use the same palette.

## Deploy with Streamlit Community Cloud

Publish the contents of this folder at the root of a GitHub repository, keeping
`requirements.txt`, `app/`, `src/`, and `.streamlit/` in their current locations.
Do not upload the local `.venv/`, caches, or secrets. The included `.gitignore`
excludes them. Dependency versions are pinned to the locally tested environment.

In [Streamlit Community Cloud](https://share.streamlit.io/), create an app using:

- Repository: the GitHub repository containing this project.
- Branch: `main`.
- Entrypoint: `app/main.py`.
- Python version in Advanced settings: `3.12`.

No API keys are required. Name lookup needs access to PubChem, and the interactive
3D viewer loads the 3Dmol.js library in the visitor's browser. After deployment,
verify name lookup, the original and modified structures, the atom colour keys,
Next/Back navigation, optimization, and both themes. Session results are temporary;
restarting the app or opening a new session requires analyzing a molecule again.

## Run tests

```bash
pytest -q
```

The test suite covers parsing/validation, isomeric-SMILES preservation, PubChem network failure, descriptor calculation, group detection, 2D highlights, 3D embedding fallback and viewer controls, every supported transformation rule, structural candidate generation and deduplication, quality screening, numerical constraints, functional-group retention, objective scoring, positional-isomer grouping, and explicit rejection reporting.

## Suggested project walkthrough

1. Analyze Aspirin and point out its formula, calculated LogP/TPSA, carboxylic acid, and ester.
2. Rotate its 3D computational conformer and use the actual zoom controls.
3. Generate What-If candidates; compare a methyl ester with the original and read the descriptor-grounded interpretation.
4. In the optimization laboratory, decrease calculated LogP while requiring carboxylic acid retention and, optionally, molecular weight `< 500`.
5. Inspect the audit trail, improving candidates, feasible alternatives, and rejected candidates with their specific screening explanations.
6. Analyze phenol and optimize a descriptor to demonstrate tied positional-isomer families rather than artificial rank differences.
7. Switch Light/Dark mode without restarting and note that the molecular drawings stay on a neutral white canvas for readability.

## Limitations and scientific assumptions

- PubChem name matching may fail for ambiguous, uncommon, or disconnected names; a friendly error is shown instead of a stack trace.
- Isomeric SMILES are preferred, but an upstream source can still supply a structure with unspecified stereochemistry; the application highlights this rather than guessing a stereoisomer.
- The scope focuses on common small organic molecules and a deliberately limited set of reliable transformations.
- A single force-field-relaxed conformer is shown; conformer ensembles and experimental geometries are outside scope.
- SMARTS patterns are educational structural rules, not an exhaustive chemical ontology.
- Descriptor results and candidate rankings are calculated estimates based on molecular structure. They do not predict experimental behavior, reaction conditions, synthesis success, safety, toxicology, pharmacology, or clinical outcomes.

## Responsible interpretation

The project is designed to teach the connection between molecular structure and calculated properties—not to make claims about what a compound will do in people or in experiments. Its strongest conclusion is always phrased at the level the software actually calculates: for example, “calculated LogP decreased” or “TPSA increased.”

## References

- [RDKit documentation](https://www.rdkit.org/docs/)
- [PubChem PUG REST documentation](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
- Wildman, S. A.; Crippen, G. M. *J. Chem. Inf. Comput. Sci.* **1999**, 39, 868–873. (Wildman–Crippen LogP)
- Ertl, P.; Rohde, B.; Selzer, P. *J. Med. Chem.* **2000**, 43, 3714–3717. (TPSA)
