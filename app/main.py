"""CHEM-WHATIF Streamlit application.

Run with: streamlit run app/main.py
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chemistry import (  # noqa: E402
    MoleculeError,
    MoleculeRecord,
    calculate_descriptors,
    canonical_smiles,
    descriptor_label,
    descriptor_value_text,
    parse_smiles,
    resolve_molecule,
)
from src.functional_groups import available_group_labels, detect_functional_groups  # noqa: E402
from src.modifications import Candidate, CandidateGeneration, generate_candidates  # noqa: E402
from src.optimization import (  # noqa: E402
    NumericalConstraint,
    OptimizationResult,
    ScreenedCandidate,
    optimize,
)
from src.reasoning import descriptor_change_sentences  # noqa: E402
from src.visualization import atom_colour_key, generate_conformer, interactive_3d_html, molecule_png  # noqa: E402


st.set_page_config(page_title="CHEM-WHATIF | Computational Laboratory", page_icon="⚗", layout="wide", initial_sidebar_state="expanded")

EXAMPLES = ("Aspirin", "Caffeine", "Paracetamol", "Ibuprofen", "Ethanol", "Acetone", "Phenol", "Benzaldehyde")
NUMERIC_PROPERTIES = ("molecular_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds")


def initialise_state() -> None:
    """Set only non-widget defaults before widgets are instantiated."""
    # Keep appearance separate from the radio's page-specific widget identity.
    # Restore the widget before it is instantiated on the newly selected page.
    st.session_state.setdefault("appearance", st.session_state.get("theme_toggle", "Light mode"))
    st.session_state.theme_toggle = st.session_state.appearance
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("candidate_generation", None)
    st.session_state.setdefault("optimization_result", None)
    # Streamlit cleans up widget state when its page is hidden. Re-register
    # workspace settings before rendering to retain selections across pages.
    workspace_prefixes = ("whatif_candidate_", "objective_", "constraint_", "retain_groups_")
    for key in list(st.session_state):
        if key.startswith(workspace_prefixes):
            st.session_state[key] = st.session_state[key]


def palette_for(theme: str) -> dict[str, str]:
    if theme == "Dark mode":
        return {
            "bg": "#09111e", "surface": "#101c2f", "surface2": "#16253b", "text": "#edf4ff", "muted": "#a8b7cc",
            "border": "#2b405d", "accent": "#57c6b9", "accent_ink": "#062d2a", "success": "#4ed09b", "warning": "#ffcd72", "danger": "#ff8891",
        }
    return {
        "bg": "#f7f9fc", "surface": "#ffffff", "surface2": "#eff5f9", "text": "#152336", "muted": "#526275",
        "border": "#d9e2ec", "accent": "#097d72", "accent_ink": "#ffffff", "success": "#087f5b", "warning": "#a56000", "danger": "#b42335",
    }


def apply_theme(theme: str) -> dict[str, str]:
    """Apply a complete CSS palette rather than changing just the page background."""
    p = palette_for(theme)
    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stAppViewContainer"] {{ background: {p['bg']}; color: {p['text']}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] > div:first-child {{ background: {p['surface']}; border-right: 1px solid {p['border']}; }}
        [data-testid="stSidebar"] * {{ color: {p['text']}; }}
        html, body, p, span, label, .stMarkdown {{ color: {p['text']}; }}
        h1, h2, h3 {{ color: {p['text']} !important; letter-spacing: -0.02em; }}
        .hero {{ background: linear-gradient(120deg, {p['surface']}, {p['surface2']}); border: 1px solid {p['border']};
                 padding: 26px 30px; border-radius: 18px; margin: 0 0 20px 0; box-shadow: 0 10px 32px rgba(1, 18, 37, .08); }}
        .hero-kicker {{ color: {p['accent']}; text-transform: uppercase; letter-spacing: .14em; font-weight: 800; font-size: .77rem; }}
        .hero h1 {{ font-size: 2.2rem; margin: .28rem 0 .35rem 0; }}
        .hero p {{ color: {p['muted']}; max-width: 850px; margin: 0; line-height: 1.55; }}
        .metric-card {{ background:{p['surface']}; border:1px solid {p['border']}; border-radius:13px; padding:14px 16px; min-height:98px; }}
        .metric-name {{ color:{p['muted']}; font-size:.80rem; font-weight:650; }}
        .metric-value {{ color:{p['text']}; font-size:1.48rem; font-weight:760; margin-top:6px; }}
        .section-note {{ color:{p['muted']}; margin-top:-.35rem; margin-bottom:.8rem; }}
        .pill {{ display:inline-block; border:1px solid {p['border']}; background:{p['surface2']}; color:{p['text']}; border-radius:999px;
                 padding:6px 10px; margin:0 7px 7px 0; font-size:.87rem; }}
        .pill b {{ color:{p['accent']}; }}
        .lab-card {{ background:{p['surface']}; border:1px solid {p['border']}; border-radius:14px; padding:17px; margin:5px 0 12px; }}
        .status-success {{ border-left:4px solid {p['success']}; background:{p['surface2']}; padding:10px 13px; border-radius:7px; color:{p['text']}; }}
        .status-warning {{ border-left:4px solid {p['warning']}; background:{p['surface2']}; padding:10px 13px; border-radius:7px; color:{p['text']}; }}
        .status-danger {{ border-left:4px solid {p['danger']}; background:{p['surface2']}; padding:10px 13px; border-radius:7px; color:{p['text']}; }}
        .stButton > button {{ background:{p['accent']}; color:{p['accent_ink']}; border:0; border-radius:9px; font-weight:700; padding:.52rem .9rem; }}
        .stButton > button:hover {{ filter:brightness(1.07); color:{p['accent_ink']}; border:0; }}
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stTextInput input, .stNumberInput input {{ background:{p['surface']} !important; color:{p['text']} !important; border-color:{p['border']} !important; }}
        div[data-baseweb="select"] *, div[data-baseweb="input"] *, .stTextInput input {{ color:{p['text']} !important; }}
        .stDataFrame, [data-testid="stDataFrame"] {{ border:1px solid {p['border']}; border-radius:10px; overflow:hidden; }}
        [data-testid="stExpander"] {{ background:{p['surface']}; border:1px solid {p['border']}; border-radius:10px; }}
        [data-testid="stAlert"] {{ border-radius:10px; }}
        hr {{ border-color:{p['border']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return p


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_name_resolution(name: str) -> tuple[str, str, str]:
    record = resolve_molecule(name)
    return record.name, record.canonical_smiles, record.source


@st.cache_data(show_spinner=False)
def cached_2d(smiles: str) -> bytes:
    return molecule_png(parse_smiles(smiles).mol)


@st.cache_data(show_spinner=False)
def cached_3d(smiles: str) -> tuple[str, str, str]:
    result = generate_conformer(parse_smiles(smiles).mol)
    return result.molblock, result.force_field, result.optimization_status


def record_from_state() -> MoleculeRecord | None:
    state = st.session_state.analysis
    if not state:
        return None
    return parse_smiles(state["smiles"], name=state["name"], source=state["source"])


def save_analysis(record: MoleculeRecord) -> None:
    """Store a compact, safe session representation and reset dependent outputs."""
    st.session_state.analysis = {
        "name": record.name,
        "smiles": record.canonical_smiles,
        "source": record.source,
    }
    st.session_state.candidate_generation = None
    st.session_state.optimization_result = None


def render_metric_grid(descriptors: dict[str, object], keys: tuple[str, ...] = NUMERIC_PROPERTIES) -> None:
    for start in range(0, len(keys), 3):
        columns = st.columns(3)
        for column, key in zip(columns, keys[start : start + 3]):
            with column:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-name">{escape(descriptor_label(key))}</div>'
                    f'<div class="metric-value">{escape(descriptor_value_text(key, descriptors[key]))}</div></div>',
                    unsafe_allow_html=True,
                )


def display_molecule_image(mol, caption: str, highlight_atoms: tuple[int, ...] = ()) -> None:
    """Render a readable molecule without enlarging it to the page width."""
    try:
        image = molecule_png(mol, highlight_atoms=highlight_atoms) if highlight_atoms else cached_2d(canonical_smiles(mol))
        # A numeric width is capped by the parent container on narrow screens.
        # Do not stretch a 620px drawing across a full-width dashboard.
        st.image(image, caption=caption, width=480)
    except Exception:
        st.warning("The structure is valid, but a 2D image could not be generated in this browser session.")


def display_3d(mol, theme: str) -> None:
    try:
        smiles = canonical_smiles(mol)
        molblock, force_field, status = cached_3d(smiles)
        from src.visualization import ConformerResult

        # Keep the key in the main page, above the iframe, so it remains visible
        # at narrow widths and is never clipped by the embedded viewer.
        swatches = "".join(
            f'<span class="pill" style="display:inline-flex;align-items:center;gap:8px;">'
            f'<span aria-hidden="true" style="display:inline-block;width:16px;height:16px;'
            f'border-radius:50%;background:{colour};border:1px solid #64748b;"></span>'
            f'{escape(name)} ({escape(symbol)}) — {escape(colour_name)}</span>'
            for symbol, name, colour, colour_name in atom_colour_key(mol)
        )
        st.markdown(f'<div class="lab-card"><b>Atom colour key</b><div style="margin-top:10px;">{swatches}</div></div>', unsafe_allow_html=True)
        html = interactive_3d_html(ConformerResult(molblock, force_field, status), "dark" if theme == "Dark mode" else "light")
        st.iframe(html, width="stretch", height=490)
        st.caption(f"Computational conformer: {force_field}; {status}. Rotate with mouse or trackpad; use the controls inside the viewer to change zoom.")
    except Exception:
        st.info("A 3D conformer could not be generated for this molecule in the current environment.")


def comparison_table(original: dict[str, object], modified: dict[str, object]) -> pd.DataFrame:
    rows = []
    for key in NUMERIC_PROPERTIES:
        before = float(original[key])
        after = float(modified[key])
        delta = after - before
        if key in {"molecular_weight", "logp", "tpsa"}:
            delta_text = f"{delta:+.2f}"
        else:
            delta_text = f"{delta:+.0f}"
        rows.append(
            {
                "Descriptor": descriptor_label(key),
                "Original": descriptor_value_text(key, before),
                "Modified": descriptor_value_text(key, after),
                "Change": delta_text,
            }
        )
    return pd.DataFrame(rows)


def descriptor_delta_chart(original: dict[str, object], modified: dict[str, object]) -> None:
    """Show a compact visual of signed, calculated descriptor differences."""
    frame = comparison_table(original, modified)
    frame["Calculated change"] = [float(modified[key]) - float(original[key]) for key in NUMERIC_PROPERTIES]
    st.bar_chart(frame.set_index("Descriptor")["Calculated change"], color="#097d72", horizontal=True)
    st.caption("Signed calculated changes use each descriptor's own units; bar lengths are for visual orientation, not cross-unit equivalence.")


def functional_group_change_text(original_mol, modified_mol) -> tuple[list[str], list[str], list[str]]:
    """Summarize group additions/losses/retentions in professor-facing words."""
    before = {item.key: item for item in detect_functional_groups(original_mol)}
    after = {item.key: item for item in detect_functional_groups(modified_mol)}
    added: list[str] = []
    lost: list[str] = []
    retained: list[str] = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        label = (new or old).label
        old_count, new_count = old.count if old else 0, new.count if new else 0
        if new_count > old_count:
            added.append(f"{label} +{new_count - old_count}")
        elif new_count < old_count:
            lost.append(f"{label} −{old_count - new_count}")
        elif new_count:
            retained.append(f"{label} ({new_count})")
    return added, lost, retained


def display_candidate_comparison(
    original_mol,
    candidate: Candidate,
    original_descriptors: dict[str, object],
    title: str = "Molecule Comparison",
) -> None:
    st.subheader(title)
    original_column, modified_column = st.columns(2, gap="large")
    with original_column:
        display_molecule_image(original_mol, "Starting molecule — edited site highlighted", candidate.starting_highlight_atoms)
    with modified_column:
        display_molecule_image(candidate.mol, "Modified molecule — changed site highlighted", candidate.modified_highlight_atoms)
    st.markdown(f"### {candidate.transformation}")
    st.write(candidate.description)
    st.markdown(f"<div class='lab-card'><b>Why this is chemically interesting</b><br>{escape(candidate.reason)}</div>", unsafe_allow_html=True)
    st.dataframe(comparison_table(original_descriptors, candidate.descriptors), hide_index=True, width="stretch")
    chart_column, groups_column = st.columns((1.1, 1), gap="large")
    with chart_column:
        st.markdown("**Calculated descriptor changes**")
        descriptor_delta_chart(original_descriptors, candidate.descriptors)
    with groups_column:
        st.markdown("**Functional-group changes**")
        added, lost, retained = functional_group_change_text(original_mol, candidate.mol)
        st.write(f"Added: {', '.join(added) or 'None detected'}")
        st.write(f"Lost: {', '.join(lost) or 'None detected'}")
        st.write(f"Retained: {', '.join(retained) or 'None detected'}")
    st.caption("All values are computationally calculated molecular descriptors; they are not experimental measurements.")
    with st.expander("Descriptor-based interpretation", expanded=False):
        for sentence in descriptor_change_sentences(original_descriptors, candidate.descriptors):
            st.write(f"• {sentence}")


def candidate_label(candidate: Candidate, candidates: tuple[Candidate, ...]) -> str:
    """Provide a clean selector label without exposing internal identifiers."""
    matching = tuple(item for item in candidates if item.transformation == candidate.transformation)
    if len(matching) == 1:
        return candidate.transformation
    ordered = tuple(sorted(matching, key=lambda item: item.canonical_smiles))
    number = ordered.index(candidate) + 1
    suffix = f"positional isomer {number}" if candidate.structural_family == "Aromatic substitution" else f"site {number}"
    return f"{candidate.transformation} — {suffix}"


def show_screened_candidate(item: ScreenedCandidate, original: dict[str, object]) -> None:
    candidate = item.candidate
    structure_column, details_column = st.columns((1, 1.4), gap="large")
    with structure_column:
        display_molecule_image(candidate.mol, "Changed site highlighted", candidate.modified_highlight_atoms)
        st.caption(f"Computational sanity-screen score: {item.quality.score}/100")
    with details_column:
        st.markdown(f"#### {candidate.transformation}")
        st.write(candidate.description)
        if item.classification == "rejected":
            st.markdown('<div class="status-danger"><b>Rejected after screening</b></div>', unsafe_allow_html=True)
            for reason in item.rejection_reasons:
                st.write(f"• {reason}")
        elif item.classification == "improving":
            st.markdown('<div class="status-success"><b>Improving candidate</b></div>', unsafe_allow_html=True)
            st.write(item.objective_message)
        else:
            st.markdown('<div class="status-warning"><b>Feasible but not preferred</b></div>', unsafe_allow_html=True)
            st.write(item.objective_message)
        st.dataframe(comparison_table(original, candidate.descriptors), hide_index=True, width="stretch")
    with st.expander("Screening and chemical reasoning", expanded=False):
        st.markdown("**Quality checks**")
        for check in item.quality.checks:
            st.write(f"{'✓' if check.passed else '✕'} {check.label} — {check.explanation}")
        if item.constraints:
            st.markdown("**Numerical constraints**")
            for check in item.constraints:
                st.write(f"{'✓' if check.passed else '✕'} {check.explanation}")
        if item.retention:
            st.markdown("**Functional-group retention**")
            for check in item.retention:
                st.write(f"{'✓' if check.passed else '✕'} {check.explanation}")
        st.markdown("**Chemical reasoning**")
        st.write(item.chemical_reasoning)


def display_category(title: str, candidates: tuple[ScreenedCandidate, ...], css: str, original: dict[str, object], empty: str) -> None:
    st.markdown(f'<div class="{css}"><b>{title}</b> · {len(candidates)} candidate(s)</div>', unsafe_allow_html=True)
    if not candidates:
        st.caption(empty)
        return
    for item in candidates:
        with st.expander(item.candidate.transformation, expanded=False):
            show_screened_candidate(item, original)


def render_sidebar(theme: str) -> None:
    with st.sidebar:
        st.markdown("## ⚗ CHEM-WHATIF")
        st.caption("A computational laboratory for structure–property exploration")
        st.radio("Appearance", ("Light mode", "Dark mode"), key="theme_toggle", horizontal=True, on_change=save_appearance)
        st.divider()
        if st.session_state.analysis:
            st.caption(f"Current molecule: {st.session_state.analysis['name']}")
        st.markdown("Choose a molecule in Analysis, inspect its 3D structure, then explore modifications or optimize a property.")
        st.divider()
        st.caption("Built for educational molecular exploration. It does not predict synthesis, stability, safety, or clinical outcomes.")


def save_appearance() -> None:
    """Persist appearance independently of the current page's radio widget."""
    st.session_state.appearance = st.session_state.theme_toggle


def render_input_area() -> None:
    st.markdown("## Start with a molecule")
    st.markdown('<p class="section-note">Enter a familiar compound name. CHEM-WHATIF resolves its public chemical structure, validates it, and calculates every result from that structure.</p>', unsafe_allow_html=True)
    with st.form("molecule_name_form", clear_on_submit=False):
        input_column, action_column = st.columns((4, 1))
        with input_column:
            name = st.text_input("Molecule name", placeholder="e.g., Aspirin or Benzaldehyde")
        with action_column:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Analyze molecule", width="stretch")
    if submitted:
        try:
            with st.spinner("Resolving the structure and calculating its molecular profile…"):
                resolved_name, smiles, source = cached_name_resolution(name)
                save_analysis(parse_smiles(smiles, name=resolved_name, source=source))
            st.success(f"{resolved_name} is ready for exploration.")
        except MoleculeError as error:
            st.error(str(error))
        except Exception:
            st.error("The molecule could not be analyzed. Please check its spelling and try again.")
    example_column, example_action = st.columns((4, 1))
    with example_column:
        example = st.selectbox("Or choose a quick example", EXAMPLES, key="quick_example")
    with example_action:
        st.markdown("<br>", unsafe_allow_html=True)
        use_example = st.button("Analyze example", key="analyze_quick_example", width="stretch")
    if use_example:
        try:
            with st.spinner("Preparing the selected example…"):
                resolved_name, smiles, source = cached_name_resolution(example)
                save_analysis(parse_smiles(smiles, name=resolved_name, source=source))
            st.success(f"{resolved_name} is ready for exploration.")
        except MoleculeError as error:
            st.error(str(error))
        except Exception:
            st.error("The selected example could not be analyzed right now.")
    with st.expander("Technical input (optional)", expanded=False):
        st.caption("For advanced troubleshooting only. Normal use never requires chemical notation.")
        technical_smiles = st.text_input("SMILES structure", key="technical_smiles", placeholder="Optional technical structure input")
        if st.button("Analyze technical structure", key="analyze_technical"):
            try:
                save_analysis(parse_smiles(technical_smiles, name="Technical structure", source="Technical SMILES input"))
                st.success("Technical structure is ready for exploration.")
            except MoleculeError as error:
                st.error(str(error))


def render_analysis(record: MoleculeRecord, theme: str) -> None:
    descriptors = calculate_descriptors(record.mol)
    groups = detect_functional_groups(record.mol)
    st.divider()
    st.markdown(f"## {record.name} — molecular profile")
    st.caption(f"Structure source: {record.source}. All values below are RDKit-calculated molecular descriptors, not experimental measurements.")
    formula_column, atom_column, bond_column = st.columns(3)
    formula_column.metric("Molecular formula", descriptors["formula"])
    atom_column.metric("Atom count", descriptors["atom_count"])
    bond_column.metric("Bond count", descriptors["bond_count"])
    render_metric_grid(descriptors)

    structure_column, group_column = st.columns((1.08, 1), gap="large")
    with structure_column:
        st.subheader("2D molecular structure")
        display_molecule_image(record.mol, "RDKit-generated 2D structure")
    with group_column:
        st.subheader("Detected functional groups")
        st.markdown('<p class="section-note">SMARTS-based structural detection from the analyzed molecule.</p>', unsafe_allow_html=True)
        if groups:
            pills = "".join(
                f'<span class="pill"><b>{escape(group.label)}</b> — {group.count} occurrence{("s" if group.count != 1 else "")}</span>'
                for group in groups
            )
            st.markdown(pills, unsafe_allow_html=True)
            with st.expander("What the detected groups mean", expanded=False):
                for group in groups:
                    st.write(f"**{group.label}** — {group.explanation}")
        else:
            st.info("No supported functional groups were detected. The molecule can still be analyzed and may still have valid What-If transformations.")

    st.subheader("Interactive 3D molecular structure")
    st.markdown('<p class="section-note">This is an RDKit-generated, force-field-optimized conformer for exploration. It is not a claim of a unique experimentally measured molecular geometry.</p>', unsafe_allow_html=True)
    display_3d(record.mol, theme)


def render_modifications_page(record: MoleculeRecord, theme: str) -> None:
    descriptors = calculate_descriptors(record.mol)
    st.markdown("## What-If structural modification studio")
    st.markdown('<p class="section-note">Candidates are generated at real functional-group or bond sites, sanitized by RDKit, and deduplicated by chemical identity.</p>', unsafe_allow_html=True)
    analysis_key = record.canonical_smiles
    if st.button("Generate structure-aware What-If candidates", key=f"generate_{analysis_key}"):
        with st.spinner("Scanning structural sites and validating candidate molecules…"):
            st.session_state.candidate_generation = generate_candidates(record.mol)
            st.session_state.optimization_result = None
    generation: CandidateGeneration | None = st.session_state.candidate_generation
    if generation is None:
        st.info("Generate candidates to compare structural changes and use the optimization laboratory.")
    else:
        st.caption(
            f"Generation audit: {generation.generated_count} valid transformation product(s) generated; "
            f"{generation.unique_count} unique chemical structure(s) after deduplication; "
            f"{generation.invalid_or_unchanged_count} invalid or unchanged attempt(s) excluded before screening."
        )
        if not generation.candidates:
            st.info("No supported, valid structural modifications were found for this molecule. Its analysis and 3D structure remain available on the Molecule Analysis page.")
        else:
            candidate_lookup = {candidate.identifier: candidate for candidate in generation.candidates}
            selected_id = st.selectbox(
                "Choose a generated modification to inspect",
                options=list(candidate_lookup),
                format_func=lambda identifier: candidate_label(candidate_lookup[identifier], generation.candidates),
                key=f"whatif_candidate_{analysis_key}",
            )
            selected = candidate_lookup[selected_id]
            display_candidate_comparison(record.mol, selected, descriptors)
            st.subheader("Modified molecule in 3D")
            st.markdown('<p class="section-note">Explore the selected candidate with the same computational-conformer method.</p>', unsafe_allow_html=True)
            display_3d(selected.mol, theme)

    with st.expander("Technical details and computational limitations", expanded=False):
        st.markdown(
            "CHEM-WHATIF uses PubChem PUG REST for name-to-structure resolution, RDKit for validation, descriptors, "
            "functional-group SMARTS matching, 2D depictions, candidate sanitization, and ETKDG/MMFF or UFF conformers. "
            "It is an RDKit-based structure–property exploration prototype—not quantum chemistry, molecular dynamics, docking, "
            "synthesis planning, experimental measurement, toxicity prediction, or clinical prediction."
        )


def render_optimization_lab(record: MoleculeRecord, original: dict[str, object], generation: CandidateGeneration, groups) -> None:
    st.divider()
    st.markdown("## Molecular Optimization Laboratory")
    st.markdown('<p class="section-note">Starting molecule → candidate generation → quality screening → constraints → property calculation → objective evaluation → structural-family grouping → transparent classification.</p>', unsafe_allow_html=True)
    labels_to_keys = {descriptor_label(key): key for key in NUMERIC_PROPERTIES}
    control_a, control_b, control_c = st.columns((1.2, 1, 1.1))
    with control_a:
        chosen_label = st.selectbox("Property to optimize", list(labels_to_keys), key=f"objective_property_{record.canonical_smiles}")
    with control_b:
        direction_label = st.radio("Direction", ("Increase", "Decrease"), horizontal=True, key=f"objective_direction_{record.canonical_smiles}")
    with control_c:
        st.markdown("<br>", unsafe_allow_html=True)
        constraint_enabled = st.checkbox("Apply a numerical constraint", key=f"constraint_enabled_{record.canonical_smiles}")
    constraint: NumericalConstraint | None = None
    if constraint_enabled:
        constraint_a, constraint_b, constraint_c = st.columns(3)
        with constraint_a:
            constraint_label = st.selectbox("Constrain descriptor", list(labels_to_keys), index=0, key=f"constraint_property_{record.canonical_smiles}")
        with constraint_b:
            operator = st.selectbox("Condition", ("<", "<=", ">", ">=", "=="), key=f"constraint_operator_{record.canonical_smiles}")
        with constraint_c:
            value = st.number_input("Value", value=500.0 if constraint_label == "Molecular weight" else 0.0, key=f"constraint_value_{record.canonical_smiles}")
        constraint = NumericalConstraint(labels_to_keys[constraint_label], operator, float(value))
    retention_options = available_group_labels(record.mol)
    retained_keys: list[str] = []
    if retention_options:
        retained_keys = st.multiselect(
            "Require these starting functional groups to be retained (optional)",
            options=[key for key, _ in retention_options],
            format_func=lambda key: dict(retention_options)[key],
            key=f"retain_groups_{record.canonical_smiles}",
        )
    retained_groups = tuple((key, dict(retention_options)[key]) for key in retained_keys)
    if st.button("Run auditable optimization", key=f"run_optimization_{record.canonical_smiles}"):
        try:
            with st.spinner("Screening candidates, applying constraints, and evaluating the selected objective…"):
                st.session_state.optimization_result = optimize(
                    record.mol,
                    labels_to_keys[chosen_label],
                    direction_label.lower(),
                    (constraint,) if constraint else (),
                    retained_groups,
                    generation,
                )
        except Exception:
            st.error("Optimization could not be completed for the selected settings. Try a different objective or analyze the molecule again.")
    result: OptimizationResult | None = st.session_state.optimization_result
    if result is not None:
        display_optimization_result(result, original)


def display_optimization_result(result: OptimizationResult, original: dict[str, object]) -> None:
    property_name = descriptor_label(result.property_key)
    st.subheader(f"Optimization results — {property_name} to {result.direction}")
    audit_columns = st.columns(6)
    audit_values = (
        ("Generated", result.generation.generated_count), ("Unique", result.generation.unique_count),
        ("Evaluated", result.evaluated_count), ("Feasible", result.feasible_count),
        ("Improving", result.improving_count), ("Rejected", result.rejected_count),
    )
    for column, (label, value) in zip(audit_columns, audit_values):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-name">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown("### Optimization Audit Trail")
    audit_frame = pd.DataFrame(
        [
            ("Starting molecule", "1", "Original structure and descriptors calculated"),
            ("Candidate generation", str(result.generation.generated_count), "Structure-aware transformations at actual sites"),
            ("Unique structures", str(result.generation.unique_count), "Canonical-SMILES deduplication"),
            ("Quality and constraint screening", str(result.evaluated_count), "Validity, scope, size, difference, numerical and retention checks"),
            ("Objective evaluation", str(result.feasible_count), f"Calculated {property_name} evaluated for requested {result.direction}"),
            ("Final classification", str(result.improving_count + len(result.feasible_not_preferred) + result.rejected_count), "Improving / feasible but not preferred / rejected"),
        ],
        columns=["Pipeline stage", "Count", "What happened"],
    )
    st.dataframe(audit_frame, hide_index=True, width="stretch")

    if result.best_candidate:
        best = result.best_candidate
        st.markdown("### Best computational candidate")
        st.markdown('<div class="status-success"><b>Best candidate is selected from the valid candidates that improve the requested calculated descriptor.</b></div>', unsafe_allow_html=True)
        show_screened_candidate(best, original)
    else:
        st.markdown('<div class="status-warning"><b>No candidate passed all selected requirements and improved the requested calculated descriptor.</b></div>', unsafe_allow_html=True)

    if result.positional_isomer_families:
        st.markdown("### Positional-isomer families")
        st.caption("These are distinct ring-substitution positions. Their selected calculated objective is identical, so CHEM-WHATIF treats them as tied families rather than falsely ranking individual isomers.")
        for family in result.positional_isomer_families:
            st.markdown(
                f'<div class="lab-card"><b>Positional isomers · family rank {family.rank}</b><br>{escape(family.transformation)} · '
                f'{escape(descriptor_label(result.property_key))}: {escape(descriptor_value_text(result.property_key, family.objective_value))}<br>'
                'The substituent occurs at different positions on the aromatic ring; the current calculated objective is tied.</div>',
                unsafe_allow_html=True,
            )
            # Create a fresh row after every three structures so visual and
            # reading order stay aligned (1, 2, 3, then 4) for a family.
            for start in range(0, len(family.members), 3):
                row_members = family.members[start : start + 3]
                columns = st.columns(len(row_members))
                for offset, member in enumerate(row_members):
                    with columns[offset]:
                        member_number = start + offset + 1
                        display_molecule_image(member.candidate.mol, f"Positional isomer {member_number}", member.candidate.modified_highlight_atoms)
                        st.caption(member.candidate.description)

    st.markdown("### Candidate classifications")
    display_category("Improving Candidates", result.improving_candidates, "status-success", original, "No candidates improved this objective under the selected requirements.")
    display_category("Feasible but Not Preferred", result.feasible_not_preferred, "status-warning", original, "No feasible candidates were neutral or opposite to the requested objective.")
    display_category("Rejected Candidates", result.rejected_candidates, "status-danger", original, "No candidates were rejected by the current structural quality and constraint screening.")
    st.caption("The quality screen is a transparent computational sanity screen. Passing it does not prove experimental synthesizability, stability, safety, efficacy, or clinical usefulness.")


def render_page(page: str) -> None:
    """Render only the selected laboratory workspace, sharing molecular state."""
    theme = st.session_state.theme_toggle
    if page == "analysis":
        render_input_area()
    record = record_from_state()
    if record is None:
        st.info("Choose and analyze a molecule on the Molecule Analysis page to begin.")
        if page != "analysis":
            st.page_link(ANALYSIS_PAGE, label="Go to Molecule Analysis")
        return
    if page != "analysis":
        st.caption(f"Starting molecule: {record.name}")
    try:
        if page == "analysis":
            render_analysis(record, theme)
        elif page == "modifications":
            render_modifications_page(record, theme)
        else:
            # Optimization is independently usable; candidates need not be
            # manually generated on another page first.
            if st.session_state.candidate_generation is None:
                with st.spinner("Preparing structural modifications…"):
                    st.session_state.candidate_generation = generate_candidates(record.mol)
            generation = st.session_state.candidate_generation
            if not generation.candidates:
                st.info("No supported structural modifications were found for this molecule.")
                return
            render_optimization_lab(record, calculate_descriptors(record.mol), generation, detect_functional_groups(record.mol))
    except MoleculeError as error:
        st.error(str(error))
    except Exception:
        st.error("This workspace could not be loaded. Please analyze the molecule again.")


ANALYSIS_PAGE = st.Page(lambda: render_page("analysis"), title="Molecule Analysis", default=True)
LABORATORY_PAGES = (
    ANALYSIS_PAGE,
    st.Page(lambda: render_page("modifications"), title="What-If Modifications", url_path="modifications"),
    st.Page(lambda: render_page("optimization"), title="Optimization", url_path="optimization"),
)


def render_page_navigation(current_page) -> None:
    """Provide sequential navigation at the end of every workspace."""
    index = next(i for i, page in enumerate(LABORATORY_PAGES) if page.url_path == current_page.url_path)
    st.divider()
    back_column, next_column = st.columns(2)
    with back_column:
        if index > 0:
            previous = LABORATORY_PAGES[index - 1]
            if st.button(f"← Back: {previous.title}", key=f"back_{index}", width="stretch"):
                st.switch_page(previous)
    with next_column:
        if index < len(LABORATORY_PAGES) - 1:
            following = LABORATORY_PAGES[index + 1]
            if st.button(
                f"Next: {following.title} →",
                key=f"next_{index}",
                width="stretch",
                disabled=st.session_state.analysis is None,
            ):
                st.switch_page(following)
        elif st.button("Return to Molecule Analysis", key="return_to_analysis", width="stretch"):
            st.switch_page(ANALYSIS_PAGE)
    if st.session_state.analysis is None:
        st.caption("Analyze a molecule to continue to the next page.")


def main() -> None:
    initialise_state()
    theme = st.session_state.theme_toggle
    apply_theme(theme)
    render_sidebar(theme)
    # Radio writes its own state during rendering; read it again afterwards so
    # the visual palette takes effect on the next normal Streamlit rerun.
    navigation = st.navigation(list(LABORATORY_PAGES))
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Computational chemistry · information technology · responsible interpretation</div>
          <h1>CHEM-WHATIF</h1>
          <p>A computational laboratory for exploring how structure-aware molecular modifications change calculated properties. Start from a real chemical structure, examine it, generate valid What-If candidates, and optimize transparently—with no clinical or experimental claims.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    navigation.run()
    render_page_navigation(navigation)


if __name__ == "__main__":
    main()
