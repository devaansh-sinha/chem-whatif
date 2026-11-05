"""Careful, descriptor-grounded explanatory text for CHEM-WHATIF."""
from __future__ import annotations

from .chemistry import descriptor_label, descriptor_value_text


def descriptor_change_sentences(original: dict[str, object], modified: dict[str, object]) -> list[str]:
    """Translate calculated changes into cautious chemistry interpretations."""
    lines: list[str] = []
    changes = {
        key: float(modified[key]) - float(original[key])
        for key in ("molecular_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds")
    }
    if changes["logp"] < -0.005:
        lines.append("Calculated LogP decreased, so this descriptor profile is less lipophilic and more hydrophilic.")
    elif changes["logp"] > 0.005:
        lines.append("Calculated LogP increased, so this descriptor profile is more lipophilic.")
    if changes["tpsa"] > 0.005:
        lines.append("Calculated polar surface area increased, indicating a more polar descriptor profile.")
    elif changes["tpsa"] < -0.005:
        lines.append("Calculated polar surface area decreased, indicating a less polar descriptor profile.")
    if changes["hbd"] > 0:
        lines.append("Additional hydrogen-bond donor functionality is present in the calculated structure.")
    elif changes["hbd"] < 0:
        lines.append("The calculated structure has fewer hydrogen-bond donors.")
    if changes["hba"] > 0:
        lines.append("The calculated structure has more hydrogen-bond acceptors.")
    elif changes["hba"] < 0:
        lines.append("The calculated structure has fewer hydrogen-bond acceptors.")
    if changes["rotatable_bonds"] > 0:
        lines.append("The modified structure has more calculated rotatable bonds, indicating greater conformational flexibility.")
    elif changes["rotatable_bonds"] < 0:
        lines.append("The modified structure has fewer calculated rotatable bonds.")
    if changes["molecular_weight"] > 0.005:
        lines.append("The calculated molecular weight increased.")
    elif changes["molecular_weight"] < -0.005:
        lines.append("The calculated molecular weight decreased.")
    return lines or ["The selected descriptor panel did not show a material change for this transformation."]


def objective_explanation(property_key: str, direction: str, original: dict[str, object], modified: dict[str, object]) -> str:
    """Explain objective direction using real calculated values only."""
    before, after = float(original[property_key]), float(modified[property_key])
    delta = after - before
    label = descriptor_label(property_key)
    target_word = "increase" if direction == "increase" else "decrease"
    if (direction == "increase" and delta > 0.005) or (direction == "decrease" and delta < -0.005):
        return (
            f"Improvement found for {label}: the calculated value changed from "
            f"{descriptor_value_text(property_key, before)} to {descriptor_value_text(property_key, after)}, matching the requested {target_word}."
        )
    if abs(delta) <= 0.005:
        return f"{label} is tied at {descriptor_value_text(property_key, after)}; this candidate does not improve the requested {target_word}."
    return (
        f"{label} changed from {descriptor_value_text(property_key, before)} to "
        f"{descriptor_value_text(property_key, after)}, which is opposite to the requested {target_word}."
    )


def candidate_reasoning(base_reason: str, original: dict[str, object], modified: dict[str, object]) -> str:
    """Combine structural rationale and descriptor observations without bioactivity claims."""
    observations = descriptor_change_sentences(original, modified)
    return f"{base_reason} {' '.join(observations[:3])} These are computational descriptors, not experimental or clinical predictions."
