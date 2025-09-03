"""
Defines the HierarchicalAction class and related action formatting logic.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class HierarchicalAction:
    """Structured action containing decisions at all hierarchy levels."""

    strategic_action: str
    tactic_family: str
    specific_tactic: str
    parameters: List[str]
    confidence: float


def format_action_to_string(action: HierarchicalAction) -> str:
    """Formats a HierarchicalAction object into a tactic string."""
    param_str = " ".join(action.parameters) if action.parameters else ""
    tactic = action.specific_tactic

    # Format the final tactic string
    if param_str:
        # Tactics that require parameters in brackets
        if tactic in ["rw", "simp", "simp_rw"]:
            return f"[{param_str}]"
        # Tactics that take a list of identifiers
        elif tactic in ["intro", "intros", "rintro", "clear", "rename"]:
            return f"{tactic} {param_str}"
        # Tactics with a 'have' clause
        elif tactic == "have":
            return f"have h : {param_str}"
        # Default formatting for tactics with parameters
        else:
            return f"{tactic} {param_str}"
    else:
        # Return tactic without parameters
        return tactic
