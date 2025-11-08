"""Visualization adapter for StandardMCTS algorithm."""

import math
from typing import Any, Dict, TypeVar

from treequest.algos.standard_mcts import MCTSState
from treequest.algos.tree import Node

StateT = TypeVar("StateT")


class StandardMCTSAdapter:
    """Adapter for StandardMCTS algorithm."""

    def __init__(self, exploration_weight: float = math.sqrt(2)):
        self.exploration_weight = exploration_weight

    def extract_node_metrics(
        self, algo_state: MCTSState[StateT], node: Node[StateT]
    ) -> Dict[str, Any]:
        """Extract MCTS-specific metrics for a node."""

        if not isinstance(algo_state, MCTSState):
            return {}
        # Safe access with defaults to avoid KeyError / log(0) / div-by-zero
        parent_visits = (
            algo_state.visit_counts.get(node.parent.expand_idx, 1) if node.parent else 1
        )
        node_visits = algo_state.visit_counts.get(node.expand_idx, 0)
        value_sum = algo_state.value_sums.get(node.expand_idx, 0.0)
        exploitation = value_sum / node_visits if node_visits > 0 else None
        prior = algo_state.priors.get(node.expand_idx, None)
        exploration = None
        if node_visits > 0 and parent_visits > 1 and isinstance(prior, (int, float)):
            exploration = (
                self.exploration_weight
                * prior
                * math.sqrt(math.log(parent_visits) / node_visits)
            )
        uct_score = (
            (exploitation + exploration)
            if (exploitation is not None and exploration is not None)
            else None
        )
        return {
            "visits": {
                "display_name": "Visits",
                "display_value": f"{node_visits:d}",
            },
            "mean": {
                "display_name": "Mean (Exploitation)",
                "display_value": f"{exploitation:.3f}"
                if exploitation is not None
                else "N/A",
            },
            "prior": {
                "display_name": "Prior",
                "display_value": f"{prior:.3f}"
                if isinstance(prior, (int, float))
                else "N/A",
            },
            "uct_score": {
                "display_name": "UCT Score",
                "display_value": f"{uct_score:.3f}"
                if isinstance(uct_score, (int, float))
                else "N/A",
            },
        }

    def get_algorithm_name(self, algo_state: Any) -> str:
        """Get algorithm name."""
        return "StandardMCTS"
