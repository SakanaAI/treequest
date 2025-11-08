"""Visualization adapter for MultiArmedBanditUCB algorithm."""

import math
import statistics
from typing import Any, Dict, TypeVar

from treequest.algos.multi_armed_bandit_ucb import UCBState
from treequest.algos.tree import Node

StateT = TypeVar("StateT")


class MultiArmedBanditUCBAdapter:
    """Adapter for MultiArmedBanditUCB algorithm."""

    def __init__(self, exploration_weight: float = math.sqrt(2)):
        self.exploration_weight = exploration_weight

    def extract_node_metrics(
        self, algo_state: UCBState[StateT], node: Node[StateT]
    ) -> Dict[str, Any]:
        """Extract UCB-specific metrics for a node."""

        if not isinstance(algo_state, UCBState):
            return {}
        total_len = sum(len(scores) for scores in algo_state.scores_by_action.values())
        if total_len == 0:
            return {}
        actions = {
            action: {
                "len": len(scores) if scores else 0,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "mean": statistics.mean(scores) if scores else None,
                "median": statistics.median(scores) if scores else None,
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            }
            for action, scores in algo_state.scores_by_action.items()
        }
        ucb_scores = {
            action: data["mean"]
            + self.exploration_weight * math.sqrt(math.log(total_len) / data["len"])
            for action, data in actions.items()
            if data["len"] > 0
        }
        return {
            "total_len": {
                "display_name": "Total Samples Recorded",
                "display_value": str(total_len),
            },
            "action_stats": {
                "display_name": "Action Statistics",
                "display_value": "<ul>"
                + "".join(
                    f"<li><b>{action}</b>: UCB Score = "
                    + (f"{ucb_scores[action]:.3f}" if action in ucb_scores else "N/A")
                    + f" (len = {data['len']}"
                    + (
                        f", min = {data['min']:.3f}, max = {data['max']:.3f}, mean = {data['mean']:.3f}"
                        + f", median = {data['median']:.3f}, stdev = {data['stdev']:.3f}"
                        if data["len"] > 0
                        else ""
                    )
                    + ")</li>"
                    for action, data in sorted(actions.items())
                )
                + "</ul>",
            },
        }

    def get_algorithm_name(self, algo_state: Any) -> str:
        """Get algorithm name."""
        return "MultiArmedBanditUCB"
