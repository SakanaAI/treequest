"""Visualization adapter for MultiArmedBanditUCB algorithm."""

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar

from treequest.algos.multi_armed_bandit_ucb import UCBState
from treequest.algos.tree import Node

StateT = TypeVar("StateT")


@dataclass
class _ActionStats:
    length: int
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    stdev: Optional[float] = None

    def format(self) -> str:
        if (
            self.length == 0
            or self.minimum is None
            or self.maximum is None
            or self.mean is None
            or self.median is None
            or self.stdev is None
        ):
            return f"len = {self.length:d}"
        return (
            f"len = {self.length:d}, min = {self.minimum:.3f}, max = {self.maximum:.3f}, "
            f"mean = {self.mean:.3f}, median = {self.median:.3f}, "
            f"stdev = {self.stdev:.3f}"
        )


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
        actions: Dict[str, _ActionStats] = {}
        for action, scores in algo_state.scores_by_action.items():
            length = len(scores)
            if length == 0:
                actions[action] = _ActionStats(length=0)
            else:
                actions[action] = _ActionStats(
                    length=length,
                    minimum=min(scores),
                    maximum=max(scores),
                    mean=statistics.mean(scores),
                    median=statistics.median(scores),
                    stdev=statistics.stdev(scores) if length > 1 else 0.0,
                )
        ucb_scores: Dict[str, float] = {}
        for action, data in actions.items():
            if data.length == 0 or data.mean is None:
                continue
            exploration_bonus = self.exploration_weight * math.sqrt(
                math.log(total_len) / data.length
            )
            ucb_scores[action] = data.mean + exploration_bonus

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
                    + f" ({data.format()})</li>"
                    for action, data in sorted(actions.items())
                )
                + "</ul>",
            },
        }

    def get_algorithm_name(self, algo_state: Any) -> str:
        """Get algorithm name."""
        return "MultiArmedBanditUCB"
