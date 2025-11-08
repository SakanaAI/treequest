"""Visualization adapter for ABMCTSM algorithm."""

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, TypeVar

from treequest.algos.ab_mcts_m.algo import ABMCTSMState
from treequest.algos.ab_mcts_m.pymc_interface import (
    Observation,
    PruningConfig,
    is_prunable,
)
from treequest.algos.tree import Node

StateT = TypeVar("StateT")


class ABMCTSMAdapter:
    """Adapter for ABMCTSM algorithm."""

    def __init__(self) -> None:
        # Used only for the (optional) prune hint shown to users
        self._default_pruning_config = PruningConfig()

    def extract_node_metrics(
        self, algo_state: ABMCTSMState[StateT], node: Node[StateT]
    ) -> Dict[str, Any]:
        """Extract ABMCTSM-specific metrics for a node."""

        if not isinstance(algo_state, ABMCTSMState):
            return {}
        # Subtree observations under this node (excluding the node itself)
        observations = Observation.collect_all_observations_of_descendant(
            node, algo_state.all_observations
        )
        if len(observations) == 0:
            return {}
        prunable = is_prunable(node, observations, self._default_pruning_config)
        rewards_by_action = defaultdict(list)
        rewards_by_child = defaultdict(list)
        for obs in observations:
            rewards_by_action[obs.action].append(obs.reward)
            if obs.child_idx >= 0:
                rewards_by_child[obs.child_idx].append(obs.reward)
        ucb_scores = {
            action: statistics.mean(rewards)
            + math.sqrt(2 * math.log(len(algo_state.all_observations)) / len(rewards))
            for action, rewards in rewards_by_action.items()
        }
        return {
            "prunable": {
                "display_name": "Prunable",
                "display_value": "Yes" if prunable else "No",
            },
            "rewards_by_action": {
                "display_name": "Rewards by Action",
                "display_value": "<ul>"
                + "".join(
                    f"<li>{action}: len = {len(rewards)}, mean = {statistics.mean(rewards):.3f}, UCB Score = {ucb_scores[action]:.3f}</li>"
                    for action, rewards in sorted(rewards_by_action.items())
                )
                + "</ul>",
            },
            "rewards_by_child": {
                "display_name": "Rewards by Child",
                "display_value": "<ul>"
                + "".join(
                    f"<li>child #{child_idx}: n={len(rewards)}, mean={statistics.mean(rewards):.3f}</li>"
                    for child_idx, rewards in sorted(rewards_by_child.items())
                )
                + "</ul>",
            },
        }

    def get_algorithm_name(self, algo_state: ABMCTSMState[StateT]) -> str:
        """Get algorithm name."""
        return "ABMCTSM"
