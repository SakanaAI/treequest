"""Visualization adapter for ABMCTSA algorithm."""

import math
from typing import Any, Dict, TypeVar

from treequest.algos.ab_mcts_a.algo import ABMCTSAAlgoState
from treequest.algos.ab_mcts_a.prob_state import ProbabilisticDist
from treequest.algos.tree import Node

StateT = TypeVar("StateT")


def probabilistic_dist_to_str(probabilistic_dist: ProbabilisticDist) -> str:
    """Convert NodeProbState to string representation."""
    if probabilistic_dist.dist_type == "beta":
        return f"β({probabilistic_dist.params['a']:.3f}, {probabilistic_dist.params['b']:.3f})"
    elif probabilistic_dist.dist_type == "gaussian":
        sigma_square = f"σ² ~ χ⁻²({probabilistic_dist.params['nu']:.3f}, {probabilistic_dist.params['tau_square']:.3f})"
        return f"𝒩({probabilistic_dist.params['m']:.3f}, σ²/{probabilistic_dist.params['kappa']:.3f}), {sigma_square}"
    else:
        return "Unknown Distribution"


class ABMCTSAAdapter:
    """Adapter for ABMCTSA algorithm."""

    def extract_node_metrics(
        self, algo_state: ABMCTSAAlgoState[StateT], node: Node[StateT]
    ) -> Dict[str, Any]:
        """Extract ABMCTSA-specific metrics for a node."""

        if not isinstance(algo_state, ABMCTSAAlgoState):
            return {}
        thompson_state = algo_state.thompson_states.get(node)
        if thompson_state is None:
            return {}  # No metrics available
        action_probas = None
        if thompson_state.model_selection_strategy == "stack":
            action_probas = (
                "<ul>"
                + "".join(
                    [
                        f"<li>{action} ~ {probabilistic_dist_to_str(probabilistic_dist)}<ul>"
                        + f"<li>GEN ~ {probabilistic_dist_to_str(thompson_state.gen_vs_cont_probas[action]['GEN'])}</li>"
                        + f"<li>CONT ~ {probabilistic_dist_to_str(thompson_state.gen_vs_cont_probas[action]['CONT'])}<ul>"
                        + "".join(
                            [
                                f"<li>{node_id} ~ {probabilistic_dist_to_str(probabilistic_dist)}</li>"
                                for node_id, probabilistic_dist in thompson_state.node_probas[
                                    action
                                ].items()
                            ]
                        )
                        + "</ul></li></ul></li>"
                        for action, probabilistic_dist in thompson_state.action_probas.items()
                    ]
                )
                + "</ul>"
            )
        elif thompson_state.model_selection_strategy == "multiarm_bandit_thompson":
            action_probas = "<ul>" + (
                f"<li>GEN ~ {probabilistic_dist_to_str(thompson_state.gen_vs_cont_probas['shared']['GEN'])}<ul>"
                + "".join(
                    [
                        f"<li>{action} ~ {probabilistic_dist_to_str(ProbabilisticDist(thompson_state.prior_for_actions[action]))}</li>"
                        for action in algo_state.all_rewards_store.keys()
                    ]
                )
                + f"</ul></li><li>CONT ~ {probabilistic_dist_to_str(thompson_state.gen_vs_cont_probas['shared']['CONT'])}<ul>"
                + "".join(
                    [
                        f"<li>{action} ~ {probabilistic_dist_to_str(probabilistic_dist)}</li>"
                        for action, probabilistic_dist in thompson_state.node_probas[
                            "shared"
                        ].items()
                    ]
                )
                + "</ul></li>"
            )
        elif thompson_state.model_selection_strategy == "multiarm_bandit_ucb":
            all_len = sum(
                len(scores) for scores in algo_state.all_rewards_store.values()
            )
            action_probas = "<ul>" + (
                f"<li>GEN ~ {probabilistic_dist_to_str(thompson_state.gen_vs_cont_probas['shared']['GEN'])}<ul>"
                + "".join(
                    [
                        f"<li>{action}: UCB Score = {sum(scores) / len(scores) + math.sqrt(2 * math.log(all_len) / len(scores))}</li>"
                        for action, scores in algo_state.all_rewards_store.items()
                    ]
                )
                + f"</ul></li><li>CONT ~ {probabilistic_dist_to_str(thompson_state.gen_vs_cont_probas['shared']['CONT'])}<ul>"
                + "".join(
                    [
                        f"<li>{action} ~ {probabilistic_dist_to_str(probabilistic_dist)}</li>"
                        for action, probabilistic_dist in thompson_state.node_probas[
                            "shared"
                        ].items()
                    ]
                )
                + "</ul></li>"
            )
        else:  # Not reachable
            raise ValueError(
                f"Unknown model_selection_strategy: {thompson_state.model_selection_strategy}"
            )
        return {
            "action_probas": {
                "display_name": "Action Probabilities",
                "display_value": action_probas if action_probas is not None else "N/A",
            },
        }

    def get_algorithm_name(self, algo_state: ABMCTSAAlgoState[StateT]) -> str:
        """Get algorithm name."""
        return "ABMCTSA"
