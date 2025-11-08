"""Visualization adapter for BestFirstSearch algorithm."""

from typing import Any, Dict, TypeVar

from treequest.algos.best_first_search import BFSState
from treequest.algos.tree import Node

StateT = TypeVar("StateT")


class BestFirstSearchAdapter:
    """Adapter for BestFirstSearch algorithm."""

    def extract_node_metrics(
        self, algo_state: BFSState[StateT], node: Node[StateT]
    ) -> Dict[str, Any]:
        """Extract BFS-specific metrics for a node."""

        if not isinstance(algo_state, BFSState):
            return {}
        leaf_indices = {
            item.node.expand_idx: {
                "rank": rank,
                "score": item.score,
                "depth": item.node.depth,
            }
            for rank, item in enumerate(
                sorted(algo_state.leaves, key=lambda item: item.sort_index), 1
            )
        }
        return {
            "is_leaf": {
                "display_name": "Is Leaf Node",
                "display_value": "Yes" if node.expand_idx in leaf_indices else "No",
            },
            "leaf_rank": {
                "display_name": "Leaf Rank (1-indexed)",
                "display_value": str(leaf_indices[node.expand_idx]["rank"])
                if node.expand_idx in leaf_indices
                else "N/A",
            },
            "leaf_score": {
                "display_name": "Leaf Score",
                "display_value": f"{leaf_indices[node.expand_idx]['score']:.4f}"
                if node.expand_idx in leaf_indices
                else "N/A",
            },
        }

    def get_algorithm_name(self, algo_state: BFSState[StateT]) -> str:
        """Get algorithm name."""
        return "BestFirstSearch"
