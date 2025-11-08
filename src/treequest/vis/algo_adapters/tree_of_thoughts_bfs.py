"""Visualization adapter for TreeOfThoughtBFS algorithm."""

from typing import Any, Dict, TypeVar

from treequest.algos.tree import Node
from treequest.algos.tree_of_thought_bfs import ToTBFSState

StateT = TypeVar("StateT")


class TreeOfThoughtsBFSAdapter:
    """Adapter for TreeOfThoughtsBFS algorithm."""

    def extract_node_metrics(
        self, algo_state: ToTBFSState[StateT], node: Node[StateT]
    ) -> Dict[str, Any]:
        """Extract ToTBFS-specific metrics for a node."""

        if not isinstance(algo_state, ToTBFSState):
            return {}
        return {}  # Currently, no specific metrics to extract

    def get_algorithm_name(self, algo_state: ToTBFSState[StateT]) -> str:
        """Get algorithm name."""
        return "TreeOfThoughtsBFS"
