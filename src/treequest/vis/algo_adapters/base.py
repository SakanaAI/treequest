"""Base adapter protocol for extracting algorithm-specific metrics."""

from typing import Any, Dict, Protocol, TypeVar

from treequest.algos.tree import Node

StateT = TypeVar("StateT")
AlgoStateT = TypeVar("AlgoStateT", contravariant=True)


class VisualizerAdapter(Protocol[StateT, AlgoStateT]):
    """Protocol for algorithm-specific metric extraction."""

    def extract_node_metrics(
        self, algo_state: AlgoStateT, node: Node[StateT]
    ) -> Dict[str, Any]:
        """
        Extract algorithm-specific metrics for a given node.

        Args:
            algo_state: The algorithm state
            node: The node to extract metrics for

        Returns:
            Dictionary of metrics for this node
        """
        ...

    def get_algorithm_name(self, algo_state: AlgoStateT) -> str:
        """
        Get the algorithm name from the state.

        Args:
            algo_state: The algorithm state

        Returns:
            Name of the algorithm
        """
        ...
