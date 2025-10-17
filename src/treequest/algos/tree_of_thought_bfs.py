import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from functools import total_ordering
from heapq import heappop, heappush
from typing import Generic, List, TypeVar

from treequest.algos.base import Algorithm
from treequest.algos.tree import Node, Tree
from treequest.trial import Trial, TrialId, TrialStoreWithNodeQueue
from treequest.types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")


@total_ordering
class TreeOfThoughtsBFSHeapItem(Generic[StateT]):
    """
    Heap item for Tree of Thoughts BFS algorithm.

    Prioritizes:
    1. Deeper nodes (higher depth first)
    2. Higher scores when depths are equal
    """

    def __init__(self, node: Node[StateT]):
        self.node = node

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TreeOfThoughtsBFSHeapItem)
        return (
            self.node.depth == other.node.depth and self.node.score == other.node.score
        )

    def __lt__(
        self, other: "TreeOfThoughtsBFSHeapItem[StateT]"
    ) -> bool:  # heapq is a min heap
        if self.node.depth != other.node.depth:
            return (
                self.node.depth > other.node.depth
            )  # Deeper nodes are better because we need to get S_{t-1}
        return self.node.score > other.node.score  # Higher score is better


@dataclass
class ToTBFSState(Generic[StateT]):
    """State for Tree of Thoughts BFS algorithm."""

    tree: Tree[StateT]
    trial_store: TrialStoreWithNodeQueue[StateT] = dataclasses.field(
        default_factory=TrialStoreWithNodeQueue[StateT]
    )
    # Current depth level being processed
    current_depth: int = 0


class TreeOfThoughtsBFSAlgo(Algorithm[StateT, ToTBFSState[StateT]]):
    """
    Tree of Thoughts Breadth-First Search (ToT-BFS) algorithm.

    Original paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html

    This algorithm:
    1. Expands the tree breadth-first by depth level
    2. At each step, selects the b best-scoring leaf nodes of the same depth
    3. For each selected node, generates k new samples
    """

    def __init__(
        self,
        *,
        breadth_limit: int = 3,  # b: number of nodes to select at each step
        size_limit: int = 5,  # k: number of samples to generate for each node
    ):
        """
        Initialize the Tree of Thoughts BFS algorithm.

        Args:
            breadth_limit: Number of nodes to select at each step (b)
            size_limit: Number of samples to generate for each node (k)
        """
        assert size_limit > 0, "size_limit (k) should be greater than 0"
        assert breadth_limit > 0, "breadth_limit (b) should be greater than 0"
        assert size_limit >= breadth_limit, (
            "size_limit (k) should be greater than or equal to breadth_limit (b)"
        )

        self.size_limit = size_limit
        self.breadth_limit = breadth_limit

    def step(
        self,
        state: ToTBFSState[StateT],
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> ToTBFSState[StateT]:
        """
        Run one step of the ToT-BFS algorithm, expanding a single node.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        state, trial = self.ask(state, actions=list(generate_fn.keys()))
        parent = state.tree.get_node(trial.node_to_expand)
        action = trial.action

        result = generate_fn[action](parent.state)

        state = self.tell(state, trial.trial_id, result)
        return state

    def _next_nodes_and_actions(
        self,
        state: ToTBFSState[StateT],
        actions: list[str],
    ) -> list[tuple[Node[StateT], str]]:
        """
        Select the best nodes at the deepest level and queue them for expansion.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions
        """
        priority_queue: List[TreeOfThoughtsBFSHeapItem[StateT]] = []

        # Find all unexpanded leaf nodes
        leaf_nodes = [
            node
            for node in state.tree.get_nodes()
            if not node.children and node != state.tree.root
        ]

        # If we have leaf nodes, find the deepest level
        max_depth = 0
        if leaf_nodes:
            max_depth = max(node.depth for node in leaf_nodes)
            state.current_depth = max_depth

        # Add all leaf nodes at the deepest level to the priority queue
        for node in leaf_nodes:
            if node.depth == max_depth:
                heappush(priority_queue, TreeOfThoughtsBFSHeapItem(node))

        # Select the top breadth_limit nodes
        selected_nodes = []
        for _ in range(min(self.breadth_limit, len(priority_queue))):
            if not priority_queue:
                break

            selected = heappop(priority_queue)
            selected_nodes.append(selected.node)

        # For each selected node, queue up expansions for all actions
        # Distribute the size_limit across all actions
        samples_per_action = max(1, self.size_limit // len(actions))

        nodes_and_actions = []
        for node in selected_nodes:
            # Queue expansions for each action
            for action in actions:
                for _ in range(samples_per_action):
                    nodes_and_actions.append((node, action))
        return nodes_and_actions

    def init_tree(self) -> ToTBFSState[StateT]:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return ToTBFSState(tree=tree)

    def get_state_score_pairs(
        self, state: ToTBFSState[StateT]
    ) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()

    def ask_batch(
        self, state: ToTBFSState[StateT], batch_size: int, actions: list[str]
    ) -> tuple[ToTBFSState[StateT], list[Trial[StateT]]]:
        # If no nodes are queued for expansion, select nodes to expand
        if state.trial_store.is_queue_empty():
            # For the root node, just add it with the first action
            if not state.tree.root.children:
                action = actions[0]
                nodes_and_actions = [(state.tree.root, action)]
            else:
                # Otherwise, select best nodes at the deepest level
                nodes_and_actions = self._next_nodes_and_actions(state, actions)
            state.trial_store.fill_nodes_queue(nodes_and_actions)

        trials = state.trial_store.get_batch_from_queue(batch_size)
        return state, trials

    def tell(
        self,
        state: ToTBFSState[StateT],
        trial_id: TrialId,
        result: tuple[StateT, float],
    ) -> ToTBFSState[StateT]:
        _new_state, new_score = result

        finished_trial = state.trial_store.get_finished_trial(trial_id, new_score)
        if (
            finished_trial is None
        ):  # Trial is no longer valid, so we do not reflect the result to state
            return state

        parent = state.tree.get_node(finished_trial.node_to_expand)
        # Add to the tree
        state.tree.add_node(result, parent)
        # Update current depth
        state.current_depth = max(state.current_depth, parent.depth + 1)

        state.trial_store.advance_queue(finished_trial.action, parent)

        return state
