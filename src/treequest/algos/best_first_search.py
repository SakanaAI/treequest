import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Generic, List, Tuple, TypeVar

from treequest.algos.base import Algorithm
from treequest.algos.tree import Node, Tree
from treequest.trial import Trial, TrialId, TrialStoreWithNodeQueue
from treequest.types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")


@dataclass(order=True)
class BFSHeapItem(Generic[StateT]):
    """
    Heap item for Best First Search algorithm.

    Implements ordering to prioritize:
    1. Higher scores (reversed for min heap)
    2. Shallower nodes when scores are equal
    """

    # Use sort_index for comparison but keep it hidden from __init__
    sort_index: Tuple[float, int] = dataclasses.field(init=False, repr=False)
    node: Node[StateT] = dataclasses.field(compare=False)
    score: float = dataclasses.field(compare=False)

    def __post_init__(self):
        # Negative score for min heap (higher scores prioritized)
        # Node depth as secondary sort key (shallower nodes preferred)
        self.sort_index = (-self.score, self.node.depth)


@dataclass
class BFSState(Generic[StateT]):
    """State for Best First Search algorithm."""

    tree: Tree[StateT]
    leaves: List[BFSHeapItem[StateT]] = dataclasses.field(
        default_factory=list[BFSHeapItem[StateT]]
    )

    trial_store: TrialStoreWithNodeQueue[StateT] = dataclasses.field(
        default_factory=TrialStoreWithNodeQueue[StateT]
    )


class BestFirstSearchAlgo(Algorithm[StateT, BFSState[StateT]]):
    """
    Best First Search algorithm implementation.

    This algorithm:
    1. Prioritizes node expansion based on scores (higher scores first)
    2. When scores are equal, prefers shallower nodes
    3. For each selected node, generates `num_samples` samples for each available action,
       for a total of `num_samples` * number of actions new nodes.

    The expansion process:
    - Select the highest-scoring node from the priority queue
    - For that node, generate num_samples samples for each available action
    - Add all new nodes to the priority queue
    - Repeat
    """

    def __init__(self, *, num_samples: int = 2):
        """
        Initialize the Best First Search algorithm.

        Args:
            num_samples: Number of samples to generate for each action.
                        For each selected node, a total of (num_samples * number of actions)
                        new nodes will be generated.
        """
        self.num_samples = num_samples

    def step(
        self,
        state: BFSState[StateT],
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> BFSState[StateT]:
        """
        Generate one additional node and add that to a given state.

        For each selected node, this algorithm will generate:
        - num_samples samples for each available action
        - Total of (num_samples * number of actions) new nodes

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        actions = list(generate_fn.keys())

        state, trial = self.ask(state, actions)

        # Get the next node to expand
        action = trial.action
        node = state.tree.get_node(trial.node_to_expand)

        # Generate a new state and add it to the tree
        result = generate_fn[action](node.state)

        state = self.tell(state, trial.trial_id, result)

        return state

    def init_tree(self) -> BFSState[StateT]:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree[StateT] = Tree.with_root_node()
        return BFSState(tree)

    def get_state_score_pairs(
        self, state: BFSState[StateT]
    ) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree, excluding the root node.

        The root node is excluded because it typically represents an initial
        empty state with no score.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs for all non-root nodes
        """
        return state.tree.get_state_score_pairs()

    def ask_batch(
        self, state: BFSState[StateT], batch_size: int, actions: list[str]
    ) -> tuple[BFSState[StateT], list[Trial]]:
        """
        Get next nodes and actions to expand.
        To reflect the stateless design of the Algorithm class, it returns AlgoState as well.
        """
        # If queue is empty, select next node and list expansion candidates
        if state.trial_store.is_queue_empty():
            if not state.leaves:
                # If no leaves exist, use the root node
                parent = state.tree.root
            else:
                # Otherwise, pop the highest priority node from the heap
                parent = heappop(state.leaves).node

            # Queue up nodes for expansion
            # For each selected node, we generate num_samples * len(actions) new nodes
            nodes_and_actions = []
            for action in actions:
                for _ in range(self.num_samples):
                    nodes_and_actions.append((parent, action))

            state.trial_store.fill_nodes_queue(nodes_and_actions)

        trials = state.trial_store.get_batch_from_queue(batch_size)
        return state, trials

    def tell(
        self,
        state: BFSState[StateT],
        trial_id: TrialId,
        result: tuple[StateT, float],
    ) -> BFSState[StateT]:
        """
        Reflect generate_fn result to an AlgoState object.
        """
        _new_state, new_score = result

        finished_trial = state.trial_store.get_finished_trial(trial_id, new_score)
        if (
            finished_trial is None
        ):  # Trial is no longer valid, so we do not reflect the result to state
            return state

        parent_node = state.tree.get_node(finished_trial.node_to_expand)

        new_node = state.tree.add_node(result, parent_node)

        # Add the new node to the priority queue
        heappush(state.leaves, BFSHeapItem(node=new_node, score=new_score))

        state.trial_store.advance_queue(finished_trial.action, parent_node)
        return state
