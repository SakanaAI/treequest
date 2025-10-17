import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, log, sqrt
from typing import Dict, Generic, List, Optional, TypeVar

from treequest.algos.base import Algorithm
from treequest.algos.tree import Node, Tree
from treequest.trial import Trial, TrialId, TrialStoreWithNodeQueue
from treequest.types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")


def softmax(values: List[float]) -> List[float]:
    """
    Compute softmax values for a list of scores.

    Args:
        values: List of scores

    Returns:
        List of softmax probabilities
    """
    # Shift values for numerical stability (prevent overflow)
    shifted = [x - max(values) for x in values]
    exp_values = [exp(x) for x in shifted]
    sum_exp = sum(exp_values)
    return [x / sum_exp for x in exp_values]


@dataclass
class MCTSState(Generic[StateT]):
    """State for Monte Carlo Tree Search algorithm."""

    tree: Tree[StateT]
    visit_counts: Dict[int, int] = field(default_factory=dict[int, int])
    value_sums: Dict[int, float] = field(default_factory=dict[int, float])
    priors: Dict[int, float] = field(default_factory=dict[int, float])

    trial_store: TrialStoreWithNodeQueue[StateT] = field(
        default_factory=TrialStoreWithNodeQueue[StateT]
    )


class StandardMCTS(Algorithm[StateT, MCTSState[StateT]]):
    """
    Standard Monte Carlo Tree Search (MCTS) algorithm with UCT scoring.

    This implementation uses the Upper Confidence Bound for Trees (UCT)
    formula to balance exploration and exploitation.
    """

    def __init__(
        self, *, samples_per_action: int = 2, exploration_weight: float = sqrt(2)
    ):
        """
        Initialize the MCTS algorithm.

        Args:
            samples_per_action: Number of samples to generate for each action
            exploration_weight: Weight for the exploration term in UCT formula
        """
        self.samples_per_action = samples_per_action
        self.exploration_weight = exploration_weight

    def step(
        self,
        state: MCTSState[StateT],
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> MCTSState[StateT]:
        """
        Perform one step of the MCTS algorithm.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        state, trial = self.ask(state, actions=list(generate_fn.keys()))

        action = trial.action
        node = state.tree.get_node(trial.node_to_expand)

        # Simulation: Generate a new state using the selected action
        result = generate_fn[action](node.state)

        state = self.tell(state, trial.trial_id, result)

        return state

    def ask_batch(
        self, state: MCTSState[StateT], batch_size: int, actions: list[str]
    ) -> tuple[MCTSState[StateT], list[Trial]]:
        # If queue is empty, select next node and list expansion candidates
        if state.trial_store.is_queue_empty():
            # Selection: Find the most promising node to expand
            node = self._select(state)

            nodes_and_actions = []
            # For each sample, add all actions
            for _ in range(self.samples_per_action):
                for action in actions:
                    nodes_and_actions.append((node, action))

            state.trial_store.fill_nodes_queue(nodes_and_actions)

        trials = state.trial_store.get_batch_from_queue(batch_size)
        return state, trials

    def tell(
        self, state: MCTSState[StateT], trial_id: TrialId, result: tuple[StateT, float]
    ) -> MCTSState[StateT]:
        _new_state, new_score = result

        finished_trial = state.trial_store.get_finished_trial(trial_id, new_score)
        if (
            finished_trial is None
        ):  # Trial is no longer valid, so we do not reflect the result to state
            return state

        parent_node = state.tree.get_node(finished_trial.node_to_expand)
        # Add the new node to the tree
        new_node = state.tree.add_node(result, parent_node)

        # Backpropagation updates visit_counts and value_sums for the new node and all ancestors.
        self._backpropagate(state, new_node, new_score)

        # Update priors if this node has siblings
        parent = new_node.parent
        if parent and len(parent.children) > 1:
            self._update_priors(state, parent)

        state.trial_store.advance_queue(finished_trial.action, parent_node)
        return state

    def _update_priors(self, state: MCTSState[StateT], parent: Node[StateT]) -> None:
        """
        Update prior probabilities for all children of a node using softmax.

        Args:
            state: Current algorithm state
            parent: Parent node whose children's priors will be updated
        """
        children = parent.children
        scores = [child.score for child in children]
        priors = softmax(scores)

        for child, prior in zip(children, priors):
            state.priors[child.expand_idx] = prior

    def _select(self, state: MCTSState[StateT]) -> Node[StateT]:
        """
        Select a node to expand using UCT.

        Starts from the root and selects child nodes with highest UCT score
        until reaching a leaf node or a node with unexplored actions.

        Args:
            state: Current algorithm state

        Returns:
            Selected node
        """
        node = state.tree.root

        # If the tree is empty, return the root
        if not node.children:
            return node

        # Traverse down the tree selecting best child according to UCT
        while node.children:
            # We're selecting based on the UCT score, which balances exploration and exploitation.
            best_child = max(
                node.children, key=lambda child: self._uct_score(state, child, node)
            )
            node = best_child

        return node

    def _uct_score(
        self, state: MCTSState[StateT], node: Node[StateT], parent: Node[StateT]
    ) -> float:
        """
        Calculate the UCT score for a node.

        UCT = prior * average_value + exploration_weight * sqrt(log(parent_visits) / node_visits)

        Args:
            state: Current algorithm state
            node: Node to calculate score for
            parent: Parent node

        Returns:
            UCT score
        """
        # Get visit counts
        parent_visits = state.visit_counts.get(parent.expand_idx, 1)
        node_visits = state.visit_counts.get(node.expand_idx, 1)

        # Get value sum
        value_sum = state.value_sums.get(node.expand_idx, 0)

        # Calculate exploitation term
        exploitation = value_sum / node_visits

        # Get prior (default to 1.0 if not set)
        prior = state.priors.get(node.expand_idx, 1.0)

        # Calculate exploration term (weighted by prior)
        exploration = (
            self.exploration_weight * prior * sqrt(log(parent_visits) / node_visits)
        )

        return exploitation + exploration

    def _backpropagate(
        self, state: MCTSState[StateT], node: Node[StateT], score: float
    ) -> None:
        """
        Update statistics for all nodes in the path from node to root.

        Args:
            state: Current algorithm state
            node: Leaf node to start backpropagation from
            score: Score to backpropagate
        """
        current: Optional[Node[StateT]] = node
        while current is not None:
            node_id = current.expand_idx
            state.visit_counts[node_id] = state.visit_counts.get(node_id, 0) + 1
            state.value_sums[node_id] = state.value_sums.get(node_id, 0) + score
            current = current.parent

    def init_tree(self) -> MCTSState[StateT]:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree[StateT] = Tree.with_root_node()
        return MCTSState(tree=tree)

    def get_state_score_pairs(
        self, state: MCTSState[StateT]
    ) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()
