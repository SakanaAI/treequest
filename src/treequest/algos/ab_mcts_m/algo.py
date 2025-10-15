import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

from treequest.algos.ab_mcts_m.pymc_interface import (
    Observation,
    PruningConfig,
    PyMCInterface,
)
from treequest.algos.base import Algorithm
from treequest.algos.tree import Node, Tree
from treequest.types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")


@dataclass
class ABMCTSMState(Generic[StateT]):
    """State for ABMCTSM Algorithm."""

    tree: Tree[StateT]
    # Dictionary mapping node expand_idx to observation data
    all_observations: Dict[int, Observation] = field(
        default_factory=dict[int, Observation]
    )


class ABMCTSM(Algorithm[StateT, ABMCTSMState[StateT]]):
    """
    Monte Carlo Tree Search algorithm using AB-MCTS-M algorithm.

    This algorithm leverages Bayesian mixed model leveraging PyMC library.
    """

    def __init__(
        self,
        *,
        enable_pruning: bool = True,
        reward_average_priors: Optional[Union[float, Dict[str, float]]] = None,
        model_selection_strategy: str = "multiarm_bandit_thompson",
        min_subtree_size_for_pruning: int = 4,
        same_score_proportion_threshold: float = 0.75,
    ):
        """
        Initialize the AB-MCTS-M algorithm.

        Args:
            enable_pruning: Whether to enable pruning of subtrees
            reward_average_priors: Prior values for reward averages (either a single float or
                                  a dict mapping model names to prior values)
            model_selection_strategy: Strategy for model selection:
                                      "stack": Perform separate fits for each model (traditional approach)
                                      "multiarm_bandit_thompson": Use Thompson Sampling for joint selection
                                      "multiarm_bandit_ucb": Use UCB for joint selection
            min_subtree_size_for_pruning: Pruning Config, see PruningConfig class.
            same_score_proportion_threshold: Pruning Config, see PruningConfig class.
        """
        self.enable_pruning = enable_pruning
        self.pruning_config = PruningConfig(
            min_subtree_size_for_pruning=min_subtree_size_for_pruning,
            same_score_proportion_threshold=same_score_proportion_threshold,
        )
        self.reward_average_priors = reward_average_priors
        self.model_selection_strategy = model_selection_strategy

        # Create PyMCInterface as part of the algorithm itself, not the state
        self.pymc_interface = PyMCInterface(
            enable_pruning=self.enable_pruning,
            pruning_config=self.pruning_config,
            reward_average_priors=self.reward_average_priors,
            model_selection_strategy=self.model_selection_strategy,
        )

    def init_tree(self) -> ABMCTSMState[StateT]:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree[StateT] = Tree.with_root_node()

        return ABMCTSMState(tree=tree)

    def step(
        self,
        state: ABMCTSMState[StateT],
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> ABMCTSMState[StateT]:
        """
        Perform one step of AB-MCTS-M algorithm and generate a one node.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        # If the tree is empty (only root), expand the root
        if not state.tree.root.children:
            self._expand_node(state, state.tree.root, generate_fn)
            return state

        # Run one simulation step
        node = state.tree.root

        # Selection phase: traverse tree until we reach a leaf node
        while node.children:
            node, action = self._select_child(state, node, generate_fn)

            # If action is not None, it means we've generated a new node
            if action is not None:
                return state

        # Expansion phase: expand leaf node
        self._expand_node(state, node, generate_fn)

        return state

    def _select_child(
        self,
        state: ABMCTSMState[StateT],
        node: Node[StateT],
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node[StateT], Optional[str]]:
        """
        Select a child node using PyMC interface.

        Args:
            state: Current algorithm state
            node: Node to select child from
            generate_fn: Mapping of action names to generation functions

        Returns:
            Tuple of (selected node, action if new node was generated)
        """
        observations = Observation.collect_all_observations_of_descendant(
            node, state.all_observations
        )

        actions = list(generate_fn.keys())

        child_identifier = self.pymc_interface.run(
            observations,
            actions=actions,
            node=node,
            all_observations=list(state.all_observations.values()),
        )

        # If we got a string, we need to generate a new node
        if isinstance(child_identifier, str):
            new_node = self._generate_new_child(
                state, node, generate_fn, child_identifier
            )
            return new_node, child_identifier
        else:
            # Otherwise, we return the existing child
            return node.children[child_identifier], None

    def _expand_node(
        self,
        state: ABMCTSMState[StateT],
        node: Node[StateT],
        generate_fn: Mapping[str, GenerateFnType[StateT]],
    ) -> Tuple[Node[StateT], str]:
        """
        Expand a leaf node by generating a new child.

        Args:
            state: Current algorithm state
            node: Node to expand
            generate_fn: Mapping of action names to generation functions

        Returns:
            Tuple of (new node, model name used)
        """
        observations = Observation.collect_all_observations_of_descendant(
            node, state.all_observations
        )

        actions = list(generate_fn.keys())

        node_identifier = self.pymc_interface.run(
            observations,
            actions=actions,
            node=node,
            all_observations=list(state.all_observations.values()),
        )

        # Ensure we get a string model name, not an index
        if not isinstance(node_identifier, str):
            raise ValueError(
                f"Internal Error: Expected model name string but got index {node_identifier}"
            )

        new_node = self._generate_new_child(state, node, generate_fn, node_identifier)
        return new_node, node_identifier

    def _generate_new_child(
        self,
        state: ABMCTSMState[StateT],
        node: Node[StateT],
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        action: str,
    ) -> Node[StateT]:
        """
        Generate a new child node using the specified model.

        Args:
            state: Current algorithm state
            node: Parent node
            generate_fn: Mapping of action names to generation functions
            action: Name of action to use for generation

        Returns:
            Newly created node
        """
        # Generate new state and score using the selected model
        new_state, new_score = generate_fn[action](node.state)

        # Add new node to the tree
        new_node = state.tree.add_node((new_state, new_score), node)

        # Record observation
        state.all_observations[new_node.expand_idx] = Observation(
            reward=new_score, action=action, node_expand_idx=new_node.expand_idx
        )

        return new_node

    def get_state_score_pairs(
        self, state: ABMCTSMState[StateT]
    ) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()
