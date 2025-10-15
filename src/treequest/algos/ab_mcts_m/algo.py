import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

from loky import ProcessPoolExecutor  # type: ignore[import-untyped]

from treequest.algos.ab_mcts_m.numpyro_utils import initialize_numpyro
from treequest.algos.ab_mcts_m.pymc_interface import (
    Observation,
    PruningConfig,
    PyMCInterface,
)
from treequest.algos.base import Algorithm
from treequest.algos.tree import Node, Tree
from treequest.trial import Trial, TrialId, TrialStore
from treequest.types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")


def _select_one_node_and_action(args):
    state, actions, algo = args

    node, action = algo.get_expand_node_and_action(state, actions)
    return node.expand_idx, action


@dataclass
class ABMCTSMState(Generic[StateT]):
    """State for ABMCTSM Algorithm."""

    tree: Tree[StateT]
    # Dictionary mapping node expand_idx to observation data
    all_observations: Dict[int, Observation] = field(
        default_factory=dict[int, Observation]
    )
    trial_store: TrialStore = field(default_factory=TrialStore)


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
        max_cpu_devices: int = 32,
        max_process_workers: int = 8,
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
            max_cpu_devices: Maximum number of CPU devices (cores) used for PyMC sampling. It controls maximum number of parallel chain sampling.
            max_process_workers: Maximum number of parallel processes used for running PyMC sampling.
        """
        self.enable_pruning = enable_pruning
        self.pruning_config = PruningConfig(
            min_subtree_size_for_pruning=min_subtree_size_for_pruning,
            same_score_proportion_threshold=same_score_proportion_threshold,
        )
        self.reward_average_priors = reward_average_priors
        self.model_selection_strategy = model_selection_strategy
        self.max_cpu_devices = max_cpu_devices
        self.max_process_workers = max_process_workers

        # Create PyMCInterface as part of the algorithm itself, not the state
        self.pymc_interface = PyMCInterface(
            enable_pruning=self.enable_pruning,
            pruning_config=self.pruning_config,
            reward_average_priors=self.reward_average_priors,
            model_selection_strategy=self.model_selection_strategy,
            max_cpu_devices=max_cpu_devices,
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
        Perform one step of the AB-MCTS-M algorithm and generate a new node.
        """
        if not inplace:
            state = copy.deepcopy(state)

        actions = list(generate_fn.keys())
        state, trial = self.ask(state, actions)

        action = trial.action
        node = state.tree.get_node(trial.node_to_expand)
        result = generate_fn[action](node.state)

        self.tell(state, trial.trial_id, result)
        return state

    def get_expand_node_and_action(
        self,
        state: ABMCTSMState[StateT],
        actions: list[str],
    ) -> tuple[Node[StateT], str]:
        # If the tree is empty (only root), expand the root
        if not state.tree.root.children:
            return state.tree.root, self._get_generation_action(
                state, state.tree.root, actions
            )

        # Run one simulation step
        node = state.tree.root

        # Selection phase: traverse tree until we reach a leaf node or need to create a new node
        while node.children:
            node, action = self._select_child(state, node, actions)

            # If action is not None, we will generate a new node from `node``
            if action is not None:
                return node, action
        action = self._get_generation_action(state, node, actions)
        return node, action

    def _select_child(
        self,
        state: ABMCTSMState[StateT],
        node: Node[StateT],
        actions: list[str],
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

        child_identifier = self.pymc_interface.run(
            observations,
            actions=actions,
            node=node,
            all_observations=list(state.all_observations.values()),
        )

        # If we got a string, we need to generate a new node
        if isinstance(child_identifier, str):
            return node, child_identifier
        else:
            # Otherwise, we return the existing child
            return node.children[child_identifier], None

    def _get_generation_action(
        self, state: ABMCTSMState[StateT], node: Node[StateT], actions: list[str]
    ) -> str:
        observations = Observation.collect_all_observations_of_descendant(
            node, state.all_observations
        )

        selected_action = self.pymc_interface.run(
            observations,
            actions=actions,
            node=node,
            all_observations=list(state.all_observations.values()),
        )

        # Ensure we get a string model name, not an index
        if not isinstance(selected_action, str):
            raise ValueError(
                f"Internal Error: Expected model name string but got index {selected_action}"
            )
        return selected_action

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

    def ask_batch(
        self, state: ABMCTSMState[StateT], batch_size: int, actions: list[str]
    ) -> tuple[ABMCTSMState[StateT], list[Trial]]:
        if batch_size <= 0:
            raise ValueError(
                f"batch_size should be equal to or more than 1, while batch_size={batch_size} is provided."
            )

        num_workers = max(1, min(batch_size, self.max_process_workers))
        per_worker_cpu_devices = max(1, self.max_cpu_devices // num_workers)

        task_args = [(state, actions, self) for _ in range(batch_size)]

        trials: list[Trial] = []
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=lambda: initialize_numpyro(per_worker_cpu_devices),
        ) as ex:
            for node_id, action in ex.map(_select_one_node_and_action, task_args):
                trials.append(state.trial_store.create_trial(node_id, action))

        return state, trials

    def tell(
        self,
        state: ABMCTSMState[StateT],
        trial_id: TrialId,
        result: tuple[StateT, float],
    ) -> ABMCTSMState[StateT]:
        _new_state, new_score = result

        finished_trial = state.trial_store.get_finished_trial(trial_id, new_score)
        if (
            finished_trial is None
        ):  # Trial is no longer valid, so we do not reflect the result to state
            return state

        parent_node = state.tree.get_node(finished_trial.node_to_expand)

        # Add new node to the tree
        new_node = state.tree.add_node(result, parent_node)

        # Record observation
        state.all_observations[new_node.expand_idx] = Observation(
            reward=new_score,
            action=finished_trial.action,
            node_expand_idx=new_node.expand_idx,
        )

        return state
