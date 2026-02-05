import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from treequest.algos.ab_mcts_a.algo import ABMCTSA, ABMCTSAAlgoState
from treequest.algos.ab_mcts_a.prob_state import PriorConfig
from treequest.trial import Trial


def test_basic_initialization():
    """Test basic initialization of ABMCTSA."""
    # Test with default parameters
    algo = ABMCTSA()
    assert algo.prior_config is not None
    assert algo.reward_average_priors is None
    assert (
        algo.model_selection_strategy == "multiarm_bandit_thompson"
    )  # Default strategy

    # Test with custom parameters
    prior_config = PriorConfig(dist_type="beta")
    algo = ABMCTSA(prior_config=prior_config, reward_average_priors=0.7)
    assert algo.prior_config == prior_config
    assert algo.reward_average_priors == 0.7
    assert (
        algo.model_selection_strategy == "multiarm_bandit_thompson"
    )  # Default strategy

    # Test with custom model selection strategy
    algo = ABMCTSA(model_selection_strategy="multiarm_bandit_ucb")
    assert algo.model_selection_strategy == "multiarm_bandit_ucb"

    # Test initial state
    state = algo.init_tree()
    assert state.tree is not None
    assert len(state.thompson_states) == 0
    assert state.tree.root is not None
    assert state.tree.root.is_root()


def test_single_step():
    """Test a single step of the ABMCTSA."""
    # Set a fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create a simple generate function
    def generate_fn(state: Optional[str]) -> Tuple[str, float]:
        return f"Generated from test: {random.random()}", random.random()

    # Initialize algorithm and state
    algo = ABMCTSA(model_selection_strategy="stack")
    state = algo.init_tree()

    # Create a mapping with a single action
    generate_fns = {"test_action": generate_fn}

    # Run a single step
    new_state = algo.step(state, generate_fns)

    # Check that a new node was created
    assert len(new_state.tree.get_state_score_pairs()) == 1
    assert len(new_state.thompson_states) == 1

    # Check that root node has one child
    assert len(new_state.tree.root.children) == 1

    # The first child should be expanded from the root
    first_child = new_state.tree.root.children[0]
    assert first_child.parent == new_state.tree.root
    assert isinstance(first_child.state, str)
    assert "Generated from test" in first_child.state


def test_thompson_sampling_algo_with_mock():
    """Test the ABMCTSA using a mocked version to always select action_a."""

    # Create a custom version of ABMCTSA
    class MockABMCTSA(ABMCTSA):
        def ask_batch(
            self, state: ABMCTSAAlgoState, batch_size: int, actions: list[str]
        ) -> tuple[ABMCTSAAlgoState, list[Trial]]:
            # initialize all_rewards_store
            if len(state.all_rewards_store) == 0:
                for a in actions:
                    state.all_rewards_store[a] = []

            trials: list[Trial] = []
            for _ in range(batch_size):
                node, _action = self._get_expand_node_and_action(state, actions)
                trials.append(state.trial_store.create_trial(node, "action_a"))

            return state, trials

    # Create the mock algorithm
    mock_algo = MockABMCTSA()
    state = mock_algo.init_tree()

    # Create generate functions
    def generate_fn_a(state):
        return f"State A: {random.random()}", 0.8

    def generate_fn_b(state):
        return f"State B: {random.random()}", 0.4

    generate_fns = {"action_a": generate_fn_a, "action_b": generate_fn_b}

    # Run multiple steps
    n_steps = 3
    for _ in range(n_steps):
        state = mock_algo.step(state, generate_fns)

    # Check we have the expected number of nodes
    assert len(state.tree.get_state_score_pairs()) == n_steps

    # Check that all nodes were created with action_a
    for node in state.tree.get_nodes():
        if not node.is_root() and node.state is not None:
            assert "State A:" in str(node.state)


def test_model_selection_strategies():
    """Test different model selection strategies."""
    # Test stack strategy (default)
    algo_stack = ABMCTSA(model_selection_strategy="stack")
    assert algo_stack.model_selection_strategy == "stack"
    state_stack = algo_stack.init_tree()
    assert state_stack.thompson_states.default_model_selection_strategy == "stack"

    # Test multiarm_bandit_thompson strategy
    algo_thompson = ABMCTSA(model_selection_strategy="multiarm_bandit_thompson")
    assert algo_thompson.model_selection_strategy == "multiarm_bandit_thompson"
    state_thompson = algo_thompson.init_tree()
    assert (
        state_thompson.thompson_states.default_model_selection_strategy
        == "multiarm_bandit_thompson"
    )

    # Test multiarm_bandit_ucb strategy
    algo_ucb = ABMCTSA(model_selection_strategy="multiarm_bandit_ucb")
    assert algo_ucb.model_selection_strategy == "multiarm_bandit_ucb"
    state_ucb = algo_ucb.init_tree()
    assert (
        state_ucb.thompson_states.default_model_selection_strategy
        == "multiarm_bandit_ucb"
    )

    # Test invalid strategy
    try:
        ABMCTSA(model_selection_strategy="invalid_strategy")
        assert False, "Should have raised ValueError for invalid strategy"
    except ValueError:
        pass  # Expected behavior


def test_dataclass_state_with_array_field_does_not_raise():
    """
    Regression test for states containing array/tensor-like fields.

    dataclass-generated __eq__ compares fields with `==`, and objects like numpy arrays
    (and torch tensors) return non-bool values from equality, which can raise when
    coerced to bool. ABMCTSA must not rely on such equality when indexing children.
    """

    @dataclass
    class ArrayState:
        x: np.ndarray

    class DeterministicABMCTSA(ABMCTSA):
        def _get_expand_node_and_action(self, state, actions):  # type: ignore[override]
            if not state.tree.root.children:
                node = state.tree.root
            else:
                node = state.tree.root.children[0]

            action = self._get_generation_action(state, node, actions)
            return node, action

    def generate_fn(_state: Optional[ArrayState]) -> Tuple[ArrayState, float]:
        return ArrayState(x=np.array([1.0, 2.0])), 0.5

    algo = DeterministicABMCTSA(model_selection_strategy="stack")
    state = algo.init_tree()

    generate_fns = {"a": generate_fn}

    state = algo.step(state, generate_fns)
    state = algo.step(state, generate_fns)

    assert len(state.tree.get_state_score_pairs()) == 2
    assert len(state.tree.root.children) == 1
    assert len(state.tree.root.children[0].children) == 1
    assert state.tree.root.children[0].parent is state.tree.root
    assert state.tree.root.children[0].children[0].parent is state.tree.root.children[0]
