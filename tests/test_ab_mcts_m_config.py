import pytest

from treequest import ABMCTSM
from treequest.algos.ab_mcts_m.algo import ABMCTSMAdvancedConfig


def test_abmctsm_reward_range_validation_default_on():
    algo = ABMCTSM(enable_pruning=False)
    state = algo.init_tree()

    state, trial = algo.ask(state, actions=["A"])
    with pytest.raises(RuntimeError):
        algo.tell(state, trial.trial_id, ("s", 2.0))


def test_abmctsm_reward_range_validation_can_be_disabled():
    algo = ABMCTSM(
        enable_pruning=False,
        _advanced_config=ABMCTSMAdvancedConfig(validate_reward_range=False),
    )
    state = algo.init_tree()

    state, trial = algo.ask(state, actions=["A"])
    state = algo.tell(state, trial.trial_id, ("s", 2.0))

    assert state.tree.get_node(0).score == 2.0
    assert len(state.all_observations) == 1


def test_abmctsm_prior_hyperparameters_are_configurable():
    algo = ABMCTSM(
        enable_pruning=False,
        _advanced_config=ABMCTSMAdvancedConfig(
            prior_mu_alpha_sigma=0.9,
            prior_sigma_alpha_sigma=0.8,
            prior_sigma_y_sigma=0.7,
        ),
    )

    assert algo.pymc_interface.prior_mu_alpha_sigma == 0.9
    assert algo.pymc_interface.prior_sigma_alpha_sigma == 0.8
    assert algo.pymc_interface.prior_sigma_y_sigma == 0.7


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"prior_mu_alpha_sigma": 0}, "prior_mu_alpha_sigma must be > 0"),
        ({"prior_sigma_alpha_sigma": -1}, "prior_sigma_alpha_sigma must be > 0"),
        ({"prior_sigma_y_sigma": 0}, "prior_sigma_y_sigma must be > 0"),
    ],
)
def test_abmctsm_prior_hyperparameters_must_be_positive(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        ABMCTSMAdvancedConfig(**kwargs)
