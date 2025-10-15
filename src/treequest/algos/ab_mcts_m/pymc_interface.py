import random

# Import all dependencies at the top of the file
from collections import Counter
from dataclasses import dataclass, replace
from math import log, sqrt
from typing import Dict, List, Literal, Optional, Tuple, Union

from treequest.algos.ab_mcts_m._ab_mcts_m_imports import _import
from treequest.algos.ab_mcts_m.numpyro_utils import initialize_numpyro
from treequest.algos.tree import Node

if _import.is_successful():
    from treequest.algos.ab_mcts_m._ab_mcts_m_imports import (
        DataArray,
        jax,
        np,
        pd,
        pm,
        sample_numpyro_nuts,
    )


@dataclass(frozen=True)
class Observation:
    reward: float
    action: str
    node_expand_idx: int

    # Linear model group index to be used by PyMC; It is not used or set from MCTS side
    child_idx: int = -1

    @classmethod
    def to_pandas(
        cls, observations: List["Observation"], action: Optional[str]
    ) -> pd.DataFrame:
        observations_with_id = []

        # We iterate over observation to avoid time-consuming dataclasses.asdict operation
        for idx, observation in enumerate(observations):
            if action is None or observation.action == action:
                observations_with_id.append(
                    {
                        "obs_id": idx,
                        "reward": observation.reward,
                        "child_idx": observation.child_idx,
                        "action": observation.action,
                    }
                )
        return pd.DataFrame(observations_with_id)

    @classmethod
    def collect_all_observations_of_descendant(
        cls,
        parent: Node,
        all_observations: Dict[int, "Observation"],
    ) -> List["Observation"]:
        """
        A helper method to collect all the descendant observations.
        """
        observations: List["Observation"] = []
        for child_idx, child in enumerate(parent.children):
            if child.expand_idx in all_observations:
                cls.collect_all_observations(
                    child,
                    all_observations,
                    child_idx_override=child_idx,
                    observations=observations,
                )

        return observations

    @classmethod
    def collect_all_observations(
        cls,
        parent: Node,
        all_observations: Dict[int, "Observation"],
        child_idx_override: int,
        observations: List["Observation"],
    ) -> None:
        """
        A helper method to collect all the observations (including the one of the current node) to list.
        """
        if parent.expand_idx in all_observations:
            observation = replace(
                all_observations[parent.expand_idx], child_idx=child_idx_override
            )
            observations.append(observation)

        for child_idx, child in enumerate(parent.children):
            if child.expand_idx in all_observations:
                cls.collect_all_observations(
                    child,
                    all_observations,
                    child_idx_override=child_idx_override,
                    observations=observations,
                )


@dataclass
class PruningConfig:
    # min subtree size where pruning is enabled. If it is 4, the subtree with size >=4 is amenable to pruning
    min_subtree_size_for_pruning: int = 4

    # subtree is pruned if (max number of nodes with the same score inside subtree) / (# nodes of subtree) >= same_score_proportion_threshold
    same_score_proportion_threshold: float = 0.75


def is_prunable(
    node: Node, observations: List["Observation"], pruning_config: PruningConfig
) -> bool:
    """
    Check if the subtree where the node is parent is prunable from next search.
    """
    # Get all nodes in the subtree to check if observations belong to this subtree
    subtree_node_ids = [n.expand_idx for n in node.get_subtree_nodes()]

    scores = []
    for obs in observations:
        if obs.node_expand_idx in subtree_node_ids:
            scores.append(int(round(obs.reward * 100)))

    if not scores:
        return False

    _max_element, max_count = Counter(scores).most_common(1)[0]
    if (
        max_count / len(scores) >= pruning_config.same_score_proportion_threshold
        and len(scores) >= pruning_config.min_subtree_size_for_pruning
    ):
        return True
    else:
        return False


class PyMCInterface:
    """
    We leverage PyMC to perform (1) parameter fitting and (2) prediction.

    We use the fact that we can use different statistical models for (1) and (2) to make predictions
    for unobserved group (i.e. GEN node).
    """

    called_number: int = 0

    def __init__(
        self,
        enable_pruning: bool = True,
        pruning_config: Optional[PruningConfig] = None,
        reward_average_priors: Optional[float | Dict[str, float]] = None,
        model_selection_strategy: str = "multiarm_bandit_thompson",
    ):
        _import.check()

        initialize_numpyro(num_cpu_devices=4)

        self.enable_pruning = enable_pruning
        self.pruning_config = (
            pruning_config if pruning_config is not None else PruningConfig()
        )
        self.reward_average_priors = (
            reward_average_priors if reward_average_priors is not None else dict()
        )

        # Strategy for model selection:
        # "stack": Perform separate fits for each model (traditional approach)
        # "multiarm_bandit_thompson": Use Thompson Sampling for joint selection
        # "multiarm_bandit_ucb": Use UCB for joint selection
        if model_selection_strategy not in [
            "stack",
            "multiarm_bandit_thompson",
            "multiarm_bandit_ucb",
        ]:
            raise ValueError(
                f"Invalid model_selection_strategy: {model_selection_strategy}. "
                f"Must be one of: 'stack', 'multiarm_bandit_thompson', 'multiarm_bandit_ucb'"
            )
        self.model_selection_strategy = model_selection_strategy

    def run(
        self,
        observations: List[Observation],
        actions: List[str],
        node: Node,
        all_observations: List[Observation],
    ) -> Union[str, int]:
        """
        Main entry point of PyMC bayesian mixed model fitting.
        Returns the action in case GEN Node is chosen; otherwise return child_idx

        Three strategies are supported:
        1. "stack": Perform separate fits for each model and select the highest scoring option
           (either a child node or new node generation).
        2. "multiarm_bandit_thompson": First decide whether to select a child or generate a new node
           using Thompson Sampling. If generating a new node, use Thompson Sampling across
           model scores using all observations from the tree.
        3. "multiarm_bandit_ucb": First decide whether to select a child or generate a new node
           using Upper Confidence Bound.
        """
        if self.model_selection_strategy == "stack":
            return self._run_stacked_strategy(observations, actions, node)
        elif self.model_selection_strategy == "multiarm_bandit_thompson":
            return self._run_multiarm_bandit_strategy(
                observations, actions, node, all_observations, strategy="thompson"
            )
        elif self.model_selection_strategy == "multiarm_bandit_ucb":
            return self._run_multiarm_bandit_strategy(
                observations, actions, node, all_observations, strategy="ucb"
            )
        else:
            raise ValueError(
                f"Unknown model_selection_strategy: {self.model_selection_strategy}"
            )

    def _run_stacked_strategy(
        self, observations: List[Observation], actions: List[str], node: Node
    ) -> Union[str, int]:
        """
        Stacked strategy: calculate scores for each action separately
        """
        scores: Dict[Union[str, int, None], float] = dict()
        for action in actions:
            scores.update(self.calculate_score(observations, action))

        sorted_scores = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        for identifier in sorted_scores:
            if isinstance(identifier, str):
                return identifier
            else:
                if identifier is None:
                    raise RuntimeError("Internal Error: identifier should be integer")
                child_node = node.children[identifier]
                if self.enable_pruning and is_prunable(
                    child_node, observations, self.pruning_config
                ):
                    continue
                return identifier

        raise RuntimeError(
            f"Internal Error: Failed to get best option from {sorted_scores}"
        )

    def _select_best_action(
        self,
        all_observations: List[Observation],
        actions: List[str],
        strategy: Literal["thompson", "ucb"],
    ) -> str:
        """
        Select the best action using Thompson Sampling or UCB across all observations.
        This is used when we've decided to generate a new node.

        Args:
            all_observations: List of all observations from the tree
            actions: List of possible actions

        Returns:
            The best action to use for generating a new node
        """

        if strategy == "thompson":
            if len(all_observations) == 0:
                return random.choice(actions)

            observed_actions, _rewards, _coords = (
                self.preprocess_observations_for_multiarm_bandit(all_observations)
            )
            fitting_model = self._build_model_for_multiarm_bandit(
                all_observations, is_prediction_model=False
            )
            pred_model = self._build_model_for_multiarm_bandit(
                all_observations, is_prediction_model=True
            )

            # We use numpyro for sampling; It may use some amount of CPU resource
            with fitting_model:
                model_trace = sample_numpyro_nuts(
                    chains=4,
                    compute_convergence_checks=False,
                    idata_kwargs=dict(log_likelihood=False),
                    progressbar=False,
                )

            # Using the model_trace which includes sampled posterior information of parameters, we predict y (reward) values for GEN node and children nodes.
            # y represents the child node reward, and y_new represents GEN node reward
            with pred_model:
                pred_model_trace = pm.sample_posterior_predictive(
                    model_trace, var_names=["y", "y_new"], progressbar=False
                )

            action_scores: Dict[str, float] = dict()
            for action in actions:
                if action in observed_actions:
                    action_scores[action] = self.get_score(
                        pred_model_trace.posterior_predictive.y.sel(action=action)
                    )
                else:
                    action_scores[action] = self.get_score(
                        pred_model_trace.posterior_predictive.y_new
                    )
        else:
            # Calculate score for each model using Thompson Sampling or UCB
            action_scores = dict()
            for action in actions:
                scores = [
                    observation.reward
                    for observation in all_observations
                    if observation.action == action
                ]
                if not scores:
                    action_scores[action] = float("inf")
                    continue
                ucb_score = sum(scores) / len(scores) + sqrt(2) * sqrt(
                    log(len(all_observations)) / len(scores)
                )
                action_scores[action] = ucb_score

        # Return the best action
        return max(action_scores, key=action_scores.__getitem__)

    def _run_multiarm_bandit_strategy(
        self,
        observations: List[Observation],
        actions: List[str],
        node: Node,
        all_observations: List[Observation],
        strategy: Literal["thompson", "ucb"],
    ) -> Union[str, int]:
        """
        New multiarm bandit strategy with Thompson Sampling or UCB.

        This is a two-step decision process:
        1. First decide between selecting an existing child or generating a new node (GEN)
           using Thompson Sampling with action=None.
        2. Only if GEN node is chosen, then decide which action to use based on
           Thompson Sampling or UCB across all observations from the tree.
        """
        # First step: decide between existing child nodes and generating a new node (GEN)
        # Calculate scores for both child nodes and GEN node with action=None
        scores = self.calculate_score(observations, action=None)

        # Determine if we should use a child node or generate a new node
        sorted_scores = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )

        # Get the highest scoring option
        if not sorted_scores:
            raise RuntimeError("Internal Error: No scores calculated")

        best_option = next(iter(sorted_scores))

        # If the best option is to generate a new node
        if (
            best_option is None
        ):  # None represents the GEN node in calculate_score when action=None
            # Second step: decide which action to use
            return self._select_best_action(
                all_observations, actions, strategy=strategy
            )
        else:
            if isinstance(best_option, str):
                raise RuntimeError(
                    f"Internal Error: best_option {best_option} should not be a str."
                )
            # Otherwise, check if this child should be pruned
            child_node = node.children[best_option]
            if self.enable_pruning and is_prunable(
                child_node, observations, self.pruning_config
            ):
                # If pruned, get the next best option
                for identifier in sorted_scores:
                    if identifier is None:  # Skip the GEN node
                        continue
                    elif isinstance(identifier, str):
                        raise RuntimeError(
                            f"Internal Error: identifier {identifier} should not be str."
                        )
                    child_node = node.children[identifier]
                    if not (
                        self.enable_pruning
                        and is_prunable(child_node, observations, self.pruning_config)
                    ):
                        return identifier

                # If all child nodes are pruned, select best action for GEN node
                return self._select_best_action(
                    all_observations, actions, strategy=strategy
                )

            # Return the child index
            return best_option

    def calculate_score(
        self, observations: List[Observation], action: Optional[str]
    ) -> Dict[Union[str, int, None], float]:
        _child_indices, _rewards, coords = self.preprocess_observations(
            observations, action=action
        )

        # In case observations for action is empty, we sample from prior predictive
        # Prior Predictive Sampling START
        if len(coords) == 0:
            prior_model = self.build_fitting_model(
                observations, action, is_prior_model=True
            )
            with prior_model:
                prior_model_trace = pm.sample_prior_predictive(var_names=["y"])

            return {action: self.get_score(prior_model_trace.prior.y)}
        # Prior Predictive Sampling END

        fitting_model = self.build_fitting_model(observations, action)
        pred_model = self.build_prediction_model(observations, action)

        # We use numpyro for sampling; It may use some amount of CPU resource
        with fitting_model:
            model_trace = sample_numpyro_nuts(
                chains=4,
                compute_convergence_checks=False,
                idata_kwargs=dict(log_likelihood=False),
                progressbar=False,
            )

        # Using the model_trace which includes sampled posterior information of parameters,
        # we predict y (reward) values for GEN node and children nodes.
        with pred_model:
            pred_model_trace = pm.sample_posterior_predictive(
                model_trace, var_names=["y", "y_new"], progressbar=False
            )

        scores: Dict[Union[str, int, None], float] = dict()
        for child_idx in coords["child_idx"]:
            scores[child_idx] = self.get_score(
                pred_model_trace.posterior_predictive.y.sel(child_idx=child_idx)
            )

        scores[action] = self.get_score(pred_model_trace.posterior_predictive.y_new)

        self.called_number += 1
        # Clear jax cache to avoid memory leak
        # numpyro sampling leads to memory leak, so we delete cached jax arrays here
        if self.called_number % 10 == 0:
            jax.clear_caches()
            for x in jax.live_arrays():
                x.delete()

        return scores

    def get_score(self, arr: DataArray) -> float:
        return np.random.choice(arr.values.flatten())

    def build_fitting_model(
        self,
        observations: List[Observation],
        action: Optional[str],
        is_prior_model: bool = False,
    ) -> pm.Model:
        """
        Build Hierarchical PyMC model for (1) parameter fitting.
        """
        return self._build_model_impl(
            observations,
            action,
            is_prediction_model=False,
            is_prior_model=is_prior_model,
        )

    def build_prediction_model(
        self,
        observations: List[Observation],
        action: Optional[str],
    ) -> pm.Model:
        """
        Build Hierarchical PyMC model for (2) prediction.
        """
        return self._build_model_impl(observations, action, is_prediction_model=True)

    def _build_model_for_multiarm_bandit(
        self,
        observations: List[Observation],
        is_prior_model: bool = False,
        is_prediction_model: bool = False,
    ) -> pm.Model:
        actions, rewards, coords = self.preprocess_observations_for_multiarm_bandit(
            observations
        )

        with pm.Model(coords=coords if not is_prior_model else None) as model:
            # Priors START
            # Overall Goodness of the model itself; mu is set to be 0.5 (50% prob of solving the problem)
            mu_alpha = pm.Normal("mu_alpha", mu=0.5, sigma=0.2)

            # expresses the strength of score fluctuation across answers
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.2)

            # expresses the strength of score fluctuation inside answers
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.3)
            # Priors END

            group_dims = "action" if not is_prior_model else None

            # We use non-centered parameterization (see https://sjster.github.io/introduction_to_computational_statistics/docs/Production/Reparameterization.html)
            z_alpha = pm.Normal("z_alpha", mu=0, sigma=1, dims=group_dims)
            alpha = mu_alpha + z_alpha * sigma_alpha

            if is_prior_model:
                y_hat = alpha
                # This value is used by sample_prior_predictive
                _ = pm.Normal("y", mu=y_hat, sigma=sigma_y)  # noqa: F841
            elif not is_prediction_model:
                y_hat = alpha[actions]
                # This observation is used for fitting
                _ = pm.Normal("y", mu=y_hat, sigma=sigma_y, observed=rewards)  # noqa: F841
            else:
                # Expected value
                # For prediction, we sample distribution of y for each action
                y_hat = alpha[list(range(len(coords["action"])))]

                _y = pm.Normal("y", mu=y_hat, sigma=sigma_y, dims="action")

                # Prediction for unseen data (i.e. GEN node)
                z_alpha_new = pm.Normal("z_alpha_new", mu=0, sigma=1)
                alpha_new = mu_alpha + z_alpha_new * sigma_alpha
                _y_new = pm.Normal("y_new", mu=alpha_new, sigma=sigma_y)

        return model

    def _build_model_impl(
        self,
        observations: List[Observation],
        action: Optional[str],
        is_prediction_model: bool,
        is_prior_model: bool = False,
    ) -> pm.Model:
        child_indices, rewards, coords = self.preprocess_observations(
            observations, action=action
        )

        with pm.Model(coords=coords if not is_prior_model else None) as model:
            # Priors START
            # Overall Goodness of the model itself; mu is set to be 0.5 (50% prob of solving the problem)
            mu_alpha = pm.Normal(
                "mu_alpha", mu=self.get_reward_average_prior(action), sigma=0.2
            )

            # expresses the strength of score fluctuation across answers
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.2)

            # expresses the strength of score fluctuation inside answers
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.3)
            # Priors END

            group_dims = "child_idx" if not is_prior_model else None
            # We use non-centered parameterization
            z_alpha = pm.Normal("z_alpha", mu=0, sigma=1, dims=group_dims)
            alpha = mu_alpha + z_alpha * sigma_alpha

            if is_prior_model:
                # Expected value
                y_hat = alpha
                # This value is used by sample_prior_predictive
                _ = pm.Normal("y", mu=y_hat, sigma=sigma_y)  # noqa: F841

            elif not is_prediction_model:
                # Expected value
                y_hat = alpha[child_indices]
                # This observation is used for fitting
                _ = pm.Normal("y", mu=y_hat, sigma=sigma_y, observed=rewards)  # noqa: F841

            else:
                # Expected value
                # For prediction, we sample distribution of y for each child_idx
                y_hat = alpha[list(range(len(coords["child_idx"])))]
                # This value is used by sample_posterior_predictive
                _ = pm.Normal("y", mu=y_hat, sigma=sigma_y, dims="child_idx")  # noqa: F841

                # Prediction for unseen data (i.e. GEN node)
                z_alpha_new = pm.Normal("z_alpha_new", mu=0, sigma=1)
                alpha_new = mu_alpha + z_alpha_new * sigma_alpha
                # This value is used by sample_posterior_predictive
                _ = pm.Normal("y_new", mu=alpha_new, sigma=sigma_y)  # noqa: F841

        return model

    def preprocess_observations_for_multiarm_bandit(
        self, observations: List[Observation]
    ) -> Tuple[List[str], List[float], Dict[str, List[int]]]:
        """
        Extract necessary information from Observation list
        """
        df = Observation.to_pandas(observations, action=None)
        if len(df) == 0:
            return [], [], dict()

        actions, mn_actions = df["action"].factorize()

        rewards = list(df["reward"].values)

        coords = {"action": list(mn_actions)}

        return actions, rewards, coords

    def preprocess_observations(
        self, observations: List[Observation], action: Optional[str]
    ) -> Tuple[List[int], List[float], Dict[str, List[int]]]:
        """
        Extract necessary information from Observation list
        """
        df = Observation.to_pandas(observations, action=action)
        if len(df) == 0:
            return [], [], dict()

        child_indices, mn_child_indices = df["child_idx"].factorize()

        rewards = list(df["reward"].values)

        coords = {"child_idx": list(mn_child_indices)}

        return child_indices, rewards, coords

    def get_reward_average_prior(self, action: Optional[str]) -> float:
        """
        Prior parameter for reward average value for each model.
        """
        default_value = 0.5
        if isinstance(self.reward_average_priors, float | int):
            return float(self.reward_average_priors)
        else:
            if action is None:
                return default_value
            else:
                return self.reward_average_priors.get(action, default_value)
