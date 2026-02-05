import copy
import dataclasses
from collections import defaultdict
from functools import partial
from math import log, sqrt
from typing import Dict, List, Optional, TypeVar, Union

import numpy as np
from scipy.stats import beta, invgamma  # type: ignore

from treequest.algos.tree import Node

T = TypeVar("T")


def _child_index_by_identity(parent: Node, child: Node) -> int:
    """
    Return the index of `child` in `parent.children` using identity comparison.

    NOTE: Avoid `list.index()` here because it relies on `__eq__`, and user-defined
    states may include objects (e.g., numpy arrays / torch tensors) whose equality
    returns non-bool values and can raise when coerced to `bool`.
    """
    for i, n in enumerate(parent.children):
        if n is child:
            return i
    raise ValueError(
        "Child node not found in parent.children by identity. "
        f"child.expand_idx={getattr(child, 'expand_idx', None)}, "
        f"parent.expand_idx={getattr(parent, 'expand_idx', None)}"
    )


@dataclasses.dataclass
class BetaPrior:
    """
    The default is Jeffrey's prior
    """

    a: float = 0.5
    b: float = 0.5


@dataclasses.dataclass
class GaussianPrior:
    m: float = 0
    kappa: float = 1
    nu: float = 1
    tau_square: float = 0.1


@dataclasses.dataclass
class PriorConfig:
    dist_type: str = "gaussian"  # "beta" or "gaussian"
    prior: Union[BetaPrior, GaussianPrior, Dict[str, float], None] = None

    def get_params(self) -> Dict[str, float]:
        # If no prior is provided, the default will be created
        if self.prior is None:
            if self.dist_type == "gaussian":
                default_prior: Union[GaussianPrior, BetaPrior] = GaussianPrior()
            elif self.dist_type == "beta":
                default_prior = BetaPrior()
            else:
                raise NotImplementedError(f"dist_type {self.dist_type} not supported.")

            return dataclasses.asdict(default_prior)

        if self.dist_type == "gaussian":
            if isinstance(self.prior, dict):
                return dataclasses.asdict(GaussianPrior(**self.prior))
            elif isinstance(self.prior, GaussianPrior):
                return dataclasses.asdict(self.prior)
            else:
                raise ValueError(
                    f"Invalid prior {self.prior} for Gaussian Distribution."
                )
        elif self.dist_type == "beta":
            if isinstance(self.prior, dict):
                return dataclasses.asdict(BetaPrior(**self.prior))
            elif isinstance(self.prior, BetaPrior):
                return dataclasses.asdict(self.prior)
            else:
                raise ValueError(f"Invalid prior {self.prior} for Beta Distribution.")
        else:
            raise NotImplementedError(f"Invalid dist_type {self.dist_type}")

    def set_reward_average_prior(self, reward_average_prior: float) -> None:
        prior = self.prior
        if isinstance(prior, dict):
            if self.dist_type == "gaussian":
                # Create a new GaussianPrior with the reward average
                self.prior = GaussianPrior(**prior)
                if self.prior:  # This helps type checking
                    self.prior.m = reward_average_prior
            else:
                # Create a new BetaPrior with the reward average
                self.prior = BetaPrior(**prior)
                if self.prior:  # This helps type checking
                    self.prior.a = reward_average_prior
                    self.prior.b = 1 - reward_average_prior
        elif prior is None:
            if self.dist_type == "gaussian":
                self.prior = GaussianPrior(m=reward_average_prior)
            else:
                self.prior = BetaPrior(
                    a=reward_average_prior, b=1 - reward_average_prior
                )
        elif isinstance(prior, GaussianPrior):
            prior.m = reward_average_prior
        elif isinstance(prior, BetaPrior):
            prior.a = reward_average_prior
            prior.b = 1 - reward_average_prior
        else:
            raise ValueError(f"Invalid prior {prior}.")


class ProbabilisticDist:
    def __init__(self, prior_config: Optional[PriorConfig] = None):
        if prior_config is None:
            prior_config = PriorConfig()
        self.dist_type = prior_config.dist_type

        self.all_obs: List[float] = []
        self.params = prior_config.get_params()
        self.prior_params = copy.deepcopy(self.params)

    def tell_observation(self, obs: float) -> None:
        if self.dist_type == "beta":
            assert obs >= 0 and obs <= 1
            self.params["a"] += obs
            self.params["b"] += 1 - obs
        elif self.dist_type == "gaussian":
            # See Section 3.4.3.3 of Murphy's textbook "Probabilistic Machine Learning: Advanced Topics": https://probml.github.io/pml-book/book2.html
            self.all_obs.append(obs)

            n = len(self.all_obs)
            ave = float(np.mean(self.all_obs))
            var = float(np.var(self.all_obs, ddof=0)) * n

            m = self.prior_params["m"]
            kappa = self.prior_params["kappa"]
            nu = self.prior_params["nu"]
            tau_square = self.prior_params["tau_square"]

            new_kappa = kappa + n
            new_nu = nu + n
            new_m = (kappa * m + n * ave) / new_kappa
            new_tau_square = (
                nu * tau_square + var + n * kappa * (m - ave) * (m - ave) / (kappa + n)
            ) / new_nu

            self.params = {
                "m": new_m,
                "kappa": new_kappa,
                "nu": new_nu,
                "tau_square": new_tau_square,
            }
        else:
            raise NotImplementedError()

    def draw_sample(self) -> float:
        if self.dist_type == "beta":
            return beta.rvs(self.params["a"], self.params["b"])
        elif self.dist_type == "gaussian":
            sigma_square = invgamma.rvs(
                self.params["nu"] / 2,
                scale=(self.params["nu"] * self.params["tau_square"]) / 2,
            )
            mu = np.random.normal(
                self.params["m"], np.sqrt(sigma_square / self.params["kappa"])
            )
            return mu
        else:
            raise NotImplementedError()


def build_default_prob_dist(prior_config):
    """
    A trick to dump pickle for NodeProbState.

    Module level function to avoid pickle dump error
    """
    return ProbabilisticDist(prior_config)


class NodeProbState:
    def __init__(
        self,
        actions: List[str],
        prior_config: Optional[PriorConfig] = None,
        reward_average_priors: Optional[float | Dict[str, float]] = None,
        model_selection_strategy: str = "stack",
    ):
        if prior_config is None:
            prior_config = PriorConfig()

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

        # Action-level probability distributions
        self.prior_for_actions = dict()
        self.action_probas = dict()
        for action in actions:
            prior_for_action = copy.deepcopy(prior_config)
            # Overrides prior reward average values
            if isinstance(reward_average_priors, float):
                reward_average_prior = reward_average_priors
            elif isinstance(reward_average_priors, dict):
                reward_average_prior = reward_average_priors[action]
            # In case override value is not specified, use the default prior
            else:
                self.action_probas[action] = ProbabilisticDist(prior_for_action)
                self.prior_for_actions[action] = prior_for_action
                continue

            prior_for_action.set_reward_average_prior(reward_average_prior)

            self.prior_for_actions[action] = prior_for_action
            self.action_probas[action] = ProbabilisticDist(
                self.prior_for_actions[action]
            )

        # Probability distributions for decision between generating new nodes vs continuing with existing
        if model_selection_strategy.startswith("multiarm_bandit_"):
            # For multiarm bandit strategies, use shared GEN/CONT distributions across all actions
            self.gen_vs_cont_probas = {
                "shared": {
                    "GEN": ProbabilisticDist(prior_config),
                    "CONT": ProbabilisticDist(prior_config),
                }
            }
            # Store node probabilistic distributions for each action
            self.node_probas: Dict[str, Dict[int, ProbabilisticDist]] = {
                "shared": defaultdict(partial(build_default_prob_dist, prior_config))
            }
        else:
            # For stack strategy, use separate GEN/CONT distributions for each action
            self.gen_vs_cont_probas = {
                action: {
                    "GEN": ProbabilisticDist(prior_config),
                    "CONT": ProbabilisticDist(prior_config),
                }
                for action in actions
            }
            # Store node probabilistic distributions for each action
            self.node_probas = {
                action: defaultdict(partial(build_default_prob_dist, prior_config))
                for action in actions
            }

        # Store which action was used to generate each node
        self.child_node_to_action: Dict[int, str] = {}

    def update_action_reward(self, action: str, reward: float) -> None:
        """
        Update probability distributions for an action with a new reward observation.

        Args:
            action: The action name
            reward: The reward value (score) to use for the update
        """
        # Update for new node generation
        if self.model_selection_strategy.startswith("multiarm_bandit_"):
            # For multiarm bandit strategies, update shared GEN distribution
            self.gen_vs_cont_probas["shared"]["GEN"].tell_observation(reward)
        else:
            # For stack strategy, update action-specific GEN distribution
            self.gen_vs_cont_probas[action]["GEN"].tell_observation(reward)

        # Update for the action itself
        self.action_probas[action].tell_observation(reward)

    def update_node_reward(self, node: Node, reward: float) -> None:
        """
        Update probability distributions for a node with a new reward observation.

        Args:
            node: The node to update reward history
            reward: The reward value (score) to use for the update
        """
        action = self.get_action_for_child(node)

        assert node.parent is not None

        # Update for continuing with existing nodes
        child_idx = _child_index_by_identity(node.parent, node)
        if self.model_selection_strategy.startswith("multiarm_bandit_"):
            # Update the node's probability
            self.node_probas["shared"][child_idx].tell_observation(reward)
            # For multiarm bandit strategies, update shared CONT distribution
            self.gen_vs_cont_probas["shared"]["CONT"].tell_observation(reward)
        else:
            # Update the node's probability
            self.node_probas[action][child_idx].tell_observation(reward)
            # For stack strategy, update action-specific CONT distribution
            self.gen_vs_cont_probas[action]["CONT"].tell_observation(reward)

        # Update for the action itself
        self.action_probas[action].tell_observation(reward)

    def select_next(self, all_rewards_store: Dict[str, List[float]]) -> Union[str, int]:
        """
        Use the selected strategy to determine the next action or node.

        Returns:
            Either an action name (str) for generating a new node,
            or a node index (int) for continuing with an existing node
        """
        if self.model_selection_strategy == "stack":
            return self._select_next_stack()
        elif self.model_selection_strategy == "multiarm_bandit_thompson":
            return self._select_next_multiarm_bandit(
                strategy="thompson", all_rewards_store=all_rewards_store
            )
        elif self.model_selection_strategy == "multiarm_bandit_ucb":
            return self._select_next_multiarm_bandit(
                strategy="ucb", all_rewards_store=all_rewards_store
            )
        else:
            raise ValueError(
                f"Unknown model_selection_strategy: {self.model_selection_strategy}"
            )

    def _select_next_stack(self) -> Union[str, int]:
        """
        Traditional approach: first select action, then decide GEN vs CONT.

        Returns:
            Either an action name (str) for generating a new node,
            or a node index (int) for continuing with an existing node
        """
        # 1. Sample from action probabilities to select an action
        action = self.thompson_sampling(self.action_probas)

        # 2. Decide whether to generate a new node or continue with existing
        gen_or_cont = self.thompson_sampling(self.gen_vs_cont_probas[action])

        if gen_or_cont == "GEN" or len(self.node_probas[action]) == 0:
            # Generate a new node using this action
            return action
        elif gen_or_cont == "CONT":
            # Continue with an existing node
            node_idx = self.thompson_sampling(self.node_probas[action])
            return node_idx
        else:
            raise RuntimeError(f"Internal Error! Invalid gen_or_cont {gen_or_cont}")

    def _select_best_action(
        self, strategy: str, all_rewards_store: Dict[str, List[float]]
    ) -> str:
        assert len(all_rewards_store) > 0, f"Internal Error: {all_rewards_store}"
        # For single action case, we just return that action.
        if len(all_rewards_store) == 1:
            return next(iter(all_rewards_store))

        all_len = sum([len(v) for k, v in all_rewards_store.items()])

        action_scores = dict()
        if strategy == "thompson":
            for action, scores in all_rewards_store.items():
                pd = ProbabilisticDist(self.prior_for_actions[action])
                for reward in all_rewards_store[action]:
                    pd.tell_observation(reward)
                action_scores[action] = pd.draw_sample()
        elif strategy == "ucb":
            for action, scores in all_rewards_store.items():
                if not scores:
                    action_scores[action] = float("inf")
                    continue
                ucb_score = sum(scores) / len(scores) + sqrt(2) * sqrt(
                    log(all_len) / len(scores)
                )
                action_scores[action] = ucb_score
            pass
        else:
            raise ValueError(
                f"Invalid strategy {strategy}, it should be either ucb or thompson"
            )
        return max(action_scores, key=action_scores.__getitem__)

    def _select_next_multiarm_bandit(
        self, strategy: str, all_rewards_store: Dict[str, List[float]]
    ) -> Union[str, int]:
        """
        Multi-armed bandit approach: two-step decision process.

        First, decide between GEN or CONT using Thompson sampling.
        If CONT is chosen, return the node. If GEN is chosen, use multiarm bandit to pick the best action.

        Args:
            strategy: Either "thompson" for Thompson sampling or "ucb" for Upper Confidence Bound

        Returns:
            Either an action name (str) for generating a new node,
            or a node index (int) for continuing with an existing node
        """
        # Step 1: Decide between GEN or CONT using shared distributions based on Thompson Sampling
        gen_cont_options = self.gen_vs_cont_probas["shared"]
        choice = self.thompson_sampling(gen_cont_options)

        if choice == "GEN" or len(self.node_probas["shared"]) == 0:
            # Step 2a: GEN was chosen, now pick the best action using multiarm bandit
            return self._select_best_action(
                strategy=strategy, all_rewards_store=all_rewards_store
            )
        else:  # CONT
            return self.thompson_sampling(self.node_probas["shared"])

    def thompson_sampling(self, probas: Dict[T, ProbabilisticDist]) -> T:
        """
        Perform Thompson Sampling on a dictionary of probability distributions.

        Args:
            probas: Dictionary mapping keys to probability distributions

        Returns:
            The key with the highest sampled value
        """
        max_name = None
        max_val = None
        for name in probas:
            val = probas[name].draw_sample()
            if max_val is None or val > max_val:
                max_name = name
                max_val = val
        assert max_name is not None

        return max_name

    def register_new_child_node(
        self, action: str, node: Node, model_selection_strategy: str
    ) -> None:
        """
        Register a new node in the state, tracking which action generated it.

        Args:
            action: The action that generated the node
            node: The newly created node
            model_selection_strategy: Strategy for model selection. One of:
                - "stack": Perform separate fits for each model (traditional approach)
                - "multiarm_bandit_thompson": Use Thompson Sampling for joint selection
                - "multiarm_bandit_ucb": Use UCB for joint selection
        """
        # Record which action generated this node
        self.child_node_to_action[node.expand_idx] = action

        # Initialize the node's probability distribution with its score
        assert node.parent is not None
        child_idx = _child_index_by_identity(node.parent, node)
        if model_selection_strategy == "stack":
            self.node_probas[action][child_idx].tell_observation(node.score)
        else:
            self.node_probas["shared"][child_idx].tell_observation(node.score)

    def get_action_for_child(self, child: Node) -> str:
        """
        Get the action that was used to generate a node.

        Args:
            node: The node

        Returns:
            The action name
        """
        node_expand_idx = child.expand_idx
        if node_expand_idx not in self.child_node_to_action:
            raise KeyError(
                f"No action found for node index {node_expand_idx} in {self.child_node_to_action}"
            )

        return self.child_node_to_action[node_expand_idx]
