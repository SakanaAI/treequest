import abc
from collections.abc import Mapping
from typing import Generic, List, TypeVar

from treequest.trial import Trial, TrialId
from treequest.types import GenerateFnType, StateScoreType

# Type variables for node state and algorithm state
NodeStateT = TypeVar("NodeStateT")
AlgoStateT = TypeVar("AlgoStateT")


class Algorithm(Generic[NodeStateT, AlgoStateT], abc.ABC):
    """
    Algorithm base class for tree search.

    The Algorithm object itself should be stateless, other than the algorithm configuration which should be specified at object instantiation time.
    The state should be maintained and saved by the caller of `step` function.
    """

    @abc.abstractmethod
    def step(
        self,
        state: AlgoStateT,
        generate_fn: Mapping[str, GenerateFnType[NodeStateT]],
        inplace: bool = False,
    ) -> AlgoStateT:
        """
        Generate one additional node and add that to a given state.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def init_tree(self) -> AlgoStateT:
        """
        Initialize the AlgoState, e.g. creating the root-only tree etc.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_state_score_pairs(
        self, state: AlgoStateT
    ) -> List[StateScoreType[NodeStateT]]:
        """
        Get all the state-score pairs of the tree.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def ask_batch(
        self, state: AlgoStateT, batch_size: int, actions: list[str]
    ) -> tuple[AlgoStateT, list[Trial]]:
        """
        Get next nodes and actions to expand.
        To reflect the stateless design of the Algorithm class, it returns AlgoState as well.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def tell(
        self, state: AlgoStateT, trial_id: TrialId, result: tuple[NodeStateT, float]
    ) -> AlgoStateT:
        """
        Reflect generate_fn result to an AlgoState object.
        """
        raise NotImplementedError()

    def ask(self, state: AlgoStateT, actions: list[str]) -> tuple[AlgoStateT, Trial]:
        """
        Get next node and action to expand.
        """
        state, trial = self.ask_batch(state, batch_size=1, actions=actions)
        return state, trial[0]
