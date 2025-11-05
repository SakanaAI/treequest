from collections.abc import Callable
from typing import List, Optional, Tuple, TypeAlias, TypeVar

# Type alias for NodeId and TrialId
NodeId: TypeAlias = int
TrialId: TypeAlias = str

# Type variable for node state
NodeStateT = TypeVar("NodeStateT")

# Type of state-score tuple
StateScoreType = Tuple[NodeStateT, float]

# Type of `generate_fn`, which generates the child node given the state of parent node.
# In case the parent node is root, None is given as an argument.
GenerateFnType = Callable[[Optional[NodeStateT]], Tuple[NodeStateT, float]]

# Type for ranking function
RankingFnType = Callable[
    [List[StateScoreType[NodeStateT]]], List[StateScoreType[NodeStateT]]
]
