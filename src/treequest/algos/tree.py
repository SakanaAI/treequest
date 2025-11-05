import dataclasses
from typing import Generic, List, Optional, TypeVar

from treequest.types import StateScoreType, TrialId

# Type variable for state
StateT = TypeVar("StateT")


@dataclasses.dataclass
class Node(Generic[StateT]):
    state: Optional[StateT] = None

    # Scores should be in the range from 0 to 1. For root, we set it to be -1.0 as a placeholder.
    score: float = -1.0

    # Reflects the order of expansion. The root has expand_idx = -1, followed by 0,1,2,...
    expand_idx: int = -1

    parent: Optional["Node[StateT]"] = None
    children: List["Node[StateT]"] = dataclasses.field(default_factory=list)
    trial_id: Optional[TrialId] = None

    def __post_init__(self) -> None:
        if not self.is_root():
            if not (self.score <= 1.0 and self.score >= 0.0):
                raise RuntimeError(
                    f"The score value should be between 0 and 1, while {self.score} is set."
                )

            if self.state is None:
                raise RuntimeError(
                    "The node state for non-root node should not be None."
                )

    def get_subtree_nodes(self) -> List["Node[StateT]"]:
        """
        Get all the nodes of the subtree starting from self, including self itself.
        """
        nodes: List["Node[StateT]"] = []
        stack: List["Node[StateT]"] = [self]
        while stack:
            current = stack.pop()
            nodes.append(current)
            stack.extend(current.children)
        return nodes

    def is_root(self) -> bool:
        return self.parent is None

    @property
    def depth(self) -> int:
        depth = 0
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent
            depth += 1
        return depth


@dataclasses.dataclass
class Tree(Generic[StateT]):
    """
    Utility data structure for handling tree.
    """

    root: Node[StateT]
    size: int

    def __init__(self):
        self.root = Node[StateT]()
        self.size = 1

    @classmethod
    def with_root_node(cls) -> "Tree[StateT]":
        """
        Create a new tree with a properly initialized root node.

        The root node follows the convention of having:
        - state: None
        - score: -1.0
        - expand_idx: -1

        Returns:
            A new Tree instance with the root node initialized
        """
        return cls()

    def __len__(self) -> int:
        return self.size

    def get_nodes(self) -> List[Node[StateT]]:
        """
        Returns the node in ascending order of their expand_idx.
        """
        nodes = self.root.get_subtree_nodes()
        nodes.sort(key=lambda x: x.expand_idx)
        return nodes

    def get_node(self, expand_idx: int) -> Node[StateT]:
        """
        Returns the node with the given expand_idx.
        """
        # The nodes are sorted by expand_idx, so we can just index into the list.
        # expand_idx starts from -1 for the root, so we need to add 1 to get the correct index.
        return self.get_nodes()[expand_idx + 1]

    def get_state_score_pairs(self) -> List[StateScoreType[StateT]]:
        return [
            (node.state, node.score)
            for node in self.get_nodes()
            if (not node.is_root()) and (node.state is not None)
        ]

    def add_node(
        self,
        state_score: StateScoreType[StateT],
        parent: Node[StateT],
        trial_id: Optional[TrialId] = None,
    ) -> Node[StateT]:
        state, score = state_score
        node = Node(
            state=state,
            score=score,
            parent=parent,
            expand_idx=self.size - 1,
            trial_id=trial_id,
        )
        parent.children.append(node)
        self.size += 1
        return node
