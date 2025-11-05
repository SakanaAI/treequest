import pytest

from treequest.algos.tree import Node, Tree


def test_node_creation():
    """Test the creation of Node objects with various parameters."""
    # Test root node creation (no parent, state=None, score=-1.0)
    root = Node()
    assert root.state is None
    assert root.score == -1.0
    assert root.expand_idx == -1
    assert root.parent is None
    assert root.children == []
    assert root.is_root() is True
    assert root.depth == 0

    # Test non-root node creation with valid parameters
    state = "test_state"
    score = 0.5
    expand_idx = 0
    non_root = Node(state=state, score=score, expand_idx=expand_idx, parent=root)
    assert non_root.state == state
    assert non_root.score == score
    assert non_root.expand_idx == expand_idx
    assert non_root.parent == root
    assert non_root.children == []
    assert non_root.is_root() is False
    assert non_root.depth == 1

    # Add the non-root node as a child of the root
    root.children.append(non_root)
    assert root.children == [non_root]


def test_node_validation():
    """Test that Node validation works correctly."""
    root = Node()

    # Test that non-root nodes require a state
    with pytest.raises(
        RuntimeError, match="node state for non-root node should not be None"
    ):
        Node(state=None, score=0.5, expand_idx=0, parent=root)

    # Test that non-root nodes require a score between 0 and 1
    with pytest.raises(RuntimeError, match="score value should be between 0 and 1"):
        Node(state="test", score=-0.1, expand_idx=0, parent=root)

    with pytest.raises(RuntimeError, match="score value should be between 0 and 1"):
        Node(state="test", score=1.1, expand_idx=0, parent=root)


def test_node_depth():
    """Test that node depth is calculated correctly."""
    # Create a chain of nodes
    root = Node()
    node1 = Node(state="1", score=0.1, expand_idx=0, parent=root)
    node2 = Node(state="2", score=0.2, expand_idx=1, parent=node1)
    node3 = Node(state="3", score=0.3, expand_idx=2, parent=node2)

    # Check depths
    assert root.depth == 0
    assert node1.depth == 1
    assert node2.depth == 2
    assert node3.depth == 3


def test_node_get_subtree_nodes():
    """Test that get_subtree_nodes returns all nodes in the subtree."""
    # Create a tree structure
    #       root
    #      /    \
    #    node1  node2
    #    /  \      \
    #  node3 node4  node5
    root = Node()
    node1 = Node(state="1", score=0.1, expand_idx=0, parent=root)
    node2 = Node(state="2", score=0.2, expand_idx=1, parent=root)
    node3 = Node(state="3", score=0.3, expand_idx=2, parent=node1)
    node4 = Node(state="4", score=0.4, expand_idx=3, parent=node1)
    node5 = Node(state="5", score=0.5, expand_idx=4, parent=node2)

    # Set up the tree structure
    root.children = [node1, node2]
    node1.children = [node3, node4]
    node2.children = [node5]

    # Test get_subtree_nodes from different starting points
    root_subtree = root.get_subtree_nodes()
    assert len(root_subtree) == 6
    # Check that all nodes are in the subtree (can't use set because Node is not hashable)
    assert root in root_subtree
    assert node1 in root_subtree
    assert node2 in root_subtree
    assert node3 in root_subtree
    assert node4 in root_subtree
    assert node5 in root_subtree

    node1_subtree = node1.get_subtree_nodes()
    assert len(node1_subtree) == 3
    assert node1 in node1_subtree
    assert node3 in node1_subtree
    assert node4 in node1_subtree
    assert root not in node1_subtree
    assert node2 not in node1_subtree
    assert node5 not in node1_subtree

    node2_subtree = node2.get_subtree_nodes()
    assert len(node2_subtree) == 2
    assert node2 in node2_subtree
    assert node5 in node2_subtree
    assert root not in node2_subtree
    assert node1 not in node2_subtree
    assert node3 not in node2_subtree
    assert node4 not in node2_subtree

    # Leaf nodes should return just themselves
    node3_subtree = node3.get_subtree_nodes()
    assert len(node3_subtree) == 1
    assert node3_subtree == [node3]


def test_tree_creation():
    """Test the creation of Tree objects."""
    # Test default tree creation
    tree = Tree()
    assert tree.root is not None
    assert tree.root.state is None
    assert tree.root.score == -1.0
    assert tree.root.expand_idx == -1
    assert tree.root.parent is None
    assert tree.root.children == []
    assert tree.size == 1

    # Test tree creation with with_root_node class method
    tree = Tree.with_root_node()
    assert tree.root is not None
    assert tree.root.state is None
    assert tree.root.score == -1.0
    assert tree.root.expand_idx == -1
    assert tree.root.parent is None
    assert tree.root.children == []
    assert tree.size == 1


def test_tree_add_node():
    """Test adding nodes to a tree."""
    tree = Tree()

    # Add first node
    state1 = "state1"
    score1 = 0.1
    node1 = tree.add_node((state1, score1), tree.root)

    assert node1.state == state1
    assert node1.score == score1
    assert node1.parent == tree.root
    assert node1.expand_idx == 0
    assert tree.root.children == [node1]
    assert node1.trial_id is None
    assert tree.size == 2

    # Add second node as child of first node
    state2 = "state2"
    score2 = 0.2
    node2 = tree.add_node((state2, score2), node1)

    assert node2.state == state2
    assert node2.score == score2
    assert node2.parent == node1
    assert node2.expand_idx == 1
    assert node1.children == [node2]
    assert node2.trial_id is None
    assert tree.size == 3

    # Add third node as another child of root
    state3 = "state3"
    score3 = 0.3
    node3 = tree.add_node((state3, score3), tree.root)

    assert node3.state == state3
    assert node3.score == score3
    assert node3.parent == tree.root
    assert node3.expand_idx == 2
    assert tree.root.children == [node1, node3]
    assert node3.trial_id is None
    assert tree.size == 4


def test_tree_add_node_with_trial_id():
    tree = Tree()
    trial_id = "trial-123"

    node = tree.add_node(("state", 0.5), tree.root, trial_id=trial_id)

    assert tree.root.children == [node]
    assert node.trial_id == trial_id
    assert tree.size == 2


def test_tree_get_nodes():
    """Test retrieving nodes from a tree."""
    tree = Tree()

    # Add some nodes
    node1 = tree.add_node(("state1", 0.1), tree.root)
    node2 = tree.add_node(("state2", 0.2), tree.root)
    node3 = tree.add_node(("state3", 0.3), node1)

    # Test get_nodes
    nodes = tree.get_nodes()
    assert len(nodes) == 4
    assert nodes[0] == tree.root
    assert nodes[1] == node1
    assert nodes[2] == node2
    assert nodes[3] == node3

    # Test __len__
    assert len(tree) == 4

    # Test get_node by expand_idx
    assert tree.get_node(0) == node1
    assert tree.get_node(1) == node2
    assert tree.get_node(2) == node3


def test_tree_get_state_score_pairs():
    """Test retrieving state-score pairs from a tree."""
    tree = Tree()

    # Add some nodes with different states and scores
    state1, score1 = "state1", 0.1
    state2, score2 = "state2", 0.2
    state3, score3 = "state3", 0.3

    tree.add_node((state1, score1), tree.root)
    tree.add_node((state2, score2), tree.root)
    tree.add_node((state3, score3), tree.root)

    # Test get_state_score_pairs
    state_score_pairs = tree.get_state_score_pairs()
    assert len(state_score_pairs) == 3
    assert (state1, score1) in state_score_pairs
    assert (state2, score2) in state_score_pairs
    assert (state3, score3) in state_score_pairs
