# TreeQuest MCP Server

A Model Context Protocol (MCP) server for TreeQuest tree search algorithms.

## Installation

Install the TreeQuest package with MCP server dependencies:

```bash
pip install -e ".[mcp-server]"
```

## Usage

### Command Line Interface

Start the MCP server:

```bash
# Using stdio transport (default)
treequest-mcp-server

# Using SSE transport on port 8000
treequest-mcp-server --transport sse --port 8000
```

### Available Tools

#### `init_tree`
Initialize a new tree search session with specified algorithm.

**Input:**
- `algorithm`: Algorithm type (`"StandardMCTS"`, `"ABMCTSA"`, `"ABMCTSM"`)
- `params` (optional): Algorithm parameters
  - `exploration_weight`: Exploration weight for UCT (default: 1.0)
  - `samples_per_action`: Number of samples per action (default: 1)

**Output:** Session ID and initialization status

#### `step_tree`
Perform one step of tree search using provided generate functions.

**Input:**
- `session_id`: Session ID from init_tree
- `generate_functions`: Map of action names to Python code defining `generate_fn`

**Output:** Step statistics including node counts and tree size

#### `get_tree_state`
Extract current tree state and statistics.

**Input:**
- `session_id`: Session ID

**Output:** Tree state with node information and state-score pairs

#### `rank_nodes`
Get top-k nodes using TreeQuest's ranking functionality.

**Input:**
- `session_id`: Session ID
- `k` (optional): Number of top nodes to return (default: 10)

**Output:** Top-k ranked nodes with states and scores

#### `list_sessions`
List all active tree search sessions.

**Output:** List of active sessions with metadata

#### `delete_session`
Clean up a tree search session.

**Input:**
- `session_id`: Session ID to delete

**Output:** Deletion confirmation

## Example Usage

```python
# Initialize a StandardMCTS session
{
  "algorithm": "StandardMCTS",
  "params": {
    "exploration_weight": 1.4,
    "samples_per_action": 2
  }
}

# Step the tree with generate functions
{
  "session_id": "your-session-id",
  "generate_functions": {
    "action1": "def generate_fn(state): return ['option1', 'option2']"
  }
}
```

## Transport Modes

- **stdio**: Standard input/output for direct MCP client integration
- **sse**: Server-Sent Events over HTTP for web-based clients
