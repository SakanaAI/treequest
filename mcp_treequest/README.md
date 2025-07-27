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

#### `get_tree_visualization`
Generate tree visualization using Graphviz.

**Input:**
- `session_id`: Session ID
- `format` (optional): Output format (`"png"`, `"pdf"`, `"svg"`, `"dot"`) (default: "png")
- `show_scores` (optional): Whether to show scores in node labels (default: true)
- `max_label_length` (optional): Maximum length for node labels (default: 20)
- `title` (optional): Optional title for the visualization

**Output:** Visualization data with DOT source code and metadata

## Example Usage

### Basic Workflow

1. **Initialize a tree search session:**
```json
{
  "algorithm": "StandardMCTS",
  "params": {
    "exploration_weight": 1.4,
    "samples_per_action": 2
  }
}
```

2. **Step the tree with generate functions:**
```json
{
  "session_id": "your-session-id",
  "generate_functions": {
    "expand": "def generate_fn(state):\n    if state is None:\n        return [('start', 0.5)]\n    return [(state + '_child1', 0.7), (state + '_child2', 0.3)]"
  }
}
```

3. **Get tree state and rankings:**
```json
{
  "session_id": "your-session-id"
}
```

4. **Rank top nodes:**
```json
{
  "session_id": "your-session-id",
  "k": 5
}
```

5. **Generate visualization:**
```json
{
  "session_id": "your-session-id",
  "format": "png",
  "title": "My Tree Search"
}
```

### Advanced Generate Function Examples

#### Mathematical Problem Solving
```python
# Generate function for exploring mathematical expressions
{
  "session_id": "math-session",
  "generate_functions": {
    "explore_math": """
def generate_fn(state):
    import random
    if state is None:
        # Start with basic numbers
        return [(str(i), random.random()) for i in range(1, 6)]
    
    # Add operations to existing expressions
    operations = ['+', '-', '*']
    results = []
    for op in operations:
        for num in range(1, 4):
            new_expr = f"({state} {op} {num})"
            # Simple evaluation-based scoring
            try:
                score = 1.0 / (1.0 + abs(eval(state) - 10))  # Target value of 10
            except:
                score = 0.1
            results.append((new_expr, score))
    return results[:3]  # Limit branching factor
"""
  }
}
```

#### Text Generation Tree Search
```python
# Generate function for text completion
{
  "session_id": "text-session", 
  "generate_functions": {
    "generate_text": """
def generate_fn(state):
    import random
    if state is None:
        # Start with sentence beginnings
        starters = ["The", "A", "In", "On", "With"]
        return [(word, random.random()) for word in starters]
    
    # Simple word continuation based on last word
    words = state.split()
    last_word = words[-1].lower()
    
    # Basic word associations
    continuations = {
        'the': ['cat', 'dog', 'house', 'tree'],
        'a': ['big', 'small', 'red', 'blue'],
        'in': ['the', 'a', 'this', 'that'],
        'on': ['the', 'a', 'top', 'bottom']
    }
    
    next_words = continuations.get(last_word, ['and', 'or', 'but', 'then'])
    results = []
    for word in next_words[:3]:
        new_state = state + ' ' + word
        # Score based on length and randomness
        score = random.random() * (1.0 - len(words) * 0.1)
        results.append((new_state, max(0.1, score)))
    
    return results
"""
  }
}
```

#### Game State Exploration
```python
# Generate function for game tree search
{
  "session_id": "game-session",
  "generate_functions": {
    "game_moves": """
def generate_fn(state):
    import random
    if state is None:
        # Initial game state
        return [("player1_turn", 0.5)]
    
    # Parse simple game state
    if "player1_turn" in state:
        moves = ["move_left", "move_right", "move_up", "move_down"]
        results = []
        for move in moves:
            new_state = state.replace("player1_turn", f"player2_turn_after_{move}")
            # Random scoring for demonstration
            score = random.random()
            results.append((new_state, score))
        return results
    
    elif "player2_turn" in state:
        # Player 2 responses
        responses = ["counter_left", "counter_right", "block", "attack"]
        results = []
        for response in responses[:2]:  # Limit branching
            new_state = state + f"_{response}"
            score = random.random()
            results.append((new_state, score))
        return results
    
    return []  # Terminal state
"""
  }
}
```

## Transport Modes

- **stdio**: Standard input/output for direct MCP client integration
- **sse**: Server-Sent Events over HTTP for web-based clients

## Error Handling

The MCP server includes comprehensive error handling and validation:

- **Parameter Validation**: Algorithm parameters are validated for type and range
- **Session Management**: Sessions are validated before operations
- **Generate Function Security**: Basic security checks prevent dangerous operations
- **Clear Error Messages**: Detailed error messages help with debugging

### Common Error Scenarios

1. **Invalid Session ID**: Returns clear error with suggestion to use `list_sessions`
2. **Invalid Parameters**: Specific validation errors for out-of-range values
3. **Generate Function Errors**: Syntax errors and execution errors are caught and reported
4. **Missing Dependencies**: Clear messages for missing Graphviz or other dependencies

## Integration with MCP Clients

### Using with Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "treequest": {
      "command": "treequest-mcp-server",
      "args": ["--transport", "stdio"]
    }
  }
}
```

### Using with MCP CLI Tools

```bash
# List available tools
mcp list-tools

# Test a tool
echo '{"algorithm": "StandardMCTS"}' | mcp call-tool init_tree
```

## Algorithm Comparison

| Algorithm | Best For | Key Parameters |
|-----------|----------|----------------|
| **StandardMCTS** | General tree search, balanced exploration | `exploration_weight` |
| **ABMCTSA** | Problems with clear action preferences | `exploration_weight`, `samples_per_action` |
| **ABMCTSM** | Complex state spaces with multiple objectives | `exploration_weight`, `samples_per_action` |

## Performance Tips

1. **Limit Branching Factor**: Keep generate functions returning 2-5 options per state
2. **Efficient Scoring**: Make scoring functions fast as they're called frequently
3. **Session Cleanup**: Delete sessions when done to free memory
4. **Batch Operations**: Use multiple steps before checking results for efficiency

## Troubleshooting

### Server Won't Start
- Check that all dependencies are installed: `pip install -e ".[mcp-server]"`
- Verify Python version >= 3.11

### Generate Functions Fail
- Check for syntax errors in your function code
- Ensure function returns list of (state, score) tuples
- Avoid using restricted imports (os, sys, subprocess, etc.)

### Visualization Issues
- Install Graphviz system package: `apt-get install graphviz` (Linux) or `brew install graphviz` (Mac)
- Install Python package: `pip install graphviz`
