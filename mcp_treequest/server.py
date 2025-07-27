"""TreeQuest MCP Server implementation."""

import anyio
import click
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
import mcp.types as types
from mcp.server.lowlevel import Server

import treequest as tq
from treequest.types import StateScoreType, GenerateFnType


class TreeQuestSession:
    """Manages a single tree search session."""
    
    def __init__(self, algorithm_name: str, algorithm_params: Dict[str, Any]):
        self.session_id = str(uuid.uuid4())
        self.algorithm_name = algorithm_name
        self.algorithm_params = algorithm_params
        self.algorithm = self._create_algorithm()
        self.state = self.algorithm.init_tree()
        self.step_count = 0
    
    def _create_algorithm(self):
        """Create the algorithm instance based on name and parameters."""
        if self.algorithm_name == "StandardMCTS":
            return tq.StandardMCTS(**self.algorithm_params)
        elif self.algorithm_name == "ABMCTSA":
            return tq.ABMCTSA(**self.algorithm_params)
        elif self.algorithm_name == "ABMCTSM":
            return tq.ABMCTSM(**self.algorithm_params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
    
    def step_algorithm(self, generate_fns):
        """Perform a step with proper type handling."""
        self.state = self.algorithm.step(self.state, generate_fns)  # type: ignore
        self.step_count += 1
    
    def get_state_score_pairs(self):
        """Get state score pairs with proper type handling."""
        return self.algorithm.get_state_score_pairs(self.state)  # type: ignore


sessions: Dict[str, TreeQuestSession] = {}


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    """Main entry point for the TreeQuest MCP server."""
    app = Server("treequest-mcp-server")

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
        """Handle tool calls."""
        if name == "init_tree":
            return await init_tree_tool(arguments)
        elif name == "step_tree":
            return await step_tree_tool(arguments)
        elif name == "get_tree_state":
            return await get_tree_state_tool(arguments)
        elif name == "rank_nodes":
            return await rank_nodes_tool(arguments)
        elif name == "list_sessions":
            return await list_sessions_tool(arguments)
        elif name == "delete_session":
            return await delete_session_tool(arguments)
        elif name == "get_tree_visualization":
            return await get_tree_visualization_tool(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List available tools."""
        return [
            types.Tool(
                name="init_tree",
                title="Initialize Tree Search",
                description="Initialize a new tree search session with specified algorithm",
                inputSchema={
                    "type": "object",
                    "required": ["algorithm"],
                    "properties": {
                        "algorithm": {
                            "type": "string",
                            "enum": ["StandardMCTS", "ABMCTSA", "ABMCTSM"],
                            "description": "Tree search algorithm to use"
                        },
                        "params": {
                            "type": "object",
                            "description": "Algorithm-specific parameters",
                            "properties": {
                                "exploration_weight": {
                                    "type": "number",
                                    "description": "Exploration weight for UCT (default: 1.0)"
                                },
                                "samples_per_action": {
                                    "type": "integer",
                                    "description": "Number of samples per action (default: 1)"
                                }
                            }
                        }
                    }
                }
            ),
            types.Tool(
                name="step_tree",
                title="Step Tree Search",
                description="Perform one step of tree search using provided generate functions",
                inputSchema={
                    "type": "object",
                    "required": ["session_id", "generate_functions"],
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from init_tree"
                        },
                        "generate_functions": {
                            "type": "object",
                            "description": "Map of action names to generate function code",
                            "additionalProperties": {
                                "type": "string",
                                "description": "Python code for generate function"
                            }
                        }
                    }
                }
            ),
            types.Tool(
                name="get_tree_state",
                title="Get Tree State",
                description="Extract current tree state and statistics",
                inputSchema={
                    "type": "object",
                    "required": ["session_id"],
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        }
                    }
                }
            ),
            types.Tool(
                name="rank_nodes",
                title="Rank Tree Nodes",
                description="Get top-k nodes using TreeQuest's ranking functionality",
                inputSchema={
                    "type": "object",
                    "required": ["session_id"],
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of top nodes to return (default: 10)",
                            "default": 10
                        }
                    }
                }
            ),
            types.Tool(
                name="list_sessions",
                title="List Sessions",
                description="List all active tree search sessions",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="delete_session",
                title="Delete Session",
                description="Clean up a tree search session",
                inputSchema={
                    "type": "object",
                    "required": ["session_id"],
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to delete"
                        }
                    }
                }
            ),
            types.Tool(
                name="get_tree_visualization",
                title="Get Tree Visualization",
                description="Generate tree visualization using Graphviz",
                inputSchema={
                    "type": "object",
                    "required": ["session_id"],
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["png", "pdf", "svg", "dot"],
                            "description": "Output format (default: png)",
                            "default": "png"
                        },
                        "show_scores": {
                            "type": "boolean",
                            "description": "Whether to show scores in node labels (default: true)",
                            "default": True
                        },
                        "max_label_length": {
                            "type": "integer",
                            "description": "Maximum length for node labels (default: 20)",
                            "default": 20
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional title for the visualization"
                        }
                    }
                }
            )
        ]

    async def init_tree_tool(arguments: dict) -> list[types.ContentBlock]:
        """Initialize a new tree search session."""
        algorithm_name = arguments["algorithm"]
        params = arguments.get("params", {})
        
        try:
            session = TreeQuestSession(algorithm_name, params)
            sessions[session.session_id] = session
            
            result = {
                "session_id": session.session_id,
                "algorithm": algorithm_name,
                "parameters": params,
                "status": "initialized"
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error initializing tree: {str(e)}"
            )]

    async def step_tree_tool(arguments: dict) -> list[types.ContentBlock]:
        """Perform one step of tree search."""
        session_id = arguments["session_id"]
        generate_functions_code = arguments["generate_functions"]
        
        if session_id not in sessions:
            return [types.TextContent(
                type="text",
                text=f"Error: Session {session_id} not found"
            )]
        
        session = sessions[session_id]
        
        try:
            generate_fns = {}
            for action_name, code in generate_functions_code.items():
                exec_globals = {
                    "Optional": Optional,
                    "Tuple": Tuple,
                    "random": __import__("random"),
                    "math": __import__("math"),
                }
                exec(code, exec_globals)
                generate_fns[action_name] = exec_globals.get("generate_fn")
                
                if generate_fns[action_name] is None:
                    return [types.TextContent(
                        type="text",
                        text=f"Error: No 'generate_fn' function found in code for action '{action_name}'"
                    )]
            
            session.step_algorithm(generate_fns)
            
            nodes = session.state.tree.get_nodes()
            state_score_pairs = session.get_state_score_pairs()
            
            result = {
                "session_id": session_id,
                "step_count": session.step_count,
                "total_nodes": len(nodes),
                "non_root_nodes": len(state_score_pairs),
                "tree_size": len(session.state.tree),
                "status": "step_completed"
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error during step: {str(e)}"
            )]

    async def get_tree_state_tool(arguments: dict) -> list[types.ContentBlock]:
        """Get current tree state and statistics."""
        session_id = arguments["session_id"]
        
        if session_id not in sessions:
            return [types.TextContent(
                type="text",
                text=f"Error: Session {session_id} not found"
            )]
        
        session = sessions[session_id]
        
        try:
            nodes = session.state.tree.get_nodes()
            state_score_pairs = session.get_state_score_pairs()
            
            serializable_pairs = []
            for state, score in state_score_pairs:
                serializable_pairs.append({
                    "state": str(state),
                    "score": float(score)
                })
            
            result = {
                "session_id": session_id,
                "algorithm": session.algorithm_name,
                "step_count": session.step_count,
                "total_nodes": len(nodes),
                "tree_size": len(session.state.tree),
                "state_score_pairs": serializable_pairs
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error getting tree state: {str(e)}"
            )]

    async def rank_nodes_tool(arguments: dict) -> list[types.ContentBlock]:
        """Get top-k nodes using ranking functionality."""
        session_id = arguments["session_id"]
        k = arguments.get("k", 10)
        
        if session_id not in sessions:
            return [types.TextContent(
                type="text",
                text=f"Error: Session {session_id} not found"
            )]
        
        session = sessions[session_id]
        
        try:
            top_results = tq.top_k(session.state.tree, session.algorithm, k=k)
            
            serializable_results = []
            for state, score in top_results:
                serializable_results.append({
                    "state": str(state),
                    "score": float(score)
                })
            
            result = {
                "session_id": session_id,
                "k": k,
                "top_nodes": serializable_results
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error ranking nodes: {str(e)}"
            )]

    async def list_sessions_tool(arguments: dict) -> list[types.ContentBlock]:
        """List all active sessions."""
        session_list = []
        for session_id, session in sessions.items():
            session_list.append({
                "session_id": session_id,
                "algorithm": session.algorithm_name,
                "step_count": session.step_count,
                "tree_size": len(session.state.tree)
            })
        
        result = {
            "active_sessions": len(sessions),
            "sessions": session_list
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def delete_session_tool(arguments: dict) -> list[types.ContentBlock]:
        """Delete a session."""
        session_id = arguments["session_id"]
        
        if session_id not in sessions:
            return [types.TextContent(
                type="text",
                text=f"Error: Session {session_id} not found"
            )]
        
        del sessions[session_id]
        
        result = {
            "session_id": session_id,
            "status": "deleted"
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def get_tree_visualization_tool(arguments: dict) -> list[types.ContentBlock]:
        """Generate tree visualization using Graphviz."""
        session_id = arguments["session_id"]
        format_type = arguments.get("format", "png")
        show_scores = arguments.get("show_scores", True)
        max_label_length = arguments.get("max_label_length", 20)
        title = arguments.get("title")
        
        if session_id not in sessions:
            return [types.TextContent(
                type="text",
                text=f"Error: Session {session_id} not found"
            )]
        
        session = sessions[session_id]
        
        try:
            from treequest.visualization import visualize_tree_graphviz
            
            dot = visualize_tree_graphviz(
                tree=session.state.tree,
                save_path=None,  # Don't save to file, just return the dot object
                show_scores=show_scores,
                max_label_length=max_label_length,
                title=title,
                format=format_type
            )
            
            if dot is None:
                return [types.TextContent(
                    type="text",
                    text="Error: Graphviz executable not found. Please install Graphviz to use visualization."
                )]
            
            dot_source = dot.source
            
            result = {
                "session_id": session_id,
                "format": format_type,
                "dot_source": dot_source,
                "node_count": len(session.state.tree.get_nodes()),
                "visualization_generated": True
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except ImportError:
            return [types.TextContent(
                type="text",
                text="Error: Graphviz not available. Install with 'pip install graphviz' to use visualization."
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error generating visualization: {str(e)}"
            )]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            return Response()

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn
        uvicorn.run(starlette_app, host="127.0.0.1", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)

    return 0


if __name__ == "__main__":
    main()
