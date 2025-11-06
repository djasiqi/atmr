"""Agent Dispatch Intelligent - Orchestrateur autonome pour mode Fully Auto.

Ce module implémente un agent dispatch qui orchestre les assignations en continu,
utilise des tools (function calls) pour interagir avec le système, respecte des
garde-fous stricts, et génère des rapports quotidiens.
"""

from services.agent_dispatch.orchestrator import AgentOrchestrator, get_agent_for_company

__all__ = ["AgentOrchestrator", "get_agent_for_company"]

