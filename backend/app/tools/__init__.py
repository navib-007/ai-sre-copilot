"""
tools/ — Agent Tool Layer
==========================
This package contains all tools available to agents.

CONCEPT: What is a Tool?
  In LangGraph/LangChain, a "tool" is a function that an LLM can decide
  to call during its reasoning process. The agent sees tool descriptions
  and chooses which to call based on the user's request.

  Tools follow this pattern:
    1. LLM reads the tool's docstring + args schema
    2. LLM decides whether to call the tool and with what arguments
    3. Tool executes and returns a result
    4. LLM reads the result and continues reasoning

Tools in this platform:
  Phase 3: rag_tool.py    — Search the knowledge base
  Phase 4: ticket_tool.py — CRUD on support tickets
  Phase 4: incident_tool.py — Query incident history
  Phase 4: logs_tool.py    — Search application logs (simulated)
  Phase 4: metrics_tool.py — Query system metrics (simulated)
  Phase 6: execute_tool.py — Execute approved actions
"""
