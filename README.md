# Agentic AI Ops Platform

> A production-grade IT Operations AI assistant built to teach every major **Agentic Engineering concept** through hands-on implementation.

---

## ✅ Phases Completed

| Phase | Status | Milestone |
|-------|--------|-----------|
| **Phase 1: Foundation** | ✅ Done | FastAPI + SQLite + ticket/incident CRUD |
| **Phase 2: RAG Pipeline** | ✅ Done | Document upload → chunk → embed → Qdrant |
| **Phase 3: First Agent** | ✅ Done | LangGraph RAG Agent + `/api/chat` endpoint |
| **Phase 4: Multi-Agent** | ✅ Done | Supervisor + intent routing + 3 specialist agents |
| **Phase 5: Memory** | ✅ Done | Three-tier memory system (Short, Long & Semantic) |
| **Phase 6: HITL** | ✅ Done | Human-in-the-loop approvals (interrupt/resume) |
| **Phase 7: MCP** | ⏳ Pending | Model Context Protocol integration |

---

## 📁 Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI entry point
│   │   ├── config.py            ← All settings via .env (no hardcoding)
│   │   ├── db/                  ← SQLAlchemy ORM + database
│   │   ├── rag/                 ← RAG Pipeline (Phase 2)
│   │   ├── agents/              ← LangGraph Agent System (Phase 3+)
│   │   │   ├── state.py         ← Shared AgentState TypedDict
│   │   │   ├── rag_agent.py     ← RAG specialist (knowledge queries)
│   │   │   ├── ticket_agent.py  ← Ticket specialist (CRUD operations)
│   │   │   ├── incident_agent.py← Incident specialist (investigation)
│   │   │   ├── rca_agent.py     ← RCA sub-agent (direct LLM call)
│   │   │   └── supervisor.py    ← Orchestrator with intent routing
│   │   ├── tools/               ← Agent Tools (Phase 3+)
│   │   │   ├── rag_tool.py      ← Knowledge base search
│   │   │   ├── ticket_tool.py   ← Ticket CRUD tools
│   │   │   ├── incident_tool.py ← Incident management tools
│   │   │   ├── logs_tool.py     ← Log search (simulated)
│   │   │   └── metrics_tool.py  ← Metrics query (simulated)
│   │   ├── routes/
│   │   │   ├── tickets.py       ← REST API for tickets
│   │   │   ├── incidents.py     ← REST API for incidents
│   │   │   ├── documents.py     ← Document upload + search
│   │   │   └── chat.py          ← AI Agent Chat (main entry point)
│   │   └── schemas/
│   │       └── chat.py          ← ChatRequest/ChatResponse schemas
│   ├── .env.example             ← Template (copy to .env)
│   └── requirements.txt
└── README.md
```

---

## 🚀 Setup Guide

### Step 1: Create & Activate Virtual Environment
```powershell
cd "d:\GenAI\AgenticAI Project"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> ⚠️ If you get an execution policy error:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 2: Install Dependencies
```powershell
cd backend
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
```powershell
copy backend\.env.example backend\.env
```

Edit `backend\.env`:
```env
# Required
OPENAI_API_KEY=sk-your-key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key

# Phase 3-4 agent tuning (defaults work fine)
AGENT_TEMPERATURE=0.0
AGENT_MAX_TOKENS=2048
AGENT_MAX_ITERATIONS=3
RAG_MAX_CHUNKS=5
RAG_MIN_SCORE=0.30

# Phase 4 supervisor
SUPERVISOR_INTENT_CONFIDENCE_THRESHOLD=0.70
ENABLE_TICKET_AGENT=true
ENABLE_INCIDENT_AGENT=true
```

### Step 4: Run the Server
```powershell
cd "d:\GenAI\AgenticAI Project\backend"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🧪 Testing the API

Open `http://localhost:8000/docs` for interactive Swagger UI.

---

## Phase 4 Tests — Multi-Agent Supervisor

The supervisor auto-detects intent from the message. No need to specify which agent.

### Knowledge Query → RAG Agent
```
POST http://localhost:8000/api/chat/
Content-Type: application/json

{"message": "How do I fix CrashLoopBackOff in Kubernetes?", "session_id": "test-001"}
```

### Ticket Operation → Ticket Agent
```
POST http://localhost:8000/api/chat/

{"message": "Create a high-priority ticket: Redis connection timeouts in auth-service after the v3.2 deployment", "session_id": "test-002"}
```

```
POST http://localhost:8000/api/chat/

{"message": "Show me all open critical tickets", "session_id": "test-002"}
```

```
POST http://localhost:8000/api/chat/

{"message": "Resolve TKT-1 — issue was fixed by rolling back the Redis config", "session_id": "test-002"}
```

### Incident Investigation → Incident Agent (+ RCA)
```
POST http://localhost:8000/api/chat/

{"message": "P1 incident: payment service is returning 503 errors. Pod memory at 98%. Investigate.", "session_id": "test-003"}
```

Expected flow (visible in server logs):
1. Supervisor detects `incident_investigation`
2. Incident Agent creates incident record
3. Incident Agent searches logs (errors in payment-service)
4. Incident Agent queries metrics (CPU, memory, error rate)
5. Incident Agent searches incident history
6. Incident Agent looks up runbook via RAG
7. Returns structured incident report with RCA

### General Chat → Direct Response (no tool calls)
```
POST http://localhost:8000/api/chat/

{"message": "Hello! What can you help me with?", "session_id": "test-004"}
```

---

## 🔍 Phase 4 Key Concepts Learned

| Concept | Where It's Used | File |
|---------|----------------|------|
| **Supervisor Pattern** | Orchestrator routes to specialist workers | `agents/supervisor.py` |
| **Custom StateGraph** | Manual LangGraph graph (vs create_react_agent) | `agents/supervisor.py` |
| **Intent Detection** | LLM classifies user intent as JSON | `agents/supervisor.py` |
| **Conditional Edges** | LangGraph routing based on state value | `agents/supervisor.py` |
| **Data-Mutating Tools** | Ticket/incident tools write to SQLite | `tools/ticket_tool.py` |
| **Tool Factory Pattern** | Closure captures request-scoped DB session | All tool files |
| **Simulated Tools** | Log/metrics tools generate realistic fake data | `tools/logs_tool.py` |
| **Four Golden Signals** | CPU, latency, errors, saturation metrics | `tools/metrics_tool.py` |
| **Sub-Agent Pattern** | RCA Agent called by Incident Agent | `agents/rca_agent.py` |
| **Direct LLM Call** | RCA uses ainvoke() directly (no ReAct loop) | `agents/rca_agent.py` |
| **Multi-Tool Agent** | Incident Agent has 7 tools for investigation | `agents/incident_agent.py` |
| **Evidence Accumulation** | Agent builds context across multiple tool calls | `agents/incident_agent.py` |

---

## Phase 1-3 Tests (still work)

### Tickets REST API
```
GET  http://localhost:8000/api/tickets/
POST http://localhost:8000/api/tickets/
GET  http://localhost:8000/api/tickets/?priority=critical
```

### Documents & RAG
```
POST http://localhost:8000/api/documents/upload       (multipart/form-data)
POST http://localhost:8000/api/documents/search
GET  http://localhost:8000/api/documents/
```

### Chat History
```
GET http://localhost:8000/api/chat/history/{session_id}
GET http://localhost:8000/api/chat/sessions
```

---

## 🐛 Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'app'` | Run uvicorn from `backend/` directory |
| `422 Unprocessable Entity` on chat | Check `Content-Type: application/json` and non-empty `message` |
| Agent goes straight to general_chat | Check OpenAI API key is valid |
| Incident agent times out | Increase `AGENT_MAX_ITERATIONS` in .env |
| Port already in use | Add `--port 8001` to uvicorn command |
| PowerShell execution policy | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## Phase 5 Tests — Memory System

The memory system auto-loads short-term, long-term, and semantic memories on every query, and automatically extracts new memories at the end of each turn.

### Automated Tests
Run the pytest suite to verify the memory database and vector store logic:
```powershell
# Set PYTHONPATH and run the memory unit tests
cd backend
cmd /c "set PYTHONPATH=.&& ..\venv\Scripts\pytest tests/test_memory.py -v"
```

### Manual Verification Flow
1. **Explicit Preference Storage**:
   Tell the agent:
   ```json
   {"message": "Please know that I prefer very short bullet-point answers. Also, we run Kubernetes version 1.28.", "session_id": "test-mem-001"}
   ```
2. **Contextual Awareness (Within Session)**:
   In the same session, ask:
   ```json
   {"message": "What did I just say about Kubernetes?", "session_id": "test-mem-001"}
   ```
   *Expected response*: A very concise answer mentioning version 1.28.
3. **Cross-Session Recall (Long-Term Memory)**:
   Open a brand new session and ask about your preference:
   ```json
   {"message": "What is my preferred response format?", "session_id": "test-mem-002"}
   ```
   *Expected response*: The agent recalls your preference for "very short bullet-point answers" from the persistent `MemoryEntry` table, even though this is a new session.

---

## 🔍 Phase 5 Key Concepts Learned

| Concept | Description | File |
|---------|-------------|------|
| **Short-Term Memory** | Conversation history buffer loaded chronologically from database | `memory/short_term.py` |
| **Long-Term Memory** | Structured facts & preferences stored and retrieved per user | `memory/long_term.py` |
| **Semantic Memory** | Embeddings-based vector search of past Q&A pairs in Qdrant | `memory/semantic.py` |
| **Memory Consolidation** | Automatic extraction of facts & preferences at the end of each turn via LLM | `agents/memory_agent.py` |
| **Explicit Memory Tools** | Giving agents tools (`save_user_preference`, `save_environment_fact`) | `tools/memory_tool.py` |

---

## Phase 6 Tests — Human-in-the-Loop (Approvals)

The approvals workflow enforces safety barriers on high-risk operations (like `execute_remediation_action`).

### Automated Tests
Run the pytest suite to verify approvals CRUD, decision endpoints, and execution tool breakpoints:
```powershell
cd backend
cmd /c "set PYTHONPATH=.&& ..\venv\Scripts\pytest tests/test_approvals.py -v"
```

### Manual Verification Flow
1. **Trigger Incident & Remediation**:
   Report a P1 outage to the chat:
   ```json
   {"message": "P1 incident: payment-service is down and returning 503s. Investigate and fix.", "session_id": "hitl-test-001"}
   ```
   *Expected response*: The Incident Agent creates an incident record, investigates logs/metrics, and identifies that it needs to restart the pod. However, it will return:
   `"⚠️ CRITICAL: The action 'restart_pod' on target 'payment-service' requires human approval (Request ID: 1)."`
2. **List Pending Approvals**:
   Query the approvals queue to check your request:
   ```bash
   GET http://localhost:8000/api/approvals/pending
   ```
   *Expected response*: A JSON array containing the pending request details with `id=1`.
3. **Approve the Action**:
   Approve the action:
   ```bash
   POST http://localhost:8000/api/approvals/1/approve
   Content-Type: application/json
   {"comment": "Approved by senior SRE"}
   ```
4. **Resume and Execute**:
   Instruct the agent to proceed:
   ```json
   {"message": "I have approved Request ID 1. Go ahead and execute the restart.", "session_id": "hitl-test-001"}
   ```
   *Expected response*: The Incident Agent loads the checkpoint state, executes the approved restart tool successfully, mitigates the service status, and outputs a final report indicating the issue is resolved.

---

## 🔍 Phase 6 Key Concepts Learned

| Concept | Description | File |
|---------|-------------|------|
| **LangGraph Checkpointing** | Async state serialization using `AsyncSqliteSaver` in checkpoints database | `db/database.py` |
| **Approval Guardrails** | SQLite-backed `ApprovalRequest` registry to verify tool permissions | `tools/execute_tool.py` |
| **Action Lifespan Context** | Lifespan-scoped context management of persistent connections | `main.py` |
| **Approvals API** | Endpoints (`GET /approvals/pending`, `POST /approvals/{id}/approve`) for HITL gate reviews | `routes/approvals.py` |

---

## Phase 7 Tests — Model Context Protocol (MCP) Integration

Phase 7 exposes all 14 operational tools via a standardized MCP server and executes them dynamically through an MCP client transport.

### Automated Tests
Run the pytest suite to verify MCP client-server transport connection, tool discovery, and log search invocation:
```powershell
cd backend
cmd /c "set PYTHONPATH=.&& ..\venv\Scripts\pytest tests/test_mcp.py -v"
```

To run all automated test suites (Memory, Approvals, MCP):
```powershell
cd backend
cmd /c "set PYTHONPATH=.&& ..\venv\Scripts\pytest -v"
```

### Manual Verification Flow
1. **Verifying Dynamic Discovery & Execution**:
   Start the FastAPI app and trigger any normal chat flow:
   ```json
   {"message": "Show me the logs for auth-service matching keyword timeout", "session_id": "mcp-test-001"}
   ```
   *Expected response*: The supervisor correctly routes to the Incident Agent, which connects over the MCP client to call `search_logs` on the MCP server subprocess, returning formatted simulated logs.

2. **Standalone SSE Server Mode**:
   Start the MCP server as a standalone SSE HTTP server on port `8010`:
   ```powershell
   cd backend
   ..\venv\Scripts\python -m app.mcp.server --transport sse --port 8010
   ```
   This allows external tools/agents (e.g. Claude Desktop) to connect directly to the system tools by configuring their config file.

---

## 🔍 Phase 7 Key Concepts Learned

| Concept | Description | File |
|---------|-------------|------|
| **FastMCP Server** | High-level API for creating MCP servers and declaring Python functions as tools | `mcp/server.py` |
| **Stdio Transport** | Process-based tool execution where client spawns server subprocess and streams JSON-RPC | `mcp/client.py` |
| **SSE Transport** | HTTP/Server-Sent Events server allowing remote tool access on port `8010` | `mcp/server.py` |
| **Dynamic Schema Mapping** | Translating MCP JSON-Schema definitions into Pydantic models at runtime | `mcp/client.py` |
| **Decoupled Tool Execution** | Decoupling agent definitions and state from DB sessions and local code bindings | `routes/chat.py` |

---

## ➡️ Next: Phase 8 — Agent-to-Agent (A2A) Protocol



