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
| **Phase 6: HITL** | ⏳ Pending | Human-in-the-loop approvals |

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

## ➡️ Next: Phase 6 — Human-in-the-Loop (Approval Workflow)

Phase 6 adds:
1. **Approval Node** — Pausing the execution graph for high-risk actions using LangGraph `interrupt()`.
2. **State Persistence** — SQLite-based checkpointers to save graph state across server restarts.
3. **Approvals Route** — Endpoints to inspect, approve, or reject pending actions.

