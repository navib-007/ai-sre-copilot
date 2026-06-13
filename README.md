# Agentic AI Ops Platform

## Phase 1: Foundation — Setup & Run Guide

---

## 📁 Project Structure Created

```text
.
├── backend/                 ← FastAPI Backend Application
│   ├── app/                 ← Application source code
│   │   ├── main.py          ← FastAPI entry point
│   │   ├── config.py        ← Centralized settings
│   │   ├── db/              ← Database setup and ORM models
│   │   ├── rag/             ← RAG Pipeline (Chunker, Embedder, Qdrant)
│   │   ├── routes/          ← API Endpoints (Tickets, Incidents, Documents)
│   │   └── schemas/         ← Pydantic validation schemas
│   ├── data/                ← SQLite DB + Sample Docs (Ignored in Git)
│   ├── logs/                ← Application Logs (Ignored in Git)
│   ├── .env.example         ← Template for environment variables (copy to .env)
│   ├── requirements.txt     ← Python dependencies
│   └── pyproject.toml       ← Project metadata and linting rules
├── .gitignore               ← Standard Git ignore rules for Python/environments
└── README.md                ← Project documentation
```

---

## 🚀 Step-by-Step Setup

### Step 1: Create Virtual Environment
```powershell
# Navigate to backend directory
cd "d:\GenAI\AgenticAI Project\backend"

# Create virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# You should see (.venv) prefix in your terminal
```

> ⚠️ If you get "execution policy" error on PowerShell, run first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 2: Install Dependencies
```powershell
# Make sure .venv is activated (see Step 1)
pip install -r requirements.txt

# This installs: FastAPI, SQLAlchemy, LangGraph, OpenAI, Qdrant, etc.
# Takes 2-3 minutes on first install.
```

### Step 3: Configure Environment Variables
Open `backend\.env` and fill in your actual API keys:

```env
# Replace these placeholders with your real keys:
OPENAI_API_KEY=sk-your-actual-openai-key
QDRANT_URL=https://your-actual-cluster.qdrant.io
QDRANT_API_KEY=your-actual-qdrant-key
LANGCHAIN_API_KEY=your-actual-langsmith-key
```

> For Phase 1, only `OPENAI_API_KEY` is required.
> Qdrant and LangSmith are used in Phase 2+.

### Step 4: Run the Server
```powershell
# Make sure you're in backend/ and .venv is activated
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:
```
INFO  application_starting  app_name='Agentic AI Ops Platform'  environment='development'
INFO  database_initializing
INFO  database_tables_created
INFO  user_created  username='admin'  id=1  role='admin'
INFO  ticket_created  id=1  title='Database backup failing...'  priority='high'
INFO  application_started  host='0.0.0.0'  port=8000
INFO  Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 5: Verify It Works
Open your browser and visit:

| URL | What You'll See |
|-----|----------------|
| `http://localhost:8000` | Welcome JSON |
| `http://localhost:8000/health` | Health check (database: connected) |
| `http://localhost:8000/docs` | **Swagger UI** — Interactive API docs |
| `http://localhost:8000/redoc` | ReDoc docs |

---

## 🧪 Testing the API via Swagger UI

Go to `http://localhost:8000/docs` and try these:

### Test 1: List Tickets
1. Click `GET /api/tickets/`
2. Click "Try it out"
3. Click "Execute"
4. You should see 6 pre-seeded tickets

### Test 2: Create a Ticket
1. Click `POST /api/tickets/`
2. Click "Try it out"
3. Edit the request body:
```json
{
  "title": "Redis connection timeout in auth-service",
  "description": "The auth-service is getting Redis connection timeouts every 5 minutes. Error: ECONNRESET. Started after Redis upgrade to v7.2.",
  "priority": "high",
  "category": "database"
}
```
4. Click "Execute"
5. Response: `201 Created` with the new ticket

### Test 3: Filter Tickets by Priority
1. Click `GET /api/tickets/`
2. Click "Try it out"
3. Set `priority` = `critical`
4. Execute → See only critical tickets

### Test 4: Update a Ticket
1. Click `PATCH /api/tickets/{ticket_id}`
2. Use `ticket_id` = `2` (the critical Kubernetes ticket)
3. Body:
```json
{
  "status": "resolved",
  "resolution": "Restarted Redis pod and updated connection pool settings."
}
```
4. Execute → Status changes to "resolved"

### Test 5: Create an Incident
1. Click `POST /api/incidents/`
2. Body:
```json
{
  "title": "P1: API Gateway returning 503 errors",
  "description": "The API gateway is returning 503 Service Unavailable for 30% of requests. Affecting all external API consumers.",
  "severity": "P1",
  "affected_services": "api-gateway,load-balancer"
}
```

---

## 🔍 Understanding the Logs

When you make API calls, you'll see structured logs in the console:

```
INFO  request_started  request_id='a1b2c3d4'  method='POST'  path='/api/tickets/'
INFO  db_session_opened
INFO  create_ticket_request  title='Redis connection...'  priority='high'
INFO  ticket_created  ticket_id=7  title='Redis connection timeout...'
INFO  db_session_committed
INFO  db_session_closed
INFO  request_completed  request_id='a1b2c3d4'  status_code=201  duration_ms=45.3
```

Notice:
- `request_id` ties all logs for ONE request together
- Every DB operation is logged (session open/commit/close)
- Duration in milliseconds helps spot slow endpoints
- All fields are structured (key=value) for easy filtering

---

## 📚 Key Concepts Learned in Phase 1

| Concept | Where It's Used | File |
|---------|----------------|------|
| **pydantic-settings** | Centralized config from .env | `app/config.py` |
| **Structured Logging** | structlog JSON/console output | `app/logging_config.py` |
| **Async SQLAlchemy** | ORM with async/await | `app/db/database.py` |
| **ORM Models** | Python classes = DB tables | `app/db/models.py` |
| **Pydantic Schemas** | Request validation + response serialization | `app/schemas/` |
| **FastAPI Dependency Injection** | `Depends(get_db)` | `app/routes/tickets.py` |
| **Lifespan Events** | Startup/shutdown logic | `app/main.py` |
| **CORS Middleware** | Allow Streamlit to call API | `app/main.py` |
| **Pagination** | `page` + `page_size` query params | `app/routes/tickets.py` |
| **Soft Delete** | Set status=closed vs DELETE | `app/routes/tickets.py` |
| **Audit Logging** | Immutable action trail | `app/db/models.py` + routes |
| **Database Seeding** | Realistic test data | `app/db/seed.py` |
| **Health Checks** | `/health` endpoint | `app/main.py` |
| **App Factory Pattern** | `create_app()` function | `app/main.py` |

---

## 🐛 Common Issues & Fixes

### Issue: `ModuleNotFoundError: No module named 'app'`
**Fix**: Run uvicorn from inside the `backend/` directory, not from the project root.
```powershell
cd "d:\GenAI\AgenticAI Project\backend"
uvicorn app.main:app --reload
```

### Issue: `ERROR: Address already in use`
**Fix**: Port 8000 is in use. Either kill the other process or use a different port:
```powershell
uvicorn app.main:app --reload --port 8001
```

### Issue: `aiosqlite not found`
**Fix**: Make sure your virtual environment is activated:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Issue: `422 Unprocessable Entity` on POST
**Fix**: Check the request body. Pydantic validation failed.
The error response tells you exactly which field failed and why.

### Issue: PowerShell execution policy error
**Fix**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## ➡️ Next: Phase 2 — RAG Pipeline

Once Phase 1 is working, we'll add:
1. Document upload endpoint (`POST /api/documents/upload`)
2. Text chunking with `langchain_text_splitters`
3. OpenAI embeddings generation
4. Qdrant vector storage
5. Semantic search endpoint
6. Sample IT runbook documents

The RAG pipeline is what powers the knowledge base that our AI agents use to answer questions.
