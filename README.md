# Intelligent Research Agent

An autonomous research agent built with **LangGraph** that doesn't just generate text — it *decides*. Given a question, the agent chooses which tools to call, observes their results, and decides whether it has enough information to answer or needs another tool call. The reasoning loop (**ReAct**: Reason → Act → Observe) is implemented as an explicit state machine, not a hidden prompt trick.

The agent is served as a stateful HTTP API via **FastAPI**, containerized with **Docker**, observable end-to-end via **LangSmith**, and backed by a **pgvector**-powered long-term memory that lets it recall context across separate conversations.

---

## Why this project exists

Most "LLM wrapper" projects call an API and print the response. This project is built to demonstrate the layer above that: an agent that **plans its own tool use**, maintains **short-term memory within a conversation** and **long-term memory across conversations**, and is wrapped as a **production-style service** — not a notebook, not a script.

It's built without any high-level agent abstractions that hide the control flow. The state machine, the conditional routing between "call a tool" and "finish," and the memory retrieval logic are all explicit and inspectable.

---

## Architecture

```
                          ┌─────────────────────┐
                          │   FastAPI  /chat     │
                          │  (main.py, :8001)    │
                          └──────────┬───────────┘
                                     │  session_id → thread_id
                                     ▼
                     ┌───────────────────────────────┐
                     │        LangGraph Agent          │
                     │                                │
                     │   ┌────────┐   tool_calls?      │
                     │   │ agent  │───────────┐        │
                     │   │ node   │           │        │
                     │   └───┬────┘           ▼        │
                     │       │           ┌─────────┐   │
                     │       │  end      │  tools  │   │
                     │       │◄──────────│  node   │   │
                     │       ▼           └─────────┘   │
                     │   final answer                  │
                     └───────────────────────────────┘
                            │        │         │
              ┌─────────────┘        │         └──────────────┐
              ▼                      ▼                        ▼
     ┌──────────────┐      ┌──────────────────┐      ┌──────────────────┐
     │  calculate    │      │   web_search      │      │ search_knowledge  │
     │  (safe eval)  │      │   (Tavily API)    │      │ _base (RAG HTTP)  │
     └──────────────┘      └──────────────────┘      └─────────┬─────────┘
                                                                 │
                                                                 ▼
                                                     ┌────────────────────────┐
                                                     │  Multitenant-RAG API    │
                                                     │  (separate container,   │
                                                     │   :8000)                │
                                                     └────────────────────────┘

     ┌─────────────────────────────────────────────────────────────┐
     │                     Memory layer                            │
     │  • Short-term: LangGraph state, scoped per thread_id         │
     │  • Long-term: conversation summaries embedded (MiniLM) and   │
     │    stored in Postgres + pgvector, retrieved by similarity    │
     │    search and injected back as SystemMessage context         │
     └─────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────┐
     │        LangSmith — every node, tool call, token count,        │
     │        and latency in the graph is traced automatically       │
     └─────────────────────────────────────────────────────────────┘
```

---

## How the reasoning loop works

The agent is a `StateGraph` with two nodes and one conditional edge:

1. **`agent` node** — sends the running message list to `llama-3.3-70b-versatile` (via Groq) with three tools bound. The LLM decides, based on the system prompt and conversation so far, whether it needs a tool.
2. **`should_continue` conditional edge** — inspects the LLM's response. If it contains `tool_calls`, route to `tools`. Otherwise, route to `END`.
3. **`tools` node** — a prebuilt `ToolNode` executes whichever tool(s) the LLM requested, appends the result to the message list, and routes back to `agent`.

This repeats until the LLM produces a plain answer with no tool calls — the loop terminates itself based on the model's own judgment, not a fixed number of steps.

```python
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"use_tool": "tools", "end": END}
)
builder.add_edge("tools", "agent")
agent = builder.compile()
```

---

## Tools

| Tool | Purpose | Notes |
|---|---|---|
| `calculate` | Evaluates arithmetic expressions | Uses `ast.parse` + compiled `eval` restricted to an expression node — not a raw `eval()` on user input, which would be an arbitrary code execution risk |
| `web_search` | Live web results for current/real-time information | Backed by the Tavily search API, capped at top 3 results |
| `search_knowledge_base` | Retrieves from an internal document store | Calls the Multitenant-RAG project's `/retrieve` endpoint over HTTP — this agent doesn't do its own retrieval, it delegates to a separate, already-built RAG service |

The system prompt explicitly instructs the model on *when* to prefer each tool (e.g., don't call a tool for casual conversation, prefer knowledge base over web search for internal docs), since an agent with tools available will over-call them without guidance.

---

## Memory system

**Short-term (within a session):** LangGraph's `messages` state accumulates every human/AI/tool message in a conversation. In the FastAPI layer, each request's `session_id` is passed through as LangGraph's `thread_id`, so concurrent users on the same running server get fully isolated conversation state.

**Long-term (across sessions):** At the end of a conversation, the full message history is summarized by the LLM into 2–3 sentences, embedded using `sentence-transformers/all-MiniLM-L6-v2`, and stored in Postgres with the `pgvector` extension. On a new query, the agent embeds the incoming question, retrieves the top-3 most similar past summaries by cosine distance, and injects them as a `SystemMessage` — so the agent can recall relevant context from prior sessions without replaying full transcripts.

```sql
SELECT summary, embedding <=> %s::vector AS distance
FROM memories
WHERE tenant_id = %s
ORDER BY distance
LIMIT 3
```

---

## Observability with LangSmith

Tracing is enabled purely through environment variables (`LANGCHAIN_TRACING_V2=true` and related keys) — zero code changes, zero decorators. Every run produces a full trace tree: which node ran, what was sent to the LLM, what tool was called with what arguments, what it returned, token counts per step, and total latency.

This makes the agent's decisions inspectable after the fact — if it picks the wrong tool or loops unnecessarily, the exact step is visible in the trace, not buried in print statements.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | Explicit state machine over the ReAct loop instead of a hidden agent-executor abstraction — every transition is visible and debuggable |
| LLM | Groq (`llama-3.3-70b-versatile`) | Low-latency inference, important for a multi-step tool-calling loop where each step re-invokes the model |
| Web search | Tavily | Purpose-built search API for LLM agents, returns clean content instead of raw HTML |
| Long-term memory store | PostgreSQL + pgvector | Combines relational storage (tenant scoping) with vector similarity search in one database, no separate vector DB to operate |
| Embeddings | `sentence-transformers` (MiniLM) | Runs locally, no external embedding API call or cost per memory write/read |
| API layer | FastAPI | Async-ready, automatic request validation via Pydantic, auto-generated OpenAPI docs |
| Observability | LangSmith | Native LangGraph/LangChain integration via environment variables only |
| Containerization | Docker + Docker Compose | Reproducible runtime environment; the agent container reaches the RAG project's Postgres container over the host network |

---

## Project structure

```
Intelligent-Research-Agent/
├── agent/
│   ├── graph.py          # StateGraph definition, agent/tools nodes, ReAct loop, CLI entrypoint
│   ├── tools.py           # calculate, web_search, search_knowledge_base tool definitions
│   ├── memory.py          # Long-term memory: summarization, embedding, pgvector read/write
│   └── quickdbcheck.py     # Utility script for verifying the Postgres/pgvector connection
├── main.py                # FastAPI app — /chat and /health endpoints
├── Dockerfile              # Container image definition (Python 3.13-slim)
├── docker-compose.yml       # Service definition for the agent container
├── requirements.txt         # Pinned dependencies
└── .gitignore
```

---

## Running locally

### Prerequisites
- Python 3.13
- A running Postgres instance with the `pgvector` extension enabled
- API keys: Groq, Tavily, and (optionally) LangSmith

### Setup

```bash
git clone https://github.com/Shreehari17/Intelligent-Research-Agent.git
cd Intelligent-Research-Agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_db_name
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password

# Optional — LangSmith observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=intelligent-research-agent
```

### Run as a terminal agent

```bash
python -m agent.graph
```

### Run as an API

```bash
uvicorn main:app --reload --port 8001
```

Test it:

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-1", "message": "What is 144 divided by 12?"}'
```

Interactive API docs: `http://localhost:8001/docs`

---

## Running with Docker

```bash
docker-compose up --build
```

This builds the agent image and starts it on port 8001. The agent connects to Postgres via `POSTGRES_HOST` in `.env` — set this to `host.docker.internal` if Postgres is running as a separate local container (e.g. alongside a companion RAG project), or to a service name if you add Postgres directly into this `docker-compose.yml`.

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "docker-test-1", "message": "What is 144 divided by 12?"}'
```

---

## API reference

### `POST /chat`

**Request**
```json
{
  "session_id": "string",
  "message": "string"
}
```

**Response**
```json
{
  "session_id": "string",
  "response": "string"
}
```

`session_id` maps directly to LangGraph's `thread_id`, so distinct session IDs get fully isolated conversation memory even under concurrent requests to the same server.

### `GET /health`

Returns `{"status": "ok"}` — used for container health checks.

---

## Related project

This agent's `search_knowledge_base` tool calls the retrieval endpoint of [**Multitenant-RAG**](https://github.com/Shreehari17/Multitenant-RAG), a separate production-grade RAG system with multi-tenant data isolation, hybrid search, and cross-encoder reranking, built without LangChain/LlamaIndex. The two projects run as independent containers and communicate over HTTP — this agent doesn't reimplement retrieval, it delegates to a purpose-built retrieval service.

---

## Roadmap

- [ ] Streaming responses over the `/chat` endpoint (SSE or WebSocket)
- [ ] Wait-for-Postgres-ready logic in Docker Compose startup instead of relying on `depends_on` ordering alone
- [ ] Authentication on the API layer
- [ ] Configurable tool set per request/tenant
