# Personal Assistant with Subagents

Capstone for **Building Agentic AI Systems** — Track A (Personal Assistant with Subagents).

A **supervisor** agent coordinates three specialist **subagents exposed as tools**:

- **Calendar subagent** — schedules events (tool stubs, no real calendar API).
- **Knowledge-base subagent** — answers from a small local knowledge base using a real **RAG** pipeline (Agentic RAG).
- **Email subagent** — a LangGraph **functional-API** workflow that drafts an email and **pauses for human approval** before sending.

All LLM calls go through **OpenRouter**; the API key is read from an environment variable and is never hardcoded. The project is available two ways, same program:

- [`capstone.ipynb`](capstone.ipynb) — end-to-end notebook (runs top-to-bottom).
- [`capstone.py`](capstone.py) — runnable script version (`python capstone.py`).

## Team members

- Wael Alanezi
- Saud Albander
- Abdullah Alshahrani

## Architecture

```
                       ┌──────────────────────────────┐
   user request  ───▶  │  SUPERVISOR  (create_agent)  │   short-term memory: checkpointer + thread_id
                       │  state: AssistantState       │   long-term memory: user_profile / user_preferences
                       └───────────────┬──────────────┘
             subagents-as-tools ───────┼───────────────────────────────
                    │                  │                    │
            schedule_event()   answer_from_notes()   compose_and_send_email()
                    │                  │                    │
            CALENDAR subagent    KB / RAG subagent    EMAIL subagent
            (create_agent)       (Agentic RAG)        (@entrypoint / @task,
                                                       draft ▶ approve ▶ send)
```

## Rubric coverage

| # | Section | Where |
|---|---|---|
| 1 | Agent fundamentals (real tool calls + structured output) | Tools, subagents, `EmailDraft` via `with_structured_output` |
| 2 | Multi-agent / routing (supervisor + subagents-as-tools) | §5 Supervisor |
| 3 | RAG pipeline (load→split→embed→store→retrieve; Agentic RAG) | §3 KB subagent |
| 4 | Context & state (short-term checkpointer + long-term custom state) | §5–6 |
| 5 | Human-in-the-loop (`interrupt()` before send) | §4 and §6 email demos |
| 6 | Functional API + error handling (`@task`/`@entrypoint`, retry + bubble-up) | §4 Email workflow |
| 7 | Workflow pattern (**Prompt Chaining**) | §4 Email workflow |
| 8 | LangSmith observability | §0 setup + §7 |

See [`WRITEUP.md`](WRITEUP.md) for one paragraph per section explaining the pattern used and why.

## Setup & run (local)

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."      # get one at https://openrouter.ai/keys
# optional tracing:
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_TRACING="true"

jupyter notebook capstone.ipynb            # run top-to-bottom
# or, the script version:
python capstone.py
```

## Run in Google Colab

1. Upload `capstone.ipynb` (or `File ▸ Open ▸ GitHub` and point at this repo).
2. Make the knowledge base available — easiest is to clone the repo in the first cell:
   ```python
   !git clone https://github.com/wael-alanezi/agentic-personal-assistant.git
   %cd agentic-personal-assistant
   ```
   (or upload the `knowledge_base/` folder via the Files panel).
3. Add your key in the **Secrets** panel (🔑): name `OPENROUTER_API_KEY`, paste your value.
   The notebook reads `os.environ` first and falls back to Colab Secrets automatically.
4. `Runtime ▸ Run all`.

## Notes

- The calendar and email integrations are **stubs** by design (course requirement) — they return
  formatted strings instead of calling real APIs.
- Default model is the course's free `nvidia/nemotron-3-nano-30b-a3b:free`. If structured output or
  tool-calling is unreliable on the free tier, swap `llm.model` for a stronger OpenRouter model.
