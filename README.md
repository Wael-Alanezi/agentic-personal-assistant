# Personal Assistant with Subagents

A **supervisor** agent coordinates three specialist **subagents exposed as tools**:

- **Calendar subagent** — schedules events (tool stubs, no real calendar API).
- **Knowledge-base subagent** — answers from a small local knowledge base using a real **RAG** pipeline (Agentic RAG).
- **Email subagent** — a LangGraph **functional-API** workflow that drafts an email and **pauses for human approval** before sending.

All LLM calls go through **OpenRouter**; the API key is read from an environment variable and is never hardcoded. The project is available two ways, same program:

- [`capstone.ipynb`](capstone.ipynb) — end-to-end notebook (runs top-to-bottom).
- [`capstone.py`](capstone.py) — runnable script version (`python capstone.py`).

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

See [`WRITEUP.md`](WRITEUP.md) for a design note on each architectural piece and why it was chosen.

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
3. Open the Secrets panel (🔑) and add a secret named `OPENROUTER_API_KEY` with your value.
   The notebook reads `os.environ` first and falls back to Colab Secrets automatically.
4. Run `Runtime ▸ Run all`.

## Notes

- The calendar and email integrations are **stubs** by design — they return
  formatted strings instead of calling real APIs.
- Default model is the free `nvidia/nemotron-3-nano-30b-a3b:free`. If structured output or
  tool-calling is unreliable on the free tier, swap `llm.model` for a stronger OpenRouter model.
