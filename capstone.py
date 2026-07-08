"""
Personal Assistant with Subagents — Capstone (script version).

Building Agentic AI Systems, Track A. A supervisor agent coordinates three
subagents exposed as tools: a calendar subagent, a knowledge-base (RAG) subagent,
and an email subagent whose send step pauses for human approval. All LLM calls go
through OpenRouter; the API key is read from the OPENROUTER_API_KEY environment
variable and is never hardcoded.

This is the same program as capstone.ipynb, as a runnable script. Definitions run at
import time; the rubric demos run under `if __name__ == "__main__"`.

Run:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY="sk-or-..."
    python capstone.py
"""

import os
import glob

# ---------------------------------------------------------------------------
# 0 · Setup: API keys (env var first, Colab Secrets fallback) + LangSmith tracing
# ---------------------------------------------------------------------------

def _load_key(name: str):
    """Load a key from the environment, falling back to Colab Secrets. Never hardcoded."""
    val = os.environ.get(name)
    if not val:
        try:
            from google.colab import userdata  # only exists in Colab
            val = userdata.get(name)
            if val:
                os.environ[name] = val
        except Exception:
            pass
    return val


assert _load_key("OPENROUTER_API_KEY"), (
    "Set OPENROUTER_API_KEY as an environment variable (or a Colab secret)."
)

# Rubric 8: LangSmith tracing — optional, degrades gracefully if no key.
if _load_key("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "agentic-personal-assistant")
else:
    os.environ["LANGSMITH_TRACING"] = "false"

from langchain_openai import ChatOpenAI

# All LLM calls go through OpenRouter, using the course's free default model.
llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",  # OpenRouter instead of default OpenAI
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# ---------------------------------------------------------------------------
# 1 · Tools (stubs) — no real API calls
# ---------------------------------------------------------------------------

from langchain.tools import tool


@tool
def get_available_time_slots(attendees: list[str], date: str, duration_minutes: int) -> list[str]:
    """Check calendar availability for attendees on a date (ISO 'YYYY-MM-DD')."""
    # Stub: a real implementation would query Google/Outlook Calendar.
    return ["09:30", "13:00", "14:00", "16:00"]


@tool
def create_calendar_event(title: str, start_time: str, end_time: str,
                          attendees: list[str], location: str = "") -> str:
    """Create a calendar event. start_time/end_time are ISO 'YYYY-MM-DDTHH:MM:SS'."""
    # Stub: a real implementation would call a calendar API.
    return (f"EVENT CREATED: '{title}' {start_time} -> {end_time} "
            f"| attendees={attendees} | location={location or 'n/a'}")


# Low-level email send stub with a controllable failure mode (used in section 6).
class EmailSendError(Exception):
    """A transient email-provider failure (retryable)."""


FAIL_TIMES = {"n": 0}      # how many times the next send should transiently fail first
HARD_FAIL = {"on": False}  # force an unexpected, non-retryable failure


def _send_email_stub(to: str, subject: str, body: str) -> str:
    if HARD_FAIL["on"]:
        raise RuntimeError("Email provider returned HTTP 500 (unexpected).")
    if FAIL_TIMES["n"] > 0:
        FAIL_TIMES["n"] -= 1
        raise EmailSendError("Temporary network error talking to the email provider.")
    return f"EMAIL SENT to {to} | subject='{subject}' | body {len(body)} chars"


# ---------------------------------------------------------------------------
# 2 · Calendar subagent
# ---------------------------------------------------------------------------

from langchain.agents import create_agent
from langchain.messages import HumanMessage

CALENDAR_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural-language requests (e.g. 'next Tuesday at 2pm') into ISO datetimes. "
    "Assume the year is 2026 when none is given. "
    "Use get_available_time_slots to check availability, then create_calendar_event. "
    "Confirm exactly what you scheduled."
)
calendar_agent = create_agent(
    model=llm,
    tools=[get_available_time_slots, create_calendar_event],
    system_prompt=CALENDAR_PROMPT,
)

# ---------------------------------------------------------------------------
# 3 · Knowledge-base subagent — RAG pipeline (Agentic RAG)
#     load -> split -> embed -> store -> retrieve
# ---------------------------------------------------------------------------

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# 1. LOAD — read local markdown into Documents (plain Python; no extra loader dependency)
_kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
kb_paths = sorted(glob.glob(os.path.join(_kb_dir, "*.md")))
assert kb_paths, "No knowledge_base/*.md files found next to this script."
raw_docs = [
    Document(page_content=open(p, encoding="utf-8").read(),
             metadata={"source": os.path.basename(p)})
    for p in kb_paths
]

# 2. SPLIT
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splits = splitter.split_documents(raw_docs)

# 3. EMBED + 4. STORE
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(splits)


# 5. RETRIEVE — exposed as a tool so the agent chooses when to call it (Agentic RAG)
@tool
def search_knowledge_base(query: str) -> str:
    """Retrieve the most relevant snippets from the personal knowledge base
    (assistant policy, FAQ, personal notes)."""
    hits = vector_store.similarity_search(query, k=2)
    return "\n\n".join(f"[{d.metadata.get('source')}] {d.page_content}" for d in hits)


KB_PROMPT = (
    "You answer questions using ONLY the personal knowledge base. "
    "Always call search_knowledge_base first. "
    "If the answer is not in the retrieved context, say you don't know. "
    "Cite the source file in brackets."
)
kb_agent = create_agent(model=llm, tools=[search_knowledge_base], system_prompt=KB_PROMPT)

# ---------------------------------------------------------------------------
# 4 · Email subagent — LangGraph functional API workflow
#     Prompt Chaining: draft -> human approval (interrupt) -> send
#     Error handling: transient retry + unexpected bubble-up
# ---------------------------------------------------------------------------

from typing import TypedDict
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver


# Rubric 1: structured output — the draft is a typed object, not free text.
class EmailDraft(TypedDict):
    to: str
    subject: str
    body: str


def _draft_email(request: str) -> EmailDraft:
    """Structured drafting shared by the workflow and the supervisor's email tool."""
    structured = llm.with_structured_output(EmailDraft)
    return structured.invoke(
        "Draft a short, professional internal email for this request. "
        "If a team is named, use its @example.com address.\n\nRequest: " + request
    )


@task
def draft_email(request: str) -> EmailDraft:
    return _draft_email(request)


# Transient-retry strategy: only EmailSendError is retried; anything else bubbles up.
@task(retry_policy=RetryPolicy(max_attempts=3, retry_on=(EmailSendError,)))
def send_email(draft: EmailDraft) -> str:
    return _send_email_stub(draft["to"], draft["subject"], draft["body"])


email_checkpointer = InMemorySaver()


@entrypoint(checkpointer=email_checkpointer)
def email_workflow(request: str) -> dict:
    """Prompt Chaining: draft -> human approval (interrupt) -> send."""
    draft = draft_email(request).result()

    # Human-in-the-loop: pause until a human approves (and may edit) the draft.
    decision = interrupt({"action": "Approve this email before sending?", "draft": draft})
    if not decision.get("approved"):
        return {"status": "cancelled", "draft": draft}

    final = {**draft, **decision.get("edits", {})}
    result = send_email(final).result()  # unexpected errors here bubble up
    return {"status": "sent", "draft": final, "result": result}


# ---------------------------------------------------------------------------
# 5 · Supervisor — subagents-as-tools, plus short-term + long-term memory
# ---------------------------------------------------------------------------

from langchain.agents import AgentState
from langchain.tools import ToolRuntime
from langchain.messages import ToolMessage


# --- Subagents wrapped as tools ---
@tool
def schedule_event(request: str) -> str:
    """Schedule or modify calendar events from natural language (calendar subagent)."""
    out = calendar_agent.invoke({"messages": [HumanMessage(request)]})
    return out["messages"][-1].content


@tool
def answer_from_notes(request: str) -> str:
    """Answer questions from the personal knowledge base (RAG subagent)."""
    out = kb_agent.invoke({"messages": [HumanMessage(request)]})
    return out["messages"][-1].content


@tool
def compose_and_send_email(request: str) -> str:
    """Draft an email and send it ONLY after the human approves (email subagent)."""
    # interrupt() inside a tool pauses the whole supervisor run; resume with Command(resume=...).
    draft = _draft_email(request)
    decision = interrupt({"action": "Approve before sending?", "draft": draft})
    if not decision.get("approved"):
        return f"Email cancelled by the user. Draft was: {draft}"
    final = {**draft, **decision.get("edits", {})}
    return _send_email_stub(final["to"], final["subject"], final["body"])


# --- Long-term memory: custom state + tools that read/write it ---
class AssistantState(AgentState):
    user_profile: dict       # durable facts about the user (e.g. name)
    user_preferences: dict   # durable preferences that survive across turns


@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Read a durable user preference from long-term memory."""
    prefs = runtime.state.get("user_preferences", {}) or {}
    return str(prefs.get(pref_name, "not set"))


@tool
def set_user_preference(pref_name: str, value: str, runtime: ToolRuntime) -> Command:
    """Save a durable user preference to long-term memory."""
    prefs = dict(runtime.state.get("user_preferences", {}) or {})
    prefs[pref_name] = value
    return Command(update={
        "user_preferences": prefs,
        "messages": [ToolMessage(f"Saved preference {pref_name}={value}.",
                                 tool_call_id=runtime.tool_call_id)],
    })


SUPERVISOR_PROMPT = (
    "You are a personal assistant coordinating three specialists via tools: "
    "schedule_event (calendar), answer_from_notes (knowledge base), and "
    "compose_and_send_email (email). Use get_user_preference / set_user_preference to "
    "remember durable preferences. For multi-part requests, call several tools. "
    "Address the user by name when you know it."
)

supervisor_checkpointer = InMemorySaver()  # short-term memory (conversation thread)
supervisor = create_agent(
    model=llm,
    tools=[schedule_event, answer_from_notes, compose_and_send_email,
           get_user_preference, set_user_preference],
    system_prompt=SUPERVISOR_PROMPT,
    state_schema=AssistantState,           # long-term memory fields
    checkpointer=supervisor_checkpointer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pretty(result):
    for m in result["messages"]:
        m.pretty_print()


def show_interrupt(result):
    """Print the pending interrupt payload, whatever surface shape it has."""
    intr = result.get("__interrupt__") if isinstance(result, dict) else None
    if intr:
        payload = intr[0].value if hasattr(intr[0], "value") else intr[0]
        print("PAUSED for approval:", payload)
    else:
        print("Result:", result)


# ---------------------------------------------------------------------------
# Demos (rubric evidence) — run only when executed as a script
# ---------------------------------------------------------------------------

def main():
    print("\n=== RAG quick check (retrieval tool call visible) ===")
    res = kb_agent.invoke({"messages": [HumanMessage("Can the assistant send an email on its own?")]})
    pretty(res)

    print("\n=== Email workflow: approve + transient failure recovers ===")
    cfg = {"configurable": {"thread_id": "email-wf-approve"}}
    paused = email_workflow.invoke("Send Sara a note that our 1:1 is moved to 3pm today.", config=cfg)
    show_interrupt(paused)
    FAIL_TIMES["n"] = 2  # first two send attempts fail transiently -> RetryPolicy recovers
    done = email_workflow.invoke(Command(resume={"approved": True}), config=cfg)
    print("Final:", done)

    print("\n=== Email workflow: reject (nothing sent) ===")
    cfg_r = {"configurable": {"thread_id": "email-wf-reject"}}
    email_workflow.invoke("Email finance to approve a budget increase.", config=cfg_r)
    result = email_workflow.invoke(Command(resume={"approved": False}), config=cfg_r)
    print("Final:", result)

    print("\n=== Error handling: unexpected failure bubbles up ===")
    cfg_e = {"configurable": {"thread_id": "email-wf-error"}}
    email_workflow.invoke("Email the eng team about the billing migration status.", config=cfg_e)
    HARD_FAIL["on"] = True
    try:
        email_workflow.invoke(Command(resume={"approved": True}), config=cfg_e)
    except Exception as e:
        print("Bubbled up as expected:", type(e).__name__, "-", e)
    finally:
        HARD_FAIL["on"] = False

    print("\n=== Supervisor: routing + long-term memory + RAG ===")
    config_main = {"configurable": {"thread_id": "user-wael"}}
    r1 = supervisor.invoke(
        {"messages": [HumanMessage(
            "Hi, I'm Wael. Please remember I prefer afternoon meetings. "
            "Also, what is the email approval policy?")],
         "user_profile": {"name": "Wael"},
         "user_preferences": {}},
        config=config_main,
    )
    pretty(r1)

    print("\n=== Supervisor: short-term memory (same thread) ===")
    r2 = supervisor.invoke(
        {"messages": [HumanMessage("What meeting time do I prefer, and what's my name?")]},
        config=config_main,
    )
    pretty(r2)

    print("\n=== Supervisor: calendar routing ===")
    config_cal = {"configurable": {"thread_id": "user-wael-cal"}}
    r3 = supervisor.invoke(
        {"messages": [HumanMessage("Schedule a 30-minute sync with the design team next Wednesday at 2pm.")],
         "user_profile": {"name": "Wael"}, "user_preferences": {}},
        config=config_cal,
    )
    pretty(r3)

    print("\n=== Supervisor: email with human approval ===")
    config_email = {"configurable": {"thread_id": "user-wael-email"}}
    paused = supervisor.invoke(
        {"messages": [HumanMessage("Email the design team a reminder to review the v2 mockups.")],
         "user_profile": {"name": "Wael"}, "user_preferences": {}},
        config=config_email,
    )
    show_interrupt(paused)
    approved = supervisor.invoke(Command(resume={"approved": True}), config=config_email)
    pretty(approved)

    print("\nLangSmith tracing is",
          "ON — see your project dashboard." if os.environ.get("LANGSMITH_TRACING") == "true"
          else "OFF — set LANGSMITH_API_KEY and re-run to capture traces.")


if __name__ == "__main__":
    main()
