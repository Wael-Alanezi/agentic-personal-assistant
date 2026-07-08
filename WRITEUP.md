# Write-up — Pattern per Rubric Section

One paragraph per rubric section stating the pattern used and why. All patterns are taken
directly from the course lessons (Building Agentic AI Systems, L01), not an ad-hoc design.

### 1. Agent fundamentals
Every agent in the notebook is a real tool-calling agent built with `create_agent`, and the
tools do real work rather than returning hardcoded strings from the supervisor's own reasoning:
the calendar subagent actually calls `get_available_time_slots` then `create_calendar_event`,
and the KB subagent actually calls `search_knowledge_base`. We add **structured output** where it
matters most — the email draft — by typing it as `EmailDraft` (a `TypedDict`) and generating it
with `llm.with_structured_output(EmailDraft)`, so downstream code can rely on `to`/`subject`/`body`
fields instead of parsing free text. This is the L02/L06 tool-calling pattern plus the L09
structured-output pattern.

### 2. Multi-agent / routing architecture
We use the course's **supervisor + subagents-as-tools** pattern verbatim (L02) and extend it with a
third subagent. Each specialist is its own `create_agent` (calendar, knowledge base, email), and
each is wrapped as a single `@tool` (`schedule_event`, `answer_from_notes`, `compose_and_send_email`)
that the supervisor can call. The supervisor is itself a `create_agent` whose "tools" are those three
subagents, so it decomposes a request and routes each part to the right specialist — realizing the
**Routing / Orchestrator-Worker** idea without any custom framework. We chose this over a single
monolithic agent because it keeps each subagent's prompt and tools small and independently testable.

### 3. RAG pipeline
The knowledge-base subagent runs a real retrieval pipeline over local documents: **load**
(`knowledge_base/*.md` into `Document` objects), **split** (`RecursiveCharacterTextSplitter`,
1000/200), **embed** (`HuggingFaceEmbeddings`, `all-mpnet-base-v2`), **store**
(`InMemoryVectorStore`), and **retrieve** (`similarity_search(k=2)`), following L03–L06. We chose
**Agentic RAG** (retrieval exposed as a tool the agent decides to call) over **2-Step RAG** (a fixed
retrieve→generate chain) and **Hybrid RAG** (routing between the two): Agentic RAG lets the agent
skip or refine retrieval and issue multiple queries, and it composes cleanly with the
subagents-as-tools architecture, while the knowledge base is small enough that Hybrid's extra routing
machinery would add complexity for no benefit.

### 4. Context and state management
We distinguish the two memory types explicitly rather than relying on one message list.
**Short-term memory** is the conversation thread: the supervisor is compiled with an
`InMemorySaver` checkpointer, so invoking the same `thread_id` again transparently recalls earlier
turns (L07 conversation state). **Long-term memory** is a custom `AssistantState(AgentState)` with
`user_profile` and `user_preferences` fields that persist across turns and are read/written by
dedicated tools (`get_user_preference` / `set_user_preference`, the L07 `Command(update=...)`
pattern) — durable facts about the user that are conceptually separate from the raw chat history.

### 5. Human-in-the-loop
Before any email is sent, execution pauses for real human approval using `interrupt()`. In the
functional-API email workflow the `@entrypoint` calls `interrupt({...})` with the draft; the run
stops and surfaces the pending draft, and the human resumes it with
`Command(resume={"approved": True})` (or `False` to cancel, in which case nothing is sent). The same
gate is demonstrated end-to-end through the supervisor, where the `compose_and_send_email` tool
interrupts mid-run (the L08 interrupt-inside-a-tool pattern) so approval happens in the coordinated
flow, exactly matching "approve before the email subagent sends."

### 6. LangGraph functional API and error handling
The email subagent is built with the **functional API**: discrete `@task` steps (`draft_email`,
`send_email`) composed by an `@entrypoint` that calls `.result()` to sequence them (L09). It
implements **three** of the four course error-handling strategies. *Transient retry*: `send_email`
is decorated with `RetryPolicy(max_attempts=3, retry_on=EmailSendError)`, so a flaky provider is
retried and recovers. *User-fixable*: the approval `interrupt()` pauses for human input instead of
failing. *Unexpected bubble-up*: a non-retryable `RuntimeError` from the provider is not swallowed —
it propagates out of the workflow so the caller sees the real failure. Each is demonstrated with a
dedicated run in the notebook.

### 7. Workflow pattern
The email workflow explicitly implements **Prompt Chaining** (L10): a fixed, ordered sequence of
steps — draft the email → get human approval → send — where each step's output feeds the next,
encoded as `@task`s chained inside the `@entrypoint`. We chose Prompt Chaining because the email
process is inherently linear and benefits from a checkpoint (the approval gate) between well-defined
stages; it is named and isolated in the email workflow, while the supervisor separately illustrates
Routing/Orchestrator-Worker.

### 8. LangSmith observability
Tracing is enabled via the `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` environment variables (L11,
also shown in L06); LangChain auto-instruments every agent and tool call, and the notebook degrades
gracefully to tracing-off when no key is present. The trace makes the multi-agent structure legible:
the supervisor span nests child spans for each subagent tool call (e.g. `answer_from_notes` →
`search_knowledge_base` similarity search, `compose_and_send_email` → the `interrupt` pause), so we
can see routing decisions, retrieval latency, token usage, and exactly where the graph paused for
approval — none of which is visible from the final text answer alone.
