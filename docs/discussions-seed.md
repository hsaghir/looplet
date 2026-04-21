# Discussions seed

Content for the three starter threads to post in GitHub Discussions
once Discussions is enabled. Posting these *yourself* is fine and
normal — the point is to make the repo look inhabited and show people
the level of detail they can expect from answers.

---

## 1. Category: Q&A — "Why not LangGraph?"

**Title:** Why would I pick openharness over LangGraph?

**Body:**

I keep getting this question so I'm putting the answer somewhere
searchable. LangGraph is great for workflows that are *actually*
graphs — branches, joins, parallel fan-out, recoverable sub-graphs.
If that's your shape, use LangGraph; no hard feelings.

`openharness` is for the case where you don't have a graph — you
have a single tool-calling loop and you want to **step through it**.
You want `for step in loop(...)` so you can `if step.tool_result.error:
break` or mutate state between steps. LangGraph's `StateGraph` is a
great tool, but you pay for graph semantics whether you need them or
not.

Other reasons people pick `openharness`:

- 1 runtime dep vs LangGraph's ~15.
- Hooks are `Protocol` objects, not subclasses — no framework vocab
  to learn.
- Fail-closed permissions engine built in.
- Crash-resume checkpoints in one line.
- Works with any backend (protocol-based), including Ollama.

Not a competition. Different problems.

---

## 2. Category: Show and tell — "What are you building with openharness?"

**Title:** Share what you're building

**Body:**

Tell us what you're using the loop for — domain, stack, scale. We're
looking for examples to feature in the docs and to understand where
the pain points are.

Particularly curious about:

- Which backend you're using (and why).
- Which hooks you've written that might be reusable.
- Anything the framework *doesn't* do well for your use case.

If you're willing, drop a link — we'll add you to
[THIRD_PARTY_USERS.md](../THIRD_PARTY_USERS.md).

---

## 3. Category: Ideas — "What's the right MCP story?"

**Title:** How do you use (or want to use) MCP servers with openharness?

**Body:**

We ship `MCPToolAdapter` as a minimal bridge (JSON-RPC over stdio, no
MCP SDK). But there are real design questions ahead:

1. Should MCP resources/prompts surface as tools, or as a separate
   concept?
2. Authentication — do we punt to the user or ship a default?
3. Multiple MCP servers — one registry, or one per server?
4. Streaming tool results — MCP has them, we don't yet.

If you're using MCP in production, please share how you'd want this
to work. The decision we make here will be hard to walk back.
