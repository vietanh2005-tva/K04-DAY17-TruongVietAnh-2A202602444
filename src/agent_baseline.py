from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A with within-thread memory only.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        if not force_offline and self.config.model.api_key:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is not None:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            answer = result["messages"][-1].content
            return {"answer": answer, "agent_tokens": estimate_tokens(answer), "prompt_tokens": estimate_tokens(message)}
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement deterministic within-thread offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        session.prompt_tokens_processed += prompt_tokens
        answer = "Mình đã ghi nhận thông tin này trong cuộc trò chuyện hiện tại."
        lower = message.lower()
        if "tên gì" in lower or "tên mình" in lower:
            name = next((item["content"] for item in reversed(session.messages[:-1]) if "tên" in item["content"].lower()), None)
            answer = name or "Mình chưa có thông tin đó trong cuộc trò chuyện này."
        elif any(term in lower for term in ("ở đâu", "nghề gì", "style", "đồ uống", "món ăn", "nuôi con", "tóm tắt")):
            answer = "Mình chưa có đủ thông tin đó trong cuộc trò chuyện này."
        agent_tokens = estimate_tokens(answer)
        session.token_usage += agent_tokens
        session.messages.append({"role": "assistant", "content": answer})
        return {"answer": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        """Build an optional live LangChain agent when credentials exist.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
            return create_agent(build_chat_model(self.config.model), checkpointer=InMemorySaver())
        except (ImportError, ValueError):
            return None
