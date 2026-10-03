from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B with short-term, persistent, and compact memory.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        if not force_offline and self.config.model.api_key:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a deterministic response using all memory layers."""

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        answer = self._offline_response(user_id, thread_id, message)
        agent_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.compact_memory.append(thread_id, "assistant", answer)
        return {"answer": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        profile = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        recent = "\n".join(str(item["content"]) for item in context["messages"])
        return estimate_tokens(profile) + estimate_tokens(str(context["summary"])) + estimate_tokens(recent)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.facts(user_id)
        lower = message.lower()
        keys: list[str] = []
        if "tên" in lower or "tóm tắt" in lower:
            keys.append("name")
        if "nghề" in lower or "làm" in lower or "tóm tắt" in lower:
            keys.append("profession")
        if "ở đâu" in lower or "nơi ở" in lower:
            keys.append("location")
        if "đồ uống" in lower:
            keys.append("favorite_drink")
        if "món ăn" in lower:
            keys.append("favorite_food")
        if "nuôi" in lower or "con gì" in lower:
            keys.append("pet")
        if "style" in lower or "kiểu trả lời" in lower or "cách trả lời" in lower:
            keys.append("response_style")
        if "quan tâm" in lower or "kỹ thuật" in lower or "tóm tắt" in lower:
            keys.append("interests")
        values = [facts[key] for key in dict.fromkeys(keys) if key in facts]
        if values:
            return "; ".join(values) + "."
        return "Mình đã ghi nhớ thông tin ổn định này trong hồ sơ của bạn."

    def _maybe_build_langchain_agent(self):
        """Build an optional live agent with persistent-memory-friendly state.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
            return create_agent(build_chat_model(self.config.model), checkpointer=InMemorySaver())
        except (ImportError, ValueError):
            return None
