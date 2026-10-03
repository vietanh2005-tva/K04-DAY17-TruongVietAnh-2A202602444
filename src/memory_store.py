from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import unicodedata


def estimate_tokens(text: str) -> int:
    """Return a deterministic heuristic token estimate.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    cleaned = " ".join(text.split())
    return 0 if not cleaned else max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Responsibilities:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        normalized = unicodedata.normalize("NFKD", user_id)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", normalized.encode("ascii", "ignore").decode()).strip("-._")
        if not slug:
            raise ValueError("user_id must contain at least one safe character")
        path = (self.root_dir.resolve() / slug / "User.md").resolve()
        if self.root_dir.resolve() not in path.parents:
            raise ValueError("unsafe user_id")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User Profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s+([^:]+):\s*(.+)$", line)
            if match:
                result[match.group(1).strip()] = match.group(2).strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        facts = self.facts(user_id)
        facts[key] = value.strip().strip(" .")
        lines = ["# User Profile", ""] + [f"- {name}: {fact}" for name, fact in sorted(facts.items())]
        return self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable, structured profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = " ".join(message.strip().split())
    lower = text.lower()
    # Recall questions never introduce new profile facts. Keeping this guard
    # prevents phrases such as "mình tên gì?" from overwriting a real name.
    if not text or "?" in text:
        return {}
    updates: dict[str, str] = {}

    def clean_value(value: str) -> str:
        value = re.split(r"\s+(?:và|nhưng|chứ|để|vì)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
        value = re.sub(r"\s+(?:như cũ|hiện tại|trong giai đoạn này)$", "", value, flags=re.IGNORECASE)
        return value.strip().strip(" .,:;\"")

    def capture(key: str, patterns: list[str]) -> None:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = clean_value(match.group(1))
                if value:
                    updates[key] = value
                return

    capture("name", [r"(?:mình|tôi)\s+tên\s+là\s+([^,.!?]+)"])
    capture("favorite_drink", [
        r"đồ uống yêu thích(?: của (?:mình|tôi))?\s+là\s+([^,.!?]+)",
        r"(?:mình|tôi)\s+(?:vẫn\s+)?(?:thích\s+)?uống\s+([^,.!?]+)",
    ])
    capture("favorite_food", [
        r"món ăn yêu thích(?: của (?:mình|tôi))?\s+là\s+([^,.!?]+)",
        r"(?:mình|tôi)\s+(?:rất\s+)?thích\s+ăn\s+([^,.!?]+)",
    ])
    pet = re.search(
        r"(?:mình|tôi)\s+nuôi\s+(?:một\s+)?(?:bé\s+)?([^,.!?]+?)(?:\s+tên\s+([^,.!?]+))?(?:[,.!?]|$)",
        text,
        re.IGNORECASE,
    )
    if pet and not re.search(r"\b(?:gì|nào)\b", pet.group(1), re.IGNORECASE):
        species = clean_value(pet.group(1))
        pet_name = clean_value(pet.group(2) or "")
        updates["pet"] = f"{species} tên {pet_name}" if pet_name else species

    noise = any(cue in lower for cue in ("chỉ là câu đùa", "không phải nghề"))
    if not noise:
        capture("profession", [
            r"(?:nghề nghiệp(?: hiện tại)?\s*(?:thì\s+)?(?:vẫn\s+)?là|hiện làm)\s+([^,.!?]+)",
            r"(?:giờ|hiện tại)\s+(?:mình|tôi)?\s*(?:đã\s+)?chuyển sang\s+([^,.!?]+)",
            r"(?:mình|tôi)\s+(?:hiện\s+)?đang làm\s+(?!việc\s+ở\b)([^,.!?]+?)(?:\s+cho\s+[^,.!?]+)?(?:[,.!?]|$)",
        ])

    location_noise = any(cue in lower for cue in ("không phải nơi ở", "chỉ là nơi", "đi họp", "bay ra"))
    if not location_noise:
        capture("location", [
            r"nơi ở(?: hiện tại)?(?: đã)?(?: được)?\s*(?:là|cập nhật từ .*? sang)\s+([^,.!?]+)",
            r"(?:giờ|hiện tại|thực ra|từ [^,.!?]+)\s+(?:mình|tôi)?\s*(?:đang\s+)?(?:làm việc\s+)?ở\s+([^,.!?]+)",
            r"(?:mình|tôi)\s+(?:vẫn\s+|đang\s+)?ở\s+([^,.!?]+)",
        ])

    capture("interests", [
        r"(?:mình|tôi)\s+(?:đang\s+)?(?:quan tâm|hứng thú)\s+(?:nhiều\s+)?(?:đến|tới)\s+([^.!?]+)",
        r"dài hạn:\s*(?:mình|tôi)\s+thích\s+([^.!?]+)",
    ])
    style_declaration = re.search(
        r"(?:mình|tôi)\s+(?:vẫn\s+)?(?:muốn|thích|ưu tiên).*?(?:trả lời|style|cách giải thích)|hãy\s+trả lời",
        text,
        re.IGNORECASE,
    )
    if style_declaration:
        style_sentence = next(
            (sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)
             if any(word in sentence.lower() for word in ("trả lời", "style", "cách giải thích"))),
            "",
        )
        if style_sentence:
            updates["response_style"] = style_sentence
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    selected = messages[-max_items:]
    lines = []
    for item in selected:
        compact = " ".join(item.get("content", "").split())
        lines.append(f"{item.get('role', 'unknown')}: {compact[:180]}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        summary = str(thread["summary"])
        total = estimate_tokens(summary) + sum(estimate_tokens(str(item["content"])) for item in messages)
        if total > self.threshold_tokens and len(messages) > self.keep_messages:
            older = messages[:-self.keep_messages]
            new_piece = summarize_messages(older)
            # Replace the previous summary with a bounded summary of the newest
            # compacted window. Stable profile facts live separately in User.md.
            thread["summary"] = new_piece
            thread["messages"] = messages[-self.keep_messages:]
            thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        return {"messages": list(thread["messages"]), "summary": str(thread["summary"]), "compactions": int(thread["compactions"])}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
