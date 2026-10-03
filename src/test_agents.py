from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    offline = ProviderConfig("openai", "gpt-4o-mini", 0.0)
    return LabConfig(tmp_path, tmp_path / "data", state_dir, 45, 2, offline, offline)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text("user-1", "# User Profile\n\n- location: Huế")
    assert path.exists()
    assert "Huế" in store.read_text("user-1")
    assert store.edit_text("user-1", "Huế", "Đà Nẵng") is True
    assert "Đà Nẵng" in store.read_text("user-1")
    assert store.file_size("user-1") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""

    manager = CompactMemoryManager(threshold_tokens=30, keep_messages=2)
    for index in range(10):
        manager.append("thread", "user", f"message {index} " + "nội dung dài " * 8)
    assert manager.compaction_count("thread") > 0
    assert len(manager.context("thread")["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    advanced.reply("user", "a-1", "Mình tên là Việt Anh.")
    baseline.reply("user", "b-1", "Mình tên là Việt Anh.")
    assert "Việt Anh" in advanced.reply("user", "a-2", "Mình tên gì?")["answer"]
    assert "Việt Anh" not in baseline.reply("user", "b-2", "Mình tên gì?")["answer"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(20):
        message = f"Lượt {index}: " + ("đây là ngữ cảnh dài để kiểm tra compact memory " * 12)
        baseline.reply("user", "baseline-long", message)
        advanced.reply("user", "advanced-long", message)
    assert advanced.compaction_count("advanced-long") > 0
    assert advanced.prompt_token_usage("advanced-long") < baseline.prompt_token_usage("baseline-long")
