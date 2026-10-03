from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0
    lowered = answer.casefold()
    hits = sum(item.casefold() in lowered for item in expected)
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Return a lightweight deterministic quality score for offline mode."""

    recall = recall_points(answer, expected)
    concise = 1.0 if 3 <= len(answer.split()) <= 80 else 0.5
    return round(0.8 * recall + 0.2 * concise, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    start_sizes: dict[str, int] = {}
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    thread_ids: list[str] = []
    for conversation in conversations:
        user_id = conversation["user_id"]
        if hasattr(agent, "memory_file_size") and user_id not in start_sizes:
            start_sizes[user_id] = agent.memory_file_size(user_id)
        thread_id = f"{agent_name.lower()}-{conversation['id']}"
        thread_ids.append(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, turn)
        for index, question in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{index}"
            result = agent.reply(user_id, recall_thread, question["question"])
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(result["answer"], expected))
            quality_scores.append(heuristic_quality(result["answer"], expected))
            thread_ids.append(recall_thread)
    token_total = sum(agent.token_usage(thread) for thread in thread_ids)
    prompt_total = sum(agent.prompt_token_usage(thread) for thread in thread_ids)
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(max(0, agent.memory_file_size(user) - size) for user, size in start_sizes.items())
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=token_total,
        prompt_tokens_processed=prompt_total,
        recall_score=round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
        memory_growth_bytes=memory_growth,
        compactions=sum(agent.compaction_count(thread) for thread in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a readable table."""

    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
    values = [[r.agent_name, r.agent_tokens_only, r.prompt_tokens_processed, f"{r.recall_score:.3f}", f"{r.response_quality:.3f}", r.memory_growth_bytes, r.compactions] for r in rows]
    try:
        from tabulate import tabulate
        return tabulate(values, headers=headers, tablefmt="github")
    except ImportError:
        return "\n".join([" | ".join(headers)] + [" | ".join(map(str, row)) for row in values])


def main() -> None:
    """Run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    # - load both datasets from root/data
    # - initialize baseline and advanced agents
    # - run benchmarks
    # - print comparison tables
    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]
    for title, path in suites:
        conversations = load_conversations(path)
        baseline = BaselineAgent(config, force_offline=True)
        advanced = AdvancedAgent(config, force_offline=True)
        rows = [
            run_agent_benchmark("Baseline", baseline, conversations, config),
            run_agent_benchmark("Advanced", advanced, conversations, config),
        ]
        print(f"\n## {title}")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
