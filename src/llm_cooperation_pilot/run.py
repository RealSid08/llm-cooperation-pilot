"""Run the pilot through real agent harnesses: Claude Code, Codex and OpenCode.

Each decision is one fresh, stateless harness call in an empty directory, with
user instruction files, MCP servers and tools disabled. "memory" agents see the
game history in the prompt; "no_memory" agents see only the round number.

Appends one JSON line per round to results/rounds.jsonl and resumes from it.
Usage: uv run python -m llm_cooperation_pilot.run
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .game import REFERENCE_STRATEGIES, History, play_round

OPENCODE_MODELS = [
    "deepseek-v4-pro",
    "glm-5.3",
    "kimi-k3",
    "qwen3.8-max",
    "minimax-m3",
    "mimo-v2.6-pro",
    "grok-4.7",
    "gpt-6-luna",
    "longcat-2.0",
    "hy3",
    "muse-spark-1.3-contributor",
]
AGENTS = ["claude-code/sonnet", "codex/gpt-6.1-sol"] + [
    f"opencode/{m}" for m in OPENCODE_MODELS
]
OPPONENTS = ["ALLD", "TFT", "WSLS", "Random"]
ROUNDS = 10
REPS = 3
NOISE = 0.05
TIMEOUT = 300
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "rounds.jsonl"
SANDBOX = Path(tempfile.mkdtemp(prefix="coop-pilot-"))

RULES = f"""You are playing a repeated game against another player. In every round, \
both players simultaneously choose A or B.

Points per round:
- You choose A, other chooses A: you get 3, they get 3
- You choose A, other chooses B: you get 0, they get 5
- You choose B, other chooses A: you get 5, they get 0
- You choose B, other chooses B: you get 1, they get 1

The game lasts an unknown number of rounds. In each round there is a \
{int(NOISE * 100)}% chance that a player's choice is accidentally switched to the \
other option. Your goal is to maximise your own total points over the whole game.

Do not use any tools. Reply with only a single letter: A or B."""

_lock = threading.Lock()


def describe(history: History, condition: str, number: int) -> str:
    if condition == "no_memory" or not history:
        return f"{RULES}\n\nThis is round {number}. Your choice?"
    lines = "\n".join(
        f"Round {i}: you played {own}, other played {other}"
        for i, (own, other) in enumerate(history, start=1)
    )
    return (
        f"{RULES}\n\nPrevious rounds (as played, after any accidental switches):\n"
        f"{lines}\n\nThis is round {number}. Your choice?"
    )


def command(agent: str, prompt: str, workdir: Path) -> tuple[list[str], dict]:
    harness, model = agent.split("/", 1)
    env = dict(os.environ)
    if harness == "claude-code":
        return [
            "claude", "-p", prompt, "--model", model,
            "--setting-sources", "", "--tools", "", "--strict-mcp-config",
            "--no-session-persistence",
        ], env  # fmt: skip
    if harness == "codex":
        # A private CODEX_HOME keeps the user's AGENTS.md and config out.
        env["CODEX_HOME"] = str(SANDBOX / "codex-home")
        return [
            "codex", "exec", "-m", model, "-c", 'model_reasoning_effort="low"',
            "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
            "-s", "read-only", "-o", str(workdir / "last.txt"), prompt,
        ], env  # fmt: skip
    env["XDG_CONFIG_HOME"] = str(SANDBOX / "xdg")
    env["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
    return [
        "opencode", "run", "--pure", "--format", "json",
        "-m", f"opencode-go/{model}", prompt,
    ], env  # fmt: skip


def reply_text(agent: str, stdout: str, workdir: Path) -> str:
    if agent.startswith("codex/"):
        path = workdir / "last.txt"
        return path.read_text() if path.exists() else ""
    if agent.startswith("opencode/"):
        parts = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "text":
                parts.append(event["part"].get("text", ""))
        return "".join(parts)
    return stdout


def parse_move(text: str) -> str | None:
    if text.strip() in ("A", "B"):
        return text.strip()
    letters = re.findall(r"\b[AB]\b", text)
    return letters[-1] if letters else None


def ask(agent: str, prompt: str) -> tuple[str, str]:
    text = ""
    for _ in range(3):
        with tempfile.TemporaryDirectory(dir=SANDBOX) as tmp:
            workdir = Path(tmp)
            args, env = command(agent, prompt, workdir)
            try:
                done = subprocess.run(
                    args, cwd=workdir, env=env, capture_output=True, text=True,
                    timeout=TIMEOUT, stdin=subprocess.DEVNULL, check=False,
                )  # fmt: skip
                text = reply_text(agent, done.stdout, workdir)
            except subprocess.TimeoutExpired:
                text = ""
        move = parse_move(text)
        if move:
            return move, text
    raise RuntimeError(f"{agent} gave no valid move: {text[-300:]!r}")


def play_game(player_a: str, player_b: str, condition: str, rep: int) -> None:
    rng = random.Random(f"{player_a}|{player_b}|{condition}|{rep}")
    history_a: History = []
    history_b: History = []
    records = []
    for number in range(1, ROUNDS + 1):
        moves, raw = [], []
        for name, history in ((player_a, history_a), (player_b, history_b)):
            if name in REFERENCE_STRATEGIES:
                moves.append(REFERENCE_STRATEGIES[name](history, rng))
                raw.append(None)
            else:
                move, text = ask(name, describe(history, condition, number))
                moves.append(move)
                raw.append(text)
        result = play_round(number, (moves[0], moves[1]), NOISE, rng)
        history_a.append(result.executed)
        history_b.append(result.executed[::-1])
        records.append(
            {
                "player_a": player_a,
                "player_b": player_b,
                "condition": condition,
                "rep": rep,
                "round": number,
                "intended_a": result.intended[0],
                "intended_b": result.intended[1],
                "executed_a": result.executed[0],
                "executed_b": result.executed[1],
                "payoff_a": result.payoffs[0],
                "payoff_b": result.payoffs[1],
                "raw_a": raw[0],
                "raw_b": raw[1],
            }
        )
    with _lock, OUT.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"done {player_a} vs {player_b} {condition} rep {rep}", flush=True)


def jobs() -> list[tuple[str, str, str, int]]:
    planned = []
    for rep in range(REPS):
        for agent in AGENTS:
            # Without memory the opponent cannot affect choices, so one baseline.
            planned.append((agent, "Random", "no_memory", rep))
            planned += [(agent, opp, "memory", rep) for opp in OPPONENTS]
            planned.append((agent, agent, "memory", rep))
    return planned


def done_games() -> set[tuple]:
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {
            (r["player_a"], r["player_b"], r["condition"], r["rep"])
            for r in map(json.loads, f)
        }


def setup_sandbox() -> None:
    (SANDBOX / "xdg").mkdir(exist_ok=True)
    codex_home = SANDBOX / "codex-home"
    codex_home.mkdir(exist_ok=True)
    (codex_home / "auth.json").symlink_to(Path.home() / ".codex" / "auth.json")


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    setup_sandbox()
    finished = done_games()
    selected = os.environ.get("PILOT_AGENTS")
    todo = [
        job
        for job in jobs()
        if job not in finished and (not selected or job[0] in selected.split(","))
    ]
    print(f"{len(todo)} games to play, sandbox {SANDBOX}", flush=True)
    workers = int(os.environ.get("PILOT_WORKERS", "16"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(play_game, *job) for job in todo]
        for future in futures:
            try:
                future.result()
            except RuntimeError as error:  # keep other games running
                print(f"FAILED: {error}", flush=True)


if __name__ == "__main__":
    main()
