# LLM cooperation pilot

A pilot study of how language-model agents, running inside real agent harnesses, play the noisy repeated prisoner's dilemma.

## Design

- **Game:** repeated prisoner's dilemma with payoffs T=5, R=3, P=1, S=0. Actions are labelled A (cooperate) and B (defect) so wording does not suggest a choice. Games last 10 rounds and players are told the length is unknown. Each executed action is flipped with 5% probability, and players are told this.
- **Agents (13):** Claude Code (Sonnet), Codex (gpt-6.1-sol, low reasoning effort), and OpenCode with 11 open and third-party models from the OpenCode Go subscription.
- **Isolation:** every decision is a fresh, stateless harness call in an empty directory. User instruction files (CLAUDE.md, AGENTS.md), MCP servers and tools are disabled.
- **Conditions:**
  - *No memory:* the agent sees only the round number. The opponent cannot influence its choices, so there is one baseline per agent.
  - *Memory:* the agent sees the executed history of the current game. It plays ALLD, TFT, WSLS, a uniform random player, and a copy of itself.
- **Replication:** 3 games per cell. Intended and executed actions and the raw harness replies are all logged.

## Analysis

- Cooperation rate per agent and opponent.
- Mean payoff per round.
- Memory-one strategy estimates: P(A) given the previous executed outcome, pooled over memory games. Each estimate is matched to the nearest of ALLC, TFT, WSLS, GRIM or ALLD.

## Run

```sh
uv sync
uv run python -m llm_cooperation_pilot.run      # resumable; PILOT_WORKERS, PILOT_AGENTS
uv run python -m llm_cooperation_pilot.analyse  # writes results/summary.txt and a heatmap
uv run pytest
```

Requires authenticated `claude`, `codex` and `opencode` CLIs.

## Limits

This is a pilot. Three games per cell gives wide uncertainty, model versions behind each harness change over time, and a 10-round horizon shows only short-run behaviour.

## Status (6 October 2026)

Claude Code and Codex are complete: 18 games each. The OpenCode Go run stopped partway when the subscription returned "Go usage limit exceeded". Five OpenCode models have partial data. The runner resumes from `results/rounds.jsonl` once quota is available.
