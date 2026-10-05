"""Summarise results/rounds.jsonl into tables and a heatmap.

Usage: uv run python -m llm_cooperation_pilot.analyse
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .game import C, memory_one

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Reference memory-one strategies: P(A | previous own, other) for AA, AB, BA, BB.
STATES = ["AA", "AB", "BA", "BB"]
REFERENCE = {
    "ALLC": (1, 1, 1, 1),
    "TFT": (1, 0, 1, 0),
    "WSLS": (1, 0, 0, 1),
    "GRIM": (1, 0, 0, 0),
    "ALLD": (0, 0, 0, 0),
}
COLUMNS = ["No memory", "vs ALLD", "vs TFT", "vs WSLS", "vs Random", "Self-play"]
RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def load() -> pd.DataFrame:
    with (RESULTS / "rounds.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    # Give every agent decision its own row, from that agent's perspective.
    decisions = []
    for r in rows:
        for me, other in (("a", "b"), ("b", "a")):
            agent = r[f"player_{me}"]
            if r[f"raw_{me}"] is None:
                continue
            opponent = r[f"player_{other}"]
            decisions.append(
                {
                    "agent": agent,
                    "column": (
                        "No memory"
                        if r["condition"] == "no_memory"
                        else "Self-play"
                        if opponent == agent
                        else f"vs {opponent}"
                    ),
                    "game": (
                        r["player_a"],
                        r["player_b"],
                        r["condition"],
                        r["rep"],
                        me,
                    ),
                    "round": r["round"],
                    "intended": r[f"intended_{me}"],
                    "own_exec": r[f"executed_{me}"],
                    "other_exec": r[f"executed_{other}"],
                    "payoff": r[f"payoff_{me}"],
                }
            )
    return pd.DataFrame(decisions)


def memory_one_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    with_memory = df[df["column"] != "No memory"].sort_values(["game", "round"])
    for agent, frame in with_memory.groupby("agent"):
        pairs = []
        for _, game in frame.groupby("game"):
            previous = ""
            for r in game.itertuples():
                pairs.append((r.intended, previous))
                previous = r.own_exec + r.other_exec
        counts = memory_one(pairs)
        probs = {
            s: counts[s][0] / counts[s][1] if counts[s][1] else math.nan for s in STATES
        }
        observed = [s for s in STATES if counts[s][1]]
        nearest = min(
            REFERENCE,
            key=lambda name: sum(
                (probs[s] - REFERENCE[name][STATES.index(s)]) ** 2 for s in observed
            ),
        )
        rows.append(
            {
                "agent": agent,
                "first_move_A": counts[""][0] / counts[""][1],
                **{f"P(A|{s})": probs[s] for s in STATES},
                **{f"n({s})": counts[s][1] for s in STATES},
                "nearest": nearest,
            }
        )
    return pd.DataFrame(rows).set_index("agent")


def heatmap(rates: pd.DataFrame, path: Path) -> None:
    mpl.rcParams.update({"font.family": "Helvetica", "font.size": 8.5})
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(rates) + 1.0), dpi=200)
    ax.imshow(rates.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), value in pd.DataFrame(rates.to_numpy()).stack().items():
        ax.text(
            j, i, f"{value:.0%}", ha="center", va="center", fontsize=7.5,
            color="#ffffff" if value > 0.55 else "#0b0b0b",
        )  # fmt: skip
    ax.set_xticks(range(len(rates.columns)), rates.columns)
    ax.set_yticks(range(len(rates)), rates.index)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors="#52514e")
    ax.set_xticks([x - 0.5 for x in range(1, len(rates.columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rates))], minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def main() -> None:
    df = load()
    rates = (
        df.assign(coop=df["intended"] == C)
        .pivot_table(index="agent", columns="column", values="coop", aggfunc="mean")
        .reindex(columns=COLUMNS)
    )
    rates = rates.sort_values("vs TFT", ascending=False)
    payoffs = df.pivot_table(
        index="agent", columns="column", values="payoff", aggfunc="mean"
    )
    games = df.groupby(["agent", "column"])["game"].nunique().unstack()
    m1 = memory_one_table(df).reindex(rates.index)

    pd.options.display.width = 200
    pd.options.display.float_format = "{:.2f}".format
    report = "\n\n".join(
        [
            "Cooperation rate (intended A):\n" + rates.to_string(),
            "Mean payoff per round:\n"
            + payoffs.reindex(index=rates.index, columns=COLUMNS).to_string(),
            "Games per cell:\n"
            + games.reindex(index=rates.index, columns=COLUMNS).to_string(),
            "Memory-one estimates (memory games, pooled):\n" + m1.to_string(),
        ]
    )
    (RESULTS / "summary.txt").write_text(report + "\n")
    rates.to_csv(RESULTS / "cooperation_rates.csv")
    m1.to_csv(RESULTS / "memory_one.csv")
    heatmap(rates, RESULTS / "cooperation_heatmap.png")
    profile_chart(
        m1,
        {
            "claude-code/sonnet": "Claude Code (Sonnet)",
            "codex/gpt-6.1-sol": "Codex (gpt-6.1-sol)",
        },
        RESULTS / "memory_one_profile.png",
    )
    print(report)


def profile_chart(m1: pd.DataFrame, agents: dict[str, str], path: Path) -> None:
    """Grouped bars of P(cooperate | previous outcome) for selected agents."""
    mpl.rcParams.update({"font.family": "Helvetica", "font.size": 8.5})
    labels = [
        "Both\ncooperated",
        "Agent was\nexploited",
        "Agent\nexploited",
        "Both\ndefected",
    ]
    colours = ["#2a78d6", "#eb6834"]
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 2.3), dpi=220)
    for i, (agent, name) in enumerate(agents.items()):
        values = [m1.loc[agent, f"P(A|{s})"] for s in STATES]
        xs = [x + (i - 0.5) * (width + 0.02) for x in range(len(STATES))]
        ax.bar(
            xs,
            values,
            width,
            color=colours[i],
            label=name,
            edgecolor="#fcfcfb",
            linewidth=1,
        )
        for x, v in zip(xs, values, strict=True):
            ax.text(
                x,
                v + 0.03,
                f"{v:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#52514e",
            )
    ax.set_xticks(range(len(STATES)), labels)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_ylabel("Cooperated next round", color="#52514e")
    ax.tick_params(length=0, colors="#52514e")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper right", ncols=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    main()
