"""Noisy repeated prisoner's dilemma with neutral action labels.

Action "A" is cooperation and "B" is defection. Players are never told these
names, only the payoff table, so behaviour reflects incentives rather than the
connotations of the words.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

C, D = "A", "B"
PAYOFFS = {(C, C): (3, 3), (C, D): (0, 5), (D, C): (5, 0), (D, D): (1, 1)}

# A history is a list of (own executed action, opponent executed action).
History = list[tuple[str, str]]
Policy = Callable[[History, random.Random], str]


def all_c(history: History, rng: random.Random) -> str:
    return C


def all_d(history: History, rng: random.Random) -> str:
    return D


def tit_for_tat(history: History, rng: random.Random) -> str:
    return history[-1][1] if history else C


def win_stay_lose_shift(history: History, rng: random.Random) -> str:
    if not history:
        return C
    own, other = history[-1]
    return own if other == C else (D if own == C else C)


def random_policy(history: History, rng: random.Random) -> str:
    return rng.choice([C, D])


REFERENCE_STRATEGIES: dict[str, Policy] = {
    "ALLC": all_c,
    "ALLD": all_d,
    "TFT": tit_for_tat,
    "WSLS": win_stay_lose_shift,
    "Random": random_policy,
}


def flip(action: str) -> str:
    return D if action == C else C


@dataclass
class Round:
    round: int
    intended: tuple[str, str]
    executed: tuple[str, str]
    payoffs: tuple[int, int]


def play_round(
    number: int, intended: tuple[str, str], noise: float, rng: random.Random
) -> Round:
    executed = tuple(flip(a) if rng.random() < noise else a for a in intended)
    return Round(number, intended, executed, PAYOFFS[executed])


def memory_one(rounds: list[tuple[str, str]]) -> dict[str, tuple[int, int]]:
    """Count cooperation conditioned on the previous executed outcome.

    `rounds` is a sequence of (own intended action, previous executed outcome as
    two letters, e.g. "AB" meaning own A, opponent B), with the first round's
    outcome given as "". Returns {state: (cooperations, total)}.
    """
    counts = {state: [0, 0] for state in ("", "AA", "AB", "BA", "BB")}
    for action, previous in rounds:
        counts[previous][1] += 1
        counts[previous][0] += action == C
    return {state: (c, n) for state, (c, n) in counts.items()}
