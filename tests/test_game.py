import random

from llm_cooperation_pilot.game import (
    C,
    D,
    memory_one,
    play_round,
    tit_for_tat,
    win_stay_lose_shift,
)


def test_tit_for_tat_copies_opponent():
    rng = random.Random(0)
    assert tit_for_tat([], rng) == C
    assert tit_for_tat([(C, D)], rng) == D
    assert tit_for_tat([(D, C)], rng) == C


def test_win_stay_lose_shift():
    rng = random.Random(0)
    assert win_stay_lose_shift([(C, C)], rng) == C
    assert win_stay_lose_shift([(C, D)], rng) == D
    assert win_stay_lose_shift([(D, C)], rng) == D
    assert win_stay_lose_shift([(D, D)], rng) == C


def test_noise_flips_and_payoffs_follow_executed_actions():
    result = play_round(1, (C, C), noise=1.0, rng=random.Random(0))
    assert result.executed == (D, D)
    assert result.payoffs == (1, 1)
    assert play_round(1, (C, D), noise=0.0, rng=random.Random(0)).payoffs == (0, 5)


def test_memory_one_counts():
    counts = memory_one([(C, ""), (D, "AA"), (C, "AA"), (C, "BB")])
    assert counts[""] == (1, 1)
    assert counts["AA"] == (1, 2)
    assert counts["BB"] == (1, 1)
    assert counts["AB"] == (0, 0)
