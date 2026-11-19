"""Stress schedules (Section IV-E/F) and the neutral-conversation control."""
from __future__ import annotations

from itertools import combinations

CODES = {"P": "time_pressure", "E": "emotional_distress", "U": "moral_uncertainty",
         "D": "deception", "C": "conflict_of_interest"}
ORDER = ["P", "E", "U", "D", "C"]

# Eq. (trajectory schedules): each stressor appears twice in every round position.
TRAJECTORY = {
    "pi1": "PEUDC", "pi2": "EUDCP", "pi3": "UDCPE", "pi4": "DCPEU", "pi5": "CPEUD",
    "pi6": "CDUEP", "pi7": "PCDUE", "pi8": "EPCDU", "pi9": "UEPCD", "pi10": "DUEPC",
}


def trajectory_schedules() -> dict[str, list[str]]:
    return {k: [CODES[c] for c in v] for k, v in TRAJECTORY.items()}


def order_schedules() -> dict[str, list[str]]:
    out = {}
    for a, b in combinations(ORDER, 2):
        out[f"{a}>{b}"] = [CODES[a], CODES[b]]
        out[f"{b}>{a}"] = [CODES[b], CODES[a]]
    return out


def order_pairs() -> list[tuple[str, str, str]]:
    """(pair label, schedule i->j, schedule j->i) for the 10 unordered pairs."""
    return [(f"{a}/{b}", f"{a}>{b}", f"{b}>{a}") for a, b in combinations(ORDER, 2)]


def neutral_schedule(rounds: int = 5) -> dict[str, list[str]]:
    return {"neutral": ["neutral"] * rounds}


def check_balance() -> bool:
    """Every stressor appears exactly twice in every round position."""
    sched = list(TRAJECTORY.values())
    for pos in range(5):
        col = [s[pos] for s in sched]
        if any(col.count(c) != 2 for c in ORDER):
            return False
    return all(sorted(s) == sorted(ORDER) for s in sched)
