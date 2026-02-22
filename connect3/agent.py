from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


State = Tuple[int, ...]


@dataclass
class QLearningAgent:
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.2
    q: Dict[Tuple[State, int], float] = None

    def __post_init__(self):
        if self.q is None:
            self.q = {}

    def get_q(self, s: State, a: int) -> float:
        return float(self.q.get((s, a), 0.0))

    def set_q(self, s: State, a: int, v: float) -> None:
        self.q[(s, a)] = float(v)

    def best_action(self, s: State, valid_actions: List[int]) -> int:
        qs = [self.get_q(s, a) for a in valid_actions]
        max_q = max(qs)
        best = [a for a, qv in zip(valid_actions, qs) if qv == max_q]
        return int(np.random.choice(best))

    def act(self, s: State, valid_actions: List[int], rng: np.random.Generator, greedy: bool = False) -> int:
        if greedy:
            return self.best_action(s, valid_actions)

        if rng.random() < self.epsilon:
            return int(rng.choice(valid_actions))
        return self.best_action(s, valid_actions)

    def update(self, s: State, a: int, r: float, s2: State, valid_actions_s2: List[int], done: bool) -> None:
        old = self.get_q(s, a)
        if done:
            target = r
        else:
            next_best = max(self.get_q(s2, a2) for a2 in valid_actions_s2) if valid_actions_s2 else 0.0
            target = r + self.gamma * next_best
        new = (1 - self.alpha) * old + self.alpha * target
        self.set_q(s, a, new)