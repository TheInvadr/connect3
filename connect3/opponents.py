from __future__ import annotations
from typing import List, Optional
import numpy as np
from .env import Connect3Env


def random_opponent(env: Connect3Env, rng: np.random.Generator) -> int:
    actions = env.valid_actions()
    return int(rng.choice(actions))


def _winning_move(env: Connect3Env, player: int) -> Optional[int]:
    for a in env.valid_actions():
        backup = env.clone_board()
        env.drop(a, player=player)
        w = env.check_winner()
        env.board[:] = backup
        if w == player:
            return a
    return None


def heuristic_opponent(env: Connect3Env, rng: np.random.Generator) -> int:

    win = _winning_move(env, player=-1)
    if win is not None:
        return win

    block = _winning_move(env, player=+1)
    if block is not None:
        return block

    return random_opponent(env, rng)