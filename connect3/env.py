from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class StepResult:
    state: Tuple[int, ...]
    reward: float
    done: bool
    info: Dict[str, object]


class Connect3Env:
    """
    4x4 "Connect-3" drop-token game.
    Agent is +1, Opponent is -1.
    """

    ROWS = 4
    COLS = 4
    CONNECT_N = 3

    def __init__(self, step_penalty: float = 0.0):
        self.step_penalty = float(step_penalty)
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)

    def reset(self) -> Tuple[int, ...]:
        self.board[:] = 0
        return self.get_state()

    def clone_board(self) -> np.ndarray:
        return self.board.copy()

    def get_state(self) -> Tuple[int, ...]:
        # row-major flatten
        return tuple(int(x) for x in self.board.flatten())

    def valid_actions(self) -> List[int]:
        # action = column index, valid if top cell is empty
        return [c for c in range(self.COLS) if self.board[0, c] == 0]

    def drop(self, col: int, player: int) -> bool:
        if col < 0 or col >= self.COLS:
            return False
        if self.board[0, col] != 0:
            return False
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = np.int8(player)
                return True
        return False

    def is_full(self) -> bool:
        return not any(self.board[0, c] == 0 for c in range(self.COLS))

    def check_winner(self) -> int:

        B = self.board
        R, C, N = self.ROWS, self.COLS, self.CONNECT_N

        # Horizontal
        for r in range(R):
            for c in range(C - N + 1):
                window = B[r, c:c+N]
                s = int(window.sum())
                if abs(s) == N and np.all(window != 0):
                    return int(np.sign(s))

        # Vertical
        for c in range(C):
            for r in range(R - N + 1):
                window = B[r:r+N, c]
                s = int(window.sum())
                if abs(s) == N and np.all(window != 0):
                    return int(np.sign(s))

        # Diagonal down-right
        for r in range(R - N + 1):
            for c in range(C - N + 1):
                window = np.array([B[r+i, c+i] for i in range(N)], dtype=np.int8)
                s = int(window.sum())
                if abs(s) == N and np.all(window != 0):
                    return int(np.sign(s))

        # Diagonal down-left
        for r in range(R - N + 1):
            for c in range(N - 1, C):
                window = np.array([B[r+i, c-i] for i in range(N)], dtype=np.int8)
                s = int(window.sum())
                if abs(s) == N and np.all(window != 0):
                    return int(np.sign(s))

        return 0

    def step(self, agent_action: int, opponent_action: Optional[int]) -> StepResult:

        info: Dict[str, object] = {}

        # Agent move
        ok = self.drop(agent_action, player=+1)
        if not ok:
            # Illegal move: strong penalty and terminate (teaches quickly)
            return StepResult(self.get_state(), reward=-1.0, done=True, info={"illegal_move": True})

        winner = self.check_winner()
        if winner == +1:
            return StepResult(self.get_state(), reward=1.0, done=True, info={"winner": +1})

        if self.is_full():
            return StepResult(self.get_state(), reward=0.0, done=True, info={"winner": 0})

        # Opponent move
        if opponent_action is not None:
            ok2 = self.drop(opponent_action, player=-1)
            if not ok2:
                # Opponent illegal shouldn’t happen if policy uses valid_actions,
                # but if it does, treat as agent win.
                return StepResult(self.get_state(), reward=1.0, done=True, info={"winner": +1, "opponent_illegal": True})

            winner = self.check_winner()
            if winner == -1:
                return StepResult(self.get_state(), reward=-1.0, done=True, info={"winner": -1})

            if self.is_full():
                return StepResult(self.get_state(), reward=0.0, done=True, info={"winner": 0})

        # Non-terminal transition
        # Optional small step penalty to encourage faster wins.
        return StepResult(self.get_state(), reward=float(self.step_penalty), done=False, info=info)