from __future__ import annotations
import numpy as np


def board_to_emoji(board: np.ndarray) -> str:

    mapping = {1: "🔴", -1: "🟡", 0: "⚪"}
    rows = []
    for r in range(board.shape[0]):
        rows.append(" ".join(mapping[int(x)] for x in board[r]))
    footer = "   ".join([f"{i}" for i in range(board.shape[1])])
    return "\n".join(rows) + "\n\n" + footer


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default