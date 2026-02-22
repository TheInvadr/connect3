from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .env import Connect3Env
from .agent import QLearningAgent
from .opponents import random_opponent, heuristic_opponent


OpponentFn = Callable[[Connect3Env, np.random.Generator], int]


@dataclass
class TrainConfig:
    episodes: int = 10000
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon_start: float = 0.4
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.7  # fraction of training over which epsilon decays
    step_penalty: float = 0.0
    opponent: str = "random"  # "random" or "heuristic"
    seed: int = 42


def get_opponent(name: str) -> OpponentFn:
    if name == "heuristic":
        return heuristic_opponent
    return random_opponent


def train_q_agent(cfg: TrainConfig) -> Tuple[QLearningAgent, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    env = Connect3Env(step_penalty=cfg.step_penalty)

    agent = QLearningAgent(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon_start)

    opp_fn = get_opponent(cfg.opponent)

    results: List[Dict[str, float]] = []
    decay_episodes = max(1, int(cfg.episodes * cfg.epsilon_decay_frac))

    def eps_for_ep(ep: int) -> float:
        if ep >= decay_episodes:
            return cfg.epsilon_end
        t = ep / decay_episodes
        return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)

    for ep in range(cfg.episodes):
        agent.epsilon = eps_for_ep(ep)

        s = env.reset()
        done = False

        while not done:
            valid = env.valid_actions()
            a = agent.act(s, valid, rng=rng, greedy=False)

            opp_a = opp_fn(env, rng)  # opponent chooses based on current board
            res = env.step(agent_action=a, opponent_action=opp_a)

            s2 = res.state
            done = res.done
            valid2 = env.valid_actions() if not done else []

            agent.update(s, a, res.reward, s2, valid2, done)
            s = s2

        winner = res.info.get("winner", None)
        win = 1.0 if winner == +1 else 0.0
        loss = 1.0 if winner == -1 else 0.0
        draw = 1.0 if winner == 0 else 0.0

        results.append({
            "episode": ep + 1,
            "epsilon": agent.epsilon,
            "reward": float(res.reward),
            "win": win,
            "loss": loss,
            "draw": draw,
        })

    df = pd.DataFrame(results)
    df["win_rate_200"] = df["win"].rolling(200, min_periods=1).mean()
    df["loss_rate_200"] = df["loss"].rolling(200, min_periods=1).mean()
    df["draw_rate_200"] = df["draw"].rolling(200, min_periods=1).mean()
    return agent, df


def evaluate(agent: QLearningAgent, episodes: int = 2000, opponent: str = "random", seed: int = 123) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    env = Connect3Env(step_penalty=0.0)
    opp_fn = get_opponent(opponent)

    wins = losses = draws = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        if rng.random() < 0.5:
            opp_first = opp_fn(env, rng)
            env.drop(opp_first, player=-1)
            if env.check_winner() == -1:
                results.append({
                    "episode": ep + 1,
                    "epsilon": agent.epsilon,
                    "reward": -1.0,
                    "win": 0.0,
                    "loss": 1.0,
                    "draw": 0.0,
                })
                continue
            if env.is_full():
                results.append({
                    "episode": ep + 1,
                    "epsilon": agent.epsilon,
                    "reward": 0.0,
                    "win": 0.0,
                    "loss": 0.0,
                    "draw": 1.0,
                })
                continue

            s = env.get_state()
        while not done:
            valid = env.valid_actions()
            a = agent.act(s, valid, rng=rng, greedy=True)  
            opp_a = opp_fn(env, rng)
            res = env.step(a, opp_a)
            s = res.state
            done = res.done

        w = res.info.get("winner", 0)
        if w == +1:
            wins += 1
        elif w == -1:
            losses += 1
        else:
            draws += 1

    total = wins + losses + draws
    return {
        "episodes": total,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
    }