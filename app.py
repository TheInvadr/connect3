import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from connect3.env import Connect3Env
from connect3.agent import QLearningAgent
from connect3.train import TrainConfig, train_q_agent, evaluate
from connect3.opponents import random_opponent, heuristic_opponent
from connect3.utils import board_to_emoji

import time

st.set_page_config(page_title="Connect-3 RL (4x4)", layout="wide")
st.title("🎮 4×4 Connect-3 — Q-Learning Agent")
st.caption("A simplified reinforcement learning project: tabular Q-learning + ε-greedy exploration + interactive demo. By Rahul Nadipalli")

# -----------------------
# Session state
# -----------------------
if "pending_agent" not in st.session_state:
    st.session_state.pending_agent = False
if "pending_state_index" not in st.session_state:
    st.session_state.pending_state_index = None
if "trained_agent" not in st.session_state:
    st.session_state.trained_agent = None
if "train_df" not in st.session_state:
    st.session_state.train_df = None
if "play_env" not in st.session_state:
    st.session_state.play_env = Connect3Env(step_penalty=0.0)
    st.session_state.play_env.reset()
if "play_done" not in st.session_state:
    st.session_state.play_done = False
if "play_message" not in st.session_state:
    st.session_state.play_message = ""
if "human_starts" not in st.session_state:
    st.session_state.human_starts = True


tab_train, tab_play = st.tabs(["Train", "Play"])

with tab_train:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Training controls")

        episodes = st.slider("Episodes", 1000, 50000, 12000, step=1000)
        opponent = st.selectbox("Opponent", ["random", "heuristic"], index=0)
        alpha = st.slider("Learning rate (α)", 0.05, 0.8, 0.2, step=0.05)
        gamma = st.slider("Discount (γ)", 0.5, 0.99, 0.95, step=0.01)
        eps_start = st.slider("ε start", 0.05, 0.9, 0.4, step=0.05)
        eps_end = st.slider("ε end", 0.0, 0.2, 0.05, step=0.01)
        eps_decay_frac = st.slider("ε decay fraction", 0.2, 1.0, 0.7, step=0.05)
        step_penalty = st.select_slider("Step penalty (optional)", options=[0.0, -0.005, -0.01, -0.02], value=0.0)

        train_btn = st.button("Train", type="primary")

    with right:
        st.subheader("Results")

        if train_btn:
            cfg = TrainConfig(
                episodes=episodes,
                opponent=opponent,
                alpha=alpha,
                gamma=gamma,
                epsilon_start=eps_start,
                epsilon_end=eps_end,
                epsilon_decay_frac=eps_decay_frac,
                step_penalty=step_penalty,
                seed=42,
            )
            with st.spinner("Training..."):
                agent, df = train_q_agent(cfg)

            st.session_state.trained_agent = agent
            st.session_state.train_df = df

        if st.session_state.train_df is None:
            st.info("Train an agent to see learning curves and evaluation results.")
        else:
            df = st.session_state.train_df

            c1, c2, c3 = st.columns(3)
            c1.metric("Episodes", f"{int(df['episode'].iloc[-1]):,}")
            c2.metric("Final ε", f"{df['epsilon'].iloc[-1]:.3f}")
            c3.metric("Win rate (last 200)", f"{df['win_rate_200'].iloc[-1]:.3f}")

            fig = px.line(
                df,
                x="episode",
                y=["win_rate_200", "loss_rate_200", "draw_rate_200"],
                title="Moving outcome rates (window=200)",
            )
            st.plotly_chart(fig, use_container_width=True)
            agent = st.session_state.trained_agent
            eval_random = evaluate(agent, episodes=2000, opponent="random", seed=1)
            eval_heur = evaluate(agent, episodes=2000, opponent="heuristic", seed=2)

            st.markdown("### Evaluation (greedy policy)")
            st.write(pd.DataFrame([{
                "Opponent": "random",
                "Win rate": eval_random["win_rate"],
                "Loss rate": eval_random["loss_rate"],
                "Draw rate": eval_random["draw_rate"],
            },{
                "Opponent": "heuristic",
                "Win rate": eval_heur["win_rate"],
                "Loss rate": eval_heur["loss_rate"],
                "Draw rate": eval_heur["draw_rate"],
            }]))

# -----------------------
# PLAY TAB
# -----------------------
with tab_play:
    st.subheader("Play against the trained agent")

    if st.session_state.trained_agent is None:
        st.warning("Train an agent first in the **Train** tab.")
    else:
        env: Connect3Env = st.session_state.play_env
        agent: QLearningAgent = st.session_state.trained_agent
        rng = np.random.default_rng(123)

        top = st.columns([1, 1, 2])
        with top[0]:
            if st.button("Reset game"):
                env.reset()
                st.session_state.play_done = False
                st.session_state.play_message = ""
        with top[1]:
            st.session_state.human_starts = st.toggle("Human starts", value=st.session_state.human_starts)

    
        if (not st.session_state.human_starts) and (not st.session_state.play_done) and np.all(env.board == 0):
            s = env.get_state()
            a = agent.act(s, env.valid_actions(), rng=rng, greedy=True)
            env.drop(a, player=+1)

        st.markdown("#### Board")
        st.code(board_to_emoji(env.board))

        # Column buttons for human move
        cols = st.columns(env.COLS)
        human_move = None
        for c in range(env.COLS):
            with cols[c]:
                if st.button(f"Drop in {c}", disabled=st.session_state.play_done or (env.board[0, c] != 0)):
                    human_move = c

        def finalize_if_terminal(env):
            w = env.check_winner()
            if w != 0:
                st.session_state.play_done = True
                st.session_state.play_message = "You win! 🟡" if w == -1 else "Agent wins! 🔴"
                return True
            if env.is_full():
                st.session_state.play_done = True
                st.session_state.play_message = "Draw."
                return True
            return False


        import plotly.graph_objects as go
        import numpy as np


        if st.session_state.pending_agent and (not st.session_state.play_done):
            st.markdown("### Agent turn")


            s = env.get_state()
            valid = env.valid_actions()

            all_actions = list(range(env.COLS))
            q_list = []
            for a in all_actions:
                if a in valid:
                    q_list.append(agent.get_q(s, a))
                else:
                    q_list.append(None) 
            finite = [v for v in q_list if isinstance(v, (int, float))]
            max_abs = float(np.max(np.abs(finite))) if finite else 0.0
            ypad = max(0.5, max_abs * 1.2)  # ensures non-flat axis



            with st.spinner("Agent thinking..."):
                time.sleep(0.4)

                a = agent.act(s, valid, rng=rng, greedy=True)
                env.drop(a, player=+1)

            st.session_state.pending_agent = False
            finalize_if_terminal(env)
            st.rerun()


        # --- Human plays---
        if human_move is not None and (not st.session_state.play_done) and (not st.session_state.pending_agent):
            ok = env.drop(human_move, player=-1)
            if not ok:
                st.session_state.play_message = "Invalid move (column full)."
                st.rerun()
            else:
                if not finalize_if_terminal(env):
                    st.session_state.pending_agent = True
                st.rerun()

        if st.session_state.play_message:
            st.success(st.session_state.play_message)



