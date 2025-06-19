#!/usr/bin/env python
"""
Simple Tabular Q-learning agent. Tabular Q(s,a) with ε-greedy exploration.

The agent must choose out of the existing DOM elements which one to use the *click* action on (simplified action space).

We state a seed when resetting the environment to simpllify the state space. If we do not, then the DOM elements aka the
state representation is not deterministic. 

-----
python simple_q_learning.py --env miniwob/click-button-v1
python simple_q_learning.py --env miniwob/click-link
python simple_q_learning.py --episodes 1000 
# ex: python simple_q_learning.py --env miniwob/click-test --episodes 100 --eval 30 --gui
"""
from __future__ import annotations

import argparse
import collections
import random
from typing import DefaultDict, Tuple

import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

# Q-learning hyper-parameters
ALPHA = 0.20
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
MAX_ELEMS = 40


# convert an observation dict → *hashable* state representation.
# state = (utterance,  (text_0, text_1, ... text_{k-1}))
# ex: "click the "Yes" button" → ("click the", "yes", "no", " ")
def state_repr(obs: dict) -> Tuple[str, Tuple[str, ...]]:
    utterance = obs.get("utterance", "").strip().lower()
    texts = tuple((e.get("text", "").strip().lower())
                  for e in obs["dom_elements"][:MAX_ELEMS])
    return (utterance, texts)


# q-learning loop
def q_learning(env_id: str,
               episodes: int = 500,
               seed: int | None = 42,
               verbose: bool = True,
               gui: bool = False):
    gym.register_envs(miniwob)
    render_mode = "human" if gui else None
    env = gym.make(env_id, render_mode=render_mode)

    # q-table: defaultdict mapping *state* → array[ num_actions ]
    q_table: DefaultDict[Tuple[str, Tuple[
        str, ...]], list[float]] = collections.defaultdict(
            lambda: [0.0] *
            MAX_ELEMS  # we pre-allocate with MAX_ELEMS; unused entries ignored
        )

    eps = EPS_START
    rng = random.Random(seed)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed)
        s = state_repr(obs)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # ε-greedy action selection
            if rng.random() < eps:
                a = rng.randrange(min(len(obs["dom_elements"]), MAX_ELEMS))
            else:
                # take argmax over the currently *known* Q-values for s
                q_vals = q_table[s][:len(obs["dom_elements"])]
                a = int(max(range(len(q_vals)), key=q_vals.__getitem__))

            # **click** the chosen DOM element.
            ref = obs["dom_elements"][a]["ref"]
            action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT,
                                                 ref=ref)

            # environment step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            s_next = state_repr(next_obs)

            # q-learning update         Q(s,a) ← (1-α)Q + α[r + γ max_{a'} Q(s',a')]
            max_next_q = max(q_table[s_next]) if not done else 0.0
            td_target = reward + GAMMA * max_next_q
            td_error = td_target - q_table[s][a]
            q_table[s][a] += ALPHA * td_error

            # move to next state
            obs, s = next_obs, s_next
            total_reward += reward
            steps += 1

            # exponential ε-decay after **each** environment step
            eps = max(EPS_END, eps * EPS_DECAY)

        if verbose and ep % 10 == 0:
            print(
                f"Episode {ep:4d} | R = {total_reward:6.2f} | steps = {steps:3d} | ε = {eps:.3f}"
            )

    env.close()
    return q_table


# evaluation function to test the trained policy
def evaluate(env_id: str,
             q_table,
             num_episodes: int = 10,
             seed: int | None = 42,
             gui: bool = True):
    """evaluate a trained Q-table on a MiniWoB task.
    
    Args:
        env_id: MiniWoB environment ID.
        q_table: Trained Q-table from q_learning.
        num_episodes: Number of evaluation episodes.
        seed: Random seed.
        gui: Whether to show the browser window.
        
    Returns:
        Average reward and success rate.
    """
    # Setup environment
    render_mode = "human" if gui else None
    env = gym.make(env_id, render_mode=render_mode)

    total_reward = 0.0
    successes = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        s = state_repr(obs)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            # always take the best action according to Q-table (no exploration)
            q_vals = q_table[s][:len(obs["dom_elements"])]
            # if not q_vals or all(v == 0 for v in q_vals):  # If state not seen during training
            #     a = random.randrange(min(len(obs["dom_elements"]), MAX_ELEMS))
            #     print(f"Warning: Unseen state, taking random action")
            # else:
            #     a = int(max(range(len(q_vals)), key=q_vals.__getitem__))

            # execute action
            ref = obs["dom_elements"][a]["ref"]
            action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT,
                                                 ref=ref)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            s = state_repr(next_obs)
            obs = next_obs
            episode_reward += reward
            steps += 1

        # count as success if reward is positive
        if episode_reward > 0:
            successes += 1

        total_reward += episode_reward
        print(
            f"Eval episode {ep:3d} | Reward: {episode_reward:6.2f} | Steps: {steps:3d}"
        )

    env.close()

    avg_reward = total_reward / num_episodes
    success_rate = successes / num_episodes
    print(f"\nEvaluation results over {num_episodes} episodes:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Success rate: {success_rate:.2%}")

    return avg_reward, success_rate


# args to simplify the CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal tabular Q-learning for MiniWoB++")
    parser.add_argument(
        "--env",
        default="miniwob/click-button-v1",
        help=
        "MiniWoB env ID (e.g. 'miniwob/click-button-v1' or shorthand 'click-button')"
    )
    parser.add_argument("--episodes",
                        type=int,
                        default=500,
                        help="Number of training episodes")
    parser.add_argument("--quiet",
                        action="store_true",
                        help="Suppress per-episode logs")
    parser.add_argument("--gui",
                        action="store_true",
                        help="Show browser window during training")
    parser.add_argument(
        "--eval",
        type=int,
        default=0,
        help=
        "Number of evaluation episodes to run after training (0 to skip evaluation)"
    )
    args = parser.parse_args()

    # allow *shorthand* environment names like "enter-text" or "click-button"
    # by automatically converting them to the full Gym ID "miniwob/<id>-v1".
    env_id = args.env
    if "/" not in env_id:  # missing namespace -> treat as shorthand
        env_id = f"miniwob/{env_id}" + ("-v1"
                                        if not env_id.endswith("-v1") else "")
    elif env_id.startswith("miniwob/") and "-v" not in env_id.split("/")[-1]:
        env_id = env_id + "-v1"

    # train the agent
    q_table = q_learning(env_id=env_id,
                         episodes=args.episodes,
                         verbose=not args.quiet,
                         gui=args.gui)

    # evaluate if requested
    if args.eval > 0:
        print("\n" + "=" * 50)
        print(f"Evaluating trained policy on {env_id}...")
        evaluate(env_id=env_id,
                 q_table=q_table,
                 num_episodes=args.eval,
                 gui=True)  # Always show GUI for evaluation
