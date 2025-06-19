#!/usr/bin/env python
"""
Sequential Q-learning agent which uses the same tabular q-learning agent for each task.
However, this agent stores the immediate prev_action in its state representation and can
accurately learn with q-learning how to properly complete the click-button-sequence task 
which requires the agent to click the buttons in the correct order.

Usage:
python new_sequential_q_learning.py --env miniwob/click-button-v1 --episodes 100 --eval 30 --gui
python new_sequential_q_learning.py --env click-link --episodes 500 --gui
"""
from __future__ import annotations

import argparse
import collections
import random
from typing import DefaultDict, Tuple, Optional

import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

ALPHA = 0.20
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
MAX_ELEMS = 40
MAX_STEPS = 20


def state_repr(obs: dict, prev_action: Optional[int] = None) -> Tuple:
    utterance = obs.get("utterance", "").strip().lower()
    texts = tuple((e.get("text", "").strip().lower())
                  for e in obs["dom_elements"][:MAX_ELEMS])
    return (utterance, texts, prev_action)


def q_learning(env_id: str,
               episodes: int = 500,
               seed: int | None = 42,
               verbose: bool = True,
               gui: bool = False):
    gym.register_envs(miniwob)
    render_mode = "human" if gui else None
    env = gym.make(env_id, render_mode=render_mode)

    q_table: DefaultDict[Tuple, list[float]] = collections.defaultdict(
        lambda: [0.0] * MAX_ELEMS)

    eps = EPS_START
    rng = random.Random(seed)
    successes = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed)
        prev_action = None
        s = state_repr(obs, prev_action)
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            num_elements = len(obs["dom_elements"])
            max_actions = min(num_elements, MAX_ELEMS)

            if rng.random() < eps:
                a = rng.randrange(max_actions)
            else:
                q_vals = q_table[s][:max_actions]
                if len(q_vals) == 0:
                    a = 0
                else:
                    a = int(max(range(len(q_vals)), key=q_vals.__getitem__))

            # additional bounds checking
            if a < 0 or a >= num_elements:
                if verbose:
                    print(
                        f"    Warning: Invalid action {a}, using action 0 instead"
                    )
                a = 0

            # ensure we have at least one DOM element
            if num_elements == 0:
                if verbose:
                    print(
                        f"    Warning: No DOM elements available, skipping step"
                    )
                break

            # execute action
            try:
                ref = obs["dom_elements"][a]["ref"]
                action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT,
                                                     ref=ref)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                s_next = state_repr(next_obs, a)

                max_next_q = max(q_table[s_next]) if not done else 0.0
                td_target = reward + GAMMA * max_next_q
                td_error = td_target - q_table[s][a]
                q_table[s][a] += ALPHA * td_error

                obs, s = next_obs, s_next
                prev_action = a
                total_reward += reward
                steps += 1
                eps = max(EPS_END, eps * EPS_DECAY)

            except Exception as e:
                if "already done" in str(e):
                    break
                elif "getBoundingClientRect" in str(e):
                    if verbose:
                        print(
                            f"Warning: Element click failed (invalid element), skipping step"
                        )
                    break
                else:
                    if verbose:
                        print(f"Warning: Unexpected error: {e}")
                    break

        if total_reward > 0:
            successes += 1

        if verbose and ep % 10 == 0:
            success_rate = successes / ep
            print(
                f"Episode {ep:4d} | R = {total_reward:6.2f} | steps = {steps:3d} | ε = {eps:.3f} | Success = {success_rate:.2%}"
            )

    env.close()
    return q_table


def evaluate(env_id: str,
             q_table,
             num_episodes: int = 10,
             seed: int | None = 42,
             gui: bool = True):
    render_mode = "human" if gui else None
    env = gym.make(env_id, render_mode=render_mode)

    total_reward = 0.0
    successes = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        prev_action = None
        s = state_repr(obs, prev_action)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            num_elements = len(obs["dom_elements"])
            max_actions = min(num_elements, MAX_ELEMS)

            q_vals = q_table[s][:max_actions]
            if not q_vals or all(v == 0 for v in q_vals):
                a = random.randrange(max_actions)
                print(f"Warning: Unseen state, taking random action")
            else:
                a = int(max(range(len(q_vals)), key=q_vals.__getitem__))

            if a < 0 or a >= num_elements:
                print(f"Warning: Invalid action {a}, using action 0 instead")
                a = 0

            if num_elements == 0:
                print(f"Warning: No DOM elements available, ending episode")
                break

            try:
                ref = obs["dom_elements"][a]["ref"]
                action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT,
                                                     ref=ref)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                done = terminated or truncated
                s = state_repr(next_obs, a)
                obs = next_obs
                prev_action = a
                episode_reward += reward
                steps += 1

            except Exception as e:
                if "already done" in str(e):
                    break
                else:
                    print(f"Warning: Error during evaluation: {e}")
                    break

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Sequential tabular Q-learning for MiniWoB++ with enhanced state representation"
    )
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

    # allow shorthand environment names like "click-button" or "click-dialog"
    env_id = args.env
    if "/" not in env_id:  # missing namespace -> treat as shorthand
        env_id = f"miniwob/{env_id}" + ("-v1"
                                        if not env_id.endswith("-v1") else "")
    elif env_id.startswith("miniwob/") and "-v" not in env_id.split("/")[-1]:
        env_id = env_id + "-v1"

    print(f"Training sequential Q-learning agent on {env_id}")
    print(f"Episodes: {args.episodes}, GUI: {args.gui}, Eval: {args.eval}")
    print("Enhanced with previous action tracking for sequential dependencies")
    print("=" * 60)

    # train agent
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
                 gui=True)  # show GUI for evaluation
