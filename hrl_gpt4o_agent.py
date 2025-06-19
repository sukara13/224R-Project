"""hrl_gpt4o_agent.py

Hierarchical Reinforcement Learning wrapper for gpt4o_miniwob_agent.py

This script adds an HRL component that:
1. Runs the original GPT-4o agent for 5 learning trials
2. Extracts common action patterns/workflows from successful attempts
3. Uses these workflows to guide a final evaluation attempt

Usage:
    python hrl_gpt4o_agent.py --task click-button-sequence --render
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional
import importlib.util
import gymnasium as gym

try:
    from gpt4o_miniwob_agent import (gpt_decide, json_to_miniwob_action,
                                     format_action_summary, get_task_html)
except ImportError:
    print("[ERROR] Could not import gpt4o_miniwob_agent.py")
    print("Make sure gpt4o_miniwob_agent.py is in the same directory")
    sys.exit(1)


class ActionPattern:
    """Represents a learned action pattern/workflow."""

    def __init__(self, steps: List[Dict[str, Any]], success_rate: float):
        self.steps = steps
        self.success_rate = success_rate
        self.length = len(steps)

    def __str__(self):
        return f"Pattern(length={self.length}, success={self.success_rate:.2f})"


class WorkflowLearner:
    """Learns workflows from multiple agent attempts."""

    def __init__(self):
        self.attempts: List[Dict[str, Any]] = []
        self.patterns: List[ActionPattern] = []

    def add_attempt(self, actions: List[Dict[str, Any]], reward: float,
                    success: bool):
        """Add an attempt to the learning history."""
        self.attempts.append({
            'actions': actions,
            'reward': reward,
            'success': success,
            'length': len(actions)
        })
        print(
            f"[HRL] Added attempt: {len(actions)} actions, reward={reward:.2f}, success={success}"
        )

    def extract_action_signature(self, action: Dict[str, Any],
                                 dom_elements: List[Dict[str, Any]]) -> str:
        """Create a semantic signature for an action."""
        action_type = action.get('type', 'noop')

        if action_type == 'click':
            element_idx = action.get('element_idx', 0)
            if 0 <= element_idx < len(dom_elements):
                element = dom_elements[element_idx]
                element_text = element.get('text', '').strip().lower()
                element_tag = element.get('tag', 'unknown')
                # create semantic signature: type-tag-text
                return f"click-{element_tag}-{element_text}"
            return f"click-unknown-{element_idx}"
        elif action_type == 'type':
            text = action.get('text', '').strip().lower()
            return f"type-{text}"
        else:
            return action_type

    def find_common_patterns(self) -> List[ActionPattern]:
        """Extract common action patterns from successful attempts."""
        if not self.attempts:
            return []

        # focus on successful attempts
        successful_attempts = [a for a in self.attempts if a['success']]
        if not successful_attempts:
            print("[HRL] No successful attempts to learn from")
            return []

        print(
            f"[HRL] Learning from {len(successful_attempts)} successful attempts"
        )

        # for simplicity, find the most common successful action sequence
        sequence_counter = Counter()

        for attempt in successful_attempts:
            # convert actions to a simple sequence signature
            sequence = []
            for action in attempt['actions']:
                action_type = action.get('type', 'noop')
                if action_type == 'click':
                    element_idx = action.get('element_idx', 0)
                    sequence.append(f"click_{element_idx}")
                elif action_type == 'type':
                    text = action.get('text', '')[:10]
                    sequence.append(f"type_{text}")
                else:
                    sequence.append(action_type)

            sequence_tuple = tuple(sequence)
            sequence_counter[sequence_tuple] += 1

        # convert top patterns to ActionPattern objects
        patterns = []
        total_successful = len(successful_attempts)

        for sequence, count in sequence_counter.most_common(
                3):  # top 3 patterns
            success_rate = count / total_successful

            # reconstruct action steps from the most recent successful attempt using this pattern
            for attempt in reversed(successful_attempts):  # most recent first
                attempt_sequence = []
                for action in attempt['actions']:
                    action_type = action.get('type', 'noop')
                    if action_type == 'click':
                        element_idx = action.get('element_idx', 0)
                        attempt_sequence.append(f"click_{element_idx}")
                    elif action_type == 'type':
                        text = action.get('text', '')[:10]
                        attempt_sequence.append(f"type_{text}")
                    else:
                        attempt_sequence.append(action_type)

                if tuple(attempt_sequence) == sequence:
                    pattern = ActionPattern(attempt['actions'], success_rate)
                    patterns.append(pattern)
                    print(f"[HRL] Learned pattern: {pattern}")
                    break

        return patterns

    def get_best_workflow(self) -> Optional[ActionPattern]:
        """Get the best workflow pattern."""
        if not self.patterns:
            self.patterns = self.find_common_patterns()

        if not self.patterns:
            return None

        # return pattern with highest success rate
        best_pattern = max(self.patterns, key=lambda p: p.success_rate)
        print(f"[HRL] Selected best workflow: {best_pattern}")
        return best_pattern


def run_learning_phase(env,
                       task_name: str,
                       task_html: str,
                       max_steps: int = 50,
                       num_attempts: int = 5) -> WorkflowLearner:
    """Run multiple learning attempts to extract workflows."""
    learner = WorkflowLearner()

    print(f"\n{'='*60}")
    print(f"HRL LEARNING PHASE - {num_attempts} attempts")
    print(f"{'='*60}")

    for attempt_num in range(1, num_attempts + 1):
        print(f"\n--- Learning Attempt {attempt_num}/{num_attempts} ---")

        obs, _ = env.reset()
        instruction = obs.get("utterance", "") or obs.get("instruction", "")

        previous_actions: List[Dict[str, Any]] = []
        episode_reward = 0.0
        success = False

        for step in range(max_steps):
            dom = obs.get("dom", "")
            dom_elements = obs.get("dom_elements", [])

            # use original GPT-4o decision making
            action_dict = gpt_decide(dom, task_name, dom_elements,
                                     previous_actions, instruction, task_html)
            previous_actions.append(action_dict)

            try:
                action = json_to_miniwob_action(env, action_dict, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward = reward

                if terminated or truncated:
                    success = terminated and reward > 0
                    print(
                        f"Learning attempt {attempt_num} finished: reward={reward:.2f}, success={success}"
                    )
                    break

            except Exception as e:
                print(f"[ERROR] Learning attempt {attempt_num} failed: {e}")
                break

        # add this attempt to the learner
        learner.add_attempt(previous_actions, episode_reward, success)

    return learner


def run_evaluation_with_workflow(env,
                                 task_name: str,
                                 task_html: str,
                                 workflow: ActionPattern,
                                 max_steps: int = 50) -> Tuple[float, bool]:
    """Run evaluation guided by learned workflow."""
    print(f"\n{'='*60}")
    print(f"HRL EVALUATION PHASE - Using learned workflow")
    print(f"{'='*60}")

    obs, _ = env.reset()
    instruction = obs.get("utterance", "") or obs.get("instruction", "")

    print(f"Task: {instruction}")
    print(
        f"Using workflow with {workflow.length} steps (success rate: {workflow.success_rate:.2f})"
    )

    previous_actions: List[Dict[str, Any]] = []
    workflow_step = 0
    episode_reward = 0.0

    for step in range(max_steps):
        dom = obs.get("dom", "")
        dom_elements = obs.get("dom_elements", [])

        # try to follow learned workflow first, then fall back to GPT-4o
        action_dict = None

        if workflow_step < len(workflow.steps):
            # try to use workflow action
            workflow_action = workflow.steps[workflow_step]

            # validate that the workflow action is still applicable
            if workflow_action.get('type') == 'click':
                workflow_idx = workflow_action.get('element_idx', 0)
                if 0 <= workflow_idx < len(dom_elements):
                    # Workflow action is valid
                    action_dict = workflow_action.copy()
                    print(
                        f"[HRL] Using workflow step {workflow_step}: {format_action_summary(action_dict, dom_elements)}"
                    )
                    workflow_step += 1

        # if workflow action not available/valid, use GPT-4o
        if action_dict is None:
            action_dict = gpt_decide(dom, task_name, dom_elements,
                                     previous_actions, instruction, task_html)
            print(
                f"[HRL] Workflow not applicable, using GPT-4o: {format_action_summary(action_dict, dom_elements)}"
            )

        previous_actions.append(action_dict)

        try:
            action = json_to_miniwob_action(env, action_dict, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward = reward

            if terminated or truncated:
                success = terminated and reward > 0
                print(
                    f"Evaluation finished: reward={reward:.2f}, success={success}"
                )
                return episode_reward, success

        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return 0.0, False

    print(f"Evaluation reached max steps: reward={episode_reward:.2f}")
    return episode_reward, False


def main():
    """Main HRL training and evaluation loop."""
    parser = argparse.ArgumentParser(description="HRL GPT-4o MiniWoB agent")
    parser.add_argument("--task",
                        type=str,
                        default="click-button-sequence",
                        help="MiniWoB task name")
    parser.add_argument("--learning-attempts",
                        type=int,
                        default=15,
                        help="Number of learning attempts")
    parser.add_argument("--max-steps",
                        type=int,
                        default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--render",
                        action="store_true",
                        help="Render environment window")
    args = parser.parse_args()

    if args.task.startswith("miniwob/"):
        env_id = args.task
    else:
        task_name = args.task
        if task_name.startswith("miniwob/"):
            task_name = task_name[len("miniwob/"):]
        env_id = f"miniwob/{task_name}-v1" if "-v" not in task_name else f"miniwob/{task_name}"

    print(f"[INFO] Using environment ID: {env_id}")

    try:
        env = gym.make(env_id, render_mode="human" if args.render else None)
    except gym.error.Error as exc:
        print(f"[ERROR] Failed to make env '{env_id}': {exc}")
        sys.exit(1)

    # get task HTML
    task_name = args.task.replace("miniwob/", "").replace("-v1", "")
    task_html = get_task_html(task_name)

    # run learning phase
    learner = run_learning_phase(env, args.task, task_html, args.max_steps,
                                 args.learning_attempts)

    # extract best workflow
    best_workflow = learner.get_best_workflow()

    if best_workflow is None:
        print(
            "\n[HRL] No workflow learned, running standard GPT-4o evaluation")
        # run one more standard attempt
        obs, _ = env.reset()
        instruction = obs.get("utterance", "") or obs.get("instruction", "")
        previous_actions = []

        for step in range(args.max_steps):
            dom = obs.get("dom", "")
            dom_elements = obs.get("dom_elements", [])

            action_dict = gpt_decide(dom, args.task, dom_elements,
                                     previous_actions, instruction, task_html)
            previous_actions.append(action_dict)

            try:
                action = json_to_miniwob_action(env, action_dict, obs)
                obs, reward, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    print(
                        f"Standard evaluation: reward={reward:.2f}, success={terminated and reward > 0}"
                    )
                    break
            except Exception as e:
                print(f"[ERROR] Standard evaluation failed: {e}")
                break
    else:
        # run evaluation with learned workflow
        eval_reward, eval_success = run_evaluation_with_workflow(
            env, args.task, task_html, best_workflow, args.max_steps)

        print(f"\n{'='*60}")
        print(f"HRL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Learning attempts: {args.learning_attempts}")
        print(f"Learned workflow steps: {best_workflow.length}")
        print(f"Workflow success rate: {best_workflow.success_rate:.2f}")
        print(f"Evaluation reward: {eval_reward:.2f}")
        print(f"Evaluation success: {eval_success}")

    env.close()


if __name__ == "__main__":
    main()
