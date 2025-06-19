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
import time  # Add this import for timestamps

try:
    from gpt4o_miniwob_agent import (gpt_decide, json_to_miniwob_action,
                                     format_action_summary, get_task_html)
except ImportError:
    print("[ERROR] Could not import gpt4o_miniwob_agent.py")
    print("Make sure gpt4o_miniwob_agent.py is in the same directory")
    sys.exit(1)


class ActionPattern:
    """Represents a learned action pattern/workflow."""

    def __init__(self,
                 steps: List[Dict[str, Any]],
                 success_rate: float,
                 semantic_pattern: Optional[List[str]] = None):
        self.steps = steps
        self.success_rate = success_rate
        self.length = len(steps)
        self.semantic_pattern = semantic_pattern or []

    def __str__(self):
        return f"Pattern(length={self.length}, success={self.success_rate:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for JSON serialization."""
        return {
            "steps": self.steps,
            "success_rate": self.success_rate,
            "length": self.length,
            "semantic_pattern": self.semantic_pattern
        }

    def print_detailed(self,
                       dom_elements: Optional[List[Dict[str, Any]]] = None):
        """Print detailed workflow information."""
        print(
            f"\n--- Workflow Details (Success Rate: {self.success_rate:.2f}) ---"
        )
        if self.semantic_pattern:
            print("Semantic Pattern:")
            for i, pattern_step in enumerate(self.semantic_pattern):
                print(f"  Step {i+1}: {pattern_step}")
        print("Concrete Steps:")
        for i, step in enumerate(self.steps):
            action_type = step.get('type', 'noop')
            if action_type == 'click':
                element_idx = step.get('element_idx', 0)
                if dom_elements and 0 <= element_idx < len(dom_elements):
                    element = dom_elements[element_idx]
                    element_text = element.get('text', '').strip()
                    element_tag = element.get('tag', 'unknown')
                    print(
                        f"  Step {i+1}: Click {element_tag} '{element_text}' (idx={element_idx})"
                    )
                else:
                    print(
                        f"  Step {i+1}: Click element at index {element_idx}")
            elif action_type == 'type':
                text = step.get('text', '')
                print(f"  Step {i+1}: Type '{text}'")
            else:
                print(f"  Step {i+1}: {action_type}")
        print("--- End Workflow ---\n")


class SemanticWorkflowLearner:
    """Enhanced workflow learner that extracts semantic, generalizable patterns."""

    def __init__(self):
        self.attempts: List[Dict[str, Any]] = []
        self.patterns: List[ActionPattern] = []

    def add_attempt(self,
                    actions: List[Dict[str, Any]],
                    reward: float,
                    success: bool,
                    dom_states: List[List[Dict[str, Any]]] = None):
        """Add an attempt to the learning history."""
        self.attempts.append({
            'actions': actions,
            'reward': reward,
            'success': success,
            'length': len(actions),
            'dom_states': dom_states or []
        })
        print(
            f"[Semantic HRL] Added attempt: {len(actions)} actions, reward={reward:.2f}, success={success}"
        )

    def extract_semantic_signature(self, action: Dict[str, Any],
                                   dom_elements: List[Dict[str, Any]]) -> str:
        """Create a semantic signature for an action that generalizes."""
        action_type = action.get('type', 'noop')

        if action_type == 'click':
            element_idx = action.get('element_idx', 0)
            if 0 <= element_idx < len(dom_elements):
                element = dom_elements[element_idx]
                element_text = element.get('text', '').strip().lower()
                element_tag = element.get('tag', 'unknown')
                element_classes = element.get('classes', [])

                # Create semantic patterns based on element characteristics
                if any(cls in ['login', 'submit', 'button'] for cls in element_classes) or \
                   any(word in element_text for word in ['login', 'submit', 'sign in', 'enter']):
                    return "click_login_button"
                elif element_tag == 'input' and any(
                        word in element_text
                        for word in ['username', 'user', 'email']):
                    return "click_username_field"
                elif element_tag == 'input' and any(
                        word in element_text for word in ['password', 'pass']):
                    return "click_password_field"
                elif element_tag == 'button':
                    if 'next' in element_text or 'continue' in element_text:
                        return "click_next_button"
                    elif any(num in element_text for num in
                             ['1', '2', '3', '4', '5', 'one', 'two', 'three']):
                        return "click_numbered_button"
                    else:
                        return f"click_button_{element_text}" if element_text else "click_button"
                elif element_tag in ['input', 'textarea']:
                    return "click_input_field"
                else:
                    return f"click_{element_tag}"
            return f"click_unknown_{element_idx}"
        elif action_type == 'type':
            text = action.get('text', '').strip()
            # Generalize typing patterns
            if len(text) > 8 and any(c.isalpha()
                                     for c in text) and any(c.isdigit()
                                                            for c in text):
                return "type_password"
            elif len(text) > 3 and text.isalpha():
                return "type_username"
            elif text.isdigit():
                return "type_number"
            else:
                return "type_text"
        else:
            return action_type

    def find_semantic_patterns(self) -> List[ActionPattern]:
        """Extract semantic patterns that can generalize across task instances."""
        if not self.attempts:
            return []

        successful_attempts = [a for a in self.attempts if a['success']]
        if not successful_attempts:
            print("[Semantic HRL] No successful attempts to learn from")
            return []

        print(
            f"[Semantic HRL] Learning semantic patterns from {len(successful_attempts)} successful attempts"
        )

        # Extract semantic sequences
        semantic_sequences = []
        for attempt in successful_attempts:
            semantic_seq = []
            for i, action in enumerate(attempt['actions']):
                # Use the corresponding DOM state if available
                dom_elements = []
                if attempt.get('dom_states') and i < len(
                        attempt['dom_states']):
                    dom_elements = attempt['dom_states'][i]

                semantic_sig = self.extract_semantic_signature(
                    action, dom_elements)
                semantic_seq.append(semantic_sig)

            semantic_sequences.append((semantic_seq, attempt))

        # Find common semantic patterns
        pattern_counter = Counter()
        for semantic_seq, attempt in semantic_sequences:
            pattern_counter[tuple(semantic_seq)] += 1

        # Convert to generalized patterns
        patterns = []
        total_successful = len(successful_attempts)

        for pattern_seq, count in pattern_counter.most_common(3):
            success_rate = count / total_successful

            # Find a representative concrete action sequence
            for semantic_seq, attempt in semantic_sequences:
                if tuple(semantic_seq) == pattern_seq:
                    pattern = ActionPattern(steps=attempt['actions'],
                                            success_rate=success_rate,
                                            semantic_pattern=list(pattern_seq))
                    patterns.append(pattern)
                    print(
                        f"[Semantic HRL] Learned semantic pattern: {pattern}")
                    print(f"  Semantic sequence: {' -> '.join(pattern_seq)}")
                    break

        return patterns

    def get_best_workflow(self) -> Optional[ActionPattern]:
        """Get the best semantic workflow pattern."""
        if not self.patterns:
            self.patterns = self.find_semantic_patterns()

        if not self.patterns:
            return None

        best_pattern = max(self.patterns, key=lambda p: p.success_rate)
        print(
            f"[Semantic HRL] Selected best semantic workflow: {best_pattern}")
        return best_pattern

    def print_all_workflows(self):
        """Print detailed information about all learned semantic workflows."""
        if not self.patterns:
            self.patterns = self.find_semantic_patterns()

        if not self.patterns:
            print("[Semantic HRL] No workflows have been learned yet.")
            return

        print(f"\n{'='*60}")
        print(f"LEARNED SEMANTIC WORKFLOWS ({len(self.patterns)} total)")
        print(f"{'='*60}")

        for i, pattern in enumerate(self.patterns, 1):
            print(f"\nSemantic Workflow #{i}:")
            print(f"  Success Rate: {pattern.success_rate:.2f}")
            if pattern.semantic_pattern:
                print(f"  Semantic Pattern:")
                for j, sem_step in enumerate(pattern.semantic_pattern):
                    print(f"    {j+1}. {sem_step}")

    def save_workflows_to_file(self,
                               task_name: str,
                               filename: Optional[str] = None):
        """Save learned semantic workflows to a JSON file."""
        if not self.patterns:
            print("[Semantic HRL] No workflows to save.")
            return

        if filename is None:
            timestamp = int(time.time())
            filename = f"semantic_hrl_workflows_{task_name}_{timestamp}.json"

        workflows_data = {
            "task_name": task_name,
            "timestamp": int(time.time()),
            "num_workflows": len(self.patterns),
            "num_attempts": len(self.attempts),
            "successful_attempts":
            len([a for a in self.attempts if a['success']]),
            "workflows": [pattern.to_dict() for pattern in self.patterns]
        }

        try:
            with open(filename, 'w') as f:
                json.dump(workflows_data, f, indent=2)
            print(
                f"[Semantic HRL] Saved {len(self.patterns)} semantic workflows to {filename}"
            )
        except Exception as e:
            print(f"[Semantic HRL] Failed to save workflows: {e}")


def extract_task_parameters(instruction: str) -> Dict[str, str]:
    """Extract parameters like username, password from task instruction."""
    params = {}

    # Common patterns for login tasks
    if 'login' in instruction.lower():
        # Look for username patterns
        import re
        username_match = re.search(
            r'(?:username|user|login as|for)\s+([a-zA-Z0-9_]+)', instruction,
            re.IGNORECASE)
        if username_match:
            params['username'] = username_match.group(1)

        # Look for password patterns
        password_match = re.search(r'(?:password|pass)\s+([a-zA-Z0-9_]+)',
                                   instruction, re.IGNORECASE)
        if password_match:
            params['password'] = password_match.group(1)

    return params


def find_semantic_element(semantic_target: str,
                          dom_elements: List[Dict[str, Any]]) -> Optional[int]:
    """Find an element that matches the semantic description."""
    for i, element in enumerate(dom_elements):
        element_text = element.get('text', '').strip().lower()
        element_tag = element.get('tag', 'unknown')
        element_classes = element.get('classes', [])

        if semantic_target == "click_login_button":
            if any(cls in ['login', 'submit', 'button'] for cls in element_classes) or \
               any(word in element_text for word in ['login', 'submit', 'sign in', 'enter']):
                return i
        elif semantic_target == "click_username_field":
            if element_tag == 'input' and any(
                    word in element_text
                    for word in ['username', 'user', 'email']):
                return i
        elif semantic_target == "click_password_field":
            if element_tag == 'input' and any(
                    word in element_text for word in ['password', 'pass']):
                return i
        elif semantic_target == "click_next_button":
            if element_tag == 'button' and ('next' in element_text
                                            or 'continue' in element_text):
                return i
        elif semantic_target == "click_numbered_button":
            if element_tag == 'button' and any(
                    num in element_text for num in ['1', '2', '3', '4', '5']):
                return i
        elif semantic_target == "click_input_field":
            if element_tag in ['input', 'textarea']:
                return i

    return None


def run_learning_phase(env,
                       task_name: str,
                       task_html: str,
                       max_steps: int = 50,
                       num_attempts: int = 5) -> SemanticWorkflowLearner:
    """Run multiple learning attempts to extract semantic workflows."""
    learner = SemanticWorkflowLearner()

    print(f"\n{'='*60}")
    print(f"SEMANTIC HRL LEARNING PHASE - {num_attempts} attempts")
    print(f"{'='*60}")

    for attempt_num in range(1, num_attempts + 1):
        print(f"\n--- Learning Attempt {attempt_num}/{num_attempts} ---")

        obs, _ = env.reset()
        instruction = obs.get("utterance", "") or obs.get("instruction", "")

        previous_actions: List[Dict[str, Any]] = []
        dom_states: List[List[Dict[str, Any]]] = []
        episode_reward = 0.0
        success = False

        for step in range(max_steps):
            dom = obs.get("dom", "")
            dom_elements = obs.get("dom_elements", [])
            dom_states.append(list(dom_elements) if dom_elements else
                              [])  # Store DOM state as list

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

        # add this attempt to the learner with DOM states
        learner.add_attempt(previous_actions, episode_reward, success,
                            dom_states)

    return learner


def run_evaluation_with_semantic_workflow(
        env,
        task_name: str,
        task_html: str,
        workflow: ActionPattern,
        max_steps: int = 50) -> Tuple[float, bool]:
    """Run evaluation with semantic workflow that adapts to task parameters."""
    print(f"\n{'='*60}")
    print(f"SEMANTIC HRL EVALUATION PHASE - Using generalized workflow")
    print(f"{'='*60}")

    obs, _ = env.reset()
    instruction = obs.get("utterance", "") or obs.get("instruction", "")

    # Extract task-specific parameters
    task_params = extract_task_parameters(instruction)
    print(f"Task: {instruction}")
    print(f"Extracted parameters: {task_params}")

    print(
        f"Using semantic workflow with {workflow.length} steps (success rate: {workflow.success_rate:.2f})"
    )

    # Print the semantic workflow
    if workflow.semantic_pattern:
        print(f"\n--- SEMANTIC WORKFLOW TO FOLLOW ---")
        for i, pattern_step in enumerate(workflow.semantic_pattern):
            print(f"  Step {i+1}: {pattern_step}")
        print("--- END SEMANTIC WORKFLOW ---\n")

    previous_actions: List[Dict[str, Any]] = []
    workflow_step = 0
    episode_reward = 0.0

    for step in range(max_steps):
        dom = obs.get("dom", "")
        dom_elements = obs.get("dom_elements", [])

        action_dict = None

        # Try to follow semantic workflow
        if workflow_step < len(workflow.semantic_pattern):
            semantic_action = workflow.semantic_pattern[workflow_step]

            if semantic_action.startswith('click_'):
                # Find element semantically
                element_idx = find_semantic_element(semantic_action,
                                                    dom_elements)
                if element_idx is not None:
                    action_dict = {"type": "click", "element_idx": element_idx}
                    element_text = dom_elements[element_idx].get('text',
                                                                 '').strip()
                    print(
                        f"[Semantic HRL] ✓ Step {workflow_step+1}/{len(workflow.semantic_pattern)}: {semantic_action} → Click '{element_text}' (idx={element_idx})"
                    )
                    workflow_step += 1
                else:
                    print(
                        f"[Semantic HRL] ✗ Could not find element for: {semantic_action}"
                    )

            elif semantic_action.startswith('type_'):
                # Determine what to type based on semantic action and task parameters
                text_to_type = None
                if semantic_action == 'type_username' and 'username' in task_params:
                    text_to_type = task_params['username']
                elif semantic_action == 'type_password' and 'password' in task_params:
                    text_to_type = task_params['password']

                if text_to_type:
                    action_dict = {"type": "type", "text": text_to_type}
                    print(
                        f"[Semantic HRL] ✓ Step {workflow_step+1}/{len(workflow.semantic_pattern)}: {semantic_action} → Type '{text_to_type}'"
                    )
                    workflow_step += 1
                else:
                    print(
                        f"[Semantic HRL] ✗ Could not determine text for: {semantic_action}"
                    )

        # If semantic workflow not applicable, use GPT-4o
        if action_dict is None:
            action_dict = gpt_decide(dom, task_name, dom_elements,
                                     previous_actions, instruction, task_html)
            print(
                f"[Semantic HRL] → Using GPT-4o: {format_action_summary(action_dict, dom_elements)}"
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
                if success:
                    print(
                        f"[Semantic HRL] ✓ Successfully completed task using semantic workflow!"
                    )
                else:
                    print(
                        f"[Semantic HRL] ✗ Task failed despite using semantic workflow"
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
                        default=8,
                        help="Number of learning attempts")
    parser.add_argument("--max-steps",
                        type=int,
                        default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--render",
                        action="store_true",
                        help="Render environment window")
    parser.add_argument("--save-workflows",
                        action="store_true",
                        help="Save learned workflows to JSON file")
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

    # Print all learned workflows
    learner.print_all_workflows()

    # Save workflows if requested
    if args.save_workflows:
        learner.save_workflows_to_file(args.task)

    # extract best workflow
    best_workflow = learner.get_best_workflow()

    if best_workflow is None:
        print(
            "\n[Semantic HRL] No semantic workflow learned, running standard GPT-4o evaluation"
        )
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
        # run evaluation with learned semantic workflow
        eval_reward, eval_success = run_evaluation_with_semantic_workflow(
            env, args.task, task_html, best_workflow, args.max_steps)

        print(f"\n{'='*60}")
        print(f"SEMANTIC HRL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Learning attempts: {args.learning_attempts}")
        print(f"Learned semantic workflow steps: {best_workflow.length}")
        print(
            f"Semantic workflow success rate: {best_workflow.success_rate:.2f}"
        )
        if best_workflow.semantic_pattern:
            print(
                f"Semantic pattern: {' -> '.join(best_workflow.semantic_pattern)}"
            )
        print(f"Evaluation reward: {eval_reward:.2f}")
        print(f"Evaluation success: {eval_success}")

        # Print final workflow details
        print(f"\n--- FINAL SEMANTIC WORKFLOW USED ---")
        best_workflow.print_detailed()

    env.close()


if __name__ == "__main__":
    main()
