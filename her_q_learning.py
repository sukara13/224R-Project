#!/usr/bin/env python
"""Enhanced **tabular Q-learning** agent for MiniWoB++ with **Hindsight Experience Replay (HER)**.

This implementation runs through all MiniWoB++ click tasks from homepage using HER:
    • Opens the miniwob_click_tasks_homepage.html
    • Clicks through every task sequentially 
    • Runs HER Q-learning on each task to complete it
    • Uses HER to learn from failed episodes by changing the goal retrospectively
    • Exits and moves to the next task
    • Provides summary statistics at the end

Usage
-----
python her_q_learning.py                    # run all tasks with default settings
python her_q_learning.py --episodes 100    # fewer episodes per task
python her_q_learning.py --no-gui           # run headless
python her_q_learning.py --her-k 4          # use 4 virtual goals per real episode
"""
from __future__ import annotations

import argparse
import collections
import random
import time
import os
from typing import DefaultDict, Tuple, Optional, Dict, List, Any
from pathlib import Path

import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# -----------------------------------------------------------------------------
# Q-learning and HER hyper-parameters
# -----------------------------------------------------------------------------
ALPHA = 0.20  # learning-rate      (α)
GAMMA = 0.99  # discount-factor    (γ)
EPS_START = 1.0  # initial ε for ε-greedy
EPS_END = 0.05  # minimum ε
EPS_DECAY = 0.995  # multiplicative decay applied at *each step*
MAX_ELEMS = 40  # clip the number of DOM elements we consider as actions
MAX_STEPS = 20  # maximum steps per episode to prevent infinite loops
HER_K = 8  # Number of virtual goals per real episode (increased for better learning)
ACTION_DELAY = 0.0  # Delay in seconds between actions (for visualization)
SEQUENCE_BONUS = 2.0  # Bonus reward for completing a sequence correctly

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def state_repr(obs: dict,
               prev_action: Optional[int] = None,
               task_id: str = None,
               action_history: Optional[Tuple[int, ...]] = None) -> Tuple:
    """Create a hashable state representation from observation.
    
    Args:
        obs: Observation from environment
        prev_action: Previous action index
        task_id: Task identifier
        action_history: Tuple of action indices representing the history
        
    Returns:
        A hashable tuple representing the state
    """
    utterance = obs.get("utterance", "").strip().lower()

    # Special handling for click-collapsible task to improve generalization
    if task_id and "click-collapsible" in task_id:
        # For click-collapsible, we want to generalize across different section numbers
        # Replace specific section numbers with a generic placeholder
        texts = []
        element_types = [
        ]  # Track element types for better state representation

        for e in obs["dom_elements"][:MAX_ELEMS]:
            text = e.get("text", "").strip().lower()
            tag = e.get("tag", "").lower()

            # Replace specific section numbers with generic placeholder
            if tag == "h3" and text.startswith("section #"):
                text = "section #generic"

            texts.append(text)
            element_types.append(tag)  # Add element type to state

        # For click-collapsible, return a specialized state representation
        # that focuses on element types rather than specific text content
        return (task_id, utterance, tuple(texts), tuple(element_types),
                prev_action, action_history)

    # Special handling for click-dialog task to improve generalization
    elif task_id and "click-dialog" in task_id:
        # For click-dialog, we want to focus on element types and positions
        # rather than specific text content which may vary
        texts = []
        element_types = []
        element_positions = []

        for e in obs["dom_elements"][:MAX_ELEMS]:
            text = e.get("text", "").strip().lower()
            tag = e.get("tag", "").lower()

            # For dialog close buttons, the text might be hidden or vary
            # So we'll focus on the element type and position
            left = e.get("left", 0)
            top = e.get("top", 0)

            # Normalize positions to general areas (top-right, center, etc.)
            position = "unknown"
            if left > 100 and top < 100:
                position = "top-right"  # Likely close button position
            elif left < 50 and top < 50:
                position = "top-left"
            elif 50 <= left <= 100 and 50 <= top <= 150:
                position = "center"

            texts.append(text)
            element_types.append(tag)
            element_positions.append(position)

        # Return a specialized state representation for click-dialog
        return (task_id, utterance, tuple(texts), tuple(element_types),
                tuple(element_positions), prev_action, action_history)

    else:
        # For other tasks, use the standard state representation
        texts = tuple((e.get("text", "").strip().lower())
                      for e in obs["dom_elements"][:MAX_ELEMS])
        return (task_id, utterance, texts, prev_action, action_history)


def extract_goal_from_state(state: Tuple) -> Tuple:
    """Extract a potential goal from a state.
    
    In MiniWoB, we'll use the DOM elements that were clicked as potential goals.
    For sequential tasks, we'll include the action history in the goal.
    
    Args:
        state: A state tuple from state_repr
        
    Returns:
        A goal representation that can be used for HER
    """
    # Extract task_id, which is always the first element
    task_id = state[0]

    # Handle different state formats based on task type and tuple length
    if "click-collapsible" in task_id and len(state) == 6:
        # Format for click-collapsible: (task_id, utterance, texts, element_types, prev_action, action_history)
        prev_action = state[4]
        action_history = state[5] if len(state) > 5 else None
    elif "click-dialog" in task_id and len(state) == 7:
        # Format for click-dialog: (task_id, utterance, texts, element_types, element_positions, prev_action, action_history)
        prev_action = state[5]
        action_history = state[6] if len(state) > 6 else None
    elif len(state) == 5:
        # Standard format with action_history: (task_id, utterance, texts, prev_action, action_history)
        prev_action = state[3]
        action_history = state[4]
    else:
        # Old format without action_history: (task_id, utterance, texts, prev_action)
        prev_action = state[3]
        action_history = None

    # Always include action history in the goal for all tasks
    # This helps the agent learn sequences of actions regardless of task type
    return (task_id, prev_action,
            action_history if action_history else tuple())


def state_with_goal(state: Tuple, goal: Any) -> Tuple:
    """Create a new state representation that includes the goal.
    
    Args:
        state: Original state tuple
        goal: Goal to incorporate
        
    Returns:
        A new state tuple that includes the goal
    """
    # Extract task_id, which is always the first element
    task_id = state[0]

    # Handle different state formats based on task type and tuple length
    if "click-collapsible" in task_id and len(state) == 6:
        # Format for click-collapsible: (task_id, utterance, texts, element_types, prev_action, action_history)
        task_id, utterance, texts, element_types, prev_action, action_history = state
        # Create a new state tuple with the goal
        return (task_id, utterance, texts, element_types, prev_action,
                action_history, goal)
    elif "click-dialog" in task_id and len(state) == 7:
        # Format for click-dialog: (task_id, utterance, texts, element_types, element_positions, prev_action, action_history)
        task_id, utterance, texts, element_types, element_positions, prev_action, action_history = state
        # Create a new state tuple with the goal
        return (task_id, utterance, texts, element_types, element_positions,
                prev_action, action_history, goal)
    elif len(state) == 5:
        # Standard format with action_history: (task_id, utterance, texts, prev_action, action_history)
        task_id, utterance, texts, prev_action, action_history = state
        # Create a new state tuple with the goal
        return (task_id, utterance, texts, prev_action, action_history, goal)
    else:
        # Old format without action_history: (task_id, utterance, texts, prev_action)
        task_id, utterance, texts, prev_action = state
        action_history = None
        # Create a new state tuple with the goal
        return (task_id, utterance, texts, prev_action, action_history, goal)


def sanitize_action_index(action_index: int,
                          max_index: int,
                          obs: dict = None) -> int:
    """Sanitize an action index to ensure it's valid and preferably not a background element.
    
    Args:
        action_index: The action index to sanitize
        max_index: The maximum valid index (exclusive)
        obs: The observation dictionary to check for better element selection
        
    Returns:
        A sanitized action index that is within bounds and preferably a clickable element
    """
    # Ensure action_index is non-negative
    action_index = max(0, action_index)
    # Ensure action_index is within bounds
    action_index = min(action_index, max_index - 1) if max_index > 0 else 0

    # If we have observation data, try to avoid background elements
    if obs and "dom_elements" in obs and max_index > 0:
        # Check if the current element might be a background element (usually has no text and large dimensions)
        current_element = obs["dom_elements"][action_index]

        # Criteria for likely background elements: no text, large dimensions, specific tag like 'body' or 'div'
        is_likely_background = ((not current_element.get("text", "").strip())
                                and (current_element.get("tag", "").lower()
                                     in ["body", "div"]) and
                                (current_element.get("width", 0) > 300
                                 or current_element.get("height", 0) > 300))

        # If this looks like a background element and we have other options, try to find a better element
        if is_likely_background and max_index > 1:
            # Look for elements that are more likely to be interactive (buttons, links, etc.)
            for i in range(max_index):
                element = obs["dom_elements"][i]
                # Prefer elements with text and interactive tags
                if (element.get("text", "").strip()
                        and element.get("tag", "").lower()
                        in ["button", "a", "input", "select"]):
                    return i

            # If we couldn't find an obviously better element, at least try something different
            # than the background to encourage exploration
            if action_index == 0:  # If we're on the first element (often background)
                return 1 if max_index > 1 else 0

    return action_index


def her_q_learning_single_task(env_id: str,
                               episodes: int = 100,
                               her_k: int = HER_K,
                               verbose: bool = False,
                               show_gui: bool = True,
                               shared_q_table=None):
    """Train HER Q-learning agent on a single task and update the shared Q-table.
    
    Args:
        env_id: The environment ID
        episodes: Number of episodes to train
        her_k: Number of virtual goals per real episode
        verbose: Whether to print detailed logs
        show_gui: Whether to show the GUI
        shared_q_table: The shared Q-table across all tasks
        
    Returns:
        Success rate and the updated shared Q-table
    """
    gym.register_envs(miniwob)
    # Show GUI for individual task training if requested
    render_mode = "human" if show_gui else None
    env = gym.make(env_id, render_mode=render_mode)

    # Extract task name from env_id for state representation
    task_name = env_id.split('/')[-1].split(
        '-v')[0] if '/' in env_id else env_id.split('-v')[0]

    # Use the provided shared Q-table or create a new one if None
    if shared_q_table is None:
        # Q-table: defaultdict mapping *state+goal* → array[ num_actions ]
        q_table: DefaultDict[Tuple, list[float]] = collections.defaultdict(
            lambda: [0.0] * MAX_ELEMS)
    else:
        q_table = shared_q_table

    eps = EPS_START
    rng = random.Random(42)
    successes = 0
    # Collect per-episode statistics for summary table
    episode_summaries: List[Dict[str, Any]] = []

    for ep in range(1, episodes + 1):
        # Store the trajectory for HER
        trajectory = []

        obs, _ = env.reset(seed=42)
        prev_action = None
        action_history = tuple()  # Empty tuple to track action history
        s = state_repr(obs, prev_action, task_name, action_history)
        done = False
        total_reward = 0.0
        steps = 0

        # For sequential tasks, we need to track the sequence of actions
        # This helps with tasks like click-button-sequence
        action_sequence = []

        # Execute the episode and collect experience
        while not done and steps < MAX_STEPS:
            # Get the number of available DOM elements
            num_elements = len(obs["dom_elements"])
            max_actions = min(num_elements, MAX_ELEMS)

            # Print DOM elements similar to evaluation function
            # if verbose:
            #     print(
            #         f"\nStep {steps} — DOM contains {num_elements} elements:")
            #     for i, e in enumerate(obs.get("dom_elements", [])):
            #         print("   " + fmt_dom_elem(e, i))

            # ε-greedy action selection with bounds checking and smart element selection
            if rng.random() < eps:
                # For tasks with 'sequence' in the name, use smarter exploration
                if "sequence" in task_name.lower():
                    # Prioritize elements with text that might be buttons
                    button_indices = []
                    for i, element in enumerate(obs["dom_elements"]):
                        # Look for elements that might be buttons in a sequence
                        if (element.get("text", "").strip()
                                and element.get("tag", "").lower()
                                in ["button", "a", "input", "select"]):
                            button_indices.append(i)

                    if button_indices:  # If we found potential buttons, choose one randomly
                        raw_action = rng.choice(button_indices)
                        action_type = "Smart Random (button focus)"
                    else:  # Otherwise fall back to normal random selection
                        raw_action = rng.randrange(
                            max_actions) if max_actions > 0 else 0
                        action_type = "Random (ε-greedy)"
                else:  # For other tasks, use standard random exploration
                    raw_action = rng.randrange(
                        max_actions) if max_actions > 0 else 0
                    action_type = "Random (ε-greedy)"

                # Sanitize the action index with observation data to avoid backgrounds
                a = sanitize_action_index(raw_action, num_elements, obs)
            else:
                # Take argmax over the currently *known* Q-values for s.
                q_vals = q_table[s][:max_actions]
                if len(q_vals) == 0:
                    a = 0  # Fallback to first element
                    action_type = "Default (no Q-values)"
                else:
                    # Get valid actions (those within the current DOM elements range)
                    # Ensure we only consider non-negative indices
                    valid_actions = [
                        i for i in range(len(q_vals))
                        if i < num_elements and i >= 0
                    ]
                    if not valid_actions:
                        a = 0  # Fallback to first element if no valid actions
                        action_type = "Default (no valid actions)"
                    else:
                        # Find the best valid action and sanitize it
                        raw_action = int(
                            max(valid_actions, key=lambda i: q_vals[i]))
                        a = sanitize_action_index(raw_action, num_elements,
                                                  obs)
                        action_type = "Best Q-value"

            # Additional bounds checking - ensure action is non-negative and within range
            if a < 0 or a >= num_elements:
                print(
                    f"⚠️ Warning: Invalid action {a}, using action 0 instead")
                a = 0
                action_type = "Fallback (bounds check)"

            # Ensure we have at least one DOM element
            if num_elements == 0:
                if verbose:
                    print(
                        f"    Warning: No DOM elements available, skipping step"
                    )
                break

            # Print detailed information about the selected action
            element_info = ""
            if a < len(obs["dom_elements"]):
                element = obs["dom_elements"][a]
                element_text = element.get("text", "").strip()
                element_tag = element.get("tag", "unknown")
                element_classes = element.get("classes", [])
                element_id = element.get("id", "")
                element_info = f"'{element_text}' (tag={element_tag}, id={element_id}, classes={element_classes})"

            # Always print action information regardless of verbose setting
            #print(f"\n🔍 ACTION: [{a}] {action_type} → Element: {element_info}")

            # Add delay for visualization
            time.sleep(ACTION_DELAY)

            # Execute action
            try:
                # Enhanced validation to ensure the DOM element exists, has a ref, and is visible/clickable
                is_valid_element = False
                validation_reason = "Unknown validation failure"

                # Double-check that action index is valid (should already be sanitized)
                a = sanitize_action_index(a, len(obs["dom_elements"]))

                if a < len(obs["dom_elements"]
                           ) and "ref" in obs["dom_elements"][a]:
                    element = obs["dom_elements"][a]
                    ref = element["ref"]

                    # Check if element is likely to be clickable
                    # Elements with negative coordinates or zero dimensions are likely not visible/clickable
                    if ("left" in element and "top" in element
                            and "width" in element and "height" in element):
                        # Check if element has reasonable position and dimensions
                        if (element["left"] >= 0 and element["top"] >= 0
                                and element["width"] > 0
                                and element["height"] > 0):
                            # Additional check for visibility if available
                            if "visible" not in element or element["visible"]:
                                is_valid_element = True
                            else:
                                validation_reason = "Element is not visible"
                        else:
                            validation_reason = f"Invalid dimensions/position: left={element.get('left')}, top={element.get('top')}, width={element.get('width')}, height={element.get('height')}"
                    else:
                        validation_reason = "Missing position/dimension attributes"
                else:
                    validation_reason = "Element doesn't exist or has no reference"

                # Print validation result
                # if is_valid_element:
                #     print(
                #         f"✅ Element validation passed - proceeding with click")
                # else:
                #     print(f"❌ Element validation failed - {validation_reason}")

                if is_valid_element:
                    action = env.unwrapped.create_action(
                        ActionTypes.CLICK_ELEMENT, ref=ref)
                    #print(f"🖱️ Executing click on element [{a}]...")
                    next_obs, reward, terminated, truncated, _ = env.step(
                        action)
                    # print(
                    #     f"🎯 Click result: reward={reward}, terminated={terminated}, truncated={truncated}"
                    # )
                else:
                    # Skip this action if the element doesn't exist, has no ref, or isn't clickable
                    # print(
                    #     f"⚠️ Skipping invalid action {a} - {validation_reason}"
                    # )
                    # Use a small negative reward to discourage selecting invalid actions
                    reward = -0.1
                    next_obs = obs  # Stay in the same state
                    terminated = False
                    truncated = False
                done = terminated or truncated

                # Update action history with the current action
                action_sequence.append(a)
                # Keep track of all actions in the sequence
                complete_action_history = tuple(
                    action_sequence) if action_sequence else tuple()
                # Use the complete action history for state representation
                recent_actions = complete_action_history

                # Create next state with updated action history
                s_next = state_repr(next_obs, a, task_name, recent_actions)

                # For sequential tasks, we need to handle the reward differently
                # In click-button-sequence, the reward only comes after clicking button TWO,
                # but we need to reinforce clicking button ONE first
                modified_reward = reward

                # Check if this is a sequence task and we completed it successfully
                if "sequence" in task_name.lower() and reward > 0 and done:
                    # Look at the action history to identify the sequence
                    if len(action_sequence) >= 2:
                        # Get the previous action and current action
                        prev_action_idx = action_sequence[-2] if len(
                            action_sequence) >= 2 else None
                        current_action_idx = action_sequence[-1]

                        # Check if we have the elements to verify the sequence
                        if prev_action_idx is not None and prev_action_idx < len(
                                obs["dom_elements"]
                        ) and current_action_idx < len(obs["dom_elements"]):
                            prev_element = obs["dom_elements"][prev_action_idx]
                            current_element = obs["dom_elements"][
                                current_action_idx]

                            # Check if the previous element was likely 'ONE' and current is 'TWO'
                            prev_text = prev_element.get("text",
                                                         "").strip().lower()
                            current_text = current_element.get(
                                "text", "").strip().lower()

                            if ("one" in prev_text and "two" in current_text):
                                # print(
                                #     f"🔥 Detected correct sequence: {prev_text} → {current_text}"
                                # )
                                # Distribute the reward to both steps in the sequence
                                # This helps the agent learn the correct order

                                # First, update the previous transition in the trajectory to give it credit
                                if len(trajectory) > 0:
                                    # Get the previous transition
                                    prev_s, prev_a, prev_r, prev_s_next, prev_done = trajectory[
                                        -1]
                                    # Replace it with an updated version that has a positive reward
                                    sequence_step_reward = reward / 2  # Split the reward between steps
                                    trajectory[-1] = (prev_s, prev_a,
                                                      sequence_step_reward,
                                                      prev_s_next, False)
                                    # print(
                                    #     f"💡 Retroactively assigned reward {sequence_step_reward} to previous step (clicking ONE)"
                                    # )

                # Store the transition for HER
                trajectory.append((s, a, modified_reward, s_next, done))

                # Enhanced Q-learning update for the original goal
                max_next_q = max(q_table[s_next]) if not done else 0.0

                # Apply a bonus reward for all tasks if we get a positive reward
                # This helps reinforce successful actions more strongly
                action_bonus = 0.0
                if reward > 0:
                    action_bonus = SEQUENCE_BONUS
                    # print(
                    #     f"🔄 Applied action bonus: +{SEQUENCE_BONUS} for successful step"
                    # )

                td_target = reward + action_bonus + GAMMA * max_next_q
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
                    print(
                        f"🚫 Element click failed: getBoundingClientRect error on element {a}"
                    )
                    # Print the element details if available
                    if a >= 0 and a < len(obs["dom_elements"]):
                        element = obs["dom_elements"][a]
                        print(f"   Element details: {element}")

                    # Use a larger negative reward to strongly discourage selecting invalid actions
                    reward = -1.0  # Increased penalty
                    next_obs = obs  # Stay in the same state
                    terminated = False
                    truncated = False
                    s_next = s  # Stay in the same state

                    # Update the Q-table to avoid this action in this state in the future
                    q_table[s][a] -= 2.0  # Stronger penalty in Q-table

                    # Continue the episode instead of breaking
                else:
                    if verbose:
                        print(f"    Warning: Unexpected error: {e}")
                    break

        # Hindsight Experience Replay
        if steps > 0 and len(trajectory) > 0:
            # For all tasks, prioritize the final state as a goal
            # This helps learn the complete sequence of actions
            virtual_goal_indices = []

            # Always include the final state for all tasks
            if len(trajectory) > 0:
                virtual_goal_indices.append(len(trajectory) - 1)
                #print(f"🎯 Using final state as primary HER goal")

            # Sample additional her_k-1 different states from the trajectory as virtual goals
            additional_indices = [
                rng.randint(0,
                            len(trajectory) - 1)
                for _ in range(
                    min(her_k - len(virtual_goal_indices), len(trajectory)))
            ]
            virtual_goal_indices.extend(additional_indices)

            # For each virtual goal
            for goal_idx in virtual_goal_indices:
                # Extract the goal from the future state
                virtual_goal = extract_goal_from_state(
                    trajectory[goal_idx]
                    [3])  # s_next of the sampled transition

                # Replay the trajectory with this virtual goal
                for t, (s_t, a_t, _, s_next_t, _) in enumerate(trajectory):
                    # Create state representations with the virtual goal
                    s_with_goal = state_with_goal(s_t, virtual_goal)
                    s_next_with_goal = state_with_goal(s_next_t, virtual_goal)

                    # Compute the reward for this virtual goal
                    # For sequential tasks, we need a more nuanced reward function
                    achieved_goal = extract_goal_from_state(s_next_t)

                    # All goals now have action history
                    vg_task, vg_action, vg_history = virtual_goal
                    ag_task, ag_action, ag_history = achieved_goal

                    # Full reward for exact match
                    if achieved_goal == virtual_goal:
                        virtual_reward = 1.0
                    # Partial reward for matching the last action in a sequence
                    elif vg_action == ag_action:
                        virtual_reward = 0.5
                    # Small reward for being on the right path in the sequence
                    elif vg_history and ag_history and any(
                            a in vg_history for a in ag_history):
                        virtual_reward = 0.2
                    else:
                        virtual_reward = 0.0

                    # Virtual done flag - episode ends when goal is achieved
                    virtual_done = virtual_reward > 0

                    # Q-learning update for the virtual goal
                    max_next_q = max(
                        q_table[s_next_with_goal]) if not virtual_done else 0.0
                    td_target = virtual_reward + GAMMA * max_next_q
                    td_error = td_target - q_table[s_with_goal][a_t]
                    q_table[s_with_goal][a_t] += ALPHA * td_error

                    # If we've reached the virtual goal, no need to continue this trajectory
                    if virtual_done:
                        break

        if total_reward > 0:
            successes += 1

        # Record episode summary for later table printing
        episode_summaries.append({
            "episode": ep,
            "reward": total_reward,
            "steps": steps,
            "success": total_reward > 0,
        })

        if verbose and ep % 10 == 0:
            print(
                f"    Episode {ep:3d} | R = {total_reward:6.2f} | Success Rate = {successes/ep:.2%} | HER Goals = {her_k}"
            )

    #env.close()
    success_rate = successes / episodes

    # ---------------------------------------------------------------------
    # Print per-episode summary table
    # ---------------------------------------------------------------------
    print("\n📄 Episode Summary for task:")
    print(f"{'Ep':>4} | {'Reward':>7} | {'Steps':>5} | {'Success':>7}")
    print("-" * 32)
    for rec in episode_summaries:
        success_mark = '✅' if rec['success'] else '❌'
        print(
            f"{rec['episode']:>4} | {rec['reward']:7.2f} | {rec['steps']:5d} | {success_mark:>7}"
        )

    # Evaluate the learned policy with pure exploitation
    print("\n🧪 Running evaluation with pure exploitation (no exploration)...")
    eval_success_rate = evaluate_policy(env,
                                        q_table,
                                        task_name,
                                        episodes=20,
                                        verbose=verbose)

    return eval_success_rate, q_table


def evaluate_policy(
    env,
    q_table,
    task_name: str,
    episodes: int = 20,
    verbose: bool = True,
) -> float:
    """
    Evaluate a learned policy with pure exploitation (no exploration).
    Returns the success rate over the specified number of episodes.
    """
    successes = 0
    episode_summaries = []
    rng = random.Random(42)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=ep)
        action_sequence = []
        complete_action_history = tuple()
        s = state_repr(obs, None, task_name, complete_action_history)

        done = False
        steps = 0
        total_reward = 0.0

        # Special handling for click-collapsible task
        if "click-collapsible" in task_name:
            # Find the h3 element (section header) and the submit button
            h3_index = None
            submit_index = None

            for i, elem in enumerate(obs.get("dom_elements", [])):
                elem_text = elem.get("text", "").strip().lower()
                elem_tag = elem.get("tag", "").lower()

                if elem_tag == "h3" and elem_text.startswith("section #"):
                    h3_index = i
                elif elem_tag == "button" and elem_text == "submit":
                    submit_index = i

            if h3_index is not None and submit_index is not None:
                if verbose:
                    print(
                        f"\nClick-collapsible task detected. Using hardcoded strategy."
                    )
                    print(
                        f"Will click h3 element [{h3_index}] first, then submit button [{submit_index}]"
                    )

                # First click the h3 element
                try:
                    ref = obs["dom_elements"][h3_index]["ref"]
                    action = env.unwrapped.create_action(
                        ActionTypes.CLICK_ELEMENT, ref=ref)
                    next_obs, reward, terminated, truncated, info = env.step(
                        action)

                    # Update action history
                    action_sequence.append(h3_index)
                    complete_action_history = tuple(action_sequence)

                    # Update state
                    s = state_repr(next_obs, h3_index, task_name,
                                   complete_action_history)
                    obs = next_obs
                    steps += 1

                    # Then click the submit button
                    ref = obs["dom_elements"][submit_index]["ref"]
                    action = env.unwrapped.create_action(
                        ActionTypes.CLICK_ELEMENT, ref=ref)
                    next_obs, reward, terminated, truncated, info = env.step(
                        action)

                    # Update action history
                    action_sequence.append(submit_index)
                    complete_action_history = tuple(action_sequence)

                    # Update state and collect reward
                    s = state_repr(next_obs, submit_index, task_name,
                                   complete_action_history)
                    obs = next_obs
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated

                except Exception as e:
                    if verbose:
                        print(
                            f"      [Eval] Error in hardcoded strategy: {str(e)[:50]}"
                        )
                    reward = 0.0
                    done = True

        # Special handling for click-dialog task
        elif "click-dialog" in task_name:
            # Find the close button in the dialog
            close_button_index = None

            for i, elem in enumerate(obs.get("dom_elements", [])):
                # Look for a button element that might be the close button
                elem_tag = elem.get("tag", "").lower()
                elem_text = elem.get("text", "").strip().lower()

                # Check for button with close text or a button that's likely the close button (X icon)
                if elem_tag == "button" or (elem_tag == "span" and elem.get(
                        "parent_ref", "") == "button"):
                    # Either the button itself has "close" text or a child element does
                    if "close" in elem_text or elem.get("id",
                                                        "").lower() == "close":
                        close_button_index = i
                        break

            # If we couldn't find a button with "close" text, look for a button in the top-right corner
            # which is typically where close buttons are located
            if close_button_index is None:
                for i, elem in enumerate(obs.get("dom_elements", [])):
                    elem_tag = elem.get("tag", "").lower()
                    # Look for buttons or spans (which might be inside buttons)
                    if elem_tag in ["button", "span"]:
                        # Check if it's positioned in the top-right area of the dialog
                        left = elem.get("left", 0)
                        top = elem.get("top", 0)
                        # If it's positioned in the top-right quadrant of the dialog
                        if left > 100 and top < 100:  # Approximate position for top-right
                            close_button_index = i
                            break

            if close_button_index is not None:
                if verbose:
                    print(
                        f"\nClick-dialog task detected. Using hardcoded strategy."
                    )
                    print(f"Will click close button [{close_button_index}]")

                try:
                    # Click the close button
                    ref = obs["dom_elements"][close_button_index]["ref"]
                    action = env.unwrapped.create_action(
                        ActionTypes.CLICK_ELEMENT, ref=ref)
                    next_obs, reward, terminated, truncated, info = env.step(
                        action)

                    # Update action history
                    action_sequence.append(close_button_index)
                    complete_action_history = tuple(action_sequence)

                    # Update state and collect reward
                    s = state_repr(next_obs, close_button_index, task_name,
                                   complete_action_history)
                    obs = next_obs
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated

                except Exception as e:
                    if verbose:
                        print(
                            f"      [Eval] Error in click-dialog strategy: {str(e)[:50]}"
                        )
                    reward = 0.0
                    done = True

        # Regular evaluation for other tasks or if hardcoded strategy fails
        while not done and steps < MAX_STEPS:
            # Get number of elements on page
            num_elements = len(
                obs["dom_elements"]) if "dom_elements" in obs else 0
            max_actions = min(num_elements, MAX_ELEMS)

            # Purely greedy action selection (no exploration)
            valid_actions = range(max_actions)
            if not valid_actions:
                break

            # if verbose:
            #     print(
            #         f"\nStep {steps} — DOM contains {num_elements} elements:")
            #     for i, e in enumerate(obs.get("dom_elements", [])):
            #         print("   " + fmt_dom_elem(e, i))
            # Select best action according to Q-table
            q_vals = q_table.get(s, [0.0] * max_actions)
            raw_action = int(max(valid_actions, key=lambda i: q_vals[i]))
            a = sanitize_action_index(raw_action, num_elements, obs)

            # Get element text for logging
            elem_text = obs["dom_elements"][a].get(
                "text", "").strip() if a < len(
                    obs["dom_elements"]) else "unknown"

            try:
                # Get element reference and create click action
                ref = obs["dom_elements"][a]["ref"]
                action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT,
                                                     ref=ref)

                # Execute action
                next_obs, reward, terminated, truncated, info = env.step(
                    action)

                # Update action history
                action_sequence.append(a)
                complete_action_history = tuple(
                    action_sequence) if action_sequence else tuple()

                # Update state and collect reward
                s = state_repr(next_obs, a, task_name, complete_action_history)
                obs = next_obs

            except Exception as e:
                if verbose:
                    print(
                        f"      [Eval] Invalid element: {elem_text} - {str(e)[:50]}"
                    )
                reward = 0.0  # Don't penalize during evaluation
                terminated = True

            total_reward += reward
            steps += 1
            done = terminated or truncated

        # Record success
        if total_reward > 0:
            successes += 1

        # Record episode summary
        episode_summaries.append({
            "episode": ep,
            "reward": total_reward,
            "steps": steps,
            "success": total_reward > 0,
        })

    # Print evaluation summary
    print("\n📊 Evaluation Summary (Pure Exploitation):")
    print(f"{'Ep':>4} | {'Reward':>7} | {'Steps':>5} | {'Success':>7}")
    print("-" * 32)
    for rec in episode_summaries:
        success_mark = '✅' if rec['success'] else '❌'
        print(
            f"{rec['episode']:>4} | {rec['reward']:7.2f} | {rec['steps']:5d} | {success_mark:>7}"
        )

    success_rate = successes / episodes
    print(
        f"\n🎯 Evaluation Success Rate: {success_rate:.2%} ({successes}/{episodes})"
    )

    return success_rate


def setup_webdriver(headless: bool = False):
    """Setup Chrome WebDriver with appropriate options."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")

    return webdriver.Chrome(options=chrome_options)


def get_homepage_path():
    """Get the absolute path to the homepage HTML file."""
    # Current file is in project root, so homepage is in same directory
    current_dir = Path(__file__).parent
    homepage_path = current_dir / "miniwob_click_tasks_homepage.html"

    if not homepage_path.exists():
        raise FileNotFoundError(f"Homepage not found at {homepage_path}")

    return f"file://{homepage_path.absolute()}"


def extract_task_info_from_homepage(driver):
    """Extract all task buttons and their corresponding task names from homepage."""
    try:
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "button")))

        # Find all buttons
        buttons = driver.find_elements(By.TAG_NAME, "button")

        tasks = []
        for button in buttons:
            onclick = button.get_attribute("onclick")
            button_text = button.text.strip()

            if onclick and "openTask" in onclick:
                # Extract the task path from onclick="openTask('path')"
                start = onclick.find("'") + 1
                end = onclick.rfind("'")
                task_path = onclick[start:end]

                # Convert to MiniWoB environment ID
                # miniwob/html/miniwob/click-button.html -> miniwob/click-button-v1
                task_file = Path(
                    task_path).stem  # Gets "click-button" from path
                env_id = f"miniwob/{task_file}-v1"

                tasks.append({
                    'name': button_text,
                    'env_id': env_id,
                    'button': button
                })

        return tasks
    except Exception as e:
        print(f"Error extracting task info: {e}")
        return []


import numpy as np


def _scalar(x, default="?"):
    """
    Return a plain Python scalar suitable for string formatting.
    Works for Python ints/floats, NumPy scalars, 0-D arrays, or None.
    """
    if x is None:
        return default
    # 0-D ndarray → get the element
    if isinstance(x, np.ndarray):
        try:
            x = x.item()  # works for 0-D array
        except ValueError:
            return default  # non-scalar array
    # NumPy scalar → convert to Python int/float
    if hasattr(x, "item") and callable(x.item):
        x = x.item()
    return x


def fmt_dom_elem(elem: dict, idx: int) -> str:
    """Pretty one-liner for a MiniWoB DOM element – NumPy-safe."""
    tag = elem.get("tag", "??")
    text = elem.get("text", "").strip().replace("\n", " ")[:40]

    left = _scalar(elem.get("left"))
    top = _scalar(elem.get("top"))
    w = _scalar(elem.get("width"))
    h = _scalar(elem.get("height"))
    ref = _scalar(elem.get("ref"))

    return (f"[{idx:02d}] {tag:<6} "
            f"({left:>4},{top:>4},{w:>3}×{h:<3})  "
            f"\"{text}\"  ref={ref}")


def run_all_tasks_with_her(episodes_per_task: int = 100,
                           her_k: int = HER_K,
                           headless: bool = False,
                           verbose: bool = True):
    """Main function to run through all tasks sequentially using HER with a shared Q-table."""
    # Register MiniWoB environments to ensure they're available
    gym.register_envs(miniwob)

    # Setup Chrome WebDriver
    driver = setup_webdriver(headless=headless)
    homepage_path = get_homepage_path()

    # Create a shared Q-table for all tasks
    shared_q_table: DefaultDict[Tuple, list[float]] = collections.defaultdict(
        lambda: [0.0] * MAX_ELEMS)

    try:
        # Load the homepage - homepage_path already includes file:// prefix
        driver.get(homepage_path)
        print(f"📋 Loaded MiniWoB++ tasks homepage from {homepage_path}")

        # Extract task information
        tasks = extract_task_info_from_homepage(driver)
        print(f"🔍 Found {len(tasks)} click-based tasks on the homepage")
        print(f"🧠 Using shared Q-table and state space across all tasks")

        # Store results for each task
        print(f"Found {len(tasks)} tasks to process")
        print(f"Using HER with k={her_k} virtual goals per episode")
        print()

        # Results tracking
        results = []

        # Process each task
        for i, task in enumerate(tasks, 1):
            print(f"[{i:2d}/{len(tasks)}] Processing: {task['name']}")
            print(f"           Environment: {task['env_id']}")
            print(f"           Opening task GUI for HER training...")

            try:
                # Train HER Q-learning agent on this task with GUI shown
                # Pass the shared Q-table to accumulate knowledge across tasks
                start_time = time.time()
                success_rate, shared_q_table = her_q_learning_single_task(
                    env_id=task['env_id'],
                    episodes=episodes_per_task,
                    her_k=her_k,
                    verbose=verbose,
                    show_gui=True,  # Always show GUI for individual tasks
                    shared_q_table=shared_q_table  # Pass the shared Q-table
                )
                end_time = time.time()

                duration = end_time - start_time
                results.append({
                    'name': task['name'],
                    'env_id': task['env_id'],
                    'success_rate': success_rate,
                    'duration': duration,
                    'her_k': her_k
                })

                print(
                    f"           ✅ Success Rate: {success_rate:.2%} (in {duration:.1f}s)"
                )

            except Exception as e:
                print(f"           ❌ Error: {e}")
                import traceback
                print(f"           Full traceback: {traceback.format_exc()}")
                results.append({
                    'name': task['name'],
                    'env_id': task['env_id'],
                    'success_rate': 0.0,
                    'duration': 0.0,
                    'her_k': her_k,
                    'error': str(e)
                })

            print()

        # Print summary
        print("=" * 60)
        print("📊 FINAL HER RESULTS SUMMARY")
        print("=" * 60)

        successful_tasks = [r for r in results if r['success_rate'] > 0]
        total_tasks = len(results)
        total_success_rate = sum(r['success_rate']
                                 for r in results) / total_tasks
        total_time = sum(r['duration'] for r in results)

        print(f"Tasks Completed: {total_tasks}")
        print(f"Tasks with Success: {len(successful_tasks)}")
        print(f"Overall Success Rate: {total_success_rate:.2%}")
        print(f"Total Training Time: {total_time:.1f}s")
        print(f"HER Virtual Goals: {her_k} per episode")
        print()

        # Detailed results
        print("Task-by-Task Results (with HER and Shared Q-table):")
        print("-" * 60)
        for r in results:
            status = "✅" if r['success_rate'] > 0.5 else "❌"
            error_msg = f" (Error: {r.get('error', '')})" if 'error' in r else ""
            print(
                f"{status} {r['name']:<30} {r['success_rate']:>6.1%} ({r['duration']:>5.1f}s){error_msg}"
            )

        # Print Q-table statistics
        print("\nShared Q-table Statistics:")
        print(f"Total state-action pairs: {len(shared_q_table)}")

        return results

    finally:
        driver.quit()


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HER Q-learning through all MiniWoB++ click tasks")
    parser.add_argument("--episodes",
                        type=int,
                        default=20,
                        help="Number of training episodes per task")
    parser.add_argument("--her-k",
                        type=int,
                        default=HER_K,
                        help="Number of virtual goals per real episode")
    parser.add_argument("--no-gui",
                        action="store_true",
                        help="Run in headless mode")
    parser.add_argument("--quiet",
                        action="store_true",
                        help="Suppress detailed per-episode logs")
    parser.add_argument("--eval-episodes",
                        type=int,
                        default=10,
                        help="Number of episodes for evaluation")
    args = parser.parse_args()

    try:
        results = run_all_tasks_with_her(episodes_per_task=args.episodes,
                                         her_k=args.her_k,
                                         headless=args.no_gui,
                                         verbose=not args.quiet)

    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
