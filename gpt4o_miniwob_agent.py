"""
Agent that uses OpenAI's GPT-4o to decide actions in MiniWoB++ web environments.

Usage:
    python gpt4o_miniwob_agent_simple.py --task click-button --render

Environment Variables:
    export OPENAI_API_KEY = Your OpenAI API key.

The agent works as follows:
    1.  Reset the requested MiniWoB++ environment.
    2.  At each step, send the current DOM along with past actions to GPT-4o.
    3.  GPT-4o returns a JSON describing the next action.
    4.  Convert that JSON to an env-compatible action and step the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import openai

import miniwob
from miniwob.action import ActionTypes


def get_task_html(task_name: str) -> str:
    try:
        miniwob_path = Path(miniwob.__file__).parent
        html_path = miniwob_path / "html" / "miniwob" / f"{task_name}.html"

        if html_path.exists():
            return html_path.read_text()

        local_path = Path("miniwob") / "html" / "miniwob" / f"{task_name}.html"
        if local_path.exists():
            return local_path.read_text()

        return ""
    except Exception as e:
        return ""


# call gpt 4o to decide next miniwob action
def gpt_decide(dom: str,
               task_name: str,
               dom_elements: List[Dict[str, Any]],
               previous_actions: List[Dict[str, Any]],
               instruction: str = "",
               task_html: str = "") -> Dict[str, Any]:
    # get DOM elements for better understanding
    dom_elements_formatted = []
    for i, elem in enumerate(dom_elements):
        elem_text = elem.get("text", "").strip()
        elem_tag = elem.get("tag", "unknown")
        elem_classes = elem.get("classes", [])
        elem_id = elem.get("id", "")
        elem_focused = elem.get("focused", False)
        elem_checked = elem.get("checked", False)

        id_part = f' id="{elem_id}"' if elem_id else ""
        class_str = " ".join(elem_classes)
        class_part = f' class="{class_str}"' if elem_classes else ""
        state_part = []
        if elem_focused:
            state_part.append("focused")
        if elem_checked:
            state_part.append("checked")
        state_str = f" ({', '.join(state_part)})" if state_part else ""

        elem_formatted = f"[{i}] {elem_tag}{id_part}{class_part}: '{elem_text}'{state_str}"
        dom_elements_formatted.append(elem_formatted)

    dom_elements_str = "\n".join(dom_elements_formatted)

    messages = [
        {
            "role":
            "system",
            "content":
            textwrap.dedent(f"""
            You are a web agent in charge of completing the MiniWoB task "{task_name}".
            Instruction: {instruction}

            Your job: given the current web page DOM, task HTML, and the sequence of previous
            actions, decide *one* logical next step that will bring you closer to
            finishing the task.

            The DOM contains elements with unique identifiers. To interact with an element,
            you need to identify it by its index in the elements list.

            You will be provided with both the structured DOM elements and the task's HTML file.
            Use both to make informed decisions about which elements to interact with.

            Respond *only* with a JSON object following this schema (no extra text):
                {{
                    "type": "click" | "type" | "noop" | "scroll",
                    "element_idx": 0,              # index of the element in the DOM (0-based)
                    "text": "str",                # required when type == "type"
                    "direction": "up" | "down"    # required when type == "scroll"
                }}
            """).strip()
        },
        {
            "role":
            "user",
            "content":
            f"Task: {task_name}\n\nTask HTML:\n{task_html}\n\nDOM Elements:\n{dom_elements_str}\n\nPrevious actions:\n{json.dumps(previous_actions, indent=2)}"
        },
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        content: str = response.choices[0].message.content.strip()
        action_dict = json.loads(content)
    except Exception as exc:
        action_dict = {"type": "noop"}

    return action_dict


# convert json to miniwob action
def json_to_miniwob_action(env, action_json: Dict[str, Any], obs: Dict[str,
                                                                       Any]):
    action_type = action_json.get("type", "noop")

    try:
        dom_elements = obs.get("dom_elements", [])
        num_elements = len(dom_elements)

        if action_type == "click":
            element_idx = action_json.get("element_idx", 0)
            if not isinstance(element_idx, int):
                try:
                    element_idx = int(element_idx)
                except (ValueError, TypeError):
                    element_idx = 0

            # ensure index is within bounds
            if element_idx < 0 or element_idx >= num_elements:
                element_idx = 0 if num_elements > 0 else None

            if element_idx is not None and num_elements > 0:
                element = dom_elements[element_idx]
                ref = element.get("ref")
                return env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT,
                                                   ref=ref)
            else:
                return env.unwrapped.create_action(ActionTypes.NONE)

        elif action_type == "type":
            element_idx = action_json.get("element_idx", 0)
            if not isinstance(element_idx, int):
                try:
                    element_idx = int(element_idx)
                except (ValueError, TypeError):
                    element_idx = 0

            # Ensure index is within bounds
            if element_idx < 0 or element_idx >= num_elements:
                element_idx = 0 if num_elements > 0 else None

            text = str(action_json.get("text", ""))

            if element_idx is not None and num_elements > 0:
                element = dom_elements[element_idx]
                ref = element.get("ref")
                return env.unwrapped.create_action(
                    ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                    ref=ref,
                    text=text)
            else:
                return env.unwrapped.create_action(ActionTypes.NONE)

        elif action_type == "scroll":
            direction = str(action_json.get("direction", "down")).upper()
            if direction not in ["UP", "DOWN"]:
                direction = "DOWN"  # Default to down if invalid
            return env.unwrapped.create_action(ActionTypes.PRESS_KEY,
                                               key=direction)

        else:  # noop
            return env.unwrapped.create_action(ActionTypes.NONE)

    except Exception as e:
        return env.unwrapped.create_action(ActionTypes.NONE)


def main() -> None:
    # CLI entry-point
    parser = argparse.ArgumentParser(description="GPT-4o MiniWoB agent")
    parser.add_argument(
        "--task",
        type=str,
        default="click-button",
        help="MiniWoB task name, e.g. click-button, book-flight, ...",
    )
    parser.add_argument("--max_steps",
                        type=int,
                        default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--render",
                        action="store_true",
                        help="Render environment window")
    args = parser.parse_args()

    # ensure correct environment ID format
    if args.task.startswith("miniwob/"):
        env_id = args.task
    else:
        # if task doesn't have the miniwob/ prefix, add it
        task_name = args.task
        if task_name.startswith("miniwob/"):
            task_name = task_name[len("miniwob/"):]
        env_id = f"miniwob/{task_name}-v1" if "-v" not in task_name else f"miniwob/{task_name}"

    try:
        env = gym.make(env_id, render_mode="human" if args.render else None)
    except gym.error.Error as exc:
        sys.stderr.write(f"[ERROR] Failed to make env '{env_id}': {exc}\n")
        sys.exit(1)

    previous_actions: List[Dict[str, Any]] = []

    # get the task HTML
    task_name = args.task.replace("miniwob/", "").replace("-v1", "")
    task_html = get_task_html(task_name)

    obs, _ = env.reset()
    instruction = obs.get("utterance", "") or obs.get("instruction", "")

    # main episode
    for t in range(args.max_steps):
        dom: str = obs.get("dom", "")  # MiniWoB observation is a dict
        dom_elements = obs.get("dom_elements", [])

        action_dict = gpt_decide(dom, args.task, dom_elements,
                                 previous_actions, instruction, task_html)
        previous_actions.append(action_dict)

        try:
            action = json_to_miniwob_action(env, action_dict, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(
                f"[INFO] Step result: reward={reward}, terminated={terminated}, truncated={truncated}"
            )
        except Exception as e:
            continue

        if args.render:
            env.render()

        if terminated or truncated:
            print(
                f"Main episode finished after step {t + 1} | reward={reward:.2f} | success={terminated}"
            )
            break
    else:
        print(
            f"Main episode finished after step {args.max_steps} | reward={reward:.2f} | success={terminated}"
        )

    env.close()


if __name__ == "__main__":
    main()
