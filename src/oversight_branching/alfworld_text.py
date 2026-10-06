import difflib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import textworld
import textworld.gym

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos


ACTION_RE = re.compile(r"ACTION\s*:\s*([^\n]+)", re.IGNORECASE)
PLAN_RE = re.compile(r"PLAN\s*:\s*([^\n]+)", re.IGNORECASE)
CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
TASK_TYPE_TO_ID = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
    "pick_and_place_with_movable_recep": 7,
}


@dataclass
class AlfworldPrefix:
    example_id: str
    split: str
    gamefile: str
    task_desc: str
    task_type: str
    history_actions: List[str]
    history_observations: List[str]
    observation: str
    admissible_commands: List[str]
    step_index: int


def split_root(split_name: str) -> str:
    root = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    return os.path.join(root, "json_2.1.1", split_name)


def load_task_description(gamefile: str) -> str:
    traj_path = os.path.join(os.path.dirname(gamefile), "traj_data.json")
    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["turk_annotations"]["anns"][0]["task_desc"]


def load_task_type(gamefile: str) -> str:
    traj_path = os.path.join(os.path.dirname(gamefile), "traj_data.json")
    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["task_type"]


def collect_gamefiles(split_name: str, task_types: List[str], limit: int, seed: int) -> List[str]:
    root = split_root(split_name)
    candidates = []
    allowed = set(task_types)
    for current_root, _, files in os.walk(root, topdown=False):
        if "traj_data.json" not in files:
            continue
        if "movable" in current_root or "Sliced" in current_root:
            continue
        traj_path = os.path.join(current_root, "traj_data.json")
        gamefile = os.path.join(current_root, "game.tw-pddl")
        if not os.path.exists(gamefile):
            continue
        with open(traj_path, "r", encoding="utf-8") as f:
            traj = json.load(f)
        if traj["task_type"] not in allowed:
            continue
        with open(gamefile, "r", encoding="utf-8") as f:
            game = json.load(f)
        if not game.get("solvable", False):
            continue
        candidates.append(gamefile)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:limit]


def make_single_game_env(gamefile: str, max_episode_steps: int = 30, include_policy_commands: bool = False):
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        policy_commands=include_policy_commands,
        extras=["gamefile"],
    )
    env_id = textworld.gym.register_games(
        [gamefile],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=wrappers,
    )
    return textworld.gym.make(env_id)


def unwrap_reset(reset_output):
    observation, infos = reset_output
    return observation[0], infos["admissible_commands"][0], bool(infos["won"][0]), infos


def unwrap_step(step_output):
    observation, reward, done, infos = step_output
    return observation[0], float(reward[0]), bool(done[0]), bool(infos["won"][0]), infos["admissible_commands"][0], infos


def history_to_text(history_actions: List[str], history_observations: List[str], max_items: int = 4) -> str:
    if not history_actions:
        return "None"
    segments = []
    start = max(0, len(history_actions) - max_items)
    for idx in range(start, len(history_actions)):
        segments.append(
            f"Step {idx + 1} action: {history_actions[idx]}\n"
            f"Step {idx + 1} observation: {history_observations[idx][:500]}"
        )
    return "\n\n".join(segments)


def build_continue_prompt(task_desc: str, observation: str, history_text: str, admissible_commands: List[str]) -> str:
    commands = "\n".join(f"- {command}" for command in admissible_commands)
    return (
        "You are controlling an ALFWorld text agent.\n"
        "Choose the single best next action to make progress on the task.\n"
        "You must copy one action exactly from the admissible command list.\n"
        "Return exactly one line and nothing else:\n"
        "ACTION: <one admissible command>\n\n"
        "Task:\n"
        f"{task_desc}\n\n"
        "Recent history:\n"
        f"{history_text}\n\n"
        "Current observation:\n"
        f"{observation}\n\n"
        "Admissible commands:\n"
        f"{commands}"
    )


def build_verify_prompt(
    task_desc: str,
    observation: str,
    history_text: str,
    proposed_action: str,
    admissible_commands: List[str],
) -> str:
    alternative_commands = [
        command
        for command in admissible_commands
        if normalize_action(command) != normalize_action(proposed_action)
    ]
    if not alternative_commands:
        alternative_commands = admissible_commands
    commands = "\n".join(f"- {command}" for command in alternative_commands)
    return (
        "You are choosing a corrective action for an ALFWorld text agent.\n"
        "The system has already considered the proposed action below.\n"
        "Your job is to propose the best alternative next action if we intervene.\n"
        "Choose an admissible command that is DIFFERENT from the proposed action.\n"
        "Prefer actions that repair a likely mistake or open a better path to the task goal.\n"
        "Return exactly one line and nothing else:\n"
        "ACTION: <one admissible command different from the proposed action>\n\n"
        "Task:\n"
        f"{task_desc}\n\n"
        "Recent history:\n"
        f"{history_text}\n\n"
        "Current observation:\n"
        f"{observation}\n\n"
        "Proposed action:\n"
        f"{proposed_action}\n\n"
        "Admissible commands:\n"
        f"{commands}"
    )


def build_reflect_retry_prompt(
    task_desc: str,
    observation: str,
    history_text: str,
    proposed_action: str,
    admissible_commands: List[str],
) -> str:
    alternative_commands = [
        command
        for command in admissible_commands
        if normalize_action(command) != normalize_action(proposed_action)
    ]
    if not alternative_commands:
        alternative_commands = admissible_commands
    commands = "\n".join(f"- {command}" for command in alternative_commands)
    return (
        "You are intervening in an ALFWorld trajectory because the previously proposed action may be wrong.\n"
        "Briefly identify the likely problem with the proposed action, then choose a better admissible next action.\n"
        "Choose an admissible command that is DIFFERENT from the proposed action.\n"
        "Return exactly two lines and nothing else:\n"
        "REFLECTION: <one short sentence>\n"
        "ACTION: <one admissible command different from the proposed action>\n\n"
        "Task:\n"
        f"{task_desc}\n\n"
        "Recent history:\n"
        f"{history_text}\n\n"
        "Current observation:\n"
        f"{observation}\n\n"
        "Proposed action:\n"
        f"{proposed_action}\n\n"
        "Admissible commands:\n"
        f"{commands}"
    )


def build_replan_prompt(
    task_desc: str,
    observation: str,
    history_text: str,
    proposed_action: str,
    admissible_commands: List[str],
    remaining_repair_steps: int,
) -> str:
    commands = "\n".join(f"- {command}" for command in admissible_commands)
    proposed_block = (
        "Possibly bad proposed action:\n"
        f"{proposed_action}\n\n"
        if proposed_action
        else ""
    )
    return (
        "You are repairing an ALFWorld trajectory after the previously proposed action may have gone off track.\n"
        "You have a small repair budget and should choose the single best next action to recover progress toward the task.\n"
        "Do not repeat the possibly bad proposed action unless it is clearly still the best move.\n"
        f"There are {remaining_repair_steps} repair step(s) left in this repair mode, including this one.\n"
        "Return exactly one line and nothing else:\n"
        "ACTION: <one admissible command>\n\n"
        "Task:\n"
        f"{task_desc}\n\n"
        "Recent history:\n"
        f"{history_text}\n\n"
        "Current observation:\n"
        f"{observation}\n\n"
        f"{proposed_block}"
        "Admissible commands:\n"
        f"{commands}"
    )


def build_plan_repair_prompt(
    task_desc: str,
    observation: str,
    history_text: str,
    proposed_action: str,
    admissible_commands: List[str],
    repair_horizon: int,
) -> str:
    commands = "\n".join(f"- {command}" for command in admissible_commands)
    return (
        "You are repairing an ALFWorld trajectory after the previously proposed action may have gone off track.\n"
        "First produce a short repair plan, then choose the best immediate next action.\n"
        f"You have up to {repair_horizon} repair steps total before normal execution resumes.\n"
        "The first action may match the previous proposal only if it still fits a better repair plan.\n"
        "Return exactly two lines and nothing else:\n"
        "PLAN: <short repair plan>\n"
        "ACTION: <one admissible command>\n\n"
        "Task:\n"
        f"{task_desc}\n\n"
        "Recent history:\n"
        f"{history_text}\n\n"
        "Current observation:\n"
        f"{observation}\n\n"
        "Possibly bad proposed action:\n"
        f"{proposed_action}\n\n"
        "Admissible commands:\n"
        f"{commands}"
    )


def build_followup_replan_prompt(
    task_desc: str,
    observation: str,
    history_text: str,
    repair_plan: str,
    admissible_commands: List[str],
    remaining_repair_steps: int,
) -> str:
    commands = "\n".join(f"- {command}" for command in admissible_commands)
    return (
        "You are continuing a bounded repair rollout in ALFWorld.\n"
        "Choose the next action that best follows the repair plan while making concrete progress.\n"
        f"There are {remaining_repair_steps} repair step(s) left, including this one.\n"
        "Return exactly one line and nothing else:\n"
        "ACTION: <one admissible command>\n\n"
        "Task:\n"
        f"{task_desc}\n\n"
        "Repair plan:\n"
        f"{repair_plan}\n\n"
        "Recent history:\n"
        f"{history_text}\n\n"
        "Current observation:\n"
        f"{observation}\n\n"
        "Admissible commands:\n"
        f"{commands}"
    )


def build_confidence_prompt(task_desc: str, observation: str, history_text: str, proposed_action: str) -> str:
    return (
        "Estimate how likely the proposed next action is to keep the ALFWorld trajectory on track toward success.\n"
        "Be calibrated and conservative. Use 100 only when the action is very likely to be the right move.\n"
        "Return exactly one line and nothing else:\n"
        "CONFIDENCE: <0-100>\n\n"
        "Task:\n"
        f"{task_desc}\n\n"
        "Recent history:\n"
        f"{history_text}\n\n"
        "Current observation:\n"
        f"{observation}\n\n"
        "Proposed action:\n"
        f"{proposed_action}"
    )


def extract_confidence(model_text: str) -> float:
    match = CONFIDENCE_RE.search(model_text)
    if not match:
        return 50.0
    value = float(match.group(1))
    return min(100.0, max(0.0, value))


def extract_plan(model_text: str) -> str:
    match = PLAN_RE.search(model_text)
    if match:
        return match.group(1).strip()
    first_line = model_text.strip().splitlines()[0].strip() if model_text.strip() else ""
    return first_line or "repair the trajectory and make task progress"


def normalize_action(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_action(model_text: str, admissible_commands: List[str]) -> str:
    if not admissible_commands:
        return "look"
    match = ACTION_RE.search(model_text)
    lines = model_text.strip().splitlines()
    candidate = match.group(1).strip() if match else (lines[0].strip() if lines else "")
    admissible_map = {normalize_action(command): command for command in admissible_commands}
    normalized_candidate = normalize_action(candidate)
    if normalized_candidate in admissible_map:
        return admissible_map[normalized_candidate]
    close = difflib.get_close_matches(normalized_candidate, admissible_map.keys(), n=1, cutoff=0.6)
    if close:
        return admissible_map[close[0]]
    for command in admissible_commands:
        if normalized_candidate and normalized_candidate in normalize_action(command):
            return command
    return admissible_commands[0]


def basic_feature_vector(
    task_desc: str,
    observation: str,
    admissible_commands: List[str],
    chosen_action: str,
    step_index: int,
    history_len: int,
) -> List[float]:
    action = chosen_action.lower()
    return [
        float(step_index),
        float(history_len),
        float(len(task_desc.split())),
        float(len(observation.split())),
        float(len(observation)),
        float(len(admissible_commands)),
        float(len(chosen_action.split())),
        1.0 if action.startswith("go to ") else 0.0,
        1.0 if action.startswith("open ") else 0.0,
        1.0 if action.startswith("close ") else 0.0,
        1.0 if action.startswith("take ") else 0.0,
        1.0 if action.startswith("put ") else 0.0,
        1.0 if action.startswith("heat ") else 0.0,
        1.0 if action.startswith("cool ") else 0.0,
        1.0 if action.startswith("clean ") else 0.0,
        1.0 if action.startswith("toggle ") else 0.0,
        1.0 if action == "look" else 0.0,
        1.0 if "nothing happens" in observation.lower() else 0.0,
    ]


def verify_feature_vector(
    observation: str,
    admissible_commands: List[str],
    continue_action: str,
    verify_action: str,
    step_index: int,
) -> List[float]:
    continue_normalized = normalize_action(continue_action)
    verify_normalized = normalize_action(verify_action)
    return [
        float(step_index),
        float(len(admissible_commands)),
        float(len(verify_action.split())),
        1.0 if continue_normalized == verify_normalized else 0.0,
        1.0 if verify_normalized.startswith("go to ") else 0.0,
        1.0 if verify_normalized.startswith("open ") else 0.0,
        1.0 if verify_normalized.startswith("take ") else 0.0,
        1.0 if verify_normalized.startswith("put ") else 0.0,
        1.0 if verify_normalized.startswith("heat ") else 0.0,
        1.0 if verify_normalized.startswith("cool ") else 0.0,
        1.0 if verify_normalized.startswith("clean ") else 0.0,
        1.0 if verify_normalized.startswith("toggle ") else 0.0,
        1.0 if "closed" in observation.lower() else 0.0,
        1.0 if "open" in observation.lower() else 0.0,
    ]


def state_text(prefix: AlfworldPrefix) -> str:
    return (
        f"Task: {prefix.task_desc}\n\n"
        f"History:\n{history_to_text(prefix.history_actions, prefix.history_observations)}\n\n"
        f"Observation:\n{prefix.observation}\n\n"
        f"Admissible commands:\n" + "\n".join(prefix.admissible_commands)
    )
