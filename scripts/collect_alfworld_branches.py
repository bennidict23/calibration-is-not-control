import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from oversight_branching.alfworld_text import (
    TASK_TYPE_TO_ID,
    AlfworldPrefix,
    basic_feature_vector,
    build_confidence_prompt,
    build_continue_prompt,
    build_followup_replan_prompt,
    build_plan_repair_prompt,
    build_replan_prompt,
    build_reflect_retry_prompt,
    build_verify_prompt,
    collect_gamefiles,
    extract_action,
    extract_confidence,
    extract_plan,
    history_to_text,
    load_task_description,
    load_task_type,
    make_single_game_env,
    unwrap_reset,
    unwrap_step,
    verify_feature_vector,
)
from oversight_branching.local_llm import LocalChatModel
from oversight_branching.policy_learning import (
    choose_expected_utility_actions,
    choose_expected_utility_lcb_actions,
    choose_two_stage_actions,
    evaluate_actions,
    evaluate_expected_utility_lcb_policy,
    evaluate_expected_utility_policy,
    evaluate_failure_policy,
    evaluate_two_stage_policy,
    evaluate_value_policy,
    expected_utility_from_probability,
    feature_matrix,
    fit_correctness_models,
    fit_failure_baseline,
    fit_two_stage_models,
    fit_value_models,
    positive_probability_stats,
    threshold_search_expected_utility_lcb,
    threshold_search_failure,
    threshold_search_two_stage,
    threshold_search_value,
)


DEFAULT_TASK_TYPES = list(TASK_TYPE_TO_ID.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect branched ALFWorld prefixes: replay to a prefix, then execute continue, the intervention, and quit."
    )
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--api-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--val-games", type=int, default=4)
    parser.add_argument("--test-games", type=int, default=8)
    parser.add_argument("--prefixes-per-game", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-episode-steps", type=int, default=30)
    parser.add_argument("--action-max-tokens", type=int, default=24)
    parser.add_argument("--confidence-max-tokens", type=int, default=8)
    parser.add_argument("--verify-max-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repair-horizon", type=int, default=3)
    parser.add_argument(
        "--verify-mode",
        type=str,
        choices=["with_preview", "no_preview"],
        default="with_preview",
    )
    parser.add_argument("--verify-cost", type=float, default=0.05)
    parser.add_argument("--wrong-penalty", type=float, default=1.00)
    parser.add_argument("--step-cost", type=float, default=0.01)
    parser.add_argument(
        "--intervention-mode",
        type=str,
        choices=["llm_verify", "expert_defer", "self_reflect_retry", "self_replan", "self_plan_replan"],
        default="expert_defer",
    )
    parser.add_argument("--expert-max-policy-steps", type=int, default=-1)
    parser.add_argument("--expert-noise-rate", type=float, default=0.0)
    parser.add_argument("--task-types", nargs="+", default=DEFAULT_TASK_TYPES)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/alfworld_run"))
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def task_one_hot(task_type: str) -> list:
    vector = [0.0] * len(DEFAULT_TASK_TYPES)
    if task_type in DEFAULT_TASK_TYPES:
        vector[DEFAULT_TASK_TYPES.index(task_type)] = 1.0
    return vector


def utility(won: bool, branch_steps: int, verify: bool, verify_cost: float, wrong_penalty: float, step_cost: float) -> float:
    action_cost = branch_steps * step_cost + (verify_cost if verify else 0.0)
    if won:
        return 1.0 - action_cost
    return -wrong_penalty - action_cost


def safe_binary_metrics(labels, scores):
    if len(set(int(label) for label in labels)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def select_prefix_indices(num_states: int, prefixes_per_game: int) -> list:
    if num_states <= prefixes_per_game:
        return list(range(num_states))
    raw = np.linspace(0, num_states - 1, prefixes_per_game)
    indices = sorted({int(round(value)) for value in raw})
    if len(indices) < prefixes_per_game:
        for idx in range(num_states):
            if idx not in indices:
                indices.append(idx)
            if len(indices) == prefixes_per_game:
                break
    return sorted(indices[:prefixes_per_game])


def generate_single(model: LocalChatModel, prompt: str, max_new_tokens: int) -> str:
    return model.generate_batch([prompt], batch_size=1, max_new_tokens=max_new_tokens)[0].text


def rollout_base_prefix_candidates(
    model: LocalChatModel,
    gamefiles,
    split: str,
    prefixes_per_game: int,
    max_episode_steps: int,
    action_max_tokens: int,
):
    candidates = []
    for gamefile in gamefiles:
        env = make_single_game_env(gamefile, max_episode_steps=max_episode_steps)
        observation, admissible_commands, _, _ = unwrap_reset(env.reset())
        task_desc = load_task_description(gamefile)
        task_type = load_task_type(gamefile)
        history_actions = []
        history_observations = []
        states = []
        done = False
        step_index = 0
        while not done and step_index < max_episode_steps:
            prefix = AlfworldPrefix(
                example_id=f"{split}:{Path(gamefile).parent.name}:{step_index}",
                split=split,
                gamefile=gamefile,
                task_desc=task_desc,
                task_type=task_type,
                history_actions=list(history_actions),
                history_observations=list(history_observations),
                observation=observation,
                admissible_commands=list(admissible_commands),
                step_index=step_index,
            )
            prompt = build_continue_prompt(
                task_desc,
                observation,
                history_to_text(history_actions, history_observations),
                admissible_commands,
            )
            continue_text = generate_single(model, prompt, max_new_tokens=action_max_tokens)
            continue_action = extract_action(continue_text, admissible_commands)
            states.append(
                {
                    "prefix": prefix,
                    "continue_text": continue_text,
                    "continue_action": continue_action,
                }
            )
            observation, _, done, _, admissible_commands, _ = unwrap_step(env.step([continue_action]))
            history_actions.append(continue_action)
            history_observations.append(observation)
            step_index += 1

        for state_index in select_prefix_indices(len(states), prefixes_per_game):
            candidates.append(states[state_index])
    return candidates


def replay_to_prefix(prefix: AlfworldPrefix, max_episode_steps: int, include_policy_commands: bool = False):
    env = make_single_game_env(
        prefix.gamefile,
        max_episode_steps=max_episode_steps,
        include_policy_commands=include_policy_commands,
    )
    observation, admissible_commands, won, infos = unwrap_reset(env.reset())
    if prefix.history_actions:
        for action in prefix.history_actions:
            observation, _, done, won, admissible_commands, infos = unwrap_step(env.step([action]))
            if done:
                break
    else:
        done = False
    return env, observation, admissible_commands, done, won, infos


def rollout_branch(
    model: LocalChatModel,
    prefix: AlfworldPrefix,
    first_action: str,
    max_episode_steps: int,
    action_max_tokens: int,
):
    env, observation, admissible_commands, done, won, _ = replay_to_prefix(
        prefix,
        max_episode_steps=max_episode_steps,
    )
    replay_matches_prefix = normalize_text(observation) == normalize_text(prefix.observation)
    history_actions = list(prefix.history_actions)
    history_observations = list(prefix.history_observations)

    if done or won:
        return {
            "won": bool(won),
            "branch_steps": 0,
            "final_observation": observation,
            "replay_matches_prefix": replay_matches_prefix,
        }

    observation, _, done, won, admissible_commands, _ = unwrap_step(env.step([first_action]))
    history_actions.append(first_action)
    history_observations.append(observation)
    branch_steps = 1

    while not done and not won:
        prompt = build_continue_prompt(
            prefix.task_desc,
            observation,
            history_to_text(history_actions, history_observations),
            admissible_commands,
        )
        continue_text = generate_single(model, prompt, max_new_tokens=action_max_tokens)
        next_action = extract_action(continue_text, admissible_commands)
        observation, _, done, won, admissible_commands, _ = unwrap_step(env.step([next_action]))
        history_actions.append(next_action)
        history_observations.append(observation)
        branch_steps += 1

    return {
        "won": bool(won),
        "branch_steps": branch_steps,
        "final_observation": observation,
        "replay_matches_prefix": replay_matches_prefix,
    }


def rollout_self_replan_branch(
    model: LocalChatModel,
    prefix: AlfworldPrefix,
    proposed_action: str,
    max_episode_steps: int,
    action_max_tokens: int,
    repair_horizon: int,
    plan_guided: bool = False,
):
    env, observation, admissible_commands, done, won, _ = replay_to_prefix(
        prefix,
        max_episode_steps=max_episode_steps,
    )
    replay_matches_prefix = normalize_text(observation) == normalize_text(prefix.observation)
    history_actions = list(prefix.history_actions)
    history_observations = list(prefix.history_observations)
    branch_steps = 0
    first_action = ""
    generated_texts = []
    repair_plan = ""

    if done or won:
        return {
            "action": first_action,
            "text": "",
            "won": bool(won),
            "branch_steps": 0,
            "final_observation": observation,
            "replay_matches_prefix": replay_matches_prefix,
            "tool_success": False,
        }

    while not done and not won:
        if branch_steps < repair_horizon:
            if plan_guided and branch_steps == 0:
                prompt = build_plan_repair_prompt(
                    prefix.task_desc,
                    observation,
                    history_to_text(history_actions, history_observations),
                    proposed_action,
                    admissible_commands,
                    repair_horizon=repair_horizon,
                )
                action_text = generate_single(
                    model,
                    prompt,
                    max_new_tokens=max(action_max_tokens, 48),
                )
                repair_plan = extract_plan(action_text)
            elif plan_guided:
                prompt = build_followup_replan_prompt(
                    prefix.task_desc,
                    observation,
                    history_to_text(history_actions, history_observations),
                    repair_plan=repair_plan,
                    admissible_commands=admissible_commands,
                    remaining_repair_steps=repair_horizon - branch_steps,
                )
                action_text = generate_single(model, prompt, max_new_tokens=action_max_tokens)
            else:
                prompt = build_replan_prompt(
                    prefix.task_desc,
                    observation,
                    history_to_text(history_actions, history_observations),
                    proposed_action if branch_steps == 0 else "",
                    admissible_commands,
                    remaining_repair_steps=repair_horizon - branch_steps,
                )
                action_text = generate_single(model, prompt, max_new_tokens=action_max_tokens)
        else:
            prompt = build_continue_prompt(
                prefix.task_desc,
                observation,
                history_to_text(history_actions, history_observations),
                admissible_commands,
            )
            action_text = generate_single(model, prompt, max_new_tokens=action_max_tokens)

        next_action = extract_action(action_text, admissible_commands)
        if branch_steps == 0:
            first_action = next_action
        generated_texts.append(action_text)
        observation, _, done, won, admissible_commands, _ = unwrap_step(env.step([next_action]))
        history_actions.append(next_action)
        history_observations.append(observation)
        branch_steps += 1

    return {
        "action": first_action,
        "text": "\n---REPLAN-STEP---\n".join(generated_texts),
        "won": bool(won),
        "branch_steps": branch_steps,
        "final_observation": observation,
        "replay_matches_prefix": replay_matches_prefix,
        "tool_success": bool(first_action),
    }


def rollout_expert_branch(
    prefix: AlfworldPrefix,
    max_episode_steps: int,
    expert_max_policy_steps: int,
    expert_noise_rate: float,
    random_seed: int,
):
    env, observation, admissible_commands, done, won, infos = replay_to_prefix(
        prefix,
        max_episode_steps=max_episode_steps,
        include_policy_commands=True,
    )
    replay_matches_prefix = normalize_text(observation) == normalize_text(prefix.observation)
    policy_commands = infos.get("policy_commands", [[]])[0] if "policy_commands" in infos else []
    first_action = policy_commands[0] if policy_commands else ""
    branch_steps = 0
    expert_steps = 0
    rng = random.Random(random_seed)

    while not done and not won and policy_commands:
        next_action = policy_commands[0]
        if expert_noise_rate > 0.0 and rng.random() < expert_noise_rate:
            distractors = [command for command in admissible_commands if command not in policy_commands]
            if distractors:
                next_action = rng.choice(distractors)
        observation, _, done, won, admissible_commands, infos = unwrap_step(env.step([next_action]))
        branch_steps += 1
        expert_steps += 1
        if expert_max_policy_steps >= 0 and expert_steps >= expert_max_policy_steps:
            break
        policy_commands = infos.get("policy_commands", [[]])[0] if "policy_commands" in infos else []

    return {
        "action": first_action,
        "won": bool(won),
        "branch_steps": branch_steps,
        "final_observation": observation,
        "replay_matches_prefix": replay_matches_prefix,
        "tool_success": bool(first_action),
        "expert_steps": expert_steps,
    }


def build_rows(
    model: LocalChatModel,
    candidates,
    batch_size: int,
    confidence_max_tokens: int,
    verify_max_tokens: int,
    action_max_tokens: int,
    verify_cost: float,
    wrong_penalty: float,
    step_cost: float,
    max_episode_steps: int,
    intervention_mode: str,
    repair_horizon: int,
    expert_max_policy_steps: int,
    expert_noise_rate: float,
    seed: int,
):
    confidence_outputs = model.generate_batch(
        [
            build_confidence_prompt(
                candidate["prefix"].task_desc,
                candidate["prefix"].observation,
                history_to_text(candidate["prefix"].history_actions, candidate["prefix"].history_observations),
                candidate["continue_action"],
            )
            for candidate in candidates
        ],
        batch_size=batch_size,
        max_new_tokens=confidence_max_tokens,
        assistant_prefix="CONFIDENCE: ",
        stop=["\n"],
    )
    verify_outputs = None
    if intervention_mode in {"llm_verify", "self_reflect_retry"}:
        verify_outputs = model.generate_batch(
            [
                (
                    build_verify_prompt(
                        candidate["prefix"].task_desc,
                        candidate["prefix"].observation,
                        history_to_text(
                            candidate["prefix"].history_actions,
                            candidate["prefix"].history_observations,
                        ),
                        candidate["continue_action"],
                        candidate["prefix"].admissible_commands,
                    )
                    if intervention_mode == "llm_verify"
                    else build_reflect_retry_prompt(
                        candidate["prefix"].task_desc,
                        candidate["prefix"].observation,
                        history_to_text(
                            candidate["prefix"].history_actions,
                            candidate["prefix"].history_observations,
                        ),
                        candidate["continue_action"],
                        candidate["prefix"].admissible_commands,
                    )
                )
                for candidate in candidates
            ],
            batch_size=batch_size,
            max_new_tokens=verify_max_tokens,
            assistant_prefix=(
                "ACTION: "
                if intervention_mode == "llm_verify"
                else "REFLECTION: "
            ),
            stop=(["\n"] if intervention_mode == "llm_verify" else None),
        )

    rows = []
    replay_match_count = 0
    if verify_outputs is None:
        verify_outputs = [None] * len(candidates)

    for candidate, confidence_output, verify_output in zip(candidates, confidence_outputs, verify_outputs):
        prefix = candidate["prefix"]
        continue_text = candidate["continue_text"]
        continue_action = candidate["continue_action"]
        confidence = extract_confidence(confidence_output.text)
        if intervention_mode in {"self_replan", "self_plan_replan"}:
            replan_result = rollout_self_replan_branch(
                model,
                prefix,
                continue_action,
                max_episode_steps=max_episode_steps,
                action_max_tokens=action_max_tokens,
                repair_horizon=repair_horizon,
                plan_guided=(intervention_mode == "self_plan_replan"),
            )
            verify_text = replan_result["text"]
            verify_action = replan_result["action"] or "replan_unavailable"
            verify_result = replan_result
            verify_tool_success = replan_result["tool_success"]
        elif intervention_mode in {"llm_verify", "self_reflect_retry"}:
            verify_text = verify_output.text
            verify_action = extract_action(verify_text, prefix.admissible_commands)
            verify_result = rollout_branch(
                model,
                prefix,
                verify_action,
                max_episode_steps=max_episode_steps,
                action_max_tokens=action_max_tokens,
            )
            verify_tool_success = True
        else:
            expert_result = rollout_expert_branch(
                prefix,
                max_episode_steps=max_episode_steps,
                expert_max_policy_steps=expert_max_policy_steps,
                expert_noise_rate=expert_noise_rate,
                random_seed=seed + prefix.step_index * 997 + len(prefix.history_actions) * 31,
            )
            verify_action = expert_result["action"] or "defer_unavailable"
            verify_text = f"DEFER_ACTION: {verify_action}"
            verify_result = expert_result
            verify_tool_success = expert_result["tool_success"]

        continue_result = rollout_branch(
            model,
            prefix,
            continue_action,
            max_episode_steps=max_episode_steps,
            action_max_tokens=action_max_tokens,
        )
        replay_match_count += int(continue_result["replay_matches_prefix"])
        task_features = task_one_hot(prefix.task_type)
        rows.append(
            {
                "example_id": prefix.example_id,
                "split": prefix.split,
                "gamefile": prefix.gamefile,
                "task_type": prefix.task_type,
                "task_desc": prefix.task_desc,
                "step_index": prefix.step_index,
                "observation": prefix.observation,
                "history_text": history_to_text(prefix.history_actions, prefix.history_observations),
                "admissible_commands": prefix.admissible_commands,
                "continue_text": continue_text,
                "continue_answer": continue_action,
                "continue_correct": continue_result["won"],
                "continue_branch_steps": continue_result["branch_steps"],
                "confidence": confidence,
                "confidence_text": confidence_output.text,
                "verify_text": verify_text,
                "verify_answer": verify_action,
                "verify_tool_success": verify_tool_success,
                "verify_correct": verify_result["won"],
                "verify_branch_steps": verify_result["branch_steps"],
                "u_continue": utility(
                    continue_result["won"],
                    continue_result["branch_steps"],
                    verify=False,
                    verify_cost=verify_cost,
                    wrong_penalty=wrong_penalty,
                    step_cost=step_cost,
                ),
                "u_verify": utility(
                    verify_result["won"],
                    verify_result["branch_steps"],
                    verify=True,
                    verify_cost=verify_cost,
                    wrong_penalty=wrong_penalty,
                    step_cost=step_cost,
                ),
                "u_quit": 0.0,
                "features": basic_feature_vector(
                    prefix.task_desc,
                    prefix.observation,
                    prefix.admissible_commands,
                    continue_action,
                    prefix.step_index,
                    len(prefix.history_actions),
                )
                + task_features,
                "verify_features": verify_feature_vector(
                    prefix.observation,
                    prefix.admissible_commands,
                    continue_action,
                    verify_action,
                    prefix.step_index,
                )
                + task_features,
                "replay_matches_prefix": continue_result["replay_matches_prefix"],
            }
        )
    return rows, replay_match_count


def save_rows(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "example_id",
                "split",
                "gamefile",
                "task_type",
                "task_desc",
                "step_index",
                "continue_answer",
                "continue_correct",
                "continue_branch_steps",
                "confidence",
                "verify_answer",
                "verify_tool_success",
                "verify_correct",
                "verify_branch_steps",
                "u_continue",
                "u_verify",
                "u_quit",
                "observation",
                "history_text",
                "continue_text",
                "confidence_text",
                "verify_text",
                "features",
                "verify_features",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["example_id"],
                    row["split"],
                    row["gamefile"],
                    row["task_type"],
                    row["task_desc"],
                    row["step_index"],
                    row["continue_answer"],
                    row["continue_correct"],
                    row["continue_branch_steps"],
                    row["confidence"],
                    row["verify_answer"],
                    row["verify_tool_success"],
                    row["verify_correct"],
                    row["verify_branch_steps"],
                    row["u_continue"],
                    row["u_verify"],
                    row["u_quit"],
                    row["observation"],
                    row["history_text"],
                    row["continue_text"],
                    row["confidence_text"],
                    row["verify_text"],
                    json.dumps(row["features"]),
                    json.dumps(row["verify_features"]),
                ]
            )


def save_policy_decisions(
    rows,
    path: Path,
    failure_scores,
    continue_scores,
    verify_value_scores,
    continue_correct_scores,
    continue_correct_uncertainties,
    verify_correct_scores,
    verify_correct_uncertainties,
    continue_expected_utilities,
    verify_expected_utilities,
    continue_lcb_expected_utilities,
    verify_lcb_expected_utilities,
    two_stage_intervene_scores,
    two_stage_verify_scores,
    failure_actions,
    value_actions,
    explicit_eu_actions,
    explicit_eu_lcb_actions,
    two_stage_actions,
    oracle_actions,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "example_id",
                "task_type",
                "step_index",
                "confidence",
                "continue_action",
                "continue_correct",
                "continue_branch_steps",
                "verify_action",
                "verify_correct",
                "verify_branch_steps",
                "u_continue",
                "u_verify",
                "u_quit",
                "failure_score",
                "continue_value_score",
                "verify_value_score",
                "continue_correct_score",
                "continue_correct_uncertainty",
                "verify_correct_score",
                "verify_correct_uncertainty",
                "continue_expected_utility",
                "verify_expected_utility",
                "continue_lcb_expected_utility",
                "verify_lcb_expected_utility",
                "two_stage_intervene_score",
                "two_stage_verify_score",
                "failure_action",
                "value_action",
                "explicit_eu_action",
                "explicit_eu_lcb_action",
                "two_stage_action",
                "oracle_action",
                "task_desc",
                "observation",
                "history_text",
                "continue_text",
                "verify_text",
            ]
        )
        for (
            row,
            failure_score,
            continue_score,
            verify_value_score,
            continue_correct_score,
            continue_correct_uncertainty,
            verify_correct_score,
            verify_correct_uncertainty,
            continue_expected_utility,
            verify_expected_utility,
            continue_lcb_expected_utility,
            verify_lcb_expected_utility,
            intervene_score,
            selector_score,
            failure_action,
            value_action,
            explicit_eu_action,
            explicit_eu_lcb_action,
            two_stage_action,
            oracle_action,
        ) in zip(
            rows,
            failure_scores,
            continue_scores,
            verify_value_scores,
            continue_correct_scores,
            continue_correct_uncertainties,
            verify_correct_scores,
            verify_correct_uncertainties,
            continue_expected_utilities,
            verify_expected_utilities,
            continue_lcb_expected_utilities,
            verify_lcb_expected_utilities,
            two_stage_intervene_scores,
            two_stage_verify_scores,
            failure_actions,
            value_actions,
            explicit_eu_actions,
            explicit_eu_lcb_actions,
            two_stage_actions,
            oracle_actions,
        ):
            writer.writerow(
                [
                    row["example_id"],
                    row["task_type"],
                    row["step_index"],
                    row["confidence"],
                    row["continue_answer"],
                    row["continue_correct"],
                    row["continue_branch_steps"],
                    row["verify_answer"],
                    row["verify_correct"],
                    row["verify_branch_steps"],
                    row["u_continue"],
                    row["u_verify"],
                    row["u_quit"],
                    float(failure_score),
                    float(continue_score),
                    float(verify_value_score),
                    float(continue_correct_score),
                    float(continue_correct_uncertainty),
                    float(verify_correct_score),
                    float(verify_correct_uncertainty),
                    float(continue_expected_utility),
                    float(verify_expected_utility),
                    float(continue_lcb_expected_utility),
                    float(verify_lcb_expected_utility),
                    float(intervene_score),
                    float(selector_score),
                    failure_action,
                    value_action,
                    explicit_eu_action,
                    explicit_eu_lcb_action,
                    two_stage_action,
                    oracle_action,
                    row["task_desc"],
                    row["observation"],
                    row["history_text"],
                    row["continue_text"],
                    row["verify_text"],
                ]
            )


def main():
    args = parse_args()
    model = LocalChatModel(
        args.model_path,
        api_base=args.api_base,
        api_model=args.api_model,
        api_timeout=args.api_timeout,
    )

    split_gamefiles = {
        "train": collect_gamefiles("train", args.task_types, args.train_games, args.seed),
        "val": collect_gamefiles("valid_seen", args.task_types, args.val_games, args.seed + 1),
        "test": collect_gamefiles("valid_unseen", args.task_types, args.test_games, args.seed + 2),
    }
    candidates = {
        split: rollout_base_prefix_candidates(
            model,
            gamefiles,
            split=split,
            prefixes_per_game=args.prefixes_per_game,
            max_episode_steps=args.max_episode_steps,
            action_max_tokens=args.action_max_tokens,
        )
        for split, gamefiles in split_gamefiles.items()
    }
    train_rows, train_replay_matches = build_rows(
        model,
        candidates["train"],
        batch_size=args.batch_size,
        confidence_max_tokens=args.confidence_max_tokens,
        verify_max_tokens=args.verify_max_tokens,
        action_max_tokens=args.action_max_tokens,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
        step_cost=args.step_cost,
        max_episode_steps=args.max_episode_steps,
        intervention_mode=args.intervention_mode,
        repair_horizon=args.repair_horizon,
        expert_max_policy_steps=args.expert_max_policy_steps,
        expert_noise_rate=args.expert_noise_rate,
        seed=args.seed,
    )
    val_rows, val_replay_matches = build_rows(
        model,
        candidates["val"],
        batch_size=args.batch_size,
        confidence_max_tokens=args.confidence_max_tokens,
        verify_max_tokens=args.verify_max_tokens,
        action_max_tokens=args.action_max_tokens,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
        step_cost=args.step_cost,
        max_episode_steps=args.max_episode_steps,
        intervention_mode=args.intervention_mode,
        repair_horizon=args.repair_horizon,
        expert_max_policy_steps=args.expert_max_policy_steps,
        expert_noise_rate=args.expert_noise_rate,
        seed=args.seed + 1000,
    )
    test_rows, test_replay_matches = build_rows(
        model,
        candidates["test"],
        batch_size=args.batch_size,
        confidence_max_tokens=args.confidence_max_tokens,
        verify_max_tokens=args.verify_max_tokens,
        action_max_tokens=args.action_max_tokens,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
        step_cost=args.step_cost,
        max_episode_steps=args.max_episode_steps,
        intervention_mode=args.intervention_mode,
        repair_horizon=args.repair_horizon,
        expert_max_policy_steps=args.expert_max_policy_steps,
        expert_noise_rate=args.expert_noise_rate,
        seed=args.seed + 2000,
    )

    failure_model = fit_failure_baseline(train_rows)
    continue_correct_model, verify_correct_model = fit_correctness_models(
        train_rows,
        verify_mode=args.verify_mode,
    )
    continue_model, verify_model = fit_value_models(
        train_rows,
        verify_mode=args.verify_mode,
    )
    two_stage_intervene_model, two_stage_verify_selector_model = fit_two_stage_models(
        train_rows,
        verify_mode=args.verify_mode,
    )

    best_failure = threshold_search_failure(failure_model, val_rows)
    best_value = threshold_search_value(
        continue_model,
        verify_model,
        val_rows,
        verify_mode=args.verify_mode,
    )
    best_two_stage = threshold_search_two_stage(
        two_stage_intervene_model,
        two_stage_verify_selector_model,
        val_rows,
        verify_mode=args.verify_mode,
    )
    x_val_base = feature_matrix(val_rows)
    x_val_verify = feature_matrix(val_rows, verify_branch=True, verify_mode=args.verify_mode)
    continue_correct_scores_val, continue_correct_uncertainties_val = positive_probability_stats(
        continue_correct_model,
        x_val_base,
    )
    verify_correct_scores_val, verify_correct_uncertainties_val = positive_probability_stats(
        verify_correct_model,
        x_val_verify,
    )
    explicit_eu_val_metrics = evaluate_expected_utility_policy(
        val_rows,
        continue_correct_scores_val,
        verify_correct_scores_val,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )
    best_explicit_eu_lcb = threshold_search_expected_utility_lcb(
        val_rows,
        continue_correct_scores_val,
        continue_correct_uncertainties_val,
        verify_correct_scores_val,
        verify_correct_uncertainties_val,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )

    x_test_base = feature_matrix(test_rows)
    x_test_verify = feature_matrix(test_rows, verify_branch=True, verify_mode=args.verify_mode)
    failure_scores_test = failure_model.predict_proba(x_test_base)[:, 1]
    continue_correct_scores_test, continue_correct_uncertainties_test = positive_probability_stats(
        continue_correct_model,
        x_test_base,
    )
    verify_correct_scores_test, verify_correct_uncertainties_test = positive_probability_stats(
        verify_correct_model,
        x_test_verify,
    )
    continue_scores_test = continue_model.predict(x_test_base)
    verify_scores_test = verify_model.predict(x_test_verify)
    two_stage_intervene_scores_test = two_stage_intervene_model.predict_proba(x_test_base)[:, 1]
    two_stage_verify_scores_test = two_stage_verify_selector_model.predict_proba(x_test_verify)[:, 1]
    continue_expected_utilities_test = expected_utility_from_probability(
        continue_correct_scores_test,
        verify=False,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )
    verify_expected_utilities_test = expected_utility_from_probability(
        verify_correct_scores_test,
        verify=True,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )
    continue_lcb_expected_utilities_test = expected_utility_from_probability(
        np.clip(
            continue_correct_scores_test
            - best_explicit_eu_lcb["beta_continue"] * continue_correct_uncertainties_test,
            0.0,
            1.0,
        ),
        verify=False,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )
    verify_lcb_expected_utilities_test = expected_utility_from_probability(
        np.clip(
            verify_correct_scores_test - best_explicit_eu_lcb["beta_verify"] * verify_correct_uncertainties_test,
            0.0,
            1.0,
        ),
        verify=True,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )

    base_metrics = evaluate_actions(test_rows, ["continue"] * len(test_rows))
    failure_metrics = evaluate_failure_policy(
        test_rows,
        failure_scores_test,
        best_failure["verify_threshold"],
        best_failure["quit_threshold"],
    )
    value_metrics = evaluate_value_policy(
        test_rows,
        continue_scores_test,
        verify_scores_test,
        best_value["verify_margin"],
        best_value["quit_floor"],
    )
    explicit_eu_metrics = evaluate_expected_utility_policy(
        test_rows,
        continue_correct_scores_test,
        verify_correct_scores_test,
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )
    explicit_eu_lcb_metrics = evaluate_expected_utility_lcb_policy(
        test_rows,
        continue_correct_scores_test,
        continue_correct_uncertainties_test,
        verify_correct_scores_test,
        verify_correct_uncertainties_test,
        beta_continue=best_explicit_eu_lcb["beta_continue"],
        beta_verify=best_explicit_eu_lcb["beta_verify"],
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )
    two_stage_metrics = evaluate_two_stage_policy(
        test_rows,
        two_stage_intervene_scores_test,
        two_stage_verify_scores_test,
        best_two_stage["intervene_threshold"],
        best_two_stage["verify_threshold"],
    )
    always_verify_metrics = evaluate_actions(test_rows, ["verify"] * len(test_rows))
    always_quit_metrics = evaluate_actions(test_rows, ["quit"] * len(test_rows))
    failure_forecast_metrics = safe_binary_metrics(
        [0 if row["continue_correct"] else 1 for row in test_rows],
        failure_scores_test,
    )
    failure_actions = [
        "quit"
        if score >= best_failure["quit_threshold"]
        else "verify"
        if score >= best_failure["verify_threshold"]
        else "continue"
        for score in failure_scores_test
    ]
    value_actions = [
        "quit"
        if max(continue_score, verify_score) < best_value["quit_floor"]
        else "verify"
        if verify_score - continue_score >= best_value["verify_margin"]
        else "continue"
        for continue_score, verify_score in zip(continue_scores_test, verify_scores_test)
    ]
    explicit_eu_actions = choose_expected_utility_actions(
        continue_expected_utilities_test,
        verify_expected_utilities_test,
    )
    explicit_eu_lcb_actions = choose_expected_utility_lcb_actions(
        continue_correct_scores_test,
        continue_correct_uncertainties_test,
        verify_correct_scores_test,
        verify_correct_uncertainties_test,
        beta_continue=best_explicit_eu_lcb["beta_continue"],
        beta_verify=best_explicit_eu_lcb["beta_verify"],
        verify_cost=args.verify_cost,
        wrong_penalty=args.wrong_penalty,
    )
    two_stage_actions = choose_two_stage_actions(
        two_stage_intervene_scores_test,
        two_stage_verify_scores_test,
        best_two_stage["intervene_threshold"],
        best_two_stage["verify_threshold"],
    )
    oracle_actions = [
        "continue"
        if row["u_continue"] >= row["u_verify"] and row["u_continue"] >= row["u_quit"]
        else "verify"
        if row["u_verify"] >= row["u_quit"]
        else "quit"
        for row in test_rows
    ]
    oracle_metrics = evaluate_actions(test_rows, oracle_actions)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows(train_rows, output_dir / "train_rows.csv")
    save_rows(val_rows, output_dir / "val_rows.csv")
    save_rows(test_rows, output_dir / "test_rows.csv")
    save_policy_decisions(
        test_rows,
        output_dir / "test_policy_decisions.csv",
        failure_scores_test,
        continue_scores_test,
        verify_scores_test,
        continue_correct_scores_test,
        continue_correct_uncertainties_test,
        verify_correct_scores_test,
        verify_correct_uncertainties_test,
        continue_expected_utilities_test,
        verify_expected_utilities_test,
        continue_lcb_expected_utilities_test,
        verify_lcb_expected_utilities_test,
        two_stage_intervene_scores_test,
        two_stage_verify_scores_test,
        failure_actions,
        value_actions,
        explicit_eu_actions,
        explicit_eu_lcb_actions,
        two_stage_actions,
        oracle_actions,
    )

    summary = {
        "config": {
            "model_path": args.model_path,
            "api_base": args.api_base,
            "api_model": args.api_model,
            "task_types": args.task_types,
            "train_games": args.train_games,
            "val_games": args.val_games,
            "test_games": args.test_games,
            "prefixes_per_game": args.prefixes_per_game,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "verify_cost": args.verify_cost,
            "wrong_penalty": args.wrong_penalty,
            "step_cost": args.step_cost,
            "intervention_mode": args.intervention_mode,
            "repair_horizon": args.repair_horizon,
            "verify_mode": args.verify_mode,
            "expert_max_policy_steps": args.expert_max_policy_steps,
            "expert_noise_rate": args.expert_noise_rate,
        },
        "dataset": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_replay_match_rate": train_replay_matches / max(1, len(train_rows)),
            "val_replay_match_rate": val_replay_matches / max(1, len(val_rows)),
            "test_replay_match_rate": test_replay_matches / max(1, len(test_rows)),
            "test_intervention_better_than_continue": int(sum(1 for row in test_rows if row["u_verify"] > row["u_continue"])),
            "test_quit_better_than_continue": int(sum(1 for row in test_rows if row["u_quit"] > row["u_continue"])),
            "test_oracle_intervention_rate": float(np.mean([1.0 if action == "verify" else 0.0 for action in oracle_actions])),
            "test_oracle_quit_rate": float(np.mean([1.0 if action == "quit" else 0.0 for action in oracle_actions])),
        },
        "offline_metrics": {
            "failure_forecast": failure_forecast_metrics,
        },
        "best_controls": {
            "failure": best_failure,
            "value": best_value,
            "explicit_eu": explicit_eu_val_metrics,
            "explicit_eu_lcb": best_explicit_eu_lcb,
            "two_stage": best_two_stage,
        },
        "policies": {
            "base": base_metrics,
            "always_verify_policy": always_verify_metrics,
            "always_quit_policy": always_quit_metrics,
            "failure_policy": failure_metrics,
            "value_policy": value_metrics,
            "explicit_eu_policy": explicit_eu_metrics,
            "explicit_eu_lcb_policy": explicit_eu_lcb_metrics,
            "two_stage_policy": two_stage_metrics,
            "oracle": oracle_metrics,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
