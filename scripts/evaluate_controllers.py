import argparse
import csv
import json
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate failure-triggered routing and the action-conditioned controller "
            "family on saved branched rows (train/val/test CSVs per run directory)."
        )
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def parse_bool(value) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def load_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row = {
                "example_id": raw.get("example_id", ""),
                "confidence": float(raw["confidence"]),
                "continue_correct": parse_bool(raw["continue_correct"]),
                "verify_correct": parse_bool(raw["verify_correct"]),
                "u_continue": float(raw["u_continue"]),
                "u_verify": float(raw["u_verify"]),
                "u_quit": float(raw["u_quit"]),
                "features": json.loads(raw["features"]),
                "verify_features": json.loads(raw["verify_features"]),
            }
            rows.append(row)
    return rows


def load_config(run_dir: Path):
    summary_path = run_dir / "summary.json"
    replayed_summary_path = run_dir / "replayed_policy_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f).get("config", {})
    if replayed_summary_path.exists():
        with open(replayed_summary_path, "r", encoding="utf-8") as f:
            return json.load(f).get("config", {})
    return {}


def infer_costs(config: dict):
    verify_cost = float(config.get("verify_cost", config.get("defer_cost", 0.05)))
    wrong_penalty = float(config.get("wrong_penalty", 1.0))
    return verify_cost, wrong_penalty


def family_from_best_controls(best_controls: dict):
    utilities = {
        "value": best_controls["value"]["utility"],
        "explicit_eu": best_controls["explicit_eu"]["utility"],
        "explicit_eu_lcb": best_controls["explicit_eu_lcb"]["utility"],
        "two_stage": best_controls["two_stage"]["utility"],
    }
    return max(utilities, key=utilities.get)


def evaluate_action_conditioned_mode(train_rows, val_rows, test_rows, verify_cost, wrong_penalty, verify_mode: str):
    continue_correct_model, verify_correct_model = fit_correctness_models(
        train_rows,
        verify_mode=verify_mode,
    )
    continue_model, verify_model = fit_value_models(
        train_rows,
        verify_mode=verify_mode,
    )
    two_stage_intervene_model, two_stage_verify_selector_model = fit_two_stage_models(
        train_rows,
        verify_mode=verify_mode,
    )

    best_value = threshold_search_value(
        continue_model,
        verify_model,
        val_rows,
        verify_mode=verify_mode,
    )
    best_two_stage = threshold_search_two_stage(
        two_stage_intervene_model,
        two_stage_verify_selector_model,
        val_rows,
        verify_mode=verify_mode,
    )

    x_val_base = feature_matrix(val_rows)
    x_val_verify = feature_matrix(val_rows, verify_branch=True, verify_mode=verify_mode)
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
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
    )
    best_explicit_eu_lcb = threshold_search_expected_utility_lcb(
        val_rows,
        continue_correct_scores_val,
        continue_correct_uncertainties_val,
        verify_correct_scores_val,
        verify_correct_uncertainties_val,
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
    )

    x_test_base = feature_matrix(test_rows)
    x_test_verify = feature_matrix(test_rows, verify_branch=True, verify_mode=verify_mode)
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
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
    )
    verify_expected_utilities_test = expected_utility_from_probability(
        verify_correct_scores_test,
        verify=True,
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
    )

    policies = {
        "value": evaluate_value_policy(
            test_rows,
            continue_scores_test,
            verify_scores_test,
            best_value["verify_margin"],
            best_value["quit_floor"],
        ),
        "explicit_eu": evaluate_expected_utility_policy(
            test_rows,
            continue_correct_scores_test,
            verify_correct_scores_test,
            verify_cost=verify_cost,
            wrong_penalty=wrong_penalty,
        ),
        "explicit_eu_lcb": evaluate_expected_utility_lcb_policy(
            test_rows,
            continue_correct_scores_test,
            continue_correct_uncertainties_test,
            verify_correct_scores_test,
            verify_correct_uncertainties_test,
            beta_continue=best_explicit_eu_lcb["beta_continue"],
            beta_verify=best_explicit_eu_lcb["beta_verify"],
            verify_cost=verify_cost,
            wrong_penalty=wrong_penalty,
        ),
        "two_stage": evaluate_two_stage_policy(
            test_rows,
            two_stage_intervene_scores_test,
            two_stage_verify_scores_test,
            best_two_stage["intervene_threshold"],
            best_two_stage["verify_threshold"],
        ),
    }

    best_controls = {
        "value": best_value,
        "explicit_eu": explicit_eu_val_metrics,
        "explicit_eu_lcb": best_explicit_eu_lcb,
        "two_stage": best_two_stage,
    }
    val_selected_family = family_from_best_controls(best_controls)
    test_best_family = min(policies, key=lambda family: policies[family]["control_regret"])

    return {
        "verify_mode": verify_mode,
        "best_controls": best_controls,
        "policies": policies,
        "val_selected_family": val_selected_family,
        "val_selected_policy": policies[val_selected_family],
        "test_best_family": test_best_family,
        "test_best_policy": policies[test_best_family],
    }


def analyze_run_dir(run_dir: Path):
    train_rows = load_rows(run_dir / "train_rows.csv")
    val_rows = load_rows(run_dir / "val_rows.csv")
    test_rows = load_rows(run_dir / "test_rows.csv")
    config = load_config(run_dir)
    verify_cost, wrong_penalty = infer_costs(config)

    failure_model = fit_failure_baseline(train_rows)
    best_failure = threshold_search_failure(failure_model, val_rows)
    x_test_base = feature_matrix(test_rows)
    failure_scores_test = failure_model.predict_proba(x_test_base)[:, 1]
    failure_policy = evaluate_failure_policy(
        test_rows,
        failure_scores_test,
        best_failure["verify_threshold"],
        best_failure["quit_threshold"],
    )
    base_policy = evaluate_actions(test_rows, ["continue"] * len(test_rows))
    oracle_actions = [
        "continue"
        if row["u_continue"] >= row["u_verify"] and row["u_continue"] >= row["u_quit"]
        else "verify"
        if row["u_verify"] >= row["u_quit"]
        else "quit"
        for row in test_rows
    ]
    oracle_policy = evaluate_actions(test_rows, oracle_actions)

    no_preview = evaluate_action_conditioned_mode(
        train_rows,
        val_rows,
        test_rows,
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
        verify_mode="no_preview",
    )
    with_preview = evaluate_action_conditioned_mode(
        train_rows,
        val_rows,
        test_rows,
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
        verify_mode="with_preview",
    )

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "config": {
            "verify_cost": verify_cost,
            "wrong_penalty": wrong_penalty,
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "test_size": len(test_rows),
        },
        "prefix_only_baselines": {
            "base": base_policy,
            "failure": failure_policy,
            "oracle": oracle_policy,
            "best_failure": best_failure,
        },
        "modes": {
            "no_preview": no_preview,
            "with_preview": with_preview,
        },
        "key_comparisons": {
            "failure_regret": failure_policy["control_regret"],
            "no_preview_val_selected_family": no_preview["val_selected_family"],
            "no_preview_val_selected_regret": no_preview["val_selected_policy"]["control_regret"],
            "with_preview_val_selected_family": with_preview["val_selected_family"],
            "with_preview_val_selected_regret": with_preview["val_selected_policy"]["control_regret"],
            "failure_minus_no_preview": (
                failure_policy["control_regret"] - no_preview["val_selected_policy"]["control_regret"]
            ),
            "failure_minus_with_preview": (
                failure_policy["control_regret"] - with_preview["val_selected_policy"]["control_regret"]
            ),
            "with_preview_minus_no_preview": (
                with_preview["val_selected_policy"]["control_regret"]
                - no_preview["val_selected_policy"]["control_regret"]
            ),
        },
    }


def main():
    args = parse_args()
    results = [analyze_run_dir(run_dir) for run_dir in args.run_dirs]
    payload = {"results": results}

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
