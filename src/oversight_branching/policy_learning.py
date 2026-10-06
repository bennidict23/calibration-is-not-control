import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def expected_utility_from_probability(
    correct_probability,
    verify: bool,
    verify_cost: float,
    wrong_penalty: float,
):
    probability = np.asarray(correct_probability, dtype=float)
    action_cost = verify_cost if verify else 0.0
    return probability * (1.0 - action_cost) + (1.0 - probability) * (-wrong_penalty - action_cost)


def base_policy_features(row) -> list:
    return row["features"] + [row["confidence"] / 100.0]


def verify_policy_features(row) -> list:
    return base_policy_features(row) + row["verify_features"]


def _feature_builder(verify_branch: bool, verify_mode: str):
    if not verify_branch:
        return base_policy_features
    if verify_mode == "with_preview":
        return verify_policy_features
    if verify_mode == "no_preview":
        return base_policy_features
    raise ValueError(f"Unknown verify_mode: {verify_mode}")


def feature_matrix(rows, verify_branch: bool = False, verify_mode: str = "with_preview") -> np.ndarray:
    builder = _feature_builder(verify_branch=verify_branch, verify_mode=verify_mode)
    return np.asarray([builder(row) for row in rows], dtype=float)


def fit_binary_classifier(x_train: np.ndarray, y_train: np.ndarray, random_state: int):
    if len(np.unique(y_train)) < 2:
        positive_probability = float(y_train[0])

        class ConstantProbabilityModel:
            def predict_proba(self, x):
                probability = np.full(len(x), positive_probability, dtype=float)
                return np.column_stack([1.0 - probability, probability])

        return ConstantProbabilityModel()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def fit_value_regressor(x_train: np.ndarray, y_train: np.ndarray, random_state: int):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def positive_probability_stats(model, x: np.ndarray):
    probabilities = model.predict_proba(x)[:, 1]
    if not hasattr(model, "estimators_"):
        return probabilities, np.zeros(len(x), dtype=float)

    per_tree = []
    for estimator in model.estimators_:
        tree_probabilities = estimator.predict_proba(x)
        if tree_probabilities.shape[1] == 1:
            tree_positive = np.full(
                len(x),
                1.0 if int(estimator.classes_[0]) == 1 else 0.0,
                dtype=float,
            )
        else:
            positive_index = int(np.where(estimator.classes_ == 1)[0][0])
            tree_positive = tree_probabilities[:, positive_index]
        per_tree.append(tree_positive)

    stacked = np.asarray(per_tree, dtype=float)
    return probabilities, stacked.std(axis=0)


def fit_failure_baseline(train_rows):
    x_train = feature_matrix(train_rows)
    y_train = np.asarray([0 if row["continue_correct"] else 1 for row in train_rows], dtype=int)
    return fit_binary_classifier(x_train, y_train, random_state=0)


def fit_correctness_models_matrix(
    continue_x_train: np.ndarray,
    verify_x_train: np.ndarray,
    continue_y_train: np.ndarray,
    verify_y_train: np.ndarray,
):
    continue_correct_model = fit_binary_classifier(continue_x_train, continue_y_train, random_state=4)
    verify_correct_model = fit_binary_classifier(verify_x_train, verify_y_train, random_state=5)
    return continue_correct_model, verify_correct_model


def fit_correctness_models(train_rows, verify_mode: str = "with_preview"):
    continue_x_train = feature_matrix(train_rows)
    verify_x_train = feature_matrix(train_rows, verify_branch=True, verify_mode=verify_mode)
    continue_y_train = np.asarray([1 if row["continue_correct"] else 0 for row in train_rows], dtype=int)
    verify_y_train = np.asarray([1 if row["verify_correct"] else 0 for row in train_rows], dtype=int)
    return fit_correctness_models_matrix(
        continue_x_train,
        verify_x_train,
        continue_y_train,
        verify_y_train,
    )


def fit_value_models_matrix(
    x_continue_train: np.ndarray,
    x_verify_train: np.ndarray,
    y_continue: np.ndarray,
    y_verify: np.ndarray,
):
    continue_model = fit_value_regressor(x_continue_train, y_continue, random_state=0)
    verify_model = fit_value_regressor(x_verify_train, y_verify, random_state=1)
    return continue_model, verify_model


def fit_value_models(train_rows, verify_mode: str = "with_preview"):
    x_continue_train = feature_matrix(train_rows)
    x_verify_train = feature_matrix(train_rows, verify_branch=True, verify_mode=verify_mode)
    y_continue = np.asarray([row["u_continue"] for row in train_rows], dtype=float)
    y_verify = np.asarray([row["u_verify"] for row in train_rows], dtype=float)
    return fit_value_models_matrix(
        x_continue_train,
        x_verify_train,
        y_continue,
        y_verify,
    )


def fit_two_stage_models_matrix(
    x_intervene_train: np.ndarray,
    y_intervene: np.ndarray,
    x_verify_train: np.ndarray,
    y_verify: np.ndarray,
):
    if len(np.unique(y_intervene)) < 2:
        intervene_constant = float(y_intervene[0])

        class ConstantInterveneModel:
            def predict_proba(self, x):
                probability = np.full(len(x), intervene_constant, dtype=float)
                return np.column_stack([1.0 - probability, probability])

        intervene_model = ConstantInterveneModel()
    else:
        intervene_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=2,
        )
        intervene_model.fit(x_intervene_train, y_intervene)

    if len(y_verify) == 0:
        verify_constant = 1.0

        class ConstantVerifySelector:
            def predict_proba(self, x):
                probability = np.full(len(x), verify_constant, dtype=float)
                return np.column_stack([1.0 - probability, probability])

        verify_selector_model = ConstantVerifySelector()
    elif len(np.unique(y_verify)) < 2:
        verify_constant = float(y_verify[0])

        class ConstantVerifySelector:
            def predict_proba(self, x):
                probability = np.full(len(x), verify_constant, dtype=float)
                return np.column_stack([1.0 - probability, probability])

        verify_selector_model = ConstantVerifySelector()
    else:
        verify_selector_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=3,
        )
        verify_selector_model.fit(x_verify_train, y_verify)

    return intervene_model, verify_selector_model


def fit_two_stage_models(train_rows, verify_mode: str = "with_preview"):
    x_intervene_train = feature_matrix(train_rows)
    y_intervene = np.asarray(
        [1 if max(row["u_verify"], row["u_quit"]) > row["u_continue"] else 0 for row in train_rows],
        dtype=int,
    )
    intervene_indices = [idx for idx, target in enumerate(y_intervene) if target == 1]
    y_verify = np.asarray(
        [1 if train_rows[idx]["u_verify"] > train_rows[idx]["u_quit"] else 0 for idx in intervene_indices],
        dtype=int,
    )
    x_verify_train = feature_matrix(
        [train_rows[idx] for idx in intervene_indices],
        verify_branch=True,
        verify_mode=verify_mode,
    )
    return fit_two_stage_models_matrix(
        x_intervene_train,
        y_intervene,
        x_verify_train,
        y_verify,
    )


def evaluate_failure_policy(rows, scores, verify_threshold, quit_threshold):
    actions = []
    for score in scores:
        if score >= quit_threshold:
            actions.append("quit")
        elif score >= verify_threshold:
            actions.append("verify")
        else:
            actions.append("continue")
    return evaluate_actions(rows, actions)


def threshold_search_failure(model, rows):
    x = feature_matrix(rows)
    scores = model.predict_proba(x)[:, 1]
    return threshold_search_failure_scores(scores, rows)


def threshold_search_failure_scores(scores, rows):
    thresholds = np.quantile(scores, np.linspace(0.0, 0.95, 12))
    best = None
    for verify_threshold in thresholds:
        for quit_threshold in thresholds:
            if quit_threshold < verify_threshold:
                continue
            metrics = evaluate_failure_policy(rows, scores, verify_threshold, quit_threshold)
            record = {
                "verify_threshold": float(verify_threshold),
                "quit_threshold": float(quit_threshold),
                **metrics,
            }
            if best is None or record["utility"] > best["utility"]:
                best = record
    return best


def evaluate_value_policy(rows, continue_scores, verify_scores, verify_margin, quit_floor):
    actions = []
    for continue_score, verify_score in zip(continue_scores, verify_scores):
        if max(continue_score, verify_score) < quit_floor:
            actions.append("quit")
        elif verify_score - continue_score >= verify_margin:
            actions.append("verify")
        else:
            actions.append("continue")
    return evaluate_actions(rows, actions)


def threshold_search_value(continue_model, verify_model, rows, verify_mode: str = "with_preview"):
    x_continue = feature_matrix(rows)
    x_verify = feature_matrix(rows, verify_branch=True, verify_mode=verify_mode)
    continue_scores = continue_model.predict(x_continue)
    verify_scores = verify_model.predict(x_verify)
    return threshold_search_value_scores(continue_scores, verify_scores, rows)


def threshold_search_value_scores(continue_scores, verify_scores, rows):
    margins = verify_scores - continue_scores
    thresholds = np.quantile(margins, np.linspace(0.0, 0.95, 12))
    best = None
    for verify_margin in thresholds:
        for quit_floor in np.linspace(-0.2, 0.4, 13):
            metrics = evaluate_value_policy(rows, continue_scores, verify_scores, verify_margin, quit_floor)
            record = {
                "verify_margin": float(verify_margin),
                "quit_floor": float(quit_floor),
                **metrics,
            }
            if best is None or record["utility"] > best["utility"]:
                best = record
    return best


def choose_expected_utility_actions(continue_expected_utilities, verify_expected_utilities):
    actions = []
    for continue_expected_utility, verify_expected_utility in zip(
        continue_expected_utilities,
        verify_expected_utilities,
    ):
        if continue_expected_utility >= verify_expected_utility and continue_expected_utility >= 0.0:
            actions.append("continue")
        elif verify_expected_utility >= 0.0:
            actions.append("verify")
        else:
            actions.append("quit")
    return actions


def choose_expected_utility_lcb_actions(
    continue_correct_scores,
    continue_correct_uncertainties,
    verify_correct_scores,
    verify_correct_uncertainties,
    beta_continue: float,
    beta_verify: float,
    verify_cost: float,
    wrong_penalty: float,
):
    continue_probability_lcb = np.clip(
        np.asarray(continue_correct_scores, dtype=float)
        - beta_continue * np.asarray(continue_correct_uncertainties, dtype=float),
        0.0,
        1.0,
    )
    verify_probability_lcb = np.clip(
        np.asarray(verify_correct_scores, dtype=float)
        - beta_verify * np.asarray(verify_correct_uncertainties, dtype=float),
        0.0,
        1.0,
    )
    continue_expected_utilities = expected_utility_from_probability(
        continue_probability_lcb,
        verify=False,
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
    )
    verify_expected_utilities = expected_utility_from_probability(
        verify_probability_lcb,
        verify=True,
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
    )
    return choose_expected_utility_actions(continue_expected_utilities, verify_expected_utilities)


def evaluate_expected_utility_policy(
    rows,
    continue_correct_scores,
    verify_correct_scores,
    verify_cost: float,
    wrong_penalty: float,
):
    continue_expected_utilities = expected_utility_from_probability(
        continue_correct_scores,
        verify=False,
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
    )
    verify_expected_utilities = expected_utility_from_probability(
        verify_correct_scores,
        verify=True,
        verify_cost=verify_cost,
        wrong_penalty=wrong_penalty,
    )
    actions = choose_expected_utility_actions(continue_expected_utilities, verify_expected_utilities)
    return evaluate_actions(rows, actions)


def threshold_search_expected_utility_lcb(
    rows,
    continue_correct_scores,
    continue_correct_uncertainties,
    verify_correct_scores,
    verify_correct_uncertainties,
    verify_cost: float,
    wrong_penalty: float,
):
    best = None
    for beta_continue in np.linspace(0.0, 1.0, 11):
        for beta_verify in np.linspace(0.0, 1.5, 16):
            metrics = evaluate_actions(
                rows,
                choose_expected_utility_lcb_actions(
                    continue_correct_scores,
                    continue_correct_uncertainties,
                    verify_correct_scores,
                    verify_correct_uncertainties,
                    float(beta_continue),
                    float(beta_verify),
                    verify_cost=verify_cost,
                    wrong_penalty=wrong_penalty,
                ),
            )
            record = {
                "beta_continue": float(beta_continue),
                "beta_verify": float(beta_verify),
                **metrics,
            }
            if best is None or record["utility"] > best["utility"]:
                best = record
    return best


def evaluate_expected_utility_lcb_policy(
    rows,
    continue_correct_scores,
    continue_correct_uncertainties,
    verify_correct_scores,
    verify_correct_uncertainties,
    beta_continue: float,
    beta_verify: float,
    verify_cost: float,
    wrong_penalty: float,
):
    return evaluate_actions(
        rows,
        choose_expected_utility_lcb_actions(
            continue_correct_scores,
            continue_correct_uncertainties,
            verify_correct_scores,
            verify_correct_uncertainties,
            beta_continue=beta_continue,
            beta_verify=beta_verify,
            verify_cost=verify_cost,
            wrong_penalty=wrong_penalty,
        ),
    )


def choose_two_stage_actions(intervene_scores, verify_selector_scores, intervene_threshold, verify_threshold):
    actions = []
    for intervene_score, verify_score in zip(intervene_scores, verify_selector_scores):
        if intervene_score < intervene_threshold:
            actions.append("continue")
        elif verify_score >= verify_threshold:
            actions.append("verify")
        else:
            actions.append("quit")
    return actions


def threshold_search_two_stage(intervene_model, verify_selector_model, rows, verify_mode: str = "with_preview"):
    x_intervene = feature_matrix(rows)
    x_verify = feature_matrix(rows, verify_branch=True, verify_mode=verify_mode)
    intervene_scores = intervene_model.predict_proba(x_intervene)[:, 1]
    verify_selector_scores = verify_selector_model.predict_proba(x_verify)[:, 1]
    return threshold_search_two_stage_scores(intervene_scores, verify_selector_scores, rows)


def threshold_search_two_stage_scores(intervene_scores, verify_selector_scores, rows):
    intervene_thresholds = np.quantile(intervene_scores, np.linspace(0.0, 0.95, 12))
    verify_thresholds = np.quantile(verify_selector_scores, np.linspace(0.0, 0.95, 12))
    best = None
    for intervene_threshold in intervene_thresholds:
        for verify_threshold in verify_thresholds:
            metrics = evaluate_actions(
                rows,
                choose_two_stage_actions(
                    intervene_scores,
                    verify_selector_scores,
                    intervene_threshold,
                    verify_threshold,
                ),
            )
            record = {
                "intervene_threshold": float(intervene_threshold),
                "verify_threshold": float(verify_threshold),
                **metrics,
            }
            if best is None or record["utility"] > best["utility"]:
                best = record
    return best


def evaluate_two_stage_policy(rows, intervene_scores, verify_selector_scores, intervene_threshold, verify_threshold):
    return evaluate_actions(
        rows,
        choose_two_stage_actions(
            intervene_scores,
            verify_selector_scores,
            intervene_threshold,
            verify_threshold,
        ),
    )


ACTION_INDEX = {"continue": 0, "verify": 1, "quit": 2}
ACTION_NAMES = ["continue", "verify", "quit"]


def fit_cost_sensitive_policy(train_rows, verify_mode: str = "with_preview"):
    """Cost-sensitive direct policy: single joint model that outputs optimal action.

    Unlike the value baseline (two separate regressors + margin thresholding),
    this trains a single classifier whose loss is weighted by the regret of
    choosing the wrong action.  The model directly maps features -> action
    without intermediate value estimation.
    """
    x_train = feature_matrix(train_rows, verify_branch=True, verify_mode=verify_mode)

    y_train = []
    sample_weights = []
    for row in train_rows:
        utilities = [row["u_continue"], row["u_verify"], row["u_quit"]]
        oracle_idx = int(np.argmax(utilities))
        y_train.append(oracle_idx)
        # Weight = regret of second-best action (how critical this decision is)
        sorted_u = sorted(utilities, reverse=True)
        margin = sorted_u[0] - sorted_u[1]
        sample_weights.append(max(margin, 0.01))

    y_train = np.asarray(y_train, dtype=int)
    sample_weights = np.asarray(sample_weights, dtype=float)

    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        constant_action = int(unique_labels[0])

        class _ConstantPolicy:
            classes_ = np.arange(3)

            def predict(self, x):
                return np.full(len(x), constant_action, dtype=int)

            def predict_proba(self, x):
                probs = np.zeros((len(x), 3), dtype=float)
                probs[:, constant_action] = 1.0
                return probs

        return _ConstantPolicy()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=7,
    )
    model.fit(x_train, y_train, sample_weight=sample_weights)
    return model


def oracle_targets_and_weights(rows):
    utilities = np.asarray(
        [
            [row["u_continue"], row["u_verify"], row["u_quit"]]
            for row in rows
        ],
        dtype=float,
    )
    y = np.argmax(utilities, axis=1).astype(int)
    sorted_utilities = np.sort(utilities, axis=1)
    margins = sorted_utilities[:, -1] - sorted_utilities[:, -2]
    sample_weights = np.maximum(margins, 0.01)
    return y, sample_weights


def fit_cost_sensitive_classifier_matrix(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    random_state: int = 7,
):
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        constant_action = int(unique_labels[0])

        class _ConstantPolicy:
            classes_ = np.arange(3)

            def predict(self, x):
                return np.full(len(x), constant_action, dtype=int)

            def predict_proba(self, x):
                probs = np.zeros((len(x), 3), dtype=float)
                probs[:, constant_action] = 1.0
                return probs

        return _ConstantPolicy()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=random_state,
    )
    model.fit(x_train, y_train, sample_weight=sample_weights)
    return model


def fit_cost_sensitive_policy_matrix(
    x_train: np.ndarray,
    rows,
    random_state: int = 7,
):
    y_train, sample_weights = oracle_targets_and_weights(rows)
    return fit_cost_sensitive_classifier_matrix(
        x_train,
        y_train,
        sample_weights,
        random_state=random_state,
    )


def aligned_action_probabilities(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)

    if probs.shape[1] >= len(ACTION_NAMES):
        return probs

    full_probs = np.zeros((len(x), len(ACTION_NAMES)), dtype=float)
    for col_idx, cls in enumerate(model.classes_):
        full_probs[:, cls] = probs[:, col_idx]
    return full_probs


def evaluate_cost_sensitive_policy(rows, model, verify_mode: str = "with_preview"):
    """Evaluate cost-sensitive policy on test data — no thresholds needed."""
    x = feature_matrix(rows, verify_branch=True, verify_mode=verify_mode)
    return evaluate_cost_sensitive_policy_matrix(rows, model, x)


def threshold_search_cost_sensitive(model, rows, verify_mode: str = "with_preview"):
    """Tune an optional quit-confidence threshold on validation data.

    The base policy uses pure argmax from predicted class probabilities.
    This additionally searches for a confidence floor: if the model's
    confidence for the top non-quit action is below the threshold, quit.
    """
    x = feature_matrix(rows, verify_branch=True, verify_mode=verify_mode)
    return threshold_search_cost_sensitive_matrix(model, x, rows)


def evaluate_cost_sensitive_policy_with_threshold(rows, model, quit_confidence_threshold, verify_mode: str = "with_preview"):
    """Evaluate cost-sensitive policy with a quit-confidence threshold."""
    x = feature_matrix(rows, verify_branch=True, verify_mode=verify_mode)
    return evaluate_cost_sensitive_policy_matrix_with_threshold(
        rows,
        model,
        x,
        quit_confidence_threshold=quit_confidence_threshold,
    )


def evaluate_cost_sensitive_policy_matrix(rows, model, x: np.ndarray):
    predictions = model.predict(x)
    actions = [ACTION_NAMES[p] for p in predictions]
    return evaluate_actions(rows, actions)


def threshold_search_cost_sensitive_matrix(model, x: np.ndarray, rows):
    probs = aligned_action_probabilities(model, x)

    base_predictions = np.argmax(probs, axis=1)
    base_actions = [ACTION_NAMES[p] for p in base_predictions]
    base_metrics = evaluate_actions(rows, base_actions)
    best = {"quit_confidence_threshold": 0.0, **base_metrics}

    for confidence_threshold in np.linspace(0.0, 0.8, 17):
        actions = []
        for i in range(len(rows)):
            p_cont, p_ver = probs[i, 0], probs[i, 1]
            if max(p_cont, p_ver) < confidence_threshold:
                actions.append("quit")
            elif p_ver > p_cont:
                actions.append("verify")
            else:
                actions.append("continue")
        metrics = evaluate_actions(rows, actions)
        record = {"quit_confidence_threshold": float(confidence_threshold), **metrics}
        if record["utility"] > best["utility"]:
            best = record

    return best


def evaluate_cost_sensitive_policy_matrix_with_threshold(
    rows,
    model,
    x: np.ndarray,
    quit_confidence_threshold: float,
):
    probs = aligned_action_probabilities(model, x)

    if quit_confidence_threshold <= 0.0:
        predictions = np.argmax(probs, axis=1)
        actions = [ACTION_NAMES[p] for p in predictions]
    else:
        actions = []
        for i in range(len(rows)):
            p_cont, p_ver = probs[i, 0], probs[i, 1]
            if max(p_cont, p_ver) < quit_confidence_threshold:
                actions.append("quit")
            elif p_ver > p_cont:
                actions.append("verify")
            else:
                actions.append("continue")

    return evaluate_actions(rows, actions)


def evaluate_actions(rows, actions):
    chosen_utilities = []
    chosen_correct = []
    catastrophic = []
    intervention = []
    verify_rate = []
    quit_rate = []
    harmful = []

    for row, action in zip(rows, actions):
        if action == "continue":
            chosen_utilities.append(row["u_continue"])
            chosen_correct.append(1.0 if row["continue_correct"] else 0.0)
            catastrophic.append(1.0 if not row["continue_correct"] else 0.0)
            intervention.append(0.0)
            verify_rate.append(0.0)
            quit_rate.append(0.0)
        elif action == "verify":
            chosen_utilities.append(row["u_verify"])
            chosen_correct.append(1.0 if row["verify_correct"] else 0.0)
            catastrophic.append(1.0 if not row["verify_correct"] else 0.0)
            intervention.append(1.0)
            verify_rate.append(1.0)
            quit_rate.append(0.0)
            harmful.append(1.0 if row["u_verify"] < row["u_continue"] else 0.0)
        else:
            chosen_utilities.append(row["u_quit"])
            chosen_correct.append(0.0)
            catastrophic.append(0.0)
            intervention.append(1.0)
            verify_rate.append(0.0)
            quit_rate.append(1.0)
            harmful.append(1.0 if row["u_quit"] < row["u_continue"] else 0.0)

    oracle = [max(row["u_continue"], row["u_verify"], row["u_quit"]) for row in rows]
    return {
        "utility": float(np.mean(chosen_utilities)),
        "success_rate": float(np.mean(chosen_correct)),
        "catastrophic_failure_rate": float(np.mean(catastrophic)),
        "intervention_rate": float(np.mean(intervention)),
        "verify_rate": float(np.mean(verify_rate)),
        "quit_rate": float(np.mean(quit_rate)),
        "harmful_intervention_rate": float(np.mean(harmful)) if harmful else 0.0,
        "control_regret": float(np.mean(np.asarray(oracle) - np.asarray(chosen_utilities))),
    }
