import json
import numpy as np
import os
from modnet.preprocessing import MODData
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .evaluate_RF_models import shap_plot_rf, get_rf_feature_importances, get_rf_hyperparams_statistics
from .evaluate_models import get_multiclass_pr_curves, get_classification_test_scores, get_twoclass_pr_curves
from .log_models import get_model_id, make_id_log_test_score_mapping


def train_rf_classifier(df, model_log: dict, target_name: str, target_type: str,
                        data_filter: str, classes_int_to_name_map: dict,
                        script_path: str, feature_dataframe_identifier: str, test_feat_df_path: str,
                        n_features: int=20, random_state_feature_selection: int=42, optimal_features: list | None = None,
                        save_models: bool = False, additional_info_for_model_id: str | None = None,
                        outer_cv: StratifiedKFold=StratifiedKFold(n_splits=10, shuffle=True),
                        inner_cv = StratifiedKFold(n_splits=10, shuffle=True),
                        folds_shapley=[], eval_metric="f1_weighted",
                        n_jobs_rf: int = 1, n_jobs_grid: int = 6, n_jobs_feature_selection: int = 6,
                        ):
    """
    Trains a binary or multiclass RF classifier on one target and documents the results
    including computation of feature importances, PR curves etc.
    Feature selection before outer CV with modnet (as prev. optimization of random seed for target NMI computation)
    -> we test additional test set left out before NMI computation as f(random seed).
    :param df: df with features AND labels
    :param target_name: df column name with quantity to learn
    :param target_type: "site" or "structure"
    :param data_filter: info on dataset (e.g., "TM_structs" or "all_structs")
    :param classes_int_to_name_map: dict that maps integer class labels to str name, e.g. "p<0.005"
    :param script_path: absolute file path for model script
    :param feature_dataframe_identifier: unique id of features (and targets) used (e.g. file path and/or commit)
    :param test_feat_df_path: file path for test set removed before analysis of NMI as f(random seed),
        see featurization dir for further info.
    :param n_features: Number of features to select.
    :param random_state_feature_selection: random seed for computation of mutual information in feature selection.
    :param optimal_features: optionally already provide optimal features and skip modnet feature selection.
    :param save_models: whether to save the outer_cv best models
    :param additional_info_for_model_id: optional add on to model id
    :param outer_cv: defines outer cv.
    :param folds_shapley: list of fold idx for which to compute shapley values
    :param model_log: dict w. model information e.g. hyperparameter grid.
        For RF models, it requires key "param_grid" with RF param grid for inner cv.
        It can contain any other logging information defined in the script (e.g., info on training dataset, its size..)
        as long as the keys are not in ["folds", "script_path", "feature_df_identifier",
         "train_scores", "test_scores_cv", "test_scores_ext", "ys_cv", "hyperparam_occus", "n_outer_splits",
         "n_inner_splits", "target_name",
         "target_type", "model_type", "classes_int_to_name_map"].
    :param inner_cv: defines inner cv, possible values see scikit learn docs
    :param eval_metric: Evaluation metric for grid search.
    :param n_jobs_rf: Number of jobs for classifier.
    :param n_jobs_grid: Number of jobs for grid.
    :param n_jobs_feature_selection: Number of jobs for feature selection.
    """
    model_id = get_model_id(target_name=target_name.removeprefix("target_"),
                            target_type=target_type, model_type="RF_class",
                            data_filter=data_filter, additional_info=additional_info_for_model_id)
    result_dir = f"../results/{model_id}"
    os.makedirs(result_dir, exist_ok=True)

    test_set_ext = pd.read_json(test_feat_df_path)

    y = df[target_name].values
    num_classes = len(set(y))

    y_true, y_pred, y_pred_proba, y_test_ids, train_f1s = [[] for _ in range(5)]

    model_log["target_name"] = target_name.removeprefix("target_")
    model_log["target_type"] = target_type
    model_log["model_type"] = "RF_class"
    model_log["data_filter"] = data_filter
    model_log["classes_int_to_name_map"] = classes_int_to_name_map
    model_log["outer_splits"] = outer_cv.get_n_splits(X=np.zeros(len(df)), y=y)
    model_log["script_path"] = script_path
    model_log["feature_df_identifier"] = feature_dataframe_identifier
    model_log["n_features"] = n_features,
    model_log["random_state_feature_selection"] = random_state_feature_selection,
    model_log["folds"] = []

    if not optimal_features:
        optimal_features = select_features(df=df, target_name=target_name, n_features=n_features,
                                           random_state=random_state_feature_selection, n_jobs=n_jobs_feature_selection)
    X_test_set_ext = test_set_ext.loc[:, optimal_features]
    y_test_set_ext = test_set_ext[target_name].values
    test_scores_ext = []  # For evaluation of external test set -> predict N-outer_split n times, average performances
    X = df.loc[:, optimal_features]

    for i, (train_ix, test_ix) in enumerate(outer_cv.split(np.zeros(len(df)), y.tolist())):
        y_train, y_test = y[train_ix], y[test_ix]
        X_train = X.iloc[train_ix, :]
        X_test = X.iloc[test_ix, :]

        best_model, search = inner_gridsearch_classification(X=X_train, y=y_train, inner_cv=inner_cv,
                                                             param_grid=model_log["param_grid"],
                                                             eval_metric=eval_metric, n_jobs_rf=n_jobs_rf,
                                                             n_jobs_grid=n_jobs_grid)

        y_true.extend(y_test.tolist())
        y_pred.extend([i.item() for i in best_model.predict(X_test)])
        y_pred_proba.extend([list(i) for i in best_model.predict_proba(X_test)])

        if num_classes > 2:
            train_f1s.append(f1_score(y_train, best_model.predict(X_train), average="weighted"))
        else:
            train_f1s.append(f1_score(y_train, best_model.predict(X_train)))

        # Evaluate on hold-out test set
        y_pred_ext = [i.item() for i in best_model.predict(X_test_set_ext)]
        y_pred_proba_ext = [list(i) for i in best_model.predict_proba(X_test_set_ext)]
        test_scores_ext.append(get_classification_test_scores(y_true=y_test_set_ext,
                                                              y_pred=y_pred_ext,
                                                              y_pred_proba=y_pred_proba_ext,))
        y_test_ids.append(test_ix)

        feature_dict = get_rf_feature_importances(best_model=best_model, X=X_train)

        if i in folds_shapley:
            shap_classwise, exc = shap_plot_rf(X_train=X_train, result_dir=result_dir, model=search, iteration=i)
            feature_dict["shap_classwise"] = shap_classwise.to_dict()
            feature_dict["shap_exc"] = exc
        if i == 0:
            model_log["n_inner_splits"] = inner_cv.get_n_splits(X=X_train, y=y_train)

        model_log["folds"].append({"best_params": search.best_params_,
                                   "feat": feature_dict})
        if save_models:
            pickle.dump(best_model, open(os.path.join(result_dir, f"best_model_{i}.pkl"), "wb"))

    model_log["train_scores"] = {
        "Mean train F1 (weighted)": np.mean(np.array(train_f1s)),
        "Median train F1 (weighted)": np.median(np.array(train_f1s)),
        "Std train F1 (weighted)": np.std(np.array(train_f1s)),
    }

    model_log["test_scores_cv"] = get_classification_test_scores(
        y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)

    model_log["test_scores_ext_all"] = test_scores_ext
    model_log["test_scores_ext_av"] = {k: np.array([d[k] for d in test_scores_ext]).mean() for k in test_scores_ext[0]}

    if not isinstance(outer_cv, dict):
        y_test_ids = "Full outer CV"
    model_log["ys_cv"] = {
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "y_true": y_true,
        "y_test_idx": y_test_ids,
    }

    param_occus = get_rf_hyperparams_statistics(model_log)
    model_log["hyperparam_occus"] = param_occus

    if num_classes > 2:
        pr_curves = get_multiclass_pr_curves(y_true, y_pred_proba)
    else:
        pr_curves = get_twoclass_pr_curves(y_true, y_pred_proba)
    pr_curves.write_html(os.path.join(result_dir, "pr_curves.html"))

    f1_test_weighted_or_minor_cv = model_log["test_scores_cv"]["f1_test_weighted"] \
        if num_classes > 2 else model_log["test_scores_cv"]["f1_test"]
    f1_test_weighted_or_minor_ext = model_log["test_scores_ext_av"]["f1_test_weighted"] \
        if num_classes > 2 else model_log["test_scores_ext_av"]["f1_test"]

    make_id_log_test_score_mapping(model_id=model_id, log_path=os.path.abspath(result_dir),
                                   f1_test_weighted_or_minor_cv=f1_test_weighted_or_minor_cv,
                                   f1_test_weighted_or_minor_ext=f1_test_weighted_or_minor_ext,
                                   )

    with open(os.path.join(result_dir, model_id+".json"), "w") as f:
            json.dump(model_log, f)


def inner_gridsearch_classification(X, y, param_grid, inner_cv=StratifiedKFold(n_splits=10, shuffle=True),
                                    eval_metric="f1_weighted", n_jobs_rf=1, n_jobs_grid=-1):
    """
    Defines inner loop for hyperparameter optimization in RF classification, refits best model on whole training set.
    :param X: training set fr. outer loop
    :param y: training set fr. outer loop
    :param param_grid: dict of hyperparameter values
    :param inner_cv: defines inner cv, possible values see scikit learn docs.
    :param eval_metric: Evaluation metric for grid search.
    :param n_jobs_rf: Number of jobs for classifier.
    :param n_jobs_grid: Number of jobs for grid.
    :return: best estimator of hyperparameter search, fitted on whole set and GridSearchCV object
    """
    rf = RandomForestClassifier(n_jobs=n_jobs_rf, class_weight="balanced")

    search = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring=eval_metric,
        cv=inner_cv,
        refit=True,
        return_train_score=True,
        n_jobs=n_jobs_grid,
    )
    result = search.fit(X, y)

    return result.best_estimator_, search


def select_features(df: pd.DataFrame, target_name: str,
                    n_features: int=20, random_state: int=42, n_jobs: int = 6) -> list:
    """
    Perform modnet feature selection.
    :param df: Training df of one outer fold, containing targets.
    :param target_name: name of target column.
    :param n_features: Number of features to select
    :param random_state: random seed for computation of mutual information.
    :param n_jobs: Number of jobs.
    :return: indices of optimal descriptors in indexing of original df.
    """
    targets = df[target_name].values
    feat = df.drop(columns=[target_name])
    md = MODData(targets=targets, structure_ids=feat.index,
                 df_featurized=feat, target_names=[target_name],
                 num_classes={target_name: len(set(targets))})
    md.feature_selection(n=n_features, n_jobs=n_jobs, random_state=random_state)
    print(md.optimal_features)
    return md.optimal_features
