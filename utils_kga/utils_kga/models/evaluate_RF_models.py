from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap

from utils_kga.general import pretty_plot

shap.initjs()

FEATURE_GROUP_MAP = {"bond_angle": "bond angle",
                     "perc": "connectivity",
                     "Valence": "valence",
                     "Unfilled": "valence",
                     "neighbor_distance": "distance",
                     "_row_": "PSE (mag)",
                     " MendeleevNumber": "PSE",
                     "ionization_energy_": "ionization energy",
                     "element_": "PSE (mag)",
                     "GSbandgap": "bandgaps of elemental solids",
                     "crystal_system": "symmetry",
                     " AtomicWeight": "PSE",
                     "spacegroup_": "symmetry",
                     " Number": "PSE",
                     "_group": "PSE (mag)",
                     "is_centrosymmetric": "symmetry",
                     "ewald_energy": "ewald energy",
                     "density": "density",
                     "GSvolume": "volumes of elemental solids",
                     "SCM": "SCM",
                     "SCM mag substructure": "SCM mag substructure",
                     "GSmagmom": "magnetic moments of elemental solids",
                     " Row": "PSE",
                     " Column": "PSE",
                     "_ligand": "ligand info",
                     "cn_oxidation": "oxidation state (mag)",
                     " oxidation": "oxidation state",
                     "MeltingT": "melting temperatures of elemental solids",
                     "Electronegativity": "electronegativity",
                     "cn_X_": "electronegativity (mag)",
                     "CovalentRadius": "covalent radii of elemental soldis",
                     "_radius_": "atomic or ionic radii",
                     "SpaceGroup": "symmetry",
                     "structural complexity": "structural complexity",
                     "_csm": "coordination polyhedra distortion",
                     "has_mag_neighbors": "percentage of isolated magnetic sites",
                     "has_mag_connections": "percentage of isolated magnetic sites",
                     "avg ionic char": "average ionic character",
                     "max ionic char": "max ionic character",
                     "packing fraction": "packing fraction",
                     "avg anion electron affinity": "average anion electron affinity",
                     "rest general matminer": "rest general matminer",
                     "rest magnetism-specific": "rest magnetism-specific"}

def shap_plot_rf(model, result_dir: str, X_train, iteration) -> (pd.DataFrame, str):
    """
    Function to extract and store the shapley values plot to identify feature influence on model predictions.
    Compute influences of all features.
    Adapted from Aakash Naik for shap 0.46.0.
    """
    exc = ""

    # Extract shapley values from the best model
    explainer = shap.TreeExplainer(
        model.best_estimator_, X_train
    )
    try:
        shap_values = explainer.shap_values(X_train, check_additivity=True)
    except Exception as e:
        shap_values = explainer.shap_values(X_train, check_additivity=False)
        exc = str(e)

    fig = shap.summary_plot(
        [shap_values[:, :, class_ind] for class_ind in range(shap_values.shape[-1])],  # v0.46.0
        features=X_train,
        feature_names=X_train.columns,
        show=False,
    )
    plt.savefig(os.path.join(result_dir, f"shap_fold_{iteration}.png"))
    plt.close()

    data = {}
    for cl in range(shap_values.shape[-1]):
        vals = np.abs([shap_values[:, :, class_ind] for class_ind in range(shap_values.shape[-1])][cl]).mean(0)
        data[f"feat_importance_cl{cl}"] = dict(zip(X_train.columns, vals))

    shap_classwise = pd.DataFrame(data)
    print(shap_classwise.values.flatten().sum())
    shap_classwise["feat_importance_all_classes"] = shap_classwise.apply(lambda row: np.sum(row), axis=1)
    shap_classwise.sort_values(by=["feat_importance_all_classes"], ascending=False, inplace=True)

    return shap_classwise, exc


def get_rf_feature_importances(best_model, X):
    """
    Get feature importances of all features from RF for one fold.
    Generally applied to best estimator of one inner (GridSearchCV) loop.
    :param best_model: best model of one inner loop iteration
    :param X: whole feat. df (without labels)
    :return: dict of feat.: feat_imp, not sorted.
    """
    feature_names = X.columns.tolist()
    feature_scores = best_model.feature_importances_.tolist()
    return {"rf_feature_importances": dict(zip(feature_names, feature_scores))}

def plot_average_rf_feature_importances(model_log) -> go.Figure:
    """
    Plot feature importances from RF averaged over all folds.
    :param model_log: model_log of train_rf_classifier
    :return: a plotly.graph_objects.Figure object
    """
    scores = {}
    for f in model_log["folds"]:
        for idx, fn in enumerate(f["feat"]["feature_names"]):
            try:
                scores[fn].append(f["feat"]["feature_scores"][idx])
            except KeyError:
                scores[fn] = [f["feat"]["feature_scores"][idx]]

    imp_dict = {k: (np.mean(np.array(v)), len(v)) for k, v in scores.items()}
    fig = go.Figure(layout=go.Layout(xaxis=go.layout.XAxis(title="Feature"),
                                     yaxis=go.layout.YAxis(title="Average importance"),
                                     ))
    fig.add_bar(x=list(imp_dict.keys()),
                y=[i[0] for i in (imp_dict.values())],
                hovertext=["Occurrences: "+str(i[1]) for i in (imp_dict.values())]
                )
    fig = pretty_plot(fig)
    return fig


def plot_grouped_summed_feature_importances(model_logs: list[dict],
                                            imp_type: str = "rf",
                                            summarize_below_perc: float = 3.0
                                            ) -> go.Figure:
    """
    Assign features to groups and plot sunburst plot of feature importances
    summed over all outer folds and models. (Inner ring: feature origin (matminer or custom magnetism-specific)).
    :param model_logs: list of model_logs of train_rf_classifier
    :param imp_type: type of feature importances to plot (Shapley values or RF feature importances)
    :param summarize_below_perc: percentage under which feature group is not displayed single, but summarized as rest
    :return: a plotly.graph_objects.Figure object
    """
    assert imp_type in ["rf", "shapley"]
    assert len(set([len(model_log["folds"]) for model_log in model_logs])) == 1, \
        ("Function plot_grouped_summed_feature_importances is right now not suited for comparing "
         "models with different number of outer folds.")
    feat_dicts = []
    if imp_type == "rf":
        for model_log in model_logs:
            feat_dicts.extend(fold["feat"]["rf_feature_importances"] for fold in model_log["folds"])
    else:
        for model_log in model_logs:
            for fold in model_log["folds"]:
                normalizer = sum(fold["feat"]["shap_classwise"]["feat_importance_all_classes"].values())
                feat_dicts.append({k: v / normalizer for k, v in
                                   fold["feat"]["shap_classwise"]["feat_importance_all_classes"].items()})

    return util_plot_grouped_feature_importances(feat_dicts=feat_dicts, summarize_below_perc=summarize_below_perc)


def util_plot_grouped_feature_importances(feat_dicts: list[dict],
                                          summarize_below_perc: float = 3.0) -> go.Figure:
    """
    Util function used both to plot grouped (over folds / models) feature importances
    from RF and from Shapley analysis.
    :param feat_dicts: list of feature importance dict (normalized to 1.0 per model / fold)
    :param summarize_below_perc: percentage under which feature group is not displayed single, but summarized as rest
    :return: a plotly.graph_objects.Figure object
    """
    def map_origin(feature: str)-> str:
        if feature.startswith("cn_") or ("sine coulomb" in feature and feature.endswith("cat")):
            return "magnetism-specific"
        return "general matminer"

    summed_importances = {k: 0.0 for k in list(FEATURE_GROUP_MAP.values())}
    origin_map_dict = {"rest general matminer": "general matminer", "rest magnetism-specific": "magnetism-specific"}
    for fold in feat_dicts:
        for feat, imp in fold.items():
            if "sine coulomb" in feat:
                om = map_origin(feat)
                fil = "SCM mag substructure" if om == "magnetism-specific" else "SCM"
                summed_importances[fil] += imp
                if fil in origin_map_dict:
                    assert map_origin(feat) == origin_map_dict[fil]
                else:
                    origin_map_dict[fil] = map_origin(feat)
                continue

            catched = False
            for fil, fil_str in FEATURE_GROUP_MAP.items():
                if fil in feat:
                    summed_importances[fil_str] += imp
                    catched = True
                    if fil in origin_map_dict:
                        assert map_origin(feat) == origin_map_dict[fil_str]
                    else:
                        origin_map_dict[fil_str] = map_origin(feat)
            if not catched:
                om = map_origin(feat)
                fil = f"rest {om}"
                summed_importances[fil] += imp
                origin_map_dict[fil] = om
                print(feat, round(imp, 2))

    assert round(sum(list(summed_importances.values())), 6) == len(feat_dicts)  # assert no double counting

    summed_importances = {k: v for k, v in summed_importances.items() if v > 0.0 or "rest " in k}
    summed_importances_n = summed_importances.copy()
    for k, v in summed_importances.items():
        if v * 100 / len(feat_dicts) < summarize_below_perc and not "rest " in k:
            del summed_importances_n[k]
            summed_importances_n[f"rest {origin_map_dict[k]}"] += v

    summed_importances_n = {k: v for k, v in summed_importances_n.items() if v > 0.0}
    assert round(sum(list(summed_importances_n.values())), 6) == len(feat_dicts)

    df = pd.DataFrame.from_dict(summed_importances_n, orient="index")
    df.reset_index(inplace=True)
    df["feat_origin"] = df["index"].apply(lambda x: origin_map_dict[x])
    color_discrete_map = {"general matminer": "#8B1811",
                          "magnetism-specific": "#009FE3"}
    fig = px.sunburst(df, path=["feat_origin", "index"], values=0,
                      color="feat_origin", color_discrete_map=color_discrete_map)
    fig.data[0].textinfo = "label+percent entry"
    fig.data[0].insidetextorientation = "horizontal"
    fig.update_layout(font_family="Arial")
    fig.update_layout(paper_bgcolor="rgba(255, 255, 255, 1)")
    fig.update_traces(textfont=dict(family=["Arial" for _ in range(len(fig.data[0]["ids"]))],
                                    size=[51 for _ in range(len(fig.data[0]["ids"]))]))
    return fig


def get_heuristic_importance_sum(model_log: dict, imp_type: str = "rf",
                                 feat_type: str = "heuristic") -> float:
    """
    :param model_log: log dict as written in train_rf_classifier
    :param imp_type: type of feature importances (Shapley values or RF feature importances)
    :param feat_type: function determining which features to count
    :return: percentage of sum of all heuristic feature importances
    """
    assert imp_type in ["rf", "shapley"]
    assert feat_type in ["perc", "bond_angle", "neighbor_distance", "heuristic"]

    if imp_type == "rf":
        feat_dicts = [fold["feat"]["rf_feature_importances"] for fold in model_log["folds"]]
    else:
        feat_dicts = []
        for fold in model_log["folds"]:
            normalizer = sum(fold["feat"]["shap_classwise"]["feat_importance_all_classes"].values())
            feat_dicts.append({k: v / normalizer for k, v
                               in fold["feat"]["shap_classwise"]["feat_importance_all_classes"].items()})
    heuristic_imp = 0.0
    for fold in feat_dicts:
        for feat, imp in fold.items():
            if f(feat_type, feat):
                heuristic_imp += imp
    return heuristic_imp / len(feat_dicts)

def f(feat_type, feat):
    if feat_type == "heuristic":
        return "perc" in feat or "bond_angle" in feat or "neighbor_distance" in feat
    return feat_type in feat

