from collections import Counter
from math import sqrt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, precision_score, recall_score, auc
from sklearn.preprocessing import label_binarize


def get_multiclass_pr_curves(y_true, y_pred_proba) -> go.Figure:
    colors = px.colors.qualitative.Plotly

    fig = go.Figure(layout=go.Layout(xaxis=go.layout.XAxis(title="Recall"),
                                     yaxis=go.layout.YAxis(title="Precision"),
                                     ))
    num_classes = len(set(y_true))
    y_true_b = label_binarize(y_true, classes=np.arange(num_classes))

    th = get_prevalence_dict(y_true)

    for i in range(num_classes):
        precision, recall, threshold = precision_recall_curve(y_true_b[:, i], np.array(y_pred_proba)[:, i])
        fig.add_trace(go.Scatter(
            x=list(recall),
            y=list(precision),
            mode="lines",
            name=str(i),
            marker_color=colors[i]
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[th[i], th[i]],
            mode="lines",
            name="thr_" + str(i),
            marker_color=colors[i]
        ))

    return fig


def get_twoclass_pr_curves(y_true, y_pred_proba) -> go.Figure:
    colors = px.colors.qualitative.Plotly

    fig = go.Figure(layout=go.Layout(xaxis=go.layout.XAxis(title="Recall"),
                                     yaxis=go.layout.YAxis(title="Precision"),
                                     ))

    th = get_prevalence_dict(y_true)

    for i in range(2):
        precision, recall, threshold = precision_recall_curve(y_true, np.array(y_pred_proba)[:, i], pos_label=i)
        fig.add_trace(go.Scatter(
            x=list(recall),
            y=list(precision),
            mode="lines",
            name=str(i),
            marker_color=colors[i]
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[th[i], th[i]],
            mode="lines",
            name="thr_" + str(i),
            marker_color=colors[i]
        ))

    return fig


def get_classification_test_scores(y_true, y_pred, y_pred_proba):
    prevalence = get_prevalence_dict(y_true)
    num_classes = len(set(y_true))
    if num_classes > 2:
        test_scores = {
            "roc_auc_test_micro": roc_auc_score(y_true, y_pred_proba, average="micro", multi_class="ovr"),
            "roc_auc_test_weighted": roc_auc_score(y_true, y_pred_proba, average="weighted", multi_class="ovr"),
        }
        classwise_roc_auc = roc_auc_score(y_true, y_pred_proba, average=None, multi_class="ovr")
        for cl in range(num_classes):
            test_scores[f"roc_auc_test_{cl}"] = classwise_roc_auc[cl]
            test_scores[f"roc_auc_test_{cl}_normalized"] = (classwise_roc_auc[cl] - prevalence[cl]) / (1.0 - prevalence[cl])

        for metric, metric_str in zip([f1_score, precision_score, recall_score], ["f1", "precision", "recall"]):
            test_scores[f"{metric_str}_test_micro"] = metric(y_true, y_pred, average="micro")
            test_scores[f"{metric_str}_test_weighted"] = metric(y_true, y_pred, average="weighted")
            classwise_metric = metric(y_true, y_pred, average=None)
            for cl in range(num_classes):
                test_scores[f"{metric_str}_test_{cl}"] = classwise_metric[cl]
                test_scores[f"{metric_str}_test_{cl}_normalized"] = (
                        (classwise_metric[cl] - prevalence[cl]) / (1.0 - prevalence[cl]))

        y_true_b = label_binarize(y_true, classes=np.arange(num_classes))
        for cl in range(num_classes):
            precision, recall, threshold = precision_recall_curve(y_true_b[:, cl], np.array(y_pred_proba)[:, cl])
            auc_full = auc(x=recall, y=precision)
            auc_norm = (auc_full - prevalence[cl]) / (1.0 - prevalence[cl])
            test_scores[f"pr_auc_test_{cl}"] = auc_full
            test_scores[f"pr_auc_test_{cl}_normalized"] = auc_norm

    else:
        test_scores = {}
        for metric, metric_str in zip([f1_score, precision_score, recall_score], ["f1", "precision", "recall"]):
            test_scores[f"{metric_str}_test"] = metric(y_true, y_pred)
        for cl in range(num_classes):
            r = roc_auc_score(y_true=y_true, y_score=np.array(y_pred_proba)[:, cl])
            test_scores[f"roc_auc_test_{cl}"] = r
            test_scores[f"roc_auc_test_{cl}_normalized"] = (r - prevalence[cl]) / (1.0 - prevalence[cl])

    return test_scores


def get_mean_baseline_f1_score(y_true):
    """As comparison of models trained on datasets with different class prevalences,
    compute baseline weighted F1 *without* the assumption that class imbalance is known in case of 3 classes.
    In case of 2 classes, minority class is known as F1 is evaluated with respect to it."""
    prev = get_prevalence_dict(y_true)
    if len(prev) > 2:
        f1_bl = sum([r * (r / (r + 0.5)) for r in prev.values()])
    else:
        r = prev[1]
        f1_bl = r / (r + 0.5)
    return f1_bl


def get_prevalence_dict(y_true):
    c = Counter(y_true)
    th = {}
    for k, v in c.items():
        th[k] = v / len(y_true)
    return th

