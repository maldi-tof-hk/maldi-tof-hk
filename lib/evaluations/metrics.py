from lib.models.base import BaseClassifier
from lib.path import MetricsPath
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    auc,
    ConfusionMatrixDisplay,
    roc_curve,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import json
from scipy import stats


def save_metrics(metrics, path: MetricsPath):
    path = path.get_path("metrics.json")
    print(f"Metrics saved to {path}")

    metrics = {
        "loss": metrics[0],
        "accuracy": metrics[1],
        "precision": metrics[2],
        "recall": metrics[3],
        "f1": metrics[4],
        "auprc": metrics[5],
        "auroc": metrics[6],
    }

    with open(path, "w") as f:
        json.dump(metrics, f)


def evaluate_metrics(y_true, y_pred, path: MetricsPath):
    precision_list, recall_list, prc_thresholds = precision_recall_curve(y_true, y_pred)
    fpr_list, tpr_list, roc_thresholds = roc_curve(y_true, y_pred)
    loss = log_loss(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred > 0.5)
    precision = precision_score(y_true, y_pred > 0.5)
    recall = recall_score(y_true, y_pred > 0.5)
    f1 = f1_score(y_true, y_pred > 0.5)
    auprc = auc(recall_list, precision_list)
    auroc = auc(fpr_list, tpr_list)

    metrics = np.array([loss, accuracy, precision, recall, f1, auprc, auroc])

    save_metrics(metrics, path)

    return metrics


def evaluate_prc(y_true, y_pred, path: MetricsPath):
    path = path.get_path("prc.png")
    print(f"PRC saved to {path}")

    precision_list, recall_list, prc_thresholds = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall_list, precision_list)

    plt.clf()
    plt.plot([0, 1, 1], [1, 1, 0.5], "c--")
    plt.plot(recall_list, precision_list, label="Model (area = {:.3f})".format(auprc))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-recall curve")
    plt.legend(loc="best")
    plt.savefig(path, dpi=300)


def evaluate_multiplex_prc(models, path: MetricsPath, enable_inset=True):
    path = path.get_path("prc_multi.png")
    print(f"PRC saved to {path}")

    plt.clf()
    prc_fig, prc_ax = plt.subplots()
    prc_ax.plot([0, 1, 1], [1, 1, 0.5], "c--")

    if enable_inset:
        prc_ax_ins = prc_ax.inset_axes(
            [0.05, 0.25, 0.6, 0.6],
            xlim=(0.8, 1),
            ylim=(0.9, 1),
            xticklabels=[],
            yticklabels=[],
        )

    prc_ax.set_xlabel("Recall")
    prc_ax.set_ylabel("Precision")
    prc_ax.set_title(f"Precision-recall curve")

    for model, y_true, y_pred in models:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        auc_precision_recall = auc(recall, precision)
        print(f"{model.name}: ", auc_precision_recall)
        prc_ax.plot(
            recall,
            precision,
            label="{} (area = {:.3f})".format(model.name, auc_precision_recall),
        )
        if enable_inset:
            prc_ax_ins.plot(recall, precision)

    prc_ax.legend(loc="best")
    if enable_inset:
        prc_ax.indicate_inset_zoom(prc_ax_ins)
    prc_fig.savefig(path, dpi=300)


def evaluate_roc(y_true, y_pred, path: MetricsPath):
    path = path.get_path("roc.png")
    print(f"ROC saved to {path}")

    fpr_list, tpr_list, roc_thresholds = roc_curve(y_true, y_pred)
    auroc = auc(fpr_list, tpr_list)

    plt.clf()
    plt.plot([0, 0, 1], [0, 1, 1], "c--")
    plt.plot(fpr_list, tpr_list, label="Model (area = {:.3f})".format(auroc))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve")
    plt.legend(loc="best")
    plt.savefig(path, dpi=300)


def evaluate_confusion_matrix(y_true, y_pred, path: MetricsPath):
    path = path.get_path("confusion_matrix.png")
    print(f"Confusion matrix saved to {path}")

    plt.clf()
    ConfusionMatrixDisplay.from_predictions(
        y_true, np.array([x >= 0.5 for x in y_pred]), display_labels=["MSSA", "MRSA"]
    )
    plt.savefig(path, dpi=300)


def bin_accuracy_by_confidence(y_true, y_pred, bin_count=20):
    """Aggregrate validation accuracy by model confidence"""
    abs_pred = np.abs(y_pred - 0.5)
    confidence_list = []
    accuracy_list = []
    for bin in range(0, bin_count):
        include = (abs_pred > bin / 2 / bin_count) & (
            abs_pred < (bin + 1) / 2 / bin_count
        )
        count = np.sum(include)
        correct = np.sum((y_pred[include] > 0.5) == y_true[include])
        confidence_list.append(bin / 2 / bin_count + 0.5)
        if count > 0:
            accuracy_list.append(correct / count)
        else:
            accuracy_list.append(0)

    return confidence_list, accuracy_list


def evaluate_output_reliability(y_true, y_pred, path: MetricsPath):
    path = path.get_path("output_reliability.png")
    print(f"Output reliability saved to {path}")

    plt.clf()
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    fig.subplots_adjust(wspace=0.3)
    axes[0].hist(np.abs(y_pred - 0.5) + 0.5, range=(0.5, 1), bins=20, log=True)
    axes[0].set_title("Model output distribution")
    axes[0].set_xlabel("Model confidence")
    axes[0].set_ylabel("Frequency")

    bin_count = 20
    acc_x, acc_y = bin_accuracy_by_confidence(y_true, y_pred, bin_count)
    axes[1].plot([0.5, 1], [0.5, 1], "c--")
    axes[1].bar(acc_x, acc_y, 0.5 / bin_count, align="edge")
    axes[1].set_ylim((0.3, 1))
    axes[1].set_title("Model confidence reliability")
    axes[1].set_xlabel("Model confidence")
    axes[1].set_ylabel("Accuracy")

    fig.savefig(path, dpi=300)
    plt.figure()


def evaluate_cv_output_reliability(folds, path: MetricsPath):
    path = path.get_path("cv_output_reliability.png")
    print(f"CV Output reliability saved to {path}")

    outputs = []
    acc_x = None
    acc_y = []

    bin_count = 20
    for fold_id, y_true, y_pred in folds:
        fold_acc_x, fold_acc_y = bin_accuracy_by_confidence(y_true, y_pred, bin_count)
        if acc_x is None:
            acc_x = fold_acc_x
        acc_y.append(fold_acc_y)
        outputs.append(y_pred)

    acc_y = np.vstack(acc_y)
    mean_list = []
    ci_list = []
    for i in range(acc_y.shape[1]):
        mean, std = np.mean(acc_y[:, i]), np.std(acc_y[:, i])
        mean_list.append(mean)
        lower_ci, upper_ci = stats.norm.interval(
            0.95, loc=mean, scale=std / np.sqrt(acc_y.shape[0])
        )
        ci_list.append((upper_ci - lower_ci) / 2)

    outputs = np.concatenate(outputs)

    plt.clf()
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    fig.subplots_adjust(wspace=0.3)
    axes[0].hist(
        np.abs(outputs - 0.5) + 0.5,
        range=(0.5, 1),
        bins=20,
        log=True,
    )
    axes[0].set_title("Model output distribution")
    axes[0].set_xlabel("Model confidence")
    axes[0].set_ylabel("Frequency")

    axes[1].plot([0.5, 1], [0.5, 1], "c--")
    axes[1].bar(acc_x, mean_list, 0.5 / bin_count, align="edge")
    axes[1].errorbar(
        np.array(acc_x) + 0.5 / bin_count / 2,
        mean_list,
        yerr=np.array(ci_list),
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    axes[1].set_ylim((0.2, 1))
    axes[1].set_title("Accuracy by model confidence")
    axes[1].set_xlabel("Model confidence")
    axes[1].set_ylabel("Validation accuracy")

    fig.savefig(path, dpi=300)
    plt.figure()


def arc(y_true, y_pred):
    """Calculate accuracy-rejection curve"""
    accepted_list = []
    accuracy_list = []
    threshold_list = []
    bin_count = 5000

    for threshold in range(0, bin_count + 1):
        threshold_idx = threshold
        threshold = threshold / bin_count / 2 + 0.5
        y_pred_filtered = y_pred[np.abs(y_pred - 0.5) + 0.5 >= threshold]
        y_true_filtered = y_true[np.abs(y_pred - 0.5) + 0.5 >= threshold]
        y_result = (y_pred_filtered >= 0.5) == y_true_filtered
        threshold_list.append(threshold)
        accepted = len(y_result) / len(y_true)
        accepted_list.append(accepted)
        if len(y_result) > 0:
            accuracy = np.sum(y_result) / len(y_result)
            accuracy_list.append(accuracy)
        else:
            accuracy_list.append(1)

    return accepted_list, accuracy_list, threshold_list


def evaluate_arc_table(y_true, y_pred, path: MetricsPath):
    path = path.get_path("arc.csv")
    print(f"ARC table saved to {path}")

    accepted_list, accuracy_list, threshold_list = arc(y_true, y_pred)
    df = pd.DataFrame(
        {
            "Threshold": threshold_list,
            "Accepted": accepted_list,
            "Accuracy": accuracy_list,
        }
    )
    df.to_csv(path, index=False)


def evaluate_arc_figure(y_true, y_pred, path: MetricsPath):
    path = path.get_path("arc.png")
    print(f"ARC figure saved to {path}")

    accepted_list, accuracy_list, threshold_list = arc(y_true, y_pred)

    plt.clf()
    plt.plot(
        1 - np.array(accepted_list),
        accuracy_list,
    )
    plt.xlabel("% sample rejected")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy-rejection curve")
    plt.savefig(path, dpi=300)


def evaluate_multiplex_arc(models, path: MetricsPath):
    path = path.get_path("arc_multi.png")
    print(f"ARC saved to {path}")

    plt.clf()
    arc_fig, arc_ax = plt.subplots()
    arc_ax.set_xlabel("% sample rejected")
    arc_ax.set_ylabel("Accuracy")
    arc_ax.set_title(f"Accuracy-rejection curve")

    for model, y_true, y_pred in models:
        accepted_list, accuracy_list, threshold_list = arc(y_true, y_pred)
        arc_ax.plot(
            1 - np.array(accepted_list),
            accuracy_list,
            label="{}".format(model.name),
        )

    arc_ax.legend(loc="best")
    arc_fig.savefig(path, dpi=300)


def evaluate_all_metrics(y_true, y_pred, path: MetricsPath):
    evaluate_metrics(y_true, y_pred, path)
    evaluate_prc(y_true, y_pred, path)
    evaluate_roc(y_true, y_pred, path)
    evaluate_confusion_matrix(y_true, y_pred, path)
    evaluate_output_reliability(y_true, y_pred, path)
    evaluate_arc_table(y_true, y_pred, path)
    evaluate_arc_figure(y_true, y_pred, path)
