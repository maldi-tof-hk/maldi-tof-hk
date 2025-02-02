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
    plt.plot([0, 1, 1], [1, 1, 0], "c--")
    plt.plot(recall_list, precision_list, label="Model (area = {:.3f})".format(auprc))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-recall curve")
    plt.legend(loc="best")
    plt.savefig(path, dpi=300)


def evaluate_multiplex_prc(models, path: MetricsPath):
    path = path.get_path("prc_multi.png")
    print(f"PRC saved to {path}")

    plt.clf()
    prc_fig, prc_ax = plt.subplots()
    prc_ax.plot([0, 1, 1], [1, 1, 0.5], "c--")

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
        prc_ax_ins.plot(recall, precision)

    prc_ax.legend(loc="best")
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


def evaluate_output_distribution(y_true, y_pred, path: MetricsPath):
    path = path.get_path("output_distribution.png")
    print(f"Output distribution saved to {path}")

    plt.clf()
    plt.hist(y_pred, range=(0, 1), bins=20, log=True)
    plt.title("Distribution of model output")
    plt.xlabel("Model output")
    plt.ylabel("Frequency")
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


def evaluate_confidence_accuracy(y_true, y_pred, path: MetricsPath):
    path = path.get_path("confidence_accuracy.png")
    print(f"Confidence accuracy saved to {path}")

    bin_count = 20
    acc_x, acc_y = bin_accuracy_by_confidence(y_true, y_pred, bin_count)
    plt.clf()
    plt.bar(acc_x, acc_y, 0.5 / bin_count, align="edge")
    plt.ylim((0.4, 1))
    plt.title("Accuracy by model confidence")
    plt.xlabel("Model confidence")
    plt.ylabel("Validation accuracy")
    plt.savefig(path, dpi=300)


def tac(y_true, y_pred):
    """Calculate threshold-accuracy curve"""
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


def evaluate_tac_table(y_true, y_pred, path: MetricsPath):
    path = path.get_path("tac.csv")
    print(f"TAC table saved to {path}")

    accepted_list, accuracy_list, threshold_list = tac(y_true, y_pred)
    df = pd.DataFrame(
        {
            "Threshold": threshold_list,
            "Accepted": accepted_list,
            "Accuracy": accuracy_list,
        }
    )
    df.to_csv(path, index=False)


def evaluate_tac_figure(y_true, y_pred, path: MetricsPath):
    path = path.get_path("tac.png")
    print(f"TAC figure saved to {path}")

    accepted_list, accuracy_list, threshold_list = tac(y_true, y_pred)
    autac = auc(accepted_list, accuracy_list)

    plt.clf()
    plt.plot([1, 0, 0], [1, 1, 0.9], "c--")
    plt.plot(
        1 - np.array(accepted_list),
        accuracy_list,
        label="Model (area = {:.3f})".format(autac),
    )
    plt.xlabel("% sample rejected")
    plt.ylabel("Filtered accuracy")
    plt.title(f"Threshold-accuracy curve")
    plt.legend(loc="best")
    plt.savefig(path, dpi=300)


def evaluate_multiplex_tac(models, path: MetricsPath):
    path = path.get_path("tac_multi.png")
    print(f"TAC saved to {path}")

    plt.clf()
    tac_fig, tac_ax = plt.subplots()
    tac_ax.set_xlabel("% sample rejected")
    tac_ax.set_ylabel("Filtered accuracy")
    tac_ax.set_title(f"Threshold-accuracy curve")
    tac_ax.plot([1, 0, 0], [1, 1, 0.9], "c--")

    for model, y_true, y_pred in models:
        accepted_list, accuracy_list, threshold_list = tac(y_true, y_pred)
        autpc = auc(accepted_list, accuracy_list)
        tac_ax.plot(
            1 - np.array(accepted_list),
            accuracy_list,
            label=f"{model.name} (area = {autpc:.3f})",
        )

    tac_ax.legend(loc="best")
    tac_fig.savefig(path, dpi=300)


def evaluate_all_metrics(y_true, y_pred, path: MetricsPath):
    evaluate_metrics(y_true, y_pred, path)
    evaluate_prc(y_true, y_pred, path)
    evaluate_roc(y_true, y_pred, path)
    evaluate_confusion_matrix(y_true, y_pred, path)
    evaluate_output_distribution(y_true, y_pred, path)
    evaluate_confidence_accuracy(y_true, y_pred, path)
    evaluate_tac_table(y_true, y_pred, path)
    evaluate_tac_figure(y_true, y_pred, path)
