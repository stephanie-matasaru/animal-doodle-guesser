import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from download_data import load_data

#  Config 
MODEL_PATH = os.path.join("model", "quickdraw_model.keras")
CATEGORIES_PATH = os.path.join("model", "categories.npy")
PLOTS_DIR = "plots"

# Load model & data 
def load_model_and_data():
    model = tf.keras.models.load_model(MODEL_PATH)
    categories = list(np.load(CATEGORIES_PATH, allow_pickle=True))

    X, y, _ = load_data()

    # Use last 15% as test set (same split logic as train.py)
    split = int(len(X) * 0.85)
    X_test, y_test = X[split:], y[split:]

    return model, X_test, y_test, categories

# Confusion matrix 
def plot_confusion_matrix(y_true, y_pred, categories):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        linewidths=0.5
    )
    plt.title("Confusion Matrix (%)", fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path}")

# Per-class accuracy bar chart
def plot_per_class_accuracy(y_true, y_pred, categories):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    colors = ["#4CAF50" if acc >= 80 else "#FF9800" if acc >= 60 else "#F44336"
              for acc in per_class_acc]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(categories, per_class_acc, color=colors, edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.axhline(y=80, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="80% threshold")
    plt.title("Per-Class Accuracy", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 110)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path}")

# Sample predictions grid 
def plot_sample_predictions(model, X_test, y_true, y_pred, categories):
    fig, axes = plt.subplots(4, 6, figsize=(14, 10))
    fig.suptitle("Sample Predictions", fontsize=16, fontweight="bold", y=1.01)

    indices = np.random.choice(len(X_test), 24, replace=False)

    for ax, idx in zip(axes.flat, indices):
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray_r")
        true_label = categories[y_true[idx]]
        pred_label = categories[y_pred[idx]]
        correct = true_label == pred_label
        color = "#2e7d32" if correct else "#c62828"
        ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=7,
                     color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")

# Main 
def evaluate():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading model and data...")
    model, X_test, y_test, categories = load_model_and_data()

    print("Running predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    overall_acc = np.mean(y_pred == y_test) * 100
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))

    # Plots
    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred, categories)
    plot_per_class_accuracy(y_test, y_pred, categories)
    plot_sample_predictions(model, X_test, y_test, y_pred, categories)

    print(f"\nAll plots saved to /{PLOTS_DIR}/")


if __name__ == "__main__":
    evaluate()