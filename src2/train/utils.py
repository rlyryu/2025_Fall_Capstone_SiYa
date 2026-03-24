"""
Utility functions for training and evaluation, including:
- Configuration loading
- Random seed setting
- Plotting confusion matrices, accuracy/loss curves
- Preparing train/validation splits from the dataset
"""

import yaml
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from dataset.loader import CustomSample
from dataset.loader_bulk import CustomSample as CustomSampleBulk

def load_config(path="configs/train_ablation.yaml"):
    """
    Load training configuration from a YAML file.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    CONFIG = {
        # Data
        "root_dir": cfg["data"]["root_dir"],
        "max_spots": cfg["data"]["max_spots"],

        # Model
        "num_genes": cfg["model"]["num_genes"],
        "num_classes": cfg["model"]["num_classes"],
        "embed_dim": cfg["model"]["embed_dim"],
        "fusion_option": cfg["model"].get("fusion_option", "concat"),
        "top_k_genes": cfg["model"].get("top_k_genes"),

        # Ablation flags (default: multimodal)
        "use_image": cfg["model"].get("use_image", True),
        "use_st": cfg["model"].get("use_st", True),
        "is_bulk": cfg["model"].get("is_bulk", False),

        # Training
        "epochs": cfg["training"]["epochs"],
        "lr": cfg["training"]["lr"],
        "weight_decay": cfg["training"]["weight_decay"],
        "batch_size": cfg["training"]["batch_size"],

        # Memory
        "batch_spots": cfg["memory"]["batch_spots"],
        "accum_steps": cfg["memory"]["accum_steps"],
        "freeze_image_encoder": cfg["memory"]["freeze_image_encoder"],

        # Misc
        "device": cfg["misc"]["device"],
        "seed": cfg["misc"]["seed"],
        "checkpoint_freq": cfg["misc"]["checkpoint_freq"],
    }

    assert CONFIG["use_image"] or CONFIG["use_st"], "At least one modality must be enabled"
    return CONFIG

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

def plot_confusion_matrix(cm, class_names=('0', '1'), title="Confusion Matrix", save_path = None):
    """
    Plot confusion matrix with counts and save/show.
    Args:
        cm: 2D array-like confusion matrix (e.g., [[TN, FP], [FN, TP]])
        class_names: Tuple of class names for axes labels
        title: Title of the plot
        save_path: If provided, saves the plot to this path. Otherwise, shows it.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    # ticks / labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                fontsize=12,
                color="white" if cm[i, j] > threshold else "black"
            )
    
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

def plot_acc_curve(history, save_prefix=None, show=False):
    """
    Plot training and validation accuracy curves.
    Args:
        history: Dict containing "train_acc" and "val_acc" lists.
        save_prefix: If provided, saves the plot with this prefix. Otherwise, shows it.
        show: If True, displays the plot. If False, saves and closes it.
    """
    epochs = np.arange(1, len(history["train_acc"])+1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_acc"], label="Train Acc")
    ax.plot(epochs, history["val_acc"], label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training/Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_acc_curve.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_loss_curve(history, save_prefix=None, show=False):
    """
    Plot training and validation loss curves.
    Args:
        history: Dict containing "train_loss" and "val_loss" lists.
        save_prefix: If provided, saves the plot with this prefix. Otherwise, shows it.
        show: If True, displays the plot. If False, saves and closes it.
    """
    epochs = np.arange(1, len(history["train_loss"])+1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training/Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_loss_curve.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

def prepare_data_splits(root_dir, seed=42, is_bulk=False):
    """
    Prepare train/validation splits based on available samples in the dataset.
    
    Returns:
        train_samples: List of CustomSample objects for training
        val_samples: List of CustomSample objects for validation
    """
    st_dir = os.path.join(root_dir, "st_preprocessed_global_hvg")
    patch_dir = os.path.join(root_dir, "patches")

    st_files = {f.replace('.h5ad', '') for f in os.listdir(st_dir) if f.endswith('.h5ad')}
    patch_files = {f.replace('.h5', '') for f in os.listdir(patch_dir) if f.endswith('.h5')}
    valid_ids = sorted(st_files & patch_files)

    print(f"Found {len(valid_ids)} valid samples")

    samples, labels = [], []
    for sid in valid_ids:
        try:
            if is_bulk:
                sample = CustomSampleBulk(root_dir, sid)
            else:
                sample = CustomSample(root_dir, sid)
            if sample.label in [0, 1]:
                samples.append(sample)
                labels.append(sample.label)
            else:
                print(f"Skipping {sid} (label={sample.label})")
        except Exception as e:
            print(f"Failed to load {sid}: {e}")

    from collections import Counter
    label_counts = Counter(labels)
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} ({100*count/len(labels):.1f}%)")

    # train/val/test split
    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.4,  # train:val:test=6:2:2
        stratify=labels,
        random_state=seed
    )

    """
    val_samples, test_samples = train_test_split(
        val_samples,
        test_size=0.5,  # val:test = 1:1
        stratify=[s.label for s in val_samples],
        random_state=seed
    )
    """
    
    # data leakage 체크
    train_ids = {s.sample_id for s in train_samples}
    val_ids   = {s.sample_id for s in val_samples}
    inter = train_ids & val_ids
    print("Overlap train/val:", len(inter))
    if len(inter) > 0:
        print("Examples:", list(sorted(inter))[:10])

    print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples

def load_split_from_txt(root_dir, split_dir, fold):
    """
    A custom split script: supports a cross-validation by reading a fixed dataset id file.
    """
    fold_dir = os.path.join(split_dir, f"fold_{fold}")

    def read_ids(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    train_ids = read_ids(os.path.join(fold_dir, "train.txt"))
    val_ids   = read_ids(os.path.join(fold_dir, "val.txt"))

    train_samples = [CustomSample(root_dir, sid) for sid in train_ids]
    val_samples   = [CustomSample(root_dir, sid) for sid in val_ids]

    return train_samples, val_samples