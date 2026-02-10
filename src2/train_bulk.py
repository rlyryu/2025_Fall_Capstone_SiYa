import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np

import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from contextlib import nullcontext

from dataset.loader_bulk import CustomSample, create_wsi_dataloader
from models.model_bulk import MultiModalMILModel


# ===============================================
# YAML Config Loader
# ===============================================
def load_config(path="configs/train_ablation.yaml"):
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

        # ✅ Ablation flags (default: multimodal)
        "use_image": cfg["model"].get("use_image", True),
        "use_st": cfg["model"].get("use_st", True),

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


# ===============================================
# Utils
# ===============================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

def plot_confusion_matrix(
  cm, 
  class_names=('0', '1'),
  title="Confusion Matrix",
  save_path = None
):
    """
    cm: np.array shape (2, 2) [[TN, FP]. [FN, TP]]
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)

    # ticks / labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # 숫자 표시
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                fontsize=12
            )
    
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

def plot_acc_curve(history, save_prefix=None, show=False):
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

# ===============================================
# Data Split
# ===============================================
def prepare_data_splits(root_dir, seed=42):
    st_dir = os.path.join(root_dir, "st_preprocessed_global_hvg")
    patch_dir = os.path.join(root_dir, "patches")

    st_files = {f.replace('.h5ad', '') for f in os.listdir(st_dir) if f.endswith('.h5ad')}
    patch_files = {f.replace('.h5', '') for f in os.listdir(patch_dir) if f.endswith('.h5')}
    valid_ids = sorted(st_files & patch_files)

    print(f"Found {len(valid_ids)} valid samples")

    samples, labels = [], []
    for sid in valid_ids:
        try:
            sample = CustomSample(root_dir, sid)
            if sample.label in [0, 1]:
                samples.append(sample)
                labels.append(sample.label)
            else:
                print(f"⚠️  Skipping {sid} (label={sample.label})")
        except Exception as e:
            print(f"Failed to load {sid}: {e}")

    from collections import Counter
    label_counts = Counter(labels)
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} ({100*count/len(labels):.1f}%)")

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.3,
        stratify=labels,
        random_state=seed
    )
    
    # data leakage 체크
    train_ids = {s.sample_id for s in train_samples}
    val_ids   = {s.sample_id for s in val_samples}
    inter = train_ids & val_ids
    print("Overlap train/val:", len(inter))
    if len(inter) > 0:
        print("Examples:", list(sorted(inter))[:10])

    print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples


# ===============================================
# Bulk encoding helper (chunk-wise, ablation-aware)
# ===============================================
def forward_bulk_early_fusion_chunkwise(model, batch, config, device):
    """
    Returns:
      logits: (num_classes,)
      mil_attn: (N_spots,) or None
      gene_attn, gene_indices: optional (bulk)
    """
    use_image = config["use_image"]
    use_st = config["use_st"]
    freeze_img = config["freeze_image_encoder"]

    images = batch["images"] if use_image else None          # (N,3,224,224)
    expr_wsi = batch["expr"] if use_st else None             # (K,)

    if use_st:
        # make sure expr_wsi is (K,)
        if expr_wsi.dim() == 2 and expr_wsi.size(0) == 1:
            expr_wsi = expr_wsi.squeeze(0)
            
    # bulk ST branch
    gene_attn = gene_indices = None
    if use_st:
        expr_wsi = expr_wsi.to(device)
        if "return_gene_attn" in model.st_encoder.forward.__code__.co_varnames:
            st_wsi = model.st_encoder(expr_wsi, return_gene_attn=False)  # training: False
        else:
            st_wsi = model.st_encoder(expr_wsi)
    else:
        st_wsi = None

    # image branch: chunkwise encode spots -> MIL
    if use_image:
        N = images.size(0)
        spot_list = []

        use_amp = True
        amp_ctx = autocast() if use_amp else nullcontext()

        for i in range(0, N, config["batch_spots"]):
            j = min(i + config["batch_spots"], N)
            img_b = images[i:j].to(device)

            with amp_ctx:
                if freeze_img:
                    with torch.no_grad():
                        img_feat = model.img_encoder(img_b)
                else:
                    img_feat = model.img_encoder(img_b)
                img_feat = model.img_head(img_feat)

            spot_list.append(img_feat)
            del img_b, img_feat
            torch.cuda.empty_cache()

        spot_embeds = torch.cat(spot_list, dim=0)  # (N,D)

        with amp_ctx:
            img_wsi, mil_attn = model.mil_pooling(spot_embeds)
            mil_attn = mil_attn.squeeze(-1)  # (N,)
        del spot_embeds
    else:
        img_wsi, mil_attn = None, None

    # WSI-level early fusion & classifier
    use_amp = True
    amp_ctx = autocast() if use_amp else nullcontext()
    with amp_ctx:
        if use_image and use_st:
            wsi_embed = model.wsi_fusion(img_wsi, st_wsi)
        elif use_image:
            wsi_embed = img_wsi
        else:
            wsi_embed = st_wsi

        logits = model.classifier(wsi_embed)

    return logits, mil_attn

# ===============================================
# Training / Validation
# ===============================================
def train_epoch(model, loader, criterion, optimizer, scaler, config, device):
    model.train()
    if config["freeze_image_encoder"] and config["use_image"]:
        model.img_encoder.eval()

    epoch_loss, correct = 0.0, 0
    optimizer.zero_grad()

    loop = tqdm(loader, desc="Training")

    # use_amp = config["use_image"]
    use_amp = True
    amp_ctx = autocast() if use_amp else nullcontext()

    for step, batch in enumerate(loop):
        label = batch["label"].to(device)
        
        with amp_ctx:
            logits, _ = forward_bulk_early_fusion_chunkwise(model, batch, config, device)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss = loss / config["accum_steps"]

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % config["accum_steps"] == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * config["accum_steps"]
        correct += int(logits.argmax().item() == label.item())

        loop.set_postfix(
            loss=f"{epoch_loss/(step+1):.4f}",
            acc=f"{100*correct/(step+1):.1f}%"
        )

        del logits, loss
        torch.cuda.empty_cache()
        
    # after loop ends: flush remaining grads once
    if (step + 1) % config["accum_steps"] != 0:
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 1.0
        )
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        
    return epoch_loss / len(loader), 100 * correct / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, config, device):
    model.eval()
    if config["freeze_image_encoder"] and config["use_image"]:
        model.img_encoder.eval()

    val_loss, correct = 0.0, 0

    y_true = []
    y_score = []  # prob of class 1
    y_pred = []   # predicted label (0/1)

    for batch in tqdm(loader, desc="Validation"):
        label = batch["label"].to(device)

        # use_amp = config["use_image"]
        use_amp = True
        amp_ctx = autocast() if use_amp else nullcontext()
        with amp_ctx:
            logits, _ = forward_bulk_early_fusion_chunkwise(model, batch, config, device)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))

        val_loss += loss.item()
        pred = logits.argmax().item()
        correct += int(pred == label.item())

        # ROC/AUC
        prob_pos = torch.softmax(logits, dim=0)[1].item()
        y_true.append(label.item())
        y_score.append(prob_pos)
        y_pred.append(pred)

        del logits, loss
        torch.cuda.empty_cache()

    val_loss = val_loss / len(loader)
    val_acc = 100 * correct / len(loader)

    # AUC
    auc = float('nan')
    try:
      auc = roc_auc_score(y_true, y_score)
      # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    except Exception:
      pass

    # precision/recall/f1 (class 1=positive로)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    p, r, f1 = float(p), float(r), float(f1)

    # confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels = [0, 1])

    return val_loss, val_acc, auc, p, r, f1, cm

# ===============================================
# Main
# ===============================================
def main():
    CONFIG = load_config("configs/train_ablation.yaml")
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])

    print("="*70)
    print("MIL Training (Ablation-ready)")
    print("="*70)
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print("="*70 + "\n")

    train_samples, val_samples = prepare_data_splits(
        CONFIG["root_dir"], CONFIG["seed"]
    )

    train_loader = create_wsi_dataloader(
        train_samples, 1, True, CONFIG["max_spots"], CONFIG["root_dir"]
    )
    val_loader = create_wsi_dataloader(
        val_samples, 1, False, CONFIG["max_spots"], CONFIG["root_dir"]
    )

    model = MultiModalMILModel(
        num_genes=CONFIG["num_genes"],
        num_classes=CONFIG["num_classes"],
        embed_dim=CONFIG["embed_dim"],
        fusion_option=CONFIG["fusion_option"],
        top_k_genes=CONFIG.get("top_k_genes"),

        # ✅ ablation flags into model
        use_image=CONFIG["use_image"],
        use_st=CONFIG["use_st"],

        # keep this behavior consistent
        freeze_image_encoder=CONFIG["freeze_image_encoder"],
    ).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = 0.0

    # checkpoint name by ablation setting (helpful)
    tag = ("img" if CONFIG["use_image"] else "") + ("st" if CONFIG["use_st"] else "")
    tag = tag if tag else "none"
    ckpt_path = f"best_model_{tag}_{CONFIG['fusion_option']}.pt"

    # history for learning curves
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        # auc + prf1은 안 해도 될 듯
        "val_auc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, CONFIG, device
        )
        val_loss, val_acc, val_auc, val_p, val_r, val_f1, cm = validate(
            model, val_loader, criterion, CONFIG, device
        )

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_precision"].append(val_p)
        history["val_recall"].append(val_r)
        history["val_f1"].append(val_f1)

        print(
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f} | "
            f"P/R/F1: {val_p:.3f}/{val_r:.3f}/{val_f1:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

            # confusion matrix -> best model일 때만 plot
            plot_confusion_matrix(
                cm,
                class_names=("0", "1"),   # Healthy / Cancer면 바꿔도 됨
                title=f"Val Confusion Matrix (Epoch {epoch+1})",
                save_path=f"confusion_matrix_epoch_{epoch+1}.png"
            )
            
            print(f"✓ Saved best model to: {ckpt_path} (val_acc={val_acc:.2f}%)")

    # plot learning curves
    plot_acc_curve(history, save_prefix=f"learning_curve_{tag}", show=False)
    plot_loss_curve(history, save_prefix=f"learning_curve_{tag}", show=False)
    
    print("\nTRAINING COMPLETE!!")


if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    main()