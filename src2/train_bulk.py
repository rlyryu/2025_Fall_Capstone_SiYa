# train_bulk.py
# -------------------------------------------------------
# Minimal-modification bulk-RNAseq training script
# - Keeps your existing WSI patch pipeline + MIL
# - Replaces ST expr/coords with per-sample bulk expression
# - Bulk is injected per chunk (no giant (N_spots,K) allocation)
#
# Expected bulk file location (choose ONE and place files accordingly):
#   1) {root_dir}/bulk_expr/{sample_id}.npy   (shape: (K,))
#   2) {root_dir}/bulk_expr/{sample_id}.pt    (torch tensor shape: (K,))
#   3) {root_dir}/bulk_expr/{sample_id}.csv   (either "gene,expr" or single-row numeric)
#
# Notes:
# - coords are dummy zeros (no true spatial meaning in bulk)
# - st_encoder is reused as-is; spatial token exists but coords are zeros
# -------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score,
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

from dataset.loader_bulk import CustomSample, create_wsi_dataloader
from models.model_bulk import MultiModalMILModel


# ===============================================
# YAML Config Loader
# ===============================================
def load_config(path="configs/train_bulk.yaml"):
    """
    Minimal changes vs train.yaml:
    - add cfg["data"]["bulk_dir"] (default: "{root_dir}/bulk_expr")
    - optionally add cfg["data"]["bulk_ext"] in {npy, pt, csv} (default: auto)
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    root_dir = cfg["data"]["root_dir"]
    bulk_dir = cfg["data"].get("bulk_dir", os.path.join(root_dir, "bulk_processed"))

    CONFIG = {
        # Data
        "root_dir": root_dir,
        "max_spots": cfg["data"]["max_spots"],
        "bulk_dir": bulk_dir,
        "bulk_ext": cfg["data"].get("bulk_ext", "auto"),  # "auto" | "npy" | "pt" | "csv"

        # Model
        "num_genes": cfg["model"]["num_genes"],
        "num_classes": cfg["model"]["num_classes"],
        "embed_dim": cfg["model"]["embed_dim"],
        "fusion_option": cfg["model"]["fusion_option"],
        "top_k_genes": cfg["model"].get("top_k_genes"),

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
    return CONFIG


# ===============================================
# Utils
# ===============================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def plot_confusion_matrix(
    cm,
    class_names=('0', '1'),
    title="Confusion Matrix",
    save_path=None
):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def _find_bulk_path(bulk_dir: str, sample_id: str, bulk_ext: str):
    """
    Returns a file path for the sample_id, or None if not found.
    """
    if bulk_ext != "auto":
        cand = os.path.join(bulk_dir, f"{sample_id}.{bulk_ext}")
        return cand if os.path.exists(cand) else None

    ext = "h5ad"
    cand = os.path.join(bulk_dir, f"{sample_id}.{ext}")
    if os.path.exists(cand):
        return cand
    return None

def load_bulk_expr_vector(bulk_dir: str, sample_id: str, num_genes: int, bulk_ext: str = "auto") -> torch.Tensor:
    """
    Loads bulk expression vector for one sample_id.
    Returns: torch.FloatTensor shape (num_genes,)
    """
    path = _find_bulk_path(bulk_dir, sample_id, bulk_ext)
    if path is None:
        raise FileNotFoundError(f"Bulk expr not found for {sample_id} in {bulk_dir} (ext={bulk_ext})")

    if path.endswith(".h5ad"):
        import scanpy as sc
        adata = sc.read_h5ad(path)

        # expr 벡터 만들기: (K,)
        X = adata.X
        if hasattr(X, "toarray"):   # sparse
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)

        # 흔한 케이스들 처리:
        # 1) (1, K): 이미 bulk 1개 샘플
        if X.ndim == 2 and X.shape[0] == 1:
            x = X[0]
        # 2) (N_cells, K): pseudo-bulk면 sum/mean으로 collapse
        elif X.ndim == 2 and X.shape[0] > 1:
            # 보통 bulk면 sum 또는 mean 둘 다 가능. "표현량"이면 sum이 더 자연스럽고,
            # normalize된 값이면 mean이 더 안전. 여기선 mean을 기본으로.
            x = X.mean(axis=0)
        else:
            raise ValueError(f"Unexpected adata.X shape for bulk h5ad: {X.shape}")

        x = np.asarray(x, dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"Unsupported bulk file: {path}")

    if x.shape[0] != num_genes:
        raise ValueError(f"Bulk vector length mismatch for {sample_id}: got {x.shape[0]}, expected {num_genes}")

    return torch.from_numpy(x)  # (K,)


# ===============================================
# Data Split
# ===============================================
def prepare_data_splits(root_dir, bulk_dir, num_genes, bulk_ext="auto", seed=42):
    """
    Minimal change:
    - Valid sample = has patches + has bulk expr file
    - ST dir existence is not required for bulk training (but your CustomSample may require it).
      If your CustomSample currently requires .h5ad, keep st_dir check as well.
    """

    # Keep your original intersection logic so CustomSample keeps working.
    # If you later remove ST dependency from CustomSample, you can drop st_dir part.
    st_dir = os.path.join(root_dir, "st_preprocessed_global_hvg")
    patch_dir = os.path.join(root_dir, "patches")

    st_files = {f.replace('.h5ad', '') for f in os.listdir(st_dir) if f.endswith('.h5ad')}
    patch_files = {f.replace('.h5', '') for f in os.listdir(patch_dir) if f.endswith('.h5')}
    candidate_ids = sorted(st_files & patch_files)

    print("[DEBUG] st_dir:", st_dir, "exists:", os.path.isdir(st_dir))
    print("[DEBUG] patch_dir:", patch_dir, "exists:", os.path.isdir(patch_dir))
    print("[DEBUG] bulk_dir:", bulk_dir, "exists:", os.path.isdir(bulk_dir))

    print("[DEBUG] st files:", len(st_files), "patch files:", len(patch_files), "candidate:", len(candidate_ids))
    if len(candidate_ids) > 0:
        sid = candidate_ids[0]
        print("[DEBUG] example sid:", sid, "bulk path:", _find_bulk_path(bulk_dir, sid, bulk_ext))

    # filter by bulk existence
    valid_ids = []
    for sid in candidate_ids:
        if _find_bulk_path(bulk_dir, sid, bulk_ext) is not None:
            valid_ids.append(sid)

    print(f"Found {len(valid_ids)} valid samples (patch + ST + bulk)")

    samples, labels = [], []
    for sid in valid_ids:
        try:
            sample = CustomSample(root_dir, sid)
            # Ensure bulk vector is loadable (catch early)
            _ = load_bulk_expr_vector(bulk_dir, sid, num_genes=num_genes, bulk_ext=bulk_ext)

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

    print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples


# ===============================================
# Training / Validation
# ===============================================
def train_epoch(model, loader, criterion, optimizer, scaler, config, device):
    model.train()
    if config["freeze_image_encoder"]:
        model.img_encoder.eval()

    epoch_loss, correct = 0.0, 0
    optimizer.zero_grad()

    loop = tqdm(loader, desc="Training")

    for step, batch in enumerate(loop):
        images = batch["images"]          # (N_spots, 3, 224, 224) on CPU
        label = batch["label"].to(device) # scalar
        sample_id = batch.get("sample_id", None)
        if sample_id is None:
            raise KeyError("Loader must provide batch['sample_id'] for bulk lookup.")
        if isinstance(sample_id, (list, tuple)):
            # In case loader returns list; batch_size=1 so take [0]
            sample_id = sample_id[0]

        # Load bulk vector once per sample
        bulk_vec = load_bulk_expr_vector(
            bulk_dir=config["bulk_dir"],
            sample_id=sample_id,
            num_genes=config["num_genes"],
            bulk_ext=config["bulk_ext"],
        ).to(device)  # (K,)

        N = images.size(0)
        spot_img_list = []

        for i in range(0, N, config["batch_spots"]):
            j = min(i + config["batch_spots"], N)
            img_b = images[i:j].to(device)
            with autocast():
                if config["freeze_image_encoder"]:
                    with torch.no_grad():
                        img_feat = model.img_encoder(img_b)
                else:
                    img_feat = model.img_encoder(img_b)
                img_feat = model.img_head(img_feat)
                

            spot_img_list.append(img_feat.detach().cpu())

        torch.cuda.empty_cache()

        img_feat_all = torch.cat(spot_img_list, dim=0).to(device)

        with autocast():
            outputs = model(img_feat_all, bulk_vec, return_gene_attn=False, return_spot_embeds=False)
            logits = outputs["logits"]
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss = loss / config["accum_steps"]
        scaler.scale(loss).backward()

        if (step + 1) % config["accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * config["accum_steps"]
        correct += int(logits.argmax().item() == label.item())

        loop.set_postfix(
            loss=f"{epoch_loss/(step+1):.4f}",
            acc=f"{100*correct/(step+1):.1f}%"
        )

        del img_feat_all, outputs, logits, loss, bulk_vec, spot_img_list
    torch.cuda.empty_cache()

    return epoch_loss / len(loader), 100 * correct / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, config, device):
    model.eval()
    val_loss, correct = 0.0, 0

    y_true = []
    y_score = []
    y_pred = []

    for batch in tqdm(loader, desc="Validation"):
        images = batch["images"]
        label = batch["label"].to(device)
        sample_id = batch.get("sample_id", None)
        if sample_id is None:
            raise KeyError("Loader must provide batch['sample_id'] for bulk lookup.")
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0]

        bulk_vec = load_bulk_expr_vector(
            bulk_dir=config["bulk_dir"],
            sample_id=sample_id,
            num_genes=config["num_genes"],
            bulk_ext=config["bulk_ext"],
        ).to(device)  # (K,)

        N = images.size(0)
        spot_img_list = []

        for i in range(0, N, config["batch_spots"]):
            j = min(i + config["batch_spots"], N)
            img_b = images[i:j].to(device)
            
            with autocast():
                if config["freeze_image_encoder"]:
                    with torch.no_grad():
                        img_feat = model.img_encoder(img_b)
                else:
                    img_feat = model.img_encoder(img_b)

                img_feat = model.img_head(img_feat)

            spot_img_list.append(img_feat.detach().cpu())

        img_feat_all = torch.cat(spot_img_list, dim=0).to(device)

        with autocast():
            outputs = model(img_feat_all, bulk_vec, return_gene_attn=False, return_spot_embeds=False)
            logits = outputs["logits"]
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))

        val_loss += loss.item()
        pred = logits.argmax().item()
        correct += int(pred == label.item())

        prob_pos = torch.softmax(logits, dim=0)[1].item()
        y_true.append(label.item())
        y_score.append(prob_pos)
        y_pred.append(pred)

        del bulk_vec, img_feat_all, outputs, logits, loss, spot_img_list
    torch.cuda.empty_cache()

    val_loss = val_loss / len(loader)
    val_acc = 100 * correct / len(loader)

    auc = float('nan')
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        pass

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    p, r, f1 = float(p), float(r), float(f1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return val_loss, val_acc, auc, p, r, f1, cm


# ===============================================
# Main
# ===============================================
def main():
    CONFIG = load_config("configs/train_bulk.yaml")
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])

    print("=" * 70)
    print("WSI + Bulk RNA-seq MIL Training")
    print("=" * 70)
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")

    # data split (now includes bulk existence check)
    train_samples, val_samples = prepare_data_splits(
        root_dir=CONFIG["root_dir"],
        bulk_dir=CONFIG["bulk_dir"],
        num_genes=CONFIG["num_genes"],
        bulk_ext=CONFIG["bulk_ext"],
        seed=CONFIG["seed"],
    )

    train_loader = create_wsi_dataloader(
        train_samples, 1, True, CONFIG["max_spots"], CONFIG["root_dir"]
    )
    val_loader = create_wsi_dataloader(
        val_samples, 1, False, CONFIG["max_spots"], CONFIG["root_dir"]
    )

    # model: keep your original multimodal architecture
    model = MultiModalMILModel(
        num_genes=CONFIG["num_genes"],
        num_classes=CONFIG["num_classes"],
        embed_dim=CONFIG["embed_dim"],
        fusion_option=CONFIG["fusion_option"],
        top_k_genes=CONFIG.get("top_k_genes"),
    ).to(device)

    if CONFIG["freeze_image_encoder"]:
        for p in model.img_encoder.parameters():
            p.requires_grad = False
        model.img_encoder.eval()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, CONFIG, device
        )
        val_loss, val_acc, val_auc, val_p, val_r, val_f1, cm = validate(
            model, val_loader, criterion, CONFIG, device
        )

        print(
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f} | "
            f"P/R/F1: {val_p:.3f}/{val_r:.3f}/{val_f1:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_bulk.pt")

            plot_confusion_matrix(
                cm,
                class_names=("0", "1"),
                title=f"Val Confusion Matrix (Epoch {epoch + 1})",
                save_path=f"confusion_matrix_bulk_epoch_{epoch + 1}.png"
            )

    print("\nTRAINING COMPLETE!!")


if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    main()