"""
This script trains the multimodal MIL model.
It supports ablation by toggling the use of image and/or ST modalities via the config YAML

Expected directory structure:
root_dir/
  ├── st_preprocessed_global_hvg/
  │     ├── sample1.h5ad
  │     ├── sample2.h5ad
  │     └── ...
  └── patches/
        ├── sample1.h5
        ├── sample2.h5
        └── ...
"""

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from contextlib import nullcontext

from dataset.loader import create_wsi_dataloader
from models.model_ablation import MultiModalMILModel
from train.utils import load_config, set_seed, plot_confusion_matrix, plot_acc_curve, plot_loss_curve, prepare_data_splits

# ===============================================
# Spot encoding helper (chunk-wise, ablation-aware)
# ===============================================
def encode_spots_chunkwise(model, batch, config, device):
    """
    Returns:
      spot_embeds: (N_spots, D) on GPU
    """
    use_image = config["use_image"]
    use_st = config["use_st"]
    freeze_img = config["freeze_image_encoder"]

    images = batch["images"] if use_image else None
    expr = batch["expr"] if use_st else None
    coords = batch["coords"] if use_st else None

    # N_spots: choose from whichever exists
    if use_image:
        N = images.size(0)
    else:
        N = expr.size(0)

    spot_embeds_list = []

    # use_amp = config["use_image"]
    use_amp = True
    amp_ctx = autocast() if use_amp else nullcontext()

    for i in range(0, N, config["batch_spots"]):
        j = min(i + config["batch_spots"], N)

        img_b = images[i:j].to(device) if use_image else None
        expr_b = expr[i:j].to(device) if use_st else None
        coord_b = coords[i:j].to(device) if use_st else None

        with amp_ctx:
            # ----- Image branch -----
            if use_image:
                if freeze_img:
                    with torch.no_grad():
                        img_feat = model.img_encoder(img_b)
                else:
                    img_feat = model.img_encoder(img_b)
                # img_head always trainable (exists when use_image=True)
                img_feat = model.img_head(img_feat)
            else:
                img_feat = None

            # ----- ST branch -----
            if use_st:
                # training에서는 gene_attn 필요 없으니 return_gene_attn=False로 두는 게 빠름
                st_feat = model.st_encoder(expr_b, coord_b, return_gene_attn=False) \
                    if "return_gene_attn" in model.st_encoder.forward.__code__.co_varnames \
                    else model.st_encoder(expr_b, coord_b)
            else:
                st_feat = None

            # ----- Routing -----
            if use_image and use_st:
                # multimodal
                fused = model.fusion(img_feat, st_feat)
                spot_embeds_chunk = fused
            elif use_image:
                # image-only
                spot_embeds_chunk = img_feat
            else:
                # st-only
                spot_embeds_chunk = st_feat

        # spot_embeds_list.append(spot_embeds_chunk.detach().cpu())
        spot_embeds_list.append(spot_embeds_chunk)

        # cleanup

        if use_image:
            del img_b, img_feat
        if use_st:
            del expr_b, coord_b, st_feat

        del spot_embeds_chunk
        torch.cuda.empty_cache()

    # spot_embeds = torch.cat(spot_embeds_list, dim=0).to(device)
    spot_embeds = torch.cat(spot_embeds_list, dim=0)

    return spot_embeds


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

        # (1) chunk-wise spot encoding with modality routing
        spot_embeds = encode_spots_chunkwise(model, batch, config, device)

        # (2) MIL + classifier (same for all ablations)

        with amp_ctx:
            wsi_embed, _ = model.mil_pooling(spot_embeds)
            logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
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

        del spot_embeds, wsi_embed, logits, loss
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

        spot_embeds = encode_spots_chunkwise(model, batch, config, device)

        # use_amp = config["use_image"]
        use_amp = True
        amp_ctx = autocast() if use_amp else nullcontext()
        with amp_ctx:
            wsi_embed, _ = model.mil_pooling(spot_embeds)
            logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))

        val_loss += loss.item()
        pred = logits.argmax().item()
        correct += int(pred == label.item())

        # ROC/AUC
        prob_pos = torch.softmax(logits, dim=0)[1].item()
        y_true.append(label.item())
        y_score.append(prob_pos)
        y_pred.append(pred)

        del spot_embeds, wsi_embed, logits, loss
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

        # ablation flags into model
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