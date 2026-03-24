"""
Integrated training script for MIL model, both bulk and non-bulk.
Supports ablation by toggling the use of image and/or ST modalities via the config YAML.

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

from dataset.loader import create_wsi_dataloader
from models.model_ablation import MultiModalMILModel

from dataset.loader_bulk import create_wsi_dataloader as create_wsi_dataloader_bulk
from models.model_bulk import MultiModalMILModel as MultiModalMILModelBulk
from train.utils import load_config, set_seed, plot_confusion_matrix, plot_acc_curve, plot_loss_curve, load_split_from_txt

# ===============================================
# Spot encoding helper (chunk-wise, ablation-aware)
# ===============================================
def encode_spots_chunkwise(model, batch, config, device):
    """
    Encode spots chunk-wise, handling ablation scenarios.
    Returns:
      spot_embeds: (N_spots, D)
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
# Bulk encoding helper (chunk-wise, ablation-aware)
# ===============================================
def forward_bulk_early_fusion_chunkwise(model, batch, config, device):
    """
    Bulk encoding with early fusion, handling ablation scenarios.
    Returns:
        logits: (num_classes,)
        mil_attn: (N_spots,) or None
        wsi_embed: (D,)
    spot_embeds: (N_spots, D) or None 
    """
    use_image = config["use_image"]
    use_st = config["use_st"]
    freeze_img = config["freeze_image_encoder"]

    spot_embeds = None
    
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

    return logits, mil_attn, wsi_embed, spot_embeds

# ===============================================
# Training / Validation
# ===============================================
def train_epoch(model, loader, criterion, optimizer, scaler, config, device, use_amp):
    """
    Training loop for one epoch, with support for gradient accumulation and mixed precision.
    """
    model.train()
    if config["freeze_image_encoder"] and config["use_image"]:
        model.img_encoder.eval()

    epoch_loss, correct = 0.0, 0
    optimizer.zero_grad()

    loop = tqdm(loader, desc="Training")
    amp_ctx = autocast() if use_amp else nullcontext()

    for step, batch in enumerate(loop):
        label = batch["label"].to(device)

        with amp_ctx:
        
            # forward -> bulk/non bulk 분기 ===========================
            if config["is_bulk"]:
                logits, _, wsi_embed = forward_bulk_early_fusion_chunkwise(
                    model, batch, config, device
                )
            else:
                # (1) chunk-wise spot encoding with modality routing
                spot_embeds = encode_spots_chunkwise(
                    model, batch, config, device
                )

                # (2) MIL + classifier (same for all ablations)
                wsi_embed, _ = model.mil_pooling(spot_embeds)
                logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
            
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss = loss / config["accum_steps"]

        # backward + optimize -> 공통 ================================
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

        del wsi_embed, logits, loss
        if not config["is_bulk"]:
            del spot_embeds

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
def validate(model, loader, criterion, config, device, use_amp,
             save_embeddings=False, embedding_dir=None):
    """
    Validation loop that computes loss, accuracy, AUC, precision/recall/f1, and confusion matrix.
    If save_embeddings=True, also saves spot and WSI embeddings for each sample in the specified directory
    """
    model.eval()
    if config["freeze_image_encoder"] and config["use_image"]:
        model.img_encoder.eval()

    val_loss, correct = 0.0, 0

    y_true = []
    y_score = []  # prob of class 1
    y_pred = []   # predicted label (0/1)

    amp_ctx = autocast() if use_amp else nullcontext()

    for batch in tqdm(loader, desc="Validation"):
        label = batch["label"].to(device)

        with amp_ctx:

            # forward -> bulk/non bulk 분기 ===========================
            if config["is_bulk"]:
                logits, _, wsi_embed, spot_embeds = forward_bulk_early_fusion_chunkwise(
                    model, batch, config, device
                )

            else:
                spot_embeds = encode_spots_chunkwise(
                    model, batch, config, device
                )
                wsi_embed, _ = model.mil_pooling(spot_embeds)
                logits = model.classifier(wsi_embed.unsqueeze(0)).squeeze(0)
            
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))             

        # save embeddings
        if save_embeddings:
            sample_id = batch["sample_id"]  # loader가 sample_id 반환해야 함

            spot_path = os.path.join(embedding_dir, "spot", f"{sample_id}.npy")
            np.save(spot_path, spot_embeds.detach().cpu().numpy())

            # wsi embedding
            wsi_path = os.path.join(embedding_dir, "wsi", f"{sample_id}.npy")
            np.save(wsi_path, wsi_embed.detach().cpu().numpy())

        val_loss += loss.item()
        pred = logits.argmax().item()
        correct += int(pred == label.item())

        # ROC/AUC
        prob_pos = torch.softmax(logits, dim=0)[1].item()
        y_true.append(label.item())
        y_score.append(prob_pos)
        y_pred.append(pred)

        del wsi_embed, logits, loss
        if not config["is_bulk"]:
            del spot_embeds

        torch.cuda.empty_cache()

    val_loss /= len(loader)
    val_acc = 100 * correct / len(loader)

    # Metrics ==========================================

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

    use_amp = (device.type == 'cuda')   # use_amp if GPU available

    results = []    # fold 전체 결과

    for fold in range(5):
        print(f"\n=== FOLD {fold} ====")

        train_samples, val_samples = load_split_from_txt(
            CONFIG["root_dir"],
            CONFIG["split_dir"],
            fold
        )

        if CONFIG["is_bulk"]:
            train_loader = create_wsi_dataloader_bulk(
                train_samples, 1, True, CONFIG["max_spots"], CONFIG["root_dir"]
            )

            val_loader = create_wsi_dataloader_bulk(
                val_samples, 1, False, CONFIG["max_spots"], CONFIG["root_dir"]
            )

            model = MultiModalMILModelBulk(
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

        else:
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
        parts = []
        if CONFIG["use_image"]:
            parts.append("img")
        if CONFIG["use_st"]:
            parts.append("st")
        if CONFIG["is_bulk"]:
            parts.append("bulk")

        tag = "_".join(parts) if parts else "none"
        
        exp_dir = os.path.join("training_outputs", f"{tag}_{CONFIG['fusion_option']}", f"fold_{fold}")
        os.makedirs(exp_dir, exist_ok=True) 
        
        embedding_root = os.path.join(exp_dir,"embeddings", "val")    # val set embedding만 저장 -> training_outputs/img_st_concat/embeddings/val
        spot_dir = os.path.join(embedding_root, "spot") # spot embedding 
        wsi_dir = os.path.join(embedding_root, "wsi")   # wsi embedding

        os.makedirs(spot_dir, exist_ok=True)
        os.makedirs(wsi_dir, exist_ok=True)

        ckpt_path = os.path.join(exp_dir, f"fold_{fold}_best_model.pt")

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

        best_val_acc = 0
        best_val_auc = 0
        best_val_f1  = 0

        for epoch in range(CONFIG["epochs"]):
            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            
            train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, scaler, CONFIG, device, use_amp
                )
            val_loss, val_acc, val_auc, val_p, val_r, val_f1, cm = validate(
                model, val_loader, criterion, CONFIG, device, use_amp
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

            # best model일 때 저장

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_val_f1  = val_f1

                torch.save(model.state_dict(), ckpt_path)

                validate(
                    model, val_loader, criterion, CONFIG, device, use_amp, save_embeddings=True, embedding_dir=embedding_root
                )

                # confusion matrix -> best model일 때만 plot
                plot_confusion_matrix(
                    cm,
                    class_names=("Healthy", "Cancer"),
                    title=f"Val Confusion Matrix (Epoch {epoch+1})",
                    save_path=os.path.join(exp_dir, f"fold_{fold}_confusion_matrix_epoch_{epoch+1}.png") # confusion_matrix_img_epoch_3.png
                )
                
                print(f"✓ Saved best model to: {ckpt_path} (val_acc={val_acc:.2f}%)")

        # plot learning curves
        plot_acc_curve(history, save_prefix=os.path.join(exp_dir, f"fold_{fold}_learning_curve"), show=False)
        plot_loss_curve(history, save_prefix=os.path.join(exp_dir, f"fold_{fold}_learning_curve"), show=False)

        results.append({
            "fold": fold,
            "val_acc": best_val_acc,
            "val_auc": best_val_auc,
            "val_f1": best_val_f1
        })

    print("\nTRAINING COMPLETE!!")

    mean_acc = np.mean([r["val_acc"] for r in results])
    mean_auc = np.mean([r["val_auc"] for r in results])
    mean_f1  = np.mean([r["val_f1"]  for r in results])

    print("\n===== FINAL RESULT =====")
    print(f"Acc: {mean_acc:.2f}")
    print(f"AUC: {mean_auc:.4f}")
    print(f"F1: {mean_f1:.4f}")


if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    main()