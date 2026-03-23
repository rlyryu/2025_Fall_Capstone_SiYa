import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from dataset.loader_bulk import CustomSample, create_wsi_dataloader, load_global_gene_order
from models.model_bulk import MultiModalMILModel

"""
A bulk ver. of test.py
"""

# Detect samples -> return
def discover_samples(root_dir):
    st_dir = os.path.join(root_dir, "st_preprocessed_global_hvg")
    patch_dir = os.path.join(root_dir, "patches")

    assert os.path.isdir(st_dir), f"ST dir not found: {st_dir}"
    assert os.path.isdir(patch_dir), f"Patch dir not found: {patch_dir}"

    sample_ids = []
    for fn in os.listdir(st_dir):
        if fn.endswith(".h5ad"):
            sid = fn[:-5]
            if os.path.exists(os.path.join(patch_dir, f"{sid}.h5")):
                sample_ids.append(sid)
    sample_ids.sort()

    samples = [CustomSample(root_dir, sid) for sid in sample_ids]
    return samples

# UMAP/PCA util
def compute_2d_embedding(X: np.ndarray, method: str = "umap", seed: int = 0):
    """
    X: (N, D)
    returns: (N, 2)
    """
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed)
            return reducer.fit_transform(X)
        except Exception as e:
            print(f"[warn] UMAP not available ({e}). Falling back to PCA.")
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)

    raise ValueError(f"Unknown method: {method}")

# top percent attn extraction util
def make_top_percent_mask(attn: torch.Tensor, top_percent: float = 0.6, min_points: int = 10):
    """
    attn: (N,) torch.Tensor
    top_percent: keep top 70% => 0.7
    min_points: safety fallback (avoid empty / too few)
    returns: mask (N,) bool torch.Tensor
    """
    attn = attn.view(-1)
    N = attn.numel()
    if N == 0:
        return torch.zeros_like(attn, dtype=torch.bool)

    # Apply threshold -> top 70%
    q = 1.0 - float(top_percent)
    q = min(max(q, 0.0), 1.0)

    thr = torch.quantile(attn, q)
    mask = attn >= thr

    # fallback: top-k
    if mask.sum().item() < min_points:
        k = min(min_points, N)
        idx = torch.topk(attn, k=k).indices
        mask = torch.zeros(N, dtype=torch.bool, device=attn.device)
        mask[idx] = True

    return mask

# IO utils
def save_patch_image(tensor_chw, out_path):
    """
    tensor_chw: torch.Tensor (3,H,W), range ~[0,1]
    """
    x = tensor_chw.detach().cpu().clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(x).save(out_path)

# Patches with high attn score -> scatter plot
def plot_attention_scatter(coords_raw, attn, top10_idx, out_path,
                           title="Spot importance (MIL attn)",
                           mask=None):
    """
    coords_raw: (N,2) torch.Tensor
    attn: (N,) torch.Tensor
    mask: (N,) bool torch.Tensor (True만 plot)
    """
    c_all = coords_raw.detach().cpu().numpy()
    a_all = attn.detach().cpu().numpy()

    if mask is not None:
        m = mask.detach().cpu().numpy().astype(bool)
    else:
        m = np.ones(len(a_all), dtype=bool)

    c = c_all[m]
    a = a_all[m]

    if len(a) == 0:
        plt.figure()
        plt.title(title + " (empty after masking)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    a_min, a_max = float(a.min()), float(a.max())
    denom = (a_max - a_min) if (a_max - a_min) > 1e-12 else 1.0
    a_n = (a - a_min) / denom
    sizes = 10 + 200 * a_n

    plt.figure()
    plt.scatter(c[:, 0], c[:, 1], s=sizes)  # 백지 위 scatter

    if top10_idx is not None and len(top10_idx) > 0:
        sel = np.array(top10_idx, dtype=np.int64)
        sel = sel[sel < len(m)]          # boundary safety
        sel = sel[m[sel]]                # mask 통과한 top10만 남김
        if len(sel) > 0:
            plt.scatter(c_all[sel, 0], c_all[sel, 1], s=250, marker="x")
            for rank, i in enumerate(sel.tolist(), start=1):
                plt.text(c_all[i, 0], c_all[i, 1], f"Top{rank}", fontsize=10)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# Important gene list
def aggregate_top_genes(gene_attn, gene_indices, mil_attn, gene_order, topk=30):
    """
    gene_attn: (N, G) torch.Tensor (per-spot attention over gene tokens)
    gene_indices: (N, G) torch.LongTensor
    mil_attn: (N,) torch.Tensor  (spot importance)
    gene_order: list[str] length K_global
    """
    N, G = gene_attn.shape
    mil_w = mil_attn.view(N, 1)

    contrib = (gene_attn * mil_w).detach().cpu().numpy()  # (N,G)
    gidx = gene_indices.detach().cpu().numpy().astype(np.int64)

    scores = {}  # gene_id -> sum score
    for i in range(N):
        for j in range(G):
            gid = int(gidx[i, j])
            scores[gid] = scores.get(gid, 0.0) + float(contrib[i, j])

    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    out = []
    for gid, sc in items:
        gname = gene_order[gid] if (0 <= gid < len(gene_order)) else f"gene_{gid}"
        out.append((gname, sc))
    return out

def aggregate_top_genes_bulk(gene_attn, gene_indices, gene_order, topk=30):
    """
    gene_attn: (K_used,) or (1,1,K_used)
    gene_indices: (K_used,)  (global gene ids)
    """
    if gene_attn.dim() == 3:
        gene_attn = gene_attn.squeeze(0).squeeze(0)
    gene_attn = gene_attn.view(-1).detach().cpu().numpy()

    gidx = gene_indices.view(-1).detach().cpu().numpy().astype(np.int64)

    items = sorted(list(zip(gidx, gene_attn)), key=lambda x: x[1], reverse=True)[:topk]
    
    out = []
    for gid, sc in items:
        gname = gene_order[gid] if (0 <= gid < len(gene_order)) else f"gene_{gid}"
        out.append((gname, float(sc)))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./xai_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model params
    parser.add_argument("--num_genes", type=int, default=2000)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--fusion_option", type=str, default="concat")
    parser.add_argument("--top_k_genes", type=int, default=512)
    parser.add_argument("--freeze_image_encoder", action="store_true")

    # 추가: ablation flags
    parser.add_argument("--use_image", action="store_true", help="Enable image modality")
    parser.add_argument("--use_st", action="store_true", help="Enable ST modality")

    # data params
    parser.add_argument("--max_spots", type=int, default=2000)
    parser.add_argument("--topk_patches", type=int, default=12)
    parser.add_argument("--topk_genes", type=int, default=30)
    parser.add_argument("--embed_2d", type=str, default="umap", choices=["umap", "pca"])

    # for local debugging
    args = parser.parse_args(args=[
        "--root_dir", r"C:\Users\rdh08\Desktop\Capstone\src2\hest_data",
        "--ckpt", r"C:\Users\rdh08\Desktop\Capstone\src2\best_model_bulk_concat.pt",
        "--use_image",
        "--use_st",
    ])

    # default=multimodal
    if (not args.use_image) and (not args.use_st):
        args.use_image = True
        args.use_st = True

    assert args.use_image or args.use_st, "At least one modality must be enabled"

    if args.use_image and args.use_st:
        print("Running in MULTIMODAL mode (image + ST)")
        mode_suffix = "st_img_bulk"
    elif args.use_image and not args.use_st:
        print("Running in IMAGE-ONLY mode")
        mode_suffix = "img"
    elif not args.use_image and args.use_st:
        print("Running in ST-ONLY mode")
        mode_suffix = "st"
    else:
        raise ValueError("Invalid modality setting")
    
    args.out_dir = f"{args.out_dir}_{mode_suffix}"
    os.makedirs(args.out_dir, exist_ok=True)

    # samples + loader
    samples = discover_samples(args.root_dir)
    if len(samples) == 0:
        raise RuntimeError("No samples discovered. Check root_dir structure.")
    loader = create_wsi_dataloader(
        samples,
        batch_size=1,
        shuffle=False,
        max_spots=args.max_spots,
        root_dir=args.root_dir,
        return_trace=True,
    )

    # gene order (fallback)
    global_gene_order = load_global_gene_order(args.root_dir)
    if global_gene_order is None:
        global_gene_order = []

    # model
    model = MultiModalMILModel(
        num_genes=args.num_genes,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        fusion_option=args.fusion_option,
        top_k_genes=args.top_k_genes,
        freeze_image_encoder=args.freeze_image_encoder,

        # ablation flags into model
        use_image=args.use_image,
        use_st=args.use_st,
    ).to(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing sample:", missing[:30])
    print("unexpected sample:", unexpected[:30])

    print(f"[load] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.eval()

    with torch.inference_mode():
        for batch in loader:
            sample_id = batch["sample_id"]
            out_dir = os.path.join(args.out_dir, sample_id)
            os.makedirs(out_dir, exist_ok=True)

            label = int(batch["label"].item())

            print(f"Processing: {sample_id}")
            
            # modality별로 필요한 것만 GPU로
            images = batch["images"].to(args.device) if args.use_image else None
            expr   = batch["expr"].to(args.device)   if args.use_st else None
            coords = None

            # XAI metadata
            barcodes = batch.get("barcodes", None)            # list[str]
            patch_indices = batch.get("patch_indices", None)  # np.ndarray
            coords_raw = batch.get("coords_raw", None)        # np.ndarray
            gene_order = batch.get("gene_order", global_gene_order)

            # gene attn - only for ST
            outputs = model(
                images,
                expr,
                return_gene_attn=bool(args.use_st),
                return_spot_embeds=True
            )

            # ---- unpack ----
            if not isinstance(outputs, dict):
                raise ValueError("Model must return a dict with keys: logits, mil_attn, spot_embeds, ...")

            logits = outputs.get("logits", None)
            mil_attn = outputs.get("mil_attn", None)
            spot_embeds = outputs.get("spot_embeds", None)

            gene_attn = outputs.get("gene_attn", None)
            gene_indices = outputs.get("gene_indices", None)

            if logits is None:
                raise ValueError("Model dict output must contain 'logits'.")

            probs = F.softmax(logits, dim=-1).detach().cpu().numpy().tolist()
            pred = int(np.argmax(probs))

            # n_spots: 
            if args.use_image:
                n_spots = int(images.shape[0])
            else:
                n_spots = int(expr.shape[0])

            # summary 
            summary = {
                "sample_id": sample_id,
                "gt_label": label,
                "pred_label": pred,
                "probs": probs,
                "num_spots_used": n_spots,
                "use_image": bool(args.use_image),
                "use_st": bool(args.use_st),
                "mil_attn_available": mil_attn is not None,
                "spot_embeds_available": spot_embeds is not None,
                "gene_xai_available": (gene_attn is not None and gene_indices is not None),
            }
            with open(os.path.join(out_dir, "pred.json"), "w") as f:
                json.dump(summary, f, indent=2)

            if mil_attn is None:
                continue

            mil_attn = mil_attn.view(-1)

            # Top-10 important spots
            top10 = torch.topk(mil_attn, k=min(10, n_spots)).indices.detach().cpu().tolist()

            # UMAP/PCA
            if spot_embeds is not None:
                X = spot_embeds.detach().cpu().numpy()
                Z = compute_2d_embedding(X, method=args.embed_2d, seed=0)

                a = mil_attn.detach().cpu().numpy()
                a_min, a_max = float(a.min()), float(a.max())
                denom = (a_max - a_min) if (a_max - a_min) > 1e-12 else 1.0
                a_n = (a - a_min) / denom

                plt.figure()
                plt.scatter(Z[:, 0], Z[:, 1], s=10 + 200 * a_n)
                plt.title(f"{args.embed_2d.upper()} of spot embeddings (size=mil_attn)")
                plt.xlabel(f"{args.embed_2d.upper()}-1")
                plt.ylabel(f"{args.embed_2d.upper()}-2")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"spot_embeds_{args.embed_2d}.png"), dpi=200)
                plt.close()

                plt.figure()
                plt.scatter(
                    Z[:, 0], Z[:, 1],
                    c=a_n,
                    s=10 + 200 * a_n,
                    cmap="viridis"
                )
                plt.colorbar(label="MIL attention")
                plt.title(f"{args.embed_2d.upper()} of spot embeddings (color+size=mil_attn)")
                plt.xlabel(f"{args.embed_2d.upper()}-1")
                plt.ylabel(f"{args.embed_2d.upper()}-2")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"spot_embeds_{args.embed_2d}_color.png"), dpi=200)
                plt.close()

            # Top-k patches
            if args.use_image and images is not None:
                k = min(args.topk_patches, n_spots)
                topk = torch.topk(mil_attn, k=k).indices.detach().cpu().tolist()

                patch_dir = os.path.join(out_dir, "top_patches")
                os.makedirs(patch_dir, exist_ok=True)

                for rank, i in enumerate(topk, start=1):
                    fn = f"rank{rank:02d}_idx{i}"
                    if barcodes is not None:
                        fn += f"_bc{barcodes[i]}"
                    if patch_indices is not None:
                        fn += f"_pidx{int(patch_indices[i])}"
                    fn += ".png"
                    save_patch_image(images[i], os.path.join(patch_dir, fn))
            
            # spot importance scatter plot
            if coords_raw is not None:
                mask70 = make_top_percent_mask(mil_attn, top_percent=0.7, min_points=20)

                plot_attention_scatter(
                    coords_raw=coords_raw,
                    attn=mil_attn.detach().cpu(),
                    top10_idx=top10,
                    out_path=os.path.join(out_dir, "patch_attn_scatter_top10.png"),
                    title="Spot importance (MIL attn) Top10",
                    mask=mask70
                )

            # Important genes
            if args.use_st and (gene_attn is not None) and (gene_indices is not None) and (len(gene_order) > 0):
                # gene_attn: (N,1,G) or (N,G)
                if gene_attn.dim() == 3:
                    gene_attn2 = gene_attn.squeeze(1)
                else:
                    gene_attn2 = gene_attn

                top_genes = aggregate_top_genes_bulk(
                    gene_attn=gene_attn2,
                    gene_indices=gene_indices,
                    gene_order=gene_order,
                    topk=args.topk_genes
                )
                with open(os.path.join(out_dir, "top_genes.csv"), "w") as f:
                    f.write("rank,gene,score\n")
                    for r, (g, sc) in enumerate(top_genes, start=1):
                        f.write(f"{r},{g},{sc:.6f}\n")

    print(f"Done. Outputs saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
