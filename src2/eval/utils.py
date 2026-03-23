import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from dataset.loader import CustomSample, create_wsi_dataloader, load_global_gene_order

def discover_samples(root_dir):
    """
    Discover sample IDs by looking for matching .h5ad in st_preprocessed_global_hvg and .h5 in patches.
    Returns a list of CustomSample instances.
    """
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

def make_top_percent_mask(attn: torch.Tensor, top_percent: float = 0.6, min_points: int = 10):
    """
    Make a boolean mask for the top X% of attention scores, 
    with a fallback to top-k if too few points are selected.
    Args:
        attn: (N,) torch.Tensor
        top_percent: keep top 70% => 0.7
        min_points: safety fallback (avoid empty / too few)
    Returns:
        mask: (N,) bool torch.Tensor
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

def save_patch_image(tensor_chw, out_path):
    """Save a single patch image from a (C,H,W) tensor."""
    x = tensor_chw.detach().cpu().clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(x).save(out_path)

def plot_attention_scatter(coords_raw, attn, top10_idx, out_path,
                           title="Spot importance (MIL attn)",
                           mask=None):
    """Plot a scatter of spot coordinates colored by attention scores."""
    if isinstance(coords_raw, np.ndarray):
        c_all = coords_raw
    else:
        c_all = coords_raw.detach().cpu().numpy()
        
    if isinstance(attn, np.ndarray):
        a_all = attn
    else:
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