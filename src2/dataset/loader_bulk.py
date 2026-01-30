import os
import h5py
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc

# -------------------------------------------------------
# Global gene order loader
# -------------------------------------------------------
def load_global_gene_order(root_dir):
    """Load global HVG gene list to ensure consistent gene ordering"""
    hvg_path = os.path.join(root_dir, "global_hvg_genes.txt")

    if os.path.exists(hvg_path):
        with open(hvg_path, "r") as f:
            global_genes = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded global gene order: {len(global_genes)} genes")
        return global_genes
    else:
        print("⚠️  global_hvg_genes_unified.txt not found, will infer from first sample")
        return None


# -------------------------------------------------------
# Custom Sample class
# -------------------------------------------------------
class CustomSample:
    def __init__(self, root, sample_id):
        self.sample_id = sample_id
        self.bulk_path = os.path.join(root, "bulk_preprocessed", f"{sample_id}.h5ad")
        self.patch_path = os.path.join(root, "patches", f"{sample_id}.h5")

        if not os.path.exists(self.bulk_path):
            raise FileNotFoundError(f"{self.bulk_path} not found.")
        if not os.path.exists(self.patch_path):
            raise FileNotFoundError(f"{self.patch_path} not found.")

        self.label = self._load_label()

    def _load_label(self):
        """Extract sample-level label from h5ad obs['disease_state']"""
        adata = sc.read_h5ad(self.bulk_path, backed="r")
        val = adata.obs["disease_state"].values[0]
        del adata

        import pandas as pd
        if pd.isna(val):
            return 0
        return int(val)


# -------------------------------------------------------
# WSI-level Dataset (unified gene order + zero-padding)
# -------------------------------------------------------
class WSIDataset(Dataset):
    """
    XAI-friendly outputs:
      - images: (N, C, H, W)
      - expr: (N, G)
      - coords_raw: (N, 2)  # original spatial coords from adata.obsm["spatial"]
      - coords_norm: (N, 2) # normalized for model input
      - barcodes: list[str] length N (aligned spot/patch barcode)
      - patch_indices: np.ndarray length N  (index into h5["img"] / h5["barcode"])
      - st_indices: np.ndarray length N     (index into adata.obs_names / expr_aligned / coords_raw before filtering)
      - sel_indices: np.ndarray length N    (indices selected after max_spots sampling, for tracing)
    """
    def __init__(self, samples, max_spots=2000, global_gene_order=None, return_trace=True):
        self.samples = samples
        self.max_spots = max_spots
        self.global_gene_order = global_gene_order
        self.return_trace = return_trace

        # Fallback: infer gene order from the first sample
        if self.global_gene_order is None:
            print("Inferring gene order from first sample...")
            adata = sc.read_h5ad(samples[0].bulk_path, backed="r")
            self.global_gene_order = adata.var_names.tolist()
            del adata
            print(f"✓ Using {len(self.global_gene_order)} genes as reference order")

    def __len__(self):
        return len(self.samples)

    # decode_barcodes() 제거

    @staticmethod
    def _normalize_coords(coords: torch.Tensor) -> torch.Tensor:
        """Min-max normalize to [0,1] per WSI for model input (keeps coords_raw separately)."""
        if coords.numel() == 0:
            return torch.zeros((0, 2), dtype=torch.float32)

        if coords.shape[0] == 1:
            return torch.tensor([[0.5, 0.5]], dtype=coords.dtype)

        c_min = coords.min(dim=0, keepdim=True).values
        c_max = coords.max(dim=0, keepdim=True).values
        c_range = c_max - c_min
        c_range[c_range == 0] = 1.0
        return (coords - c_min) / c_range

    def __getitem__(self, idx):
        sample = self.samples[idx]
        adata = None

        try:
            # 1. Load AnnData
            adata = sc.read_h5ad(sample.bulk_path, backed="r")

            # Gene mapping
            sample_genes = adata.var_names.tolist()
            sample_gene_set = set(sample_genes)
            gene_to_idx = {g: i for i, g in enumerate(sample_genes)}

            # Load expression matrix
            X_raw = adata.X[:]

            # Convert sparse -> dense
            if hasattr(X_raw, "toarray"):
                bulk_expr = X_raw.toarray()
            elif hasattr(X_raw, "todense"):
                bulk_expr = np.array(X_raw.todense())
            else:
                bulk_expr = np.asarray(X_raw)
                
            # Shape validation
            if bulk_expr.ndim == 1:
                bulk_expr = bulk_expr.reshape(1, -1)

            if bulk_expr.shape[0] != 1:
                raise ValueError(f"Expected bulk n_obs=1, got {bulk_expr.shape}")

            n_spots = bulk_expr.shape[0]

            # Zero-padded aligned expression matrix
            expr_aligned = np.zeros((1, len(self.global_gene_order)), dtype=np.float32)
            for new_idx, gene in enumerate(self.global_gene_order):
                if gene in sample_gene_set:
                    old_idx = gene_to_idx[gene]
                    expr_aligned[:, new_idx] = bulk_expr[:, old_idx]
            # barcodes_st = adata.obs_names.to_numpy()
            # coords_raw_np = np.array(adata.obsm["spatial"])  # bulk -> coord 제거
            label_val = sample.label

            del bulk_expr, X_raw
            # keep adata until we finish extracting all needed
            del adata
            adata = None
            gc.collect()

            # -------------------------
            # 2) Load patches (H5)
            # -------------------------
            with h5py.File(sample.patch_path, "r") as f:
                imgs = f["img"][:]         # (n_patches, H, W, C)
                # raw_bar = np.array(f["barcode"])

            n_patches = imgs.shape[0]

            sel = np.arange(len(imgs), dtype=np.int64)
            if self.max_spots and len(imgs) > self.max_spots:
                sel = np.random.choice(len(imgs), self.max_spots, replace=False).astype(np.int64)
            
            images_np = imgs[sel]
            N = images_np.shape[0]
            expr_np = np.repeat(expr_aligned, repeats=N, axis=0).astype(np.float32)
            
            images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
            expr = torch.from_numpy(expr_np).float()

            coords_raw_aligned_np = np.full((N, 2), 0.5, dtype=np.float32)  # bulk -> dummy coord
            
            coords_raw = torch.from_numpy(coords_raw_aligned_np).float()
            coords_norm = coords_raw

            out = {
                "images": images,
                "expr": expr,
                "coords": coords_norm,          # backward compatibility (your model input)
                "coords_norm": coords_norm,     # explicit
                "coords_raw": coords_raw,       # for visualization on original layout
                "label": torch.tensor(label_val).long(),
                "sample_id": sample.sample_id,
                "num_spots": int(images.shape[0]),
            }

            if self.return_trace:
                out.update({
                    # "barcodes": aligned_barcodes,         # list[str], length N
                    "patch_indices": sel,           # np.ndarray length N
                    # "st_indices": st_idx,                 # np.ndarray length N
                    "sel_indices": sel,                   # np.ndarray length N (after sampling)
                })

            return out

        except Exception as e:
            print(f"⚠️ Error loading {sample.sample_id}: {e}")
            import traceback
            print(traceback.format_exc())
            raise

        finally:
            if adata is not None:
                try:
                    del adata
                except:
                    pass
            gc.collect()


def wsi_collate_fn(batch):
    if len(batch) == 1:
        return batch[0]
    return batch


def create_wsi_dataloader(samples, batch_size=1, shuffle=True, max_spots=2000, root_dir=None, return_trace=True):
    """root_dir added for loading global gene order"""
    global_gene_order = None
    if root_dir is not None:
        global_gene_order = load_global_gene_order(root_dir)

    dataset = WSIDataset(
        samples,
        max_spots=max_spots,
        global_gene_order=global_gene_order,
        return_trace=return_trace
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=wsi_collate_fn
    )
    return loader


# -------------------------------------------------------
# Gene information utility
# -------------------------------------------------------
def get_gene_info(samples):
    """Retrieve gene information from the first sample"""
    sample = samples[0]

    adata = sc.read_h5ad(sample.bulk_path, backed="r")
    num_genes = adata.n_vars
    gene_names = adata.var_names.tolist()
    del adata

    print("Gene info:")
    print(f"  Total genes: {num_genes}")
    print(f"  Gene names (first 10): {gene_names[:10]}")

    return num_genes, gene_names


# -------------------------------------------------------
# Dataloader validation utility
# -------------------------------------------------------
def validate_dataloader(loader, num_batches=5):
    """Validate dataloader outputs for debugging"""
    print("\n=== Dataloader Validation ===")

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        print(f"\nBatch {i}:")
        print(f"  Sample ID: {batch['sample_id']}")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Expression shape: {batch['expr'].shape}")
        print(f"  Coords(norm) shape: {batch['coords'].shape}")
        print(f"  Coords(raw) shape: {batch['coords_raw'].shape}")
        print(f"  Label: {batch['label'].item()}")
        print(f"  Num spots: {batch['num_spots']}")

        # Range checks
        print(f"  Image range: [{batch['images'].min():.3f}, {batch['images'].max():.3f}]")
        print(f"  Expr range: [{batch['expr'].min():.3f}, {batch['expr'].max():.3f}]")
        print(f"  Coord(norm) range: [{batch['coords'].min():.3f}, {batch['coords'].max():.3f}]")

        # Zero-padding check
        nonzero_genes = (batch["expr"].sum(dim=0) != 0).sum()
        print(f"  Non-zero genes: {nonzero_genes} / {batch['expr'].shape[1]}")

        # Trace availability
        if "patch_indices" in batch:
            msg = f"Trace keys: patch_indices({len(batch['patch_indices'])})"
            if "st_indices" in batch: msg += f", st_indices({len(batch['st_indices'])})"
            if "barcodes" in batch: msg += f", barcodes({len(batch['barcodes'])})"
            print(msg)

    print("\n✓ Validation complete")
