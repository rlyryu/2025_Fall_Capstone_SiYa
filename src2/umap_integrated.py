import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

"""
per-modality UMAP: aggregate embeddings from all 6 samples, then run PCA+UMAP to visualize in 2D
"""

# CONFIG 
EMB_ROOTS = {
    "st": {
        "emb_dir": "./embeddings_st",
        "out_dir": "./xai_outputs_st",
    },
    "st_img": {
        "emb_dir": "./embeddings_st_img",
        "out_dir": "./xai_outputs_st_img",
    },
    "img": {
        "emb_dir": "./embeddings_img",
        "out_dir": "./xai_outputs_img",
    },
    "st_img_bulk": {
        "emb_dir": "./embeddings_st_img_bulk",
        "out_dir": "./xai_outputs_st_img_bulk",
    }
}

OUT_DIR = "./combined_umap"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = [
    "MEND124",
    "MEND68",
    "NCBI729",
    "MISC35",
    "NCBI828",
    "NCBI618",
]

# Metadata
sample_meta = {
    "MEND124": ("Brain", "Healthy"),
    "MEND68":  ("Brain", "Cancer"),
    "NCBI729": ("Bowel", "Healthy"),
    "MISC35":  ("Bowel", "Cancer"),
    "NCBI828": ("Liver", "Healthy"),
    "NCBI618": ("Liver", "Cancer"),
}

# Colors and markers
tissue_colors = {
    "Brain": {"Healthy": "red",     "Cancer": "orange"},
    "Bowel": {"Healthy": "green",   "Cancer": "lime"},
    "Liver": {"Healthy": "blue",    "Cancer": "skyblue"},
}

disease_markers = {
    "Healthy": "o",
    "Cancer": "x",
}

# ===== MAIN LOOP =====
for modality, paths in EMB_ROOTS.items():

    print(f"\n=== {modality} ===")

    all_embeds = []
    all_sample_ids = []

    emb_dir = paths["emb_dir"]
    out_dir = paths["out_dir"]

    os.makedirs(out_dir, exist_ok=True)

    for sid in SAMPLES:
        path = os.path.join(emb_dir, f"{sid}_{modality}_embedding.npy")
        X = np.load(path)

        all_embeds.append(X)
        all_sample_ids.extend([sid] * X.shape[0])

    X_all = np.concatenate(all_embeds, axis=0)
    sample_ids = np.array(all_sample_ids)

    # PCA 50 → UMAP
    X_pca = PCA(n_components=50, random_state=0).fit_transform(X_all)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=0)
    Z = reducer.fit_transform(X_pca)

    # ===== Plot =====
    plt.figure(figsize=(8, 6))

    for sid in SAMPLES:
        tissue, disease = sample_meta[sid]
        mask = sample_ids == sid

        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            c=tissue_colors[tissue][disease],
            marker=disease_markers[disease],
            s=12,
            alpha=0.8,
            label=f"{sid}"
        )

    plt.title(f"{modality} Combined UMAP (6 samples)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"combined_umap_{modality}.png"), dpi=300)
    plt.close()

    print("Saved.")