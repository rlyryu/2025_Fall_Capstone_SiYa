import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import LabelEncoder
import glob
import re

"""
Compute & plot UMAP using spot-level embeddings
"""

# CONFIG
ROOT = r"C:\Users\rdh08\Desktop\Capstone"

DATA = "HEST"
VER = "ver2"
FUSIONS = ["gate"]
FOLDS = [0, 1, 2, 3, 4]
MODALITY = "st"

META_PATH = os.path.join(ROOT, "HEST_v1_1_0.csv")

OUT_DIR = os.path.join(ROOT, f"modality_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# meta load
meta_df = pd.read_csv(META_PATH)
id_to_meta = meta_df.set_index("id")

# =============================================================================
# main loop
# =============================================================================
for fusion in FUSIONS:
    print(f"\n=== Fusion: {fusion} ===")

    all_embeds = []
    all_ids = []
    all_folds = []

    for fold in FOLDS:
        print(f"\n=== Collect Fold {fold} ===")

        # val ids
        val_path = os.path.join(
            ROOT,
            f"data_splits/HEST/fold_{fold}/val.txt"
        )

        with open(val_path, "r") as f:
            val_ids = [line.strip().replace(",", "") for line in f]

        # embedding
        emb_root = os.path.join(
            ROOT,
            f"training_outputs/{MODALITY}_{fusion}/fold_{fold}/embeddings/val/spot"   # ver2는 fold_i/ver2
        )

        for sid in val_ids:
            file_path = os.path.join(emb_root, f"{sid}.npy")

            if not os.path.exists(file_path):
                print(f"[WARN] missing: {file_path}")
                continue

            X = np.load(file_path)   # (num_spots, dim)

            all_embeds.append(X)
            all_ids.extend([sid] * X.shape[0])
            all_folds.extend([fold] * X.shape[0])
    # end of fold loop

    print(f"[Debugging] len(all_embeds): {len(all_embeds)}")

    # meta info concat
    organs = []
    diseases = []

    for sid in all_ids:
        row = id_to_meta.loc[sid]
        organs.append(row["organ"])
        diseases.append(row["disease_state"])

    organs = np.array(organs)
    diseases = np.array(diseases)

    X_all = np.concatenate(all_embeds, axis=0)
    all_ids_list = np.array(all_ids)

    print("Total spots:", X_all.shape)

    # PCA -> UMAP
    X_pca = PCA(n_components=50, random_state=0).fit_transform(X_all)

    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.3,
        random_state=0
    )

    Z = reducer.fit_transform(X_pca)

    # sample별
    le_sample = LabelEncoder()
    sample_encoded = le_sample.fit_transform(all_ids_list)

    plt.figure(figsize=(8, 6))

    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=sample_encoded,
        cmap="tab20",
        s=10,
        alpha=0.7
    )
    plt.title(f"{MODALITY} - Sample ID")

    handles, _ = sc.legend_elements()
    plt.legend(
        handles,
        le_sample.classes_,
        title="Sample ID",
        bbox_to_anchor=(1.02, 1),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"spot_embedding_{MODALITY}_sample.png"), dpi=300)
    plt.close()

    # disease
    le_disease = LabelEncoder()
    le_disease.fit(diseases)

    plt.figure(figsize=(8, 6))

    for d in ["Healthy", "Cancer"]:
        disease_encoded = le_disease.transform(diseases)
        sc = plt.scatter(
            Z[:, 0],
            Z[:, 1],
            c=disease_encoded,
            cmap="coolwarm",
            s=20,
            alpha=0.7
        )

    plt.title(f"{MODALITY} - Disease State")

    handles, _ = sc.legend_elements()
    plt.legend(
        handles,
        le_disease.classes_,
        title="Disease State",
        bbox_to_anchor=(1.02, 1),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"spot_embedding_{MODALITY}_disease.png"), dpi=300)
    plt.close()

    # organ
    le_organ = LabelEncoder()
    le_organ.fit(organs)
    
    plt.figure(figsize=(8, 6))

    unique_organs = np.unique(organs)

    for organ in unique_organs:
        organ_encoded = le_organ.transform(organs)
        sc = plt.scatter(
            Z[:, 0],
            Z[:, 1],
            c=organ_encoded,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )

    plt.title(f"{MODALITY} - Organ")
    handles, _ = sc.legend_elements()
    plt.legend(
        handles,
        le_organ.classes_,
        title="Organ",
        bbox_to_anchor=(1.02, 1),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"spot_embedding_{MODALITY}_organ.png"), dpi=300)
    plt.close()

    # fold별
    all_folds = np.array(all_folds)

    plt.figure(figsize=(8, 6))

    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=all_folds,
        cmap="tab10",
        s=10,
        alpha=0.7
    )

    plt.title(f"{MODALITY} - Fold")

    unique_folds = sorted(np.unique(all_folds))
    handles, _ = sc.legend_elements()
    plt.legend(
        handles,
        [f"Fold {f}" for f in unique_folds],
        title=f"Fold",
        bbox_to_anchor=(1.02, 1),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"spot_embedding_{MODALITY}_fold.png"), dpi=300)
    plt.close()

    print("Saved.")