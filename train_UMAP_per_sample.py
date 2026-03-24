import numpy as np
import pandas as pd
import umap
import glob
import re
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

"""
Compute & plot UMAP using sample-level embeddings
"""

VER = "Ver2"
DATA = "HEST"

# root = r"C:\Users\rdh08\Desktop\Capstone\st_results_ver2\ST_ver2"
root = r"C:\Users\rdh08\Desktop\Capstone\hest_ver2_results\HEST_ver2"
fusion_option = ["attn", "concat", "gate", "sim"]

# meta_path = r"C:\Users\rdh08\Desktop\Capstone\stimge_meta.csv"
meta_path = r"C:\Users\rdh08\Desktop\Capstone\HEST_v1_1_0.csv"
meta_df = pd.read_csv(meta_path)
# id_to_meta = meta_df.set_index("slide")
id_to_meta = meta_df.set_index("id")
# print(meta_df.columns)

save_dir = r"C:\Users\rdh08\Desktop\Capstone\train_UMAP_results" + f"_{DATA}_{VER}"
os.makedirs(save_dir, exist_ok=True)

def get_latest_best_npz(path):
    # filtering "best"
    files = glob.glob(os.path.join(path, "embeddings_val_best_epoch*.npz"))

    if len(files) == 0:
        raise ValueError(f"No best npz files in {path}")
    
    # extract epoch number from filename
    def get_epoch(f):
        return int(re.search(r"best_epoch(\d+)", f).group(1))

    # epoch 10 이하만
    files = [f for f in files if get_epoch(f) < 10]

    # find the file with the highest epoch number
    best_file = max(files, key=get_epoch)

    return best_file

all_option_embeds = {}  # 전체 option 임베딩

for option in fusion_option:
    all_embeds = []
    all_labels = []
    all_ids = []
    all_preds = []
    all_disease_states = []
    all_organs = []

    print(f"\nProcessing option: {option}")

    for i in range(5):
        # embedding_path = get_latest_best_npz(f"{root}/fold_{i}/{option}")
        embedding_path = get_latest_best_npz(f"{root}/fold_{i}/ver2/{option}")
        data = np.load(embedding_path, allow_pickle=True)
        print(f"Processing fold {i} with file: {embedding_path}")

        all_embeds.append(data["wsi_embeds"])
        all_labels.append(data["labels"])

        ids = data["sample_ids"]
        all_ids.extend(list(ids))

        all_preds.append(data["preds"])

        for sid in ids:
            row = id_to_meta.loc[sid]
            # all_disease_states.append(int(row["involve_cancer"]))
            all_disease_states.append(int(row["disease_state"] in ["Cancer", "Tumor"]))
            # all_organs.append(row["tissue"])
            all_organs.append(row["organ"])

    embeds = np.concatenate(all_embeds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    disease_states = np.array(all_disease_states)
    organs = np.array(all_organs)

    print(f"중복 샘플 개수: {len(all_ids) - len(set(all_ids))}")  # 중복 샘플 개수 출력
    # print("unique disease states:", set(all_disease_states))
    print("counts:", pd.Series(all_disease_states).value_counts())

    print(option, embeds.shape)

    # PCA
    pca = PCA(n_components=min(50, embeds.shape[1]), random_state=0)
    embeds_pca = pca.fit_transform(embeds)

    all_option_embeds[option] = {
    "embeds": embeds_pca,
    "labels": labels,
    "preds": preds,
    "disease": disease_states,
    "organs": organs
    }
# end of option loop

all_embeds_concat = np.concatenate(
    [v["embeds"] for v in all_option_embeds.values()],
    axis=0
)

le_disease = LabelEncoder()
le_organ = LabelEncoder()

all_disease = np.concatenate([v["disease"] for v in all_option_embeds.values()])
all_organs = np.concatenate([v["organs"] for v in all_option_embeds.values()])

le_disease.fit(all_disease)
le_organ.fit(all_organs)

# UMAP
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=0
)
reducer.fit(all_embeds_concat)

# plot

for option in fusion_option:
    data = all_option_embeds[option]

    embeds_umap = reducer.transform(data["embeds"])
    labels = data["labels"]
    preds = data["preds"]
    disease_states = data["disease"]
    organs = data["organs"]

    # 1. diase state별
    disease_encoded = le_disease.transform(disease_states)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeds_umap[:, 0], 
        embeds_umap[:, 1], 
        c=disease_encoded, 
        cmap="coolwarm", 
        s=20
    )

    plt.title(f"{VER}_{option} - Disease State")

    disease_labels = ["Healthy", "Cancer"]
    handles, _ = scatter.legend_elements()
    plt.legend(
        handles, 
        disease_labels, 
        title="Disease State", 
        bbox_to_anchor=(1.02, 1), 
    )

    plt.savefig(os.path.join(save_dir, f"{VER}_{option}_disease_state.png"), dpi=300)    
    plt.close()

    # 2. organ별
    organ_encoded = le_organ.transform(organs)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeds_umap[:, 0], embeds_umap[:, 1],
        c=organ_encoded, cmap="tab10", s=20
    )
    plt.title(f"{VER}_{option} - Organ")
    # plt.show()
    handles, _ = scatter.legend_elements()
    plt.legend(handles, le_organ.classes_, title="Organ", bbox_to_anchor=(1.02, 1))
    plt.savefig(os.path.join(save_dir, f"{VER}_{option}_organ.png"), dpi=300)
    plt.close()

    # 3. pred별
    correct = (labels == preds)

    plt.figure(figsize=(8, 6))
    plt.scatter(embeds_umap[correct, 0], embeds_umap[correct, 1],
                c="blue", s=20, label="correct")
    plt.scatter(embeds_umap[~correct, 0], embeds_umap[~correct, 1],
                c="red", s=20, label="wrong")
    plt.legend()
    plt.title(f"{VER}_{option} - Prediction Correctness")
    # plt.show()
    plt.savefig(os.path.join(save_dir, f"{VER}_{option}_pred.png"), dpi=300)
    plt.close()