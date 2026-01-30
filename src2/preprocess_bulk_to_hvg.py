import os
import numpy as np
import scanpy as sc

def load_hvg_list(root_dir: str, fname="global_hvg_genes.txt"):
    path = os.path.join(root_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"HVG list not found: {path}")
    with open(path, "r") as f:
        hvg = [line.strip() for line in f if line.strip()]
    return hvg

def to_dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X)

def convert_one(in_path: str, out_path: str, hvg_list):
    adata = sc.read_h5ad(in_path)

    # 1) bulk vector 만들기: (K_full,)
    X = to_dense(adata.X).astype(np.float32)

    if X.ndim == 2 and X.shape[0] == 1:
        x_full = X[0]
    elif X.ndim == 2 and X.shape[0] > 1:
        # pseudo-bulk면 mean (원하면 sum으로 바꿔도 됨)
        x_full = X.mean(axis=0)
    else:
        raise ValueError(f"Unexpected X shape: {X.shape} in {in_path}")

    var_names = list(map(str, adata.var_names))
    idx_map = {g: i for i, g in enumerate(var_names)}

    # 2) HVG 2000 순서대로 slice (없는 유전자는 0으로)
    out = np.zeros((len(hvg_list),), dtype=np.float32)
    missing = 0
    for t, g in enumerate(hvg_list):
        j = idx_map.get(g, None)
        if j is None:
            missing += 1
            continue
        out[t] = x_full[j]

    # 3) (1,2000) 형태로 h5ad 저장
    obs=adata.obs.copy()
    
    out_adata = sc.AnnData(
        X=out.reshape(1, -1),
        var={"gene": hvg_list},
        obs=obs,
    )
    out_adata.var_names = hvg_list

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_adata.write_h5ad(out_path)

    return missing, len(hvg_list)

def main():
    root_dir = "/content/hest_data"
    in_dir  = os.path.join(root_dir, "bulk_preprocessed")        # full genes
    out_dir = os.path.join(root_dir, "bulk_preprocessed_hvg")    # new HVG 2000

    hvg_list = load_hvg_list(root_dir)

    files = [f for f in os.listdir(in_dir) if f.endswith(".h5ad")]
    files.sort()
    print(f"Found {len(files)} bulk files in {in_dir}")
    print(f"HVG list length: {len(hvg_list)}")

    for f in files:
        sid = f[:-5]
        in_path = os.path.join(in_dir, f)
        out_path = os.path.join(out_dir, f"{sid}.h5ad")

        missing, total = convert_one(in_path, out_path, hvg_list)
        print(f"[{sid}] saved -> {out_path} | missing {missing}/{total}")

    print("DONE")

if __name__ == "__main__":
    main()
