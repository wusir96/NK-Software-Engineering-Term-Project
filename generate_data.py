"""Generate synthetic single-cell data (100 cells) for testing."""
import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(42)

N_CELLS = 100
N_GENES = 2000
N_PCA = 50

# ── obs: cell metadata ──
cell_types = ["Hepatocyte", "Cholangiocyte", "Endothelial", "Kupffer", "Stellate", "T cell", "B cell", "NK cell"]
diseases = ["Healthy", "MASLD", "Cirrhosis"]
age_groups = ["Child", "Adolescent", "Adult"]
tissues = ["Liver", "Liver"]
sexes = ["Male", "Female"]
phases = ["G1", "S", "G2M"]

obs = pd.DataFrame({
    "cell_type": np.random.choice(cell_types, N_CELLS, p=[0.25, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05]),
    "disease": np.random.choice(diseases, N_CELLS, p=[0.4, 0.35, 0.25]),
    "tissue": np.random.choice(tissues, N_CELLS),
    "sex": np.random.choice(sexes, N_CELLS),
    "donor_id": [f"D{i:03d}" for i in np.random.randint(1, 21, N_CELLS)],
    "donor_age": np.random.randint(1, 18, N_CELLS),
    "AgeGroup": np.random.choice(age_groups, N_CELLS),
    "Phase": np.random.choice(phases, N_CELLS),
    "nCount_RNA": np.random.lognormal(mean=8, sigma=0.5, size=N_CELLS).astype(int),
    "nFeature_RNA": np.random.lognormal(mean=7, sigma=0.4, size=N_CELLS).astype(int),
})
obs.index = [f"Cell_{i}" for i in range(N_CELLS)]

# ── var: gene metadata ──
var = pd.DataFrame(
    {"gene_name": [f"Gene_{i}" for i in range(N_GENES)]},
    index=[f"Gene_{i}" for i in range(N_GENES)],
)

# ── X: expression matrix (synthetic, log-normalized) ──
# Each cell type has a distinct expression pattern
np.random.seed(123)
X = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
for i, ct in enumerate(cell_types):
    mask = (obs["cell_type"] == ct).values
    n = mask.sum()
    if n == 0:
        continue
    # Each cell type has a base expression profile + noise
    base = np.abs(np.random.randn(N_GENES) * 0.5 + 1.0)
    X[mask] = np.abs(np.random.randn(n, N_GENES) * 0.3 + base)

adata = sc.AnnData(X=X, obs=obs, var=var)

# ── obsm: dimensionality reductions ──
# PCA
np.random.seed(456)
raw_pca = np.random.randn(N_CELLS, N_PCA).astype(np.float32)
adata.obsm["X_pca"] = raw_pca

# UMAP (2D embedding for visualization)
# Make clusters visible by centering coordinates per cell type
umap_coords = np.zeros((N_CELLS, 2), dtype=np.float32)
for i, ct in enumerate(cell_types):
    mask = (obs["cell_type"] == ct).values
    n = mask.sum()
    if n == 0:
        continue
    center = np.array([np.cos(i * 2 * np.pi / len(cell_types)) * 8,
                       np.sin(i * 2 * np.pi / len(cell_types)) * 8])
    umap_coords[mask] = np.random.randn(n, 2) * 1.5 + center
adata.obsm["X_umap"] = umap_coords

# adata.obsm["X_tsne"] is intentionally omitted (tsne_available=False)

# ── uns: global metadata ──
adata.uns["description"] = "Synthetic pediatric liver single-cell atlas (100 cells)"

adata.write("liver.h5ad")
print(f"Generated liver.h5ad: {N_CELLS} cells, {N_GENES} genes, PCA dim={N_PCA}")
print(adata)
