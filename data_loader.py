"""Single-cell data loading and statistics module."""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional


class CellDataLoader:
    """Load and manage single-cell .h5ad data."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.adata = None
        self.pca_embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def load(self) -> "CellDataLoader":
        self.adata = sc.read_h5ad(self.filepath)
        self.pca_embeddings = self.adata.obsm["X_pca"].copy()
        self._loaded = True
        return self

    @property
    def n_cells(self) -> int:
        return self.adata.n_obs

    @property
    def n_genes(self) -> int:
        return self.adata.n_vars

    @property
    def pca_dim(self) -> int:
        return self.pca_embeddings.shape[1]

    def get_embeddings(self) -> np.ndarray:
        return self.pca_embeddings

    def get_cell_meta(self, indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Get cell metadata for given indices (or all cells if None)."""
        if indices is None:
            return self.adata.obs
        return self.adata.obs.iloc[indices]

    def get_cell_info(self, idx: int) -> dict:
        """Get detailed info for a single cell."""
        row = self.adata.obs.iloc[idx]
        return {
            "index": int(idx),
            "cell_type": row.get("cell_type", "N/A"),
            "disease": row.get("disease", "N/A"),
            "tissue": row.get("tissue", "N/A"),
            "sex": row.get("sex", "N/A"),
            "donor_id": row.get("donor_id", "N/A"),
            "donor_age": str(row.get("donor_age", "N/A")),
            "AgeGroup": row.get("AgeGroup", "N/A"),
            "Phase": row.get("Phase", "N/A"),
            "nCount_RNA": float(row.get("nCount_RNA", 0)),
            "nFeature_RNA": float(row.get("nFeature_RNA", 0)),
        }

    def get_statistics(self) -> dict:
        """Return comprehensive data statistics."""
        obs = self.adata.obs
        stats = {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "pca_dim": int(self.pca_dim),
            "cell_types": obs["cell_type"].value_counts().to_dict(),
            "n_cell_types": int(obs["cell_type"].nunique()),
            "disease_distribution": obs["disease"].value_counts().to_dict(),
            "age_groups": obs["AgeGroup"].value_counts().to_dict() if "AgeGroup" in obs.columns else {},
            "tissue_distribution": obs["tissue"].value_counts().to_dict() if "tissue" in obs.columns else {},
            "sex_distribution": obs["sex"].value_counts().to_dict() if "sex" in obs.columns else {},
            "phase_distribution": obs["Phase"].value_counts().to_dict() if "Phase" in obs.columns else {},
            "umap_available": "X_umap" in self.adata.obsm,
            "tsne_available": "X_tsne" in self.adata.obsm,
        }
        return stats

    def get_umap_coords(self) -> np.ndarray:
        if "X_umap" in self.adata.obsm:
            return self.adata.obsm["X_umap"]
        return np.zeros((self.n_cells, 2))
