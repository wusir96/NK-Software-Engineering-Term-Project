"""Multi-dataset manager for single-cell ANN retrieval."""
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from data_loader import CellDataLoader


@dataclass
class DatasetInfo:
    filename: str
    filepath: str
    n_cells: int = 0
    pca_dim: int = 0
    cell_offset: int = 0


class DatasetManager:
    UPLOAD_DIR = "uploads"
    REGISTRY_FILE = os.path.join(UPLOAD_DIR, "datasets.json")
    UMAP_GAP = 10.0

    def __init__(self, upload_dir: str = UPLOAD_DIR):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

        self._datasets: List[DatasetInfo] = []
        self._loaders: Dict[str, CellDataLoader] = {}
        self._combined_embeddings: Optional[np.ndarray] = None
        self._cell_to_dataset: Optional[np.ndarray] = None

        self._load_registry()
        self._ensure_loaded()

    # ── registry persistence ──

    def _load_registry(self):
        if not os.path.exists(self.REGISTRY_FILE):
            return
        with open(self.REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self._datasets.append(DatasetInfo(**item))

    def _save_registry(self):
        with open(self.REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump([ds.__dict__ for ds in self._datasets], f, ensure_ascii=False, indent=2)

    # ── internal ──

    def _ensure_loaded(self):
        for ds in self._datasets:
            if os.path.exists(ds.filepath):
                try:
                    ldr = CellDataLoader(ds.filepath).load()
                    self._loaders[ds.filename] = ldr
                    ds.n_cells = ldr.n_cells
                    ds.pca_dim = ldr.pca_dim
                except Exception as e:
                    print(f"Warning: failed to load {ds.filename}: {e}")
        self._valid_datasets()
        self._rebuild_combined()

    def _valid_datasets(self):
        """Keep only datasets whose file still exists and loader is ready."""
        valid = []
        for ds in self._datasets:
            if ds.filename in self._loaders:
                valid.append(ds)
        self._datasets = valid
        self._save_registry()

    def _rebuild_combined(self):
        if not self._datasets:
            self._combined_embeddings = None
            self._cell_to_dataset = None
            return

        all_embs = []
        all_ds_indices = []
        offset = 0
        for i, ds in enumerate(self._datasets):
            ldr = self._loaders[ds.filename]
            emb = ldr.get_embeddings()
            all_embs.append(emb)
            all_ds_indices.append(np.full(emb.shape[0], i, dtype=np.int32))
            ds.cell_offset = offset
            offset += emb.shape[0]

        self._combined_embeddings = np.vstack(all_embs).astype(np.float32)
        self._cell_to_dataset = np.concatenate(all_ds_indices)

    # ── properties ──

    @property
    def n_datasets(self) -> int:
        return len(self._datasets)

    @property
    def n_cells(self) -> int:
        if self._combined_embeddings is None:
            return 0
        return self._combined_embeddings.shape[0]

    @property
    def pca_dim(self) -> int:
        if self._combined_embeddings is None:
            return 0
        return self._combined_embeddings.shape[1]

    # ── CRUD ──

    def list_datasets(self) -> List[DatasetInfo]:
        return list(self._datasets)

    def add_dataset(self, filepath: str) -> DatasetInfo:
        filename = os.path.basename(filepath)

        # duplicate check
        if any(ds.filename == filename for ds in self._datasets):
            raise ValueError(f"Dataset '{filename}' already exists")

        # load and validate
        ldr = CellDataLoader(filepath).load()

        # PCA dimension consistency
        if self._datasets and ldr.pca_dim != self._datasets[0].pca_dim:
            raise ValueError(
                f"PCA dimension mismatch: uploaded file has {ldr.pca_dim} dims "
                f"but existing datasets have {self._datasets[0].pca_dim} dims"
            )

        self._loaders[filename] = ldr
        ds = DatasetInfo(
            filename=filename,
            filepath=filepath,
            n_cells=ldr.n_cells,
            pca_dim=ldr.pca_dim,
            cell_offset=self.n_cells,
        )
        self._datasets.append(ds)

        self._rebuild_combined()
        self._save_registry()
        return ds

    def remove_dataset(self, filename: str, delete_file: bool = False):
        ds = next((d for d in self._datasets if d.filename == filename), None)
        if ds is None:
            raise ValueError(f"Dataset '{filename}' not found")

        self._datasets.remove(ds)
        self._loaders.pop(filename, None)

        if delete_file and os.path.exists(ds.filepath):
            os.remove(ds.filepath)

        self._rebuild_combined()
        self._save_registry()

    def scan_upload_dir(self):
        for fname in os.listdir(self.upload_dir):
            fpath = os.path.join(self.upload_dir, fname)
            if not fname.endswith(".h5ad"):
                continue
            if any(ds.filename == fname for ds in self._datasets):
                continue
            try:
                self.add_dataset(fpath)
            except Exception as e:
                print(f"Warning: could not auto-register {fname}: {e}")

    # ── data access ──

    def get_combined_embeddings(self) -> np.ndarray:
        return self._combined_embeddings

    def get_cell_info(self, idx: int) -> dict:
        if self._cell_to_dataset is None or idx < 0 or idx >= len(self._cell_to_dataset):
            raise IndexError(f"Cell index {idx} out of range")
        ds_idx = int(self._cell_to_dataset[idx])
        ds = self._datasets[ds_idx]
        local_idx = idx - ds.cell_offset
        ldr = self._loaders[ds.filename]
        info = ldr.get_cell_info(local_idx)
        info["index"] = idx
        info["dataset"] = ds.filename
        return info

    def get_combined_statistics(self) -> dict:
        if not self._datasets:
            return {
                "n_cells": 0, "n_genes": 0, "pca_dim": 0,
                "cell_types": {}, "n_cell_types": 0,
                "disease_distribution": {}, "age_groups": {},
                "tissue_distribution": {}, "sex_distribution": {},
                "phase_distribution": {},
                "umap_available": False, "tsne_available": False,
                "datasets": [],
            }

        merged = {
            "n_cells": 0, "n_genes": 0,
            "cell_types": {}, "disease_distribution": {},
            "age_groups": {}, "tissue_distribution": {},
            "sex_distribution": {}, "phase_distribution": {},
            "umap_available": True, "tsne_available": False,
        }

        ds_list = []
        for ds in self._datasets:
            ldr = self._loaders[ds.filename]
            s = ldr.get_statistics()
            merged["n_cells"] += s["n_cells"]
            merged["n_genes"] = max(merged["n_genes"], s["n_genes"])
            for key in ["cell_types", "disease_distribution", "age_groups",
                        "tissue_distribution", "sex_distribution", "phase_distribution"]:
                for k, v in s.get(key, {}).items():
                    merged[key][k] = merged[key].get(k, 0) + v
            ds_list.append({
                "filename": ds.filename,
                "n_cells": ds.n_cells,
                "pca_dim": ds.pca_dim,
                "cell_offset": ds.cell_offset,
            })

        merged["pca_dim"] = self._datasets[0].pca_dim
        merged["n_cell_types"] = len(merged["cell_types"])
        merged["datasets"] = ds_list
        return merged

    def get_combined_umap(self) -> dict:
        x_all, y_all = [], []
        cell_types_all, diseases_all, age_groups_all = [], [], []
        dataset_all = []
        offsets = []

        current_y_offset = 0.0

        for ds in self._datasets:
            ldr = self._loaders[ds.filename]
            coords = ldr.get_umap_coords()
            meta = ldr.get_cell_meta()
            n = coords.shape[0]

            y_range = coords[:, 1].max() - coords[:, 1].min()
            coords_shifted = coords.copy()
            coords_shifted[:, 1] = coords_shifted[:, 1] - coords[:, 1].min() + current_y_offset

            x_all.extend(coords_shifted[:, 0].tolist())
            y_all.extend(coords_shifted[:, 1].tolist())
            cell_types_all.extend(meta.get("cell_type", "").tolist())
            diseases_all.extend(meta.get("disease", "").tolist())
            age_groups_all.extend(meta.get("AgeGroup", "").tolist())
            dataset_all.extend([ds.filename] * n)
            offsets.append(float(current_y_offset))

            current_y_offset += float(y_range) + self.UMAP_GAP

        return {
            "x": x_all,
            "y": y_all,
            "cell_types": cell_types_all,
            "diseases": diseases_all,
            "age_groups": age_groups_all,
            "datasets": dataset_all,
            "offsets": offsets,
        }

    # ── conditional query filters ──

    FILTER_FIELDS = ("cell_type", "disease", "tissue", "sex", "AgeGroup", "Phase", "dataset")

    @staticmethod
    def parse_filters(raw: dict) -> dict:
        if not raw:
            return {}
        return {
            k: str(v).strip()
            for k, v in raw.items()
            if k in DatasetManager.FILTER_FIELDS and v is not None and str(v).strip()
        }

    @staticmethod
    def cell_matches_filters(cell_info: dict, filters: dict) -> bool:
        for key, expected in filters.items():
            actual = str(cell_info.get(key, ""))
            if actual.lower() != str(expected).lower():
                return False
        return True
