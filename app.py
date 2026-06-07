"""Flask web application for single-cell ANN retrieval system."""

import os

import numpy as np
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename

from dataset_manager import DatasetManager
from ann_index import ANNIndex
from user_manager import get_user_manager
from auth import create_token, login_required, role_required

app = Flask(__name__)

UPLOAD_DIR = "uploads"
DEFAULT_METHOD = "hnsw"
DEFAULT_METRIC = "l2"

# Global state
dataset_manager: DatasetManager = None
ann_index: ANNIndex = None
current_method: str = DEFAULT_METHOD
current_metric: str = DEFAULT_METRIC


def get_manager() -> DatasetManager:
    global dataset_manager
    if dataset_manager is None:
        dataset_manager = DatasetManager(upload_dir=UPLOAD_DIR)
    return dataset_manager


def get_embeddings() -> np.ndarray:
    return get_manager().get_combined_embeddings()


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize rows so cosine distance can use inner-product index."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid div-by-zero
    return vecs / norms


def get_index(method: str = None, metric: str = None) -> ANNIndex:
    global ann_index, current_method, current_metric
    m = method or current_method
    d = metric or current_metric
    emb = get_embeddings()
    if emb is None:
        raise ValueError("No datasets loaded")

    # Normalize for cosine metric (use inner-product on normalized vectors)
    store_emb = _l2_normalize(emb.copy()) if d == "cosine" else emb

    if ann_index is None or current_method != m or current_metric != d or ann_index.n_cells != store_emb.shape[0]:
        ann_index = ANNIndex(store_emb, method=m, metric=d)
        ann_index.build()
        current_method = m
        current_metric = d
    return ann_index


def _prepare_query_vec(raw: np.ndarray, metric: str) -> np.ndarray:
    """Normalize query vector if needed for the chosen metric."""
    vec = raw.astype(np.float32).reshape(1, -1)
    if metric == "cosine":
        vec = _l2_normalize(vec)
    return vec


# ── Page routes ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


# ── API: authentication ──────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    username = data.get("username", "")
    password = data.get("password", "")

    try:
        user = get_user_manager().register(username, password)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    token = create_token(user)
    resp = make_response(jsonify({
        "token": token,
        "user": user.to_public(),
    }), 201)
    resp.set_cookie("token", token, httponly=True, samesite="Lax", max_age=86400)
    return resp


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    username = data.get("username", "")
    password = data.get("password", "")

    user = get_user_manager().authenticate(username, password)
    if user is None:
        return jsonify({"error": "用户名或密码错误"}), 401

    token = create_token(user)
    resp = make_response(jsonify({
        "token": token,
        "user": user.to_public(),
    }))
    resp.set_cookie("token", token, httponly=True, samesite="Lax", max_age=86400)
    return resp


@app.route("/api/auth/me")
@login_required
def api_me():
    from flask import g
    return jsonify(g.current_user.to_public())


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    resp = make_response(jsonify({"message": "已退出登录"}))
    resp.delete_cookie("token")
    return resp


# ── API: statistics ──────────────────────────────────────────

@app.route("/api/stats")
@login_required
def api_stats():
    mgr = get_manager()
    return jsonify(mgr.get_combined_statistics())


# ── API: cell info ───────────────────────────────────────────

@app.route("/api/cell/<int:idx>")
@login_required
def api_cell(idx):
    mgr = get_manager()
    if mgr.n_cells == 0 or idx < 0 or idx >= mgr.n_cells:
        return jsonify({"error": "Cell index out of range"}), 400
    return jsonify(mgr.get_cell_info(idx))


# ── API: query similar cells ─────────────────────────────────

@app.route("/api/query", methods=["POST"])
@login_required
def api_query():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    query_idx = data.get("cell_index", None)
    query_vector = data.get("vector", None)
    k = int(data.get("k", 10))
    method = data.get("method", current_method)
    metric = data.get("metric", current_metric)
    filters = DatasetManager.parse_filters(data.get("filters", {}))

    mgr = get_manager()
    emb = get_embeddings()

    # Determine query vector
    if query_idx is not None:
        query_idx = int(query_idx)
        if mgr.n_cells == 0 or query_idx < 0 or query_idx >= mgr.n_cells:
            return jsonify({"error": "Cell index out of range"}), 400
        query_vec = emb[query_idx:query_idx + 1].copy()
    elif query_vector is not None:
        if not isinstance(query_vector, list) or len(query_vector) == 0:
            return jsonify({"error": "vector must be a non-empty list of floats"}), 400
        query_idx = None  # no self-filter when querying by raw vector
        query_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        if query_vec.shape[1] != emb.shape[1]:
            return jsonify({"error": f"Vector dimension mismatch: got {query_vec.shape[1]}, expected {emb.shape[1]}"}), 400
    else:
        return jsonify({"error": "Provide cell_index or vector for query"}), 400

    query_vec = _prepare_query_vec(query_vec, metric)
    idx = get_index(method, metric)

    # Over-fetch when filters are active so we can still return k matches
    base_k = k + 1 if query_idx is not None else k
    if filters:
        search_k = min(max(k * 50, base_k), mgr.n_cells)
    else:
        search_k = base_k

    indices, distances, query_ms = idx.query(query_vec, k=search_k)

    result_indices = []
    result_distances = []
    candidates_scanned = 0
    for i, d in zip(indices, distances):
        candidates_scanned += 1
        if query_idx is not None and int(i) == query_idx:
            continue
        if filters:
            info = mgr.get_cell_info(int(i))
            if not DatasetManager.cell_matches_filters(info, filters):
                continue
        result_indices.append(int(i))
        result_distances.append(float(d))
        if len(result_indices) >= k:
            break

    # Expand search if filters eliminated too many candidates
    if filters and len(result_indices) < k and search_k < mgr.n_cells:
        expanded_k = min(search_k * 5, mgr.n_cells)
        if expanded_k > search_k:
            indices, distances, query_ms = idx.query(query_vec, k=expanded_k)
            seen = set(result_indices)
            candidates_scanned = 0
            for i, d in zip(indices, distances):
                candidates_scanned += 1
                ci = int(i)
                if query_idx is not None and ci == query_idx:
                    continue
                if ci in seen:
                    continue
                info = mgr.get_cell_info(ci)
                if not DatasetManager.cell_matches_filters(info, filters):
                    continue
                result_indices.append(ci)
                result_distances.append(float(d))
                seen.add(ci)
                if len(result_indices) >= k:
                    break

    results = []
    for i, d in zip(result_indices, result_distances):
        info = mgr.get_cell_info(i)
        info["distance"] = round(d, 6)
        results.append(info)

    response = {
        "results": results,
        "query_time_ms": round(query_ms, 4),
        "method": method,
        "metric": metric,
        "build_time_ms": round(idx.build_time_ms, 2),
        "filters": filters,
        "candidates_scanned": candidates_scanned,
    }

    if query_idx is not None:
        response["query_cell"] = mgr.get_cell_info(query_idx)
    else:
        response["query_cell"] = None

    return jsonify(response)


# ── API: performance benchmark ───────────────────────────────

@app.route("/api/benchmark", methods=["POST"])
@login_required
def api_benchmark():
    data = request.get_json() or {}
    n_queries = int(data.get("n_queries", 100))
    k = int(data.get("k", 10))
    methods = data.get("methods", ["hnsw", "faiss_flat"])
    metric = data.get("metric", "l2")

    emb = get_embeddings()
    # Normalize for cosine
    store_emb = _l2_normalize(emb.copy()) if metric == "cosine" else emb

    n_cells = store_emb.shape[0]
    rng = np.random.RandomState(42)
    query_indices = rng.randint(0, n_cells, size=min(n_queries, n_cells))
    query_vecs = store_emb[query_indices]

    results = {}
    for method in methods:
        idx = ANNIndex(store_emb, method=method, metric=metric)
        build_ms = idx.build()
        indices, distances, query_ms = idx.batch_query(query_vecs, k=k + 1)

        results[method] = {
            "build_time_ms": round(build_ms, 2),
            "total_query_time_ms": round(query_ms, 4),
            "avg_query_time_ms": round(query_ms / n_queries, 6),
            "queries_per_second": round(n_queries / (query_ms / 1000), 2),
        }

    return jsonify(results)


# ── API: umap data for visualization ─────────────────────────

@app.route("/api/umap_data")
@login_required
def api_umap_data():
    mgr = get_manager()
    return jsonify(mgr.get_combined_umap())


# ── API: reset / change index method ─────────────────────────

@app.route("/api/rebuild_index", methods=["POST"])
@login_required
def api_rebuild():
    global ann_index, current_method, current_metric
    data = request.get_json() or {}
    method = data.get("method", DEFAULT_METHOD)
    metric = data.get("metric", DEFAULT_METRIC)
    ann_index = None
    idx = get_index(method, metric)
    return jsonify({
        "method": method,
        "metric": metric,
        "build_time_ms": round(idx.build_time_ms, 2),
        "n_cells": idx.n_cells,
        "dim": idx.dim,
    })


# ── API: dataset management ──────────────────────────────────

@app.route("/api/datasets", methods=["GET"])
@login_required
def api_list_datasets():
    mgr = get_manager()
    return jsonify([{
        "filename": ds.filename,
        "n_cells": ds.n_cells,
        "pca_dim": ds.pca_dim,
        "cell_offset": ds.cell_offset,
    } for ds in mgr.list_datasets()])


@app.route("/api/datasets/upload", methods=["POST"])
@role_required("admin", "researcher")
def api_upload_dataset():
    global ann_index

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".h5ad"):
        return jsonify({"error": "Only .h5ad files are accepted"}), 400

    filename = secure_filename(file.filename)
    mgr = get_manager()
    filepath = os.path.join(mgr.upload_dir, filename)

    try:
        file.save(filepath)
        ds = mgr.add_dataset(filepath)
    except ValueError as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Failed to load dataset: {e}"}), 400

    ann_index = None
    return jsonify({
        "filename": ds.filename,
        "n_cells": ds.n_cells,
        "pca_dim": ds.pca_dim,
        "cell_offset": ds.cell_offset,
    }), 201


@app.route("/api/datasets/<filename>", methods=["DELETE"])
@role_required("admin")
def api_delete_dataset(filename):
    global ann_index
    mgr = get_manager()

    ds = next((d for d in mgr.list_datasets() if d.filename == filename), None)
    if ds is None:
        return jsonify({"error": "Dataset not found"}), 404

    try:
        mgr.remove_dataset(filename, delete_file=True)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    ann_index = None
    return jsonify({"message": f"Dataset '{filename}' removed", "remaining": mgr.n_cells})


# ── API: user management (admin only) ────────────────────────

@app.route("/api/users", methods=["GET"])
@role_required("admin")
def api_list_users():
    return jsonify(get_user_manager().list_users())


@app.route("/api/users/<username>/role", methods=["PUT"])
@role_required("admin")
def api_update_user_role(username):
    from flask import g
    data = request.get_json() or {}
    role = data.get("role", "")
    try:
        user = get_user_manager().update_role(username, role)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if g.current_user.username == username:
        # Return new token so admin sees updated role immediately after self-change
        token = create_token(user)
        resp = make_response(jsonify({"user": user.to_public(), "token": token}))
        resp.set_cookie("token", token, httponly=True, samesite="Lax", max_age=86400)
        return resp
    return jsonify({"user": user.to_public()})


@app.route("/api/users/<username>", methods=["DELETE"])
@role_required("admin")
def api_delete_user(username):
    from flask import g
    try:
        get_user_manager().delete_user(username, g.current_user.username)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": f"用户 '{username}' 已删除"})


if __name__ == "__main__":
    print("Initializing user manager ...")
    get_user_manager().ensure_default_admin()

    print("Initializing dataset manager ...")
    mgr = get_manager()

    mgr.scan_upload_dir()

    if mgr.n_datasets == 0 and os.path.exists("liver.h5ad"):
        import shutil
        dst = os.path.join(UPLOAD_DIR, "liver.h5ad")
        shutil.copy("liver.h5ad", dst)
        mgr.scan_upload_dir()

    if mgr.n_cells > 0:
        print(f"Loaded {mgr.n_datasets} dataset(s) with {mgr.n_cells} total cells")
        print("Building default HNSW index ...")
        get_index("hnsw", "l2")
    else:
        print("Warning: No datasets found. Please upload a .h5ad file via the web UI.")

    print("Server starting at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
