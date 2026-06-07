# 单细胞 ANN 相似检索系统

基于 Flask 的单细胞 RNA-seq 近似最近邻（ANN）检索 Web 应用。用户可在 UMAP 降维图上交互选点，或使用 PCA 向量查询最相似的细胞；支持 HNSW 与 FAISS 多种索引算法、多数据集合并检索，以及用户认证与权限管理。

默认面向人类儿童肝脏单细胞图谱场景，也可上传任意符合格式要求的 `.h5ad` 数据集。

## 功能特性

- **相似细胞检索**：按细胞索引或原始 PCA 向量查询 Top-K 最近邻
- **多种 ANN 算法**：HNSW、FAISS Flat（精确）、FAISS IVF（近似）
- **多种距离度量**：L2 欧氏距离、余弦距离、内积（IP）
- **条件过滤检索**：按细胞类型、疾病、组织、性别、年龄组、细胞周期、来源数据集等筛选结果
- **UMAP 可视化**：Plotly 交互散点图，点击选点即可作为查询目标
- **多数据集管理**：上传、合并、删除多个 `.h5ad` 文件，UMAP 图自动纵向拼接
- **性能基准测试**：对比不同索引方法的构建时间与查询吞吐
- **用户与权限**：JWT 登录；角色分为普通用户（`user`）、研究员（`researcher`）、管理员（`admin`）


## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

`.h5ad` 文件需包含：

| 字段 | 位置 | 说明 |
|------|------|------|
| PCA 嵌入 | `adata.obsm["X_pca"]` | **必需**，用于 ANN 索引与检索 |
| UMAP 坐标 | `adata.obsm["X_umap"]` | 推荐，用于前端可视化 |
| 细胞元数据 | `adata.obs` | 推荐包含 `cell_type`、`disease`、`tissue`、`sex`、`AgeGroup`、`Phase` 等列 |

**方式一**：将数据文件放在项目根目录，命名为 `liver.h5ad`，启动时自动复制到 `uploads/`。

**方式二**：启动后通过 Web 界面上传（需研究员或管理员权限）。

**方式三**：生成合成测试数据（100 个细胞）：

```bash
python generate_data.py
```

生成的 `liver.h5ad` 可直接用于本地测试。

### 3. 启动服务

```bash
python app.py
```

浏览器访问 [http://127.0.0.1:5000](http://127.0.0.1:5000)。

首次启动且无用户时，系统自动创建默认管理员：

| 用户名 | 密码 |
|--------|------|
| `admin` | `admin123` |

生产环境请修改默认密码，并设置环境变量 `JWT_SECRET`。

## 使用说明

1. **注册 / 登录**：访问 `/login` 或 `/register`；也可使用默认管理员账号登录。
2. **上传数据集**（研究员 / 管理员）：在「数据集管理」区域上传 `.h5ad` 文件。
3. **选择查询方式**：
   - 在 UMAP 图上点击细胞；
   - 或手动输入细胞索引；
   - 或直接粘贴 PCA 向量。
4. **配置检索参数**：选择 ANN 方法、距离度量、Top-K 数量，可选填条件过滤。
5. **查看结果**：结果表格展示相似细胞元数据与距离；可运行性能测试对比各算法。

### 角色权限

| 角色 | 权限 |
|------|------|
| `user` | 查看统计、检索、可视化 |
| `researcher` | 上述 + 上传数据集 |
| `admin` | 上述 + 删除数据集、管理用户角色 |

## 项目结构

```
homework/
├── app.py              # Flask 主程序与 API 路由
├── data_loader.py      # 单数据集 .h5ad 加载与统计
├── dataset_manager.py  # 多数据集合并、注册表、过滤逻辑
├── ann_index.py        # HNSW / FAISS 索引封装
├── auth.py             # JWT 认证与权限装饰器
├── user_manager.py     # 用户账号持久化
├── generate_data.py    # 合成测试数据生成
├── requirements.txt    # Python 依赖
├── templates/
│   ├── index.html      # 主界面（UMAP、检索、基准测试）
│   ├── login.html
│   └── register.html
├── uploads/            # 数据集存放目录（运行时生成）
└── data/               # 用户数据 users.json（运行时生成）
```

## API 概览

除认证接口外，大部分 API 需在请求头携带 `Authorization: Bearer <token>`，或使用登录后的 Cookie。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 注册 |
| POST | `/api/auth/login` | 登录 |
| GET | `/api/auth/me` | 当前用户信息 |
| GET | `/api/stats` | 数据集统计 |
| GET | `/api/cell/<idx>` | 单细胞元数据 |
| POST | `/api/query` | 相似细胞检索 |
| POST | `/api/benchmark` | 性能基准测试 |
| GET | `/api/umap_data` | UMAP 可视化数据 |
| POST | `/api/rebuild_index` | 切换索引方法 / 度量 |
| GET | `/api/datasets` | 数据集列表 |
| POST | `/api/datasets/upload` | 上传数据集 |
| DELETE | `/api/datasets/<filename>` | 删除数据集（管理员） |
| GET/PUT/DELETE | `/api/users/...` | 用户管理（管理员） |

### 检索请求示例

```json
POST /api/query
{
  "cell_index": 42,
  "k": 10,
  "method": "hnsw",
  "metric": "l2",
  "filters": {
    "cell_type": "Hepatocyte",
    "disease": "Healthy"
  }
}
```

也可使用 `"vector": [0.1, 0.2, ...]` 代替 `cell_index` 进行向量检索。

## 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `JWT_SECRET` | `ann-retrieval-dev-secret-change-in-prod` | JWT 签名密钥 |
| `JWT_EXPIRE_HOURS` | `24` | Token 有效期（小时） |

## 技术栈

- **后端**：Flask、PyJWT、NumPy、Pandas
- **单细胞数据**：Scanpy（AnnData / `.h5ad`）
- **ANN 索引**：hnswlib、faiss-cpu
- **前端**：Plotly.js、原生 JavaScript

## 架构简述

```
.h5ad 文件 → CellDataLoader（PCA / UMAP / obs）
                ↓
         DatasetManager（多集合并、过滤、统计）
                ↓
            ANNIndex（HNSW / FAISS 建索引与查询）
                ↓
         Flask API + Web 前端（UMAP 交互、检索、基准测试）
```

## 注意事项

- `liver.h5ad` 及 `uploads/`、`data/` 目录默认不纳入版本控制，需自行准备数据。
- 多数据集合并要求 PCA 维度一致；上传时系统会校验。
- 开发模式下 `debug=True`，仅供本地调试，勿直接用于生产部署。
