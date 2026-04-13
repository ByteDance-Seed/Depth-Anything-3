# 他のリポジトリでの推論方法

## 結論

**このリポジトリのパッケージを `pip install` して使うのが最善の方法です。**

---

## 比較

### ① Transformers ライブラリ経由

| 項目 | 状況 |
|------|------|
| 対応状況 | **未対応**（2026年4月現在） |
| `transformers` の使用箇所 | `process_video_dav2.py`（DAv2用）のみ |
| DA3 固有機能 | 動画・ネストモデル等は統合されない可能性が高い |

→ **現時点では選択肢にならない。**

---

### ② このリポジトリのパッケージを使う（推奨）

`pyproject.toml` で `depth-anything-3` パッケージとして設定済みのため、他のリポジトリから簡単にインストール可能。

#### インストール

```bash
# GitHub から直接インストール
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

# またはローカルパスから
pip install -e /path/to/Depth-Anything-3
```

#### 使い方

```python
from depth_anything_3.api import DepthAnything3

# HuggingFace Hub からモデルを自動ダウンロード
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to("cuda")

# 推論
prediction = model.inference(["image1.jpg", "image2.jpg"])
depth = prediction.depth  # numpy array
```

#### メリット

- HuggingFace Hub 経由でモデルが自動ダウンロードされる（`PyTorchModelHubMixin` 使用）
- DA3 固有機能がすべて利用可能（マルチビュー、動画、ネストモデル、Gaussian Splatting 等）
- `from_pretrained` パターンで Transformers と使い勝手が近い
- API ドキュメントは `docs/API.md` を参照

---

## 利用可能なモデル

| モデル名 | パラメータ数 | 特徴 |
|----------|-------------|------|
| `da3-giant` | 1.15B | GS サポートあり |
| `da3-large` | 0.35B | 汎用（推奨） |
| `da3-base` | 0.12B | 軽量 |
| `da3-small` | 0.08B | 最軽量 |
| `da3mono-large` | 0.35B | 単眼深度専用 |
| `da3metric-large` | 0.35B | メトリック深度 + 空セグメンテーション |
| `da3nested-giant-large` | 1.40B | 全機能搭載のネストモデル |
