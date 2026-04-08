# Depth Anything V3: 時系列情報の利用とデータ構造

## 1. V3における時系列（マルチフレーム）情報の利用

### 1.1 アーキテクチャ概要

V3は **DINOv2 ViT backbone + DPT head** をベースとし、入力として `(B, N, 3, H, W)` の形状（B=バッチ, N=フレーム数）を受け取る。

backbone内では **Local Attention** と **Global Attention** が交互に適用される：

```
Layer 0 ~ alt_start-1 : Local Attention のみ（フレーム独立）
Layer alt_start 以降  : 偶数層= Local, 奇数層= Global（フレーム間cross-attention）
```

### 1.2 Local Attention（フレーム独立）

各フレームを独立に処理する。テンソル形状の変換：

```python
x = rearrange(x, "b s n c -> (b s) n c")  # フレームを独立バッチとして扱う
x = block(x)
x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
```

**各フレームのパッチトークンは、同一フレーム内でのみ互いにattendする。**

### 1.3 Global Attention（フレーム間cross-attention）

全フレームのトークンを結合して一括処理する：

```python
x = rearrange(x, "b s n c -> b (s n) c")  # 全フレームを一つのシーケンスに
x = block(x)
x = rearrange(x, "b (s n) c -> b s n c", b=B, s=S)
```

**全フレームのパッチトークンが互いにattendし、フレーム間の対応を学習する。**
これにより：
- **メトリック深度のスケール整合性** が保たれる
- **カメラポーズ（extrinsics/intrinsics）** が同時に推定される
- フレーム間の幾何的整合性が保証される

### 1.4 カメラトークンの挿入

`alt_start` 層において、CLS トークン位置にカメラトークンが挿入される：

```python
# Reference View（1番目）とSource View（2番目以降）で異なるトークン
ref_token = self.camera_token[:, :1]       # 参照フレーム用
src_token = self.camera_token[:, 1:]       # その他フレーム用
cam_token = torch.cat([ref_token, src_token], dim=1)
x[:, :, 0] = cam_token  # CLS位置に挿入
```

### 1.5 Reference View の選択

N≧一定閾値のとき、alt_start の直前で **参照ビュー** が自動選択される：
- `saddle_balanced` 等の戦略で最適なフレームが選ばれる
- 選ばれたフレームが先頭に並び替えられ、カメラトークンの参照/ソース区分に影響する

### 1.6 実用上の注意

| 条件 | 時系列情報 | 備考 |
|------|-----------|------|
| 同一 `model.inference()` 内の複数フレーム | **利用される** | Global Attentionでフレーム間対応 |
| 異なる `model.inference()` 呼び出し間 | **利用されない** | 各呼び出しは独立、スケールが一致しない可能性あり |
| `da3_streaming/` のバッチ間 | SIM(3)+ループ閉合で後処理補正 | Global Attentionではなく後付けアラインメント |

---

## 2. 出力データ構造

### 2.1 NPZ（mini_npz）— 生データ、最も自作ビューワに適切

パス: `exports/mini_npz/results.npz`

```
results.npz
├── depth       : float32 (N, H, W)      # メトリック深度 [m]（小=近, 大=遠）
├── conf        : float32 (N, H, W)      # 信頼度スコア（≥1.0）
├── extrinsics  : float32 (N, 3, 4)      # W2Cカメラ外部パラメータ [R|t]
└── intrinsics  : float32 (N, 3, 3)      # カメラ内部パラメータ K
```

**自作ビューワ向けの推奨データソース。** 全情報が数値で保存されており、任意の処理が可能。

読み込み例：
```python
import numpy as np
npz = np.load("results.npz")
depth = npz["depth"]          # (20, 280, 504), range: [0.59, 1.29]
conf = npz["conf"]            # (20, 280, 504), range: [1.00, 1.11]
extrinsics = npz["extrinsics"]  # (20, 3, 4) — W2C変換行列
intrinsics = npz["intrinsics"]  # (20, 3, 3) — fx,fy,cx,cy含む
```

フルNPZ（`exports/npz/results.npz`）にはさらに `image: uint8 (N,H,W,3)` が含まれる。

### 2.2 GLB — 3Dビューワ用（前処理済み）

パス: `scene.glb` (4.4MB)

GLBはglTF 2.0バイナリ形式。trimeshで生成される。

```
scene.glb
├── JSON chunk (13KB) — メタデータ
│   ├── scenes[0] → nodes[0] ("world") → 21 children
│   ├── nodes[1] ("geometry_0") → mesh[0]  ... 点群
│   ├── nodes[2..21] ("geometry_1..20") → mesh[1..20]  ... カメラワイヤーフレーム ×20
│   ├── meshes[0]  — mode=0 (POINTS), 282,251頂点
│   │   └── attributes: POSITION (VEC3/FLOAT), COLOR_0 (VEC4/UBYTE)
│   ├── meshes[1..20]  — mode=1 (LINES), 各16頂点
│   │   └── attributes: POSITION (VEC3/FLOAT), COLOR_0 (VEC4/UBYTE)
│   └── accessors[0..41] — 全42個
└── BIN chunk (4.5MB) — 頂点バッファ
```

**重要な特徴：**
- **全フレームの点群がマージ済み**（1つのPointCloud、282,251点）
- フレーム別のデータ分離は **されていない**
- カメラはワイヤーフレーム（LINES）としてのみ表現、ポーズ行列は失われている
- glTF座標系に変換済み（X-右, Y-上, Z-手前 ← CV座標系から変換）
- アラインメント行列 `A` で第1カメラ基準に変換 + 点群中央にセンタリング

カメラワイヤーフレームの色はフレームインデックスに応じたカラーマップ（Spectral）:
```
geometry_1  → (242, 67, 36)   frame 0 — 赤系
geometry_2  → (242, 128, 36)  frame 1
...
geometry_20 → (242, 36, 67)   frame 19 — 紫系
```

### 2.3 RRD — Rerunビューワ用（タイムライン付き）

パス: `scene.rrd` (50MB)

RRD = Rerun Recording Format v2。Rerunビューワ専用。

```
scene.rrd
├── ヘッダ: マジック "RRF2" + protobufメタデータ
├── recording_id: "depth_anything_3d"
└── 20フレーム分のタイムラインデータ:
    
    各フレーム (timeline: "frame", sequence=0..19):
    ├── "world/points"
    │   └── Points3D: positions=(141120, 3), colors=(141120, 4), radii=0.002
    ├── "world/camera"
    │   └── Transform3D: mat3x3=(3,3), translation=(3,)
    ├── "world/camera/image"
    │   ├── Pinhole: K=(3,3), width=504, height=280
    │   └── Image: (280, 504, 3) uint8
    └── "world/camera/depth_vis"
        └── Image: (280, 504, 3) uint8 colormap画像
```

**重要な特徴：**
- **フレーム別にデータが保存**されている（タイムラインで切り替え可能）
- 各フレーム141,120点（280×504ピクセル全点）
- Apache Arrow列指向フォーマットで内部保存
- Protobuf + Arrow のため直接パースは非推奨（rerun SDK経由で読み込む）

---

## 3. 自作ビューワを作るなら

### 推奨データソース: NPZ

GLBやRRDはそれぞれ特定ビューワ向けの形式であり、自作ビューワには **NPZ** が最適：

| 比較項目 | NPZ | GLB | RRD |
|---------|-----|-----|-----|
| フレーム別データ | ○ | ×（マージ済み） | ○ |
| カメラパラメータ数値 | ○ (extrinsics/intrinsics) | ×（ワイヤーフレームのみ） | ○ |
| 深度生値 | ○ (float32) | ×（点群に変換済み） | ○ |
| 信頼度 | ○ (conf) | ×（閾値フィルタ済み） | ○（間接的） |
| パース容易性 | ○ (`np.load`) | △ (glTF parser必要) | × (rerun SDK必要) |
| ファイルサイズ | 8.8MB | 4.4MB | 50MB |
| 任意の後処理 | ○ | × | × |

### 点群復元の基本手順（NPZから）

```python
import numpy as np

npz = np.load("results.npz")
depth = npz["depth"]        # (N, H, W)
K = npz["intrinsics"]       # (N, 3, 3)
ext = npz["extrinsics"]     # (N, 3, 4) — W2C

for i in range(N):
    H, W = depth[i].shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # ピクセル → カメラ座標系
    K_inv = np.linalg.inv(K[i])
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (H*W, 3)
    rays = (K_inv @ pixels.T).T       # (H*W, 3)
    
    d = depth[i].reshape(-1)           # (H*W,)
    pts_cam = rays * d[:, None]        # (H*W, 3) カメラ座標
    
    # カメラ座標 → ワールド座標
    ext44 = np.eye(4)
    ext44[:3, :4] = ext[i]
    c2w = np.linalg.inv(ext44)         # W2C → C2W
    
    pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_world = (c2w @ pts_h.T)[:3].T  # (H*W, 3)
```

### 座標系について

- **V3 出力座標系**: OpenCV準拠（X-右, Y-下, Z-前方）
- **GLB座標系**: glTF準拠（X-右, Y-上, Z-手前）— GLBエクスポート時に変換
- **NPZの座標系**: CV座標系のまま（変換なし）
