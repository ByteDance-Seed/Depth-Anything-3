# surgery_samples データセット 仕様書（Three.js ビューア向け）

DA3-LARGE で推論した手術映像 6 データセットのデータ仕様です。
Three.js を使ったカスタムビューア開発を想定した記述になっています。

---

## ディレクトリ構造

```
workspace/surgery_samples/
├── AutoLabaro/
├── Cholec80/
├── Endovis2018SubChallenge/
├── HeiChole/
├── RARP-50/
└── TMVP-SurVideo/
```

各データセット共通：

```
<DATASET>/
├── config.json                   # 推論設定（image_paths・モデル等）
├── scene.jpg                     # サムネイル
├── scene.glb                     # 全フレーム統合点群 (~11 MB)
├── input_images/                  # 入力フレーム画像（動画ソースのみ）
├── depth_vis/                     # カラー深度マップ JPEG
├── depth_raw/depth_000000.npy    # フレーム別生深度配列（NumPy）
├── exports/mini_npz/results.npz  # 全フレーム推論結果（深度・信頼度・姿勢）
└── pc_vis/
    ├── frame_0000.glb            # フレーム別点群 GLB (~1.6 MB/frame)
    ├── frame_0000_views.png      # 3×4 ビュー確認グリッド画像
    ├── frame_0001.glb
    ├── frame_0001_views.png
    └── ...  (20フレーム × 2 = 40ファイル)
```

---

## データセット一覧

| データセット | ソース形式 | 解像度 (H×W) | 深度範囲 | 信頼度範囲 | 元FPS |
|---|---|---|---|---|---|
| AutoLabaro | MP4 | 280×504 | [0.367, 1.680] | [1.000, 1.640] | 25 |
| Cholec80 | MP4 | 280×504 | [0.477, 1.336] | [1.000, 6.260] | 25 |
| Endovis2018SubChallenge | PNG連番 | 406×504 | [0.315, 1.192] | [1.000, 1.570] | — |
| HeiChole | MP4 | 280×504 | [0.503, 1.127] | [1.000, 2.760] | 25 |
| RARP-50 | AVI | 280×504 | [0.150, 1.182] | [1.000, 2.770] | 60 |
| TMVP-SurVideo | JPG連番 | 280×504 | [0.256, 1.224] | [1.000, 1.270] | — |

共通推論設定: DA3-LARGE, 20 フレーム, 処理解像度 504px, 抽出 ≈2 FPS (動画)

---

## GLB ファイル仕様（Three.js で直接読む）

### 座標系

Three.js のデフォルト座標系は glTF と同じため、**GLB をそのまま読み込むだけで正しく表示されます**。

```
glTF / Three.js: X右 ・ Y上 ・ Z手前（右手系）
```

変換は不要です。`GLTFLoader` でロードした `scene` をそのまま `add()` できます。

### `frame_NNNN.glb` の内容

| オブジェクト | Three.js の型 | 内容 |
|---|---|---|
| 点群 (`points`) | `THREE.Points` | 全ピクセル点 (H×W 点)，`position` + `color` 属性付き |
| カメラ視錐台 | `THREE.LineSegments` | 赤色ワイヤーフレーム |

点群は原点中心にセンタリング済みです。

### Three.js での読み込み例

```js
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load(`pc_vis/frame_${String(frameIdx).padStart(4,'0')}.glb`, (gltf) => {
  scene.clear();
  scene.add(gltf.scene);
});
```

---

## セグメンテーションの重畳（Three.js）

GLB の点群に対してセグメンテーションラベルを色として上書きする方法です。

### 手順

1. Python 側でセグメンテーションマスク `(H, W)` を JSON または NPZ で出力
2. Three.js 側で点群の `color` バッファを書き換える

### 点群の頂点順序

GLB 内の点群は **ラスタスキャン順**（左→右・上→下）で格納されています。
NumPy の `reshape(-1)` 順と一致しています。

```python
# Python：セグメンテーション色を JSON で出力する例
import numpy as np, json

seg_mask  = ...           # (H, W) int, ラベルID
label_rgb = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}  # ラベル→RGB (0-1)

H, W = seg_mask.shape
colors_flat = np.array([label_rgb[l] for l in seg_mask.reshape(-1)])  # (H*W, 3)
json.dump(colors_flat.tolist(), open("seg_colors.json", "w"))
```

```js
// Three.js：点群の color バッファを書き換える例
const response = await fetch('seg_colors.json');
const segColors = await response.json();  // [[r,g,b], ...]

gltf.scene.traverse((obj) => {
  if (obj.isPoints) {
    const colorAttr = obj.geometry.attributes.color;
    for (let i = 0; i < segColors.length; i++) {
      colorAttr.setXYZ(i, segColors[i][0], segColors[i][1], segColors[i][2]);
    }
    colorAttr.needsUpdate = true;
  }
});
```

**信頼度フィルタリングの場合：** NPZ の `conf` を `results.npz` から読み `(H, W)` フラット化し、
低信頼度点の alpha を 0 にする（`THREE.Points` は `alphaTest` で間引き可能）。

---

## グラフノード・エッジの追加（Three.js）

### Python 側：グラフデータを JSON で出力

```python
import json, numpy as np

# ノード: 3D 位置 (世界座標, glTF 座標系) + ラベル
nodes = [
  {"id": 0, "pos": [0.1, 0.2, -0.3], "label": "tool_tip"},
  {"id": 1, "pos": [-0.05, 0.15, -0.25], "label": "shaft"},
]
# エッジ: ノードIDのペア
edges = [[0, 1]]

json.dump({"nodes": nodes, "edges": edges}, open("graph.json", "w"))
```

> **座標変換メモ：** NPZ の点群（OpenCV系）を glTF 座標に変換するには
> `glTF.x = opencv.x`,  `glTF.y = -opencv.y`,  `glTF.z = -opencv.z`

### Three.js 側：ノード球体とエッジ線を追加

```js
import * as THREE from 'three';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const graph = await fetch('graph.json').then(r => r.json());

// ノード：球で表示
const nodeMeshes = graph.nodes.map(n => {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.01),
    new THREE.MeshBasicMaterial({ color: 0xffff00 })
  );
  mesh.position.set(...n.pos);

  // ラベル（CSS2DRenderer 使用時）
  const div = document.createElement('div');
  div.textContent = n.label;
  div.style.color = 'white';
  mesh.add(new CSS2DObject(div));

  scene.add(mesh);
  return mesh;
});

// エッジ：LineSegments で表示
const edgePositions = graph.edges.flatMap(([a, b]) => [
  ...graph.nodes[a].pos, ...graph.nodes[b].pos
]);
const edgeGeo = new THREE.BufferGeometry();
edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
scene.add(new THREE.LineSegments(edgeGeo,
  new THREE.LineBasicMaterial({ color: 0x00ff88 })));
```

---

## results.npz の仕様（Python 側の処理用）

```python
import numpy as np
npz = np.load("exports/mini_npz/results.npz")

depth      = npz["depth"]       # (20, H, W)  float32 — DA3 深度値
conf       = npz["conf"]        # (20, H, W)  float32 — 推定信頼度 (≥1.0)
extrinsics = npz["extrinsics"]  # (20, 3, 4)  float32 — [R | t]（OpenCV 外部パラメータ）
intrinsics = npz["intrinsics"]  # (20, 3, 3)  float32 — K 行列
```

内部パラメータ K の例（AutoLabaro, frame 0）：
```
fx=353.47, fy=353.61, cx=252.0, cy=140.0  （画像サイズ 504×280）
```

### OpenCV → glTF 座標変換（Python で点群を前処理して JSON 出力する場合）

```python
# カメラ座標 (OpenCV) から 3D 点を再構成
H, W = depth.shape
u, v = np.meshgrid(np.arange(W), np.arange(H))
z = depth
x = (u - K[0,2]) * z / K[0,0]
y = (v - K[1,2]) * z / K[1,1]
pts_opencv = np.stack([x, y, z], axis=-1).reshape(-1, 3)

# glTF 座標系へ変換（Three.js に渡す場合）
pts_gltf = pts_opencv * np.array([1, -1, -1])
```

---

## フレームスクラブの実装方針

```
frame_0000.glb   ← frameIdx=0
frame_0001.glb   ← frameIdx=1
...
frame_0019.glb   ← frameIdx=19
```

`<input type="range" min="0" max="19">` の `input` イベントで `GLTFLoader.load()` を呼び、
前フレームのオブジェクトを `scene.clear()` してから `gltf.scene` を `add()` するのが最もシンプルです。
フレーム切り替えを高速化したい場合は、全 GLB をあらかじめロードしてキャッシュしておく方法も有効です。

---

## pc_vis/frame_NNNN_views.png の構成（確認用）

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | 入力画像 | 深度マップ | Front | Back |
| **Row 2** | Top-Down | Right | Left | Bird's Eye 45° |
| **Row 3** | Bird's Eye Left | Bird's Eye Right | Bird's Eye Back | Diagonal |

深度カラーマップ: Spectral_r（赤=近・青=遠）
