# Depth Anything V2 vs V3: 深度マップの仕様比較

## モデル出力の仕様

| 項目 | V2 (HuggingFace Transformers) | V3 (公式リポジトリ) |
|---|---|---|
| **出力タイプ** | Disparity（逆深度） | Metric depth（メトリック深度） |
| **規約** | 大きい値 = 近い、小さい値 = 遠い | 小さい値 = 近い、大きい値 = 遠い |
| **ヘッド活性化関数** | `ReLU`（非負のdisparity） | `exp()`（正のdepth） |
| **値の範囲** | 相対的な任意単位（例：41〜599） | メトリックスケール（例：0.7〜1.2） |
| **解像度** | 入力と同じ（bicubic補間） | モデル解像度（例：280×504） |
| **保存される生データ形式** | `.npy` float32 | `.npy` float32 |
| **モデル名** | `depth-anything/Depth-Anything-V2-Large-hf` | `depth-anything/DA3-LARGE` |

## 推論パイプライン

| ステップ | V2 | V3 |
|---|---|---|
| **ロード** | `AutoModelForDepthEstimation.from_pretrained(...)` | `DepthAnything3.from_pretrained(...)` |
| **前処理** | `AutoImageProcessor`（リサイズ + 正規化） | `InputProcessor`（リサイズ + 正規化 + 内部パラメータ） |
| **フォワードパス** | `model(**inputs).predicted_depth` | `model.inference(image=[...]).depth` |
| **後処理** | `F.interpolate(..., mode="bicubic")`で元のサイズへ | モデル解像度のまま；必要に応じてリサイズ |
| **生出力** | `depth[i] = prediction.squeeze().numpy()` → **disparity** | `prediction.depth[i]` → **metric depth** |

## 共通表現への変換

V2とV3を同じ基準で比較するため、V2のdisparityをdepthに変換：

```
V2 disparity (大=近)  →  depth = 1 / disparity  →  depth (小=近)
V3 depth (小=近)      →  変換不要
```

| | V2（生データ） | V2（変換後） | V3（生データ） |
|---|---|---|---|
| **表現** | Disparity | Depth | Depth |
| **近い物体** | 大きい値 | 小さい値 | 小さい値 |
| **遠い物体** | 小さい値 | 大きい値 | 大きい値 |

## 可視化パイプライン

V3公式の `visualize_depth()` 関数は **depth**（小=近）の値に対して動作：

```
depth → 1/depth (disparity) → パーセンタイル正規化 [0,1] → 反転 (1-x) → Spectralカラーマップ
```

| ステップ | 処理 | 目的 |
|---|---|---|
| 1. 逆数変換 | `disp = 1 / depth` | 近距離の解像度を上げるためdisparityに変換 |
| 2. 正規化 | `(disp - p2) / (p98 - p2)` | 2/98パーセンタイルを使った頑健な[0,1]範囲化 |
| 3. 反転 | `1 - normed` | 近い→高いカラーマップ値（Spectralで暖色/赤）に揃える |
| 4. カラーマップ | `matplotlib Spectral(value)` | 近い = 赤/暖色、遠い = 青/寒色 |

### 両モデルへの適用

V2のdisparity → depthへ変換後（上記ステップ）、**同じ**可視化関数で両方を処理：

```python
# V2: まず変換
v2_depth = 1.0 / v2_disparity    # V3と同じ小=近へ

# 両方とも同じカラーマップ処理
v2_vis = depth_to_colormap(v2_depth, cmap="Spectral")
v3_vis = depth_to_colormap(v3_depth, cmap="Spectral")
```

## よくある間違い

| 間違い | なぜ間違いか |
|---|---|
| V2出力をdepthとして扱う | V2はdisparityを出力 — 近い/遠いの意味が逆 |
| V2のdisparityをV3の`visualize_depth()`に渡す | この関数は内部で`1/入力`を行う；disparityに適用すると再びdepthになり、その後反転されて**V2の色が反転**する |
| Spectral着色済み画像からグレースケール抽出 | Spectralは輝度に対して非単調（赤→黄→緑→青→紫）；グレースケール化すると深度の順序が破壊される |
| 事前レンダリングされたカラーマップを直接比較 | V2デフォルトはINFERNO、V3はSpectral — 異なるスケールでの視覚比較は無意味 |

## スクリプトリファレンス

| スクリプト | 目的 |
|---|---|
| `process_video_dav2.py` | 動画フレームでV2推論を実行、生の`.npy` disparity + INFERNO可視化を保存 |
| `test_v3_depth_output.py` | 画像でV3推論を実行、生の`.npy` depth + 公式可視化を保存 |
| `compare_v2_v3_depth.py` | 両モデルの生`.npy`を使った**正しい**並列比較 |
