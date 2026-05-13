<p align="center">
<p align="center">
<h1 align="center">DA3-Streaming: Memory-Efficient Inference for Videos via Chunk Streaming</h1>
</p>


This repo introduces a streaming pipeline that enables `Depth Anything 3` to process ⭐ **`long video sequences`** and **`super-large scale scenes`** ⭐ under tight CPU/GPU memory budgets by chunking frames and managing state across chunks.
Built on the ideas of `VGGT-Long`, it focuses on memory efficiency and stable online inference for near-real-time video processing.

`DA3-Streaming` is built on the [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).

### **Updates**

`[11 Nov 2025]` Code of `DA3-Streaming` release.

##  Setup, Installation & Running

### 📮 1 - Clone this project

Clone the repo using the `--recursive` flag

```cmd
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git

```

If you forgot `--recursive`

```cmd
cd <your_dir>/Depth-Anything-3/
git submodule update --init --recursive .
```

### 📦 2 - Environment Setup

#### Step 1: Dependency Installation

Install `Depth-Anything-3` first.

```cmd
pip install -r requirements.txt
```

#### Step 2: Weights Download

Download all the pre-trained weights needed:

```cmd
bash ./scripts/download_weights.sh
```


#### System dependencies you may encounter.

If you encounter an error about `libGL.so.1` (the error comes from `opencv-python`), please run the following cmd to install the system dependencies.

```cmd
sudo apt-get install -y libgl1-mesa-glx
```


### 🚀 3 - Running the code


```cmd
python da3_streaming.py --image_dir ./path_of_images
```

or

```cmd
python da3_streaming.py --image_dir ./path_of_images --config ./configs/base_config.yaml --output_dir ${OUTPUT_DIR}
```

You may run the following cmd if you got videos before `python da3_streaming.py`.

```
mkdir ./extract_images
ffmpeg -i your_video.mp4 -vf "fps=5,scale=640:-1" ./extract_images/frame_%06d.png
```


### 4 - Outputs

#### Basic Outputs

After running the code, you will get the following outputs:

- `${OUTPUT_DIR}/camera_poses.txt`: The camera poses file. Each line contains the extrinsic matrix parameters of a frame.
- `${OUTPUT_DIR}/intrinsic.txt`: The intrinsic parameters of the camera. Each line contains fx, fy, cx, cy of a frame.
- `${OUTPUT_DIR}/pcd/combined_pcd.ply`: The combined point cloud file. It contains the 3D points from all frames.

#### Additional Outputs

If setting `save_depth_conf_result` in config to `True`, you will get outputs for each frame:

- `${OUTPUT_DIR}/results_output`: The folder that contains the rgb, depth, confidence and intrinsic results for each frame. Note that the minimum value of confidence is 0.

To verify the results, you can use the following cmd to fuse point cloud from `./results_output`. The fused point cloud file will be saved as `${OUTPUT_DIR}/output.ply`.

```cmd
python npz_output_process.py --npz_folder ${OUTPUT_DIR}/results_output --pose_file ${OUTPUT_DIR}/camera_poses.txt --output_file ${OUTPUT_DIR}/output.ply
```


**Note on Space Requirements**: Please ensure your machine has sufficient disk space before running the code for `DA3-Streaming`. When finishing, the code will delete these intermediate results to prevent excessive disk usage.

## Experiment Results

We conducted some additional experiments to compare the performance differences among different architectures. Below is the comparison of `ATE RMSE [m]` on KITTI Odometry between `DA3-Streaming`, `VGGT-Long` and `Pi-Long`. All methods are evaluated with overlap equal to half chunk size, comparable resolution (~500px-width), and loop closure with similarity threshold 0.85.


| **Method**                      | **chunk size**| **AVG**  | **AVG (w/o 01)** | **00**   | **01**   | **02**   | **03** | **04** | **05**  | **06**  | **07**  | **08**  | **09**   | **10**  |
|:-------------------------------:|:--------:|:--------:|:----------------:|:--------:|:--------:|:--------:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|:--------:|:-------:|
| **Num. of Frames**             |      | 2109     | 2210             | 4542     | 1101     | 4661     | 801    | 271    | 2761    | 1101    | 1101    | 4071    | 1591     | 1201    |
| **VGGT-Long**            |120| 25.60  | 22.81          | 16.13   | 53.43   | 51.98   | 4.37  | 2.15  | 12.69  | 11.33  | 3.60   | 70.29  | 34.55   | 21.05  |
| **Pi-Long**              |120| 21.17  | 11.81          | 5.55    | 114.83  | 50.29   | 1.63  | 1.11  | 3.48   | 2.88   | 3.92   | 24.25  | 7.38    | 17.61  |
| **DA3-Streaming**        |120| <span style="text-decoration: underline;">18.63</span>  | **10.42**          | 4.48    | 100.77  | 33.41   | 3.58  | 2.39  | 3.95   | 7.59   | 2.09   | 31.20  | 8.06    | 7.44   |
| **VGGT-Long**            |60| 26.36 |	19.30           | 8.06 | 	96.96 |	34.16 |	6.83  |	4.16 |	9.15 |	4.68 |	2.68 |	63.15 |	32.24 |	27.87  |
| **Pi-Long**              |60| 30.63 |	17.10           | 7.82 |	165.92 |	73.59 |	3.67 |	0.91 |	5.16 |	3.89 |	3.57 |	33.97 |	17.01 |	21.41  |
| **DA3-Streaming**        |60| **16.83**  | <span style="text-decoration: underline;">10.64</span>          | 5.13 | 78.76 | 35.64 | 5.38 | 3.18 | 3.04 | 2.83 | 2.32 | 26.55 | 8.86 | 13.42   |


In `DA3-Streaming`, we restructured the code and accelerated it using GPU technology. Currently, our method achieves a running speed of nearly `10 FPS` without the Keyframe strategy (the test time has excluded warm-up, model loading and ply result saving). Following results are evaluated on KITTI sequences 00, 05, and 08, totaling 11,373 frames, using an NVIDIA A100 GPU.

| **Method**         | **Time** | **FPS** |
|:------------------:|:------------------------:|:------------------------:|
| **VGGT-Long** | 65min 08sec               | 2.91   |
| **Pi-long**   | 60min 09sec               | 3.15   |
| **DA3-Streaming**  | 22min 17sec                |**8.51**   |



Although the current pipeline is not an SLAM system, `DA3-Streaming` still has a certain degree of accuracy compared to the uncalibrated SLAM method on TUM RGB-D. `DA3-Streaming`, `VGGT-Long` and `Pi-Long` are evaluated with chunk size 120, overlap 60, comparable resolution (~500px-width) and with loop closure.

| **Methods**                  | **AVG** | **360** | **desk** | **desk2** | **floor** | **plant** | **room** | **rpy** | **teddy** | **xyz** |
|:----------------------------:|:-------:|:--------:|:---------:|:---------:|:---------:|:--------:|:-------:|:---------:|:-------:|:-------:|
| **Droid-SLAM Uncalibrated**  | 0.163  | 0.202  | 0.032   | 0.091    | 0.064    | 0.045    | 0.918   | 0.056  | 0.045    | 0.012  |
| **Mast3r-SLAM Uncalibrated** | **0.060**  | 0.070  | 0.035   | 0.055    | 0.056    | 0.035    | 0.118   | 0.041  | 0.114    | 0.020  |
| **VGGT-Long**                | 0.110  | 0.118  |	0.058  |	0.111 	| 0.118 	 | 0.071 	  | 0.155 	| 0.140  | 0.120 	  | 0.099  |
| **Pi-long**                  | 0.094  | 0.115  | 0.047   | 0.052    | 0.160 	 | 0.085 	  | 0.114 	| 0.143  | 0.081 	  | 0.052  |
| **DA3-Streaming**            | <span style="text-decoration: underline;">0.087</span>  | 0.059  | 0.034   | 0.042    | 0.107    | 0.060    | 0.105   | 0.206  | 0.126    | 0.044  |


We also evaluate `DA3-Streaming` with different chunk sizes on KITTI (w/o 01) with resolution 504x154 and TUM RGB-D with resolution 504x378. Overlap is set to half of chunk size.

| | **Chunk size** | **120** | **90** | **60** | **30** |
|:-------:|:---------:|:--------:|:-----:|:------:|:------:|
| **KITTI (504x154)** | **Peak VRAM [GB]** | 15.9 | 14.3 | 12.7 | 11.5 |
|| **ATE RMSE [m]** | 10.42 | 9.38 | 10.64 | 19.39 |
| **TUM RGB-D (504x378)** |  **Peak VRAM [GB]** | 28.3 | 25.1 | 21.2 | 18.7 |
|| **ATE RMSE [m]** | 0.087 | 0.091 | 0.127 | 0.227 |

## Acknowledgements

Our project is based on [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).


```markdown
# DA3-Live

Real-time 3D reconstruction from video/webcam.

## Run

```bash
bash setup_da3.sh

# Video
python da3_live.py --video video.mp4 --headless

# Webcam
python da3_live.py --video 0 --headless

# Images
python da3_live.py --image_dir /path/to/images --headless
```

Output: `exps/video_*/pcd/combined_pcd.ply` + `frames_result/output_sidebyside.mp4`

## How It Works

1. Read `chunk_size` frames
2. Run DA3 inference (depth + poses)
3. Align with previous chunk via Sim3
4. Accumulate point cloud, render preview
5. Discard old frames, repeat
6. Save final aligned PLYs

**Video/webcam**: true streaming, bounded RAM  
**Images**: batch mode, supports loop closure
```