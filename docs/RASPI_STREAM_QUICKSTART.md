# Raspberry Pi Stream Quickstart

This guide shows the shortest path to stream frames from Raspberry Pi (or any Linux edge device) to the DA3 backend.

## 1. Minimal setup on the edge device

```bash
cd ~/Documents/GithubRepo/Depth-Anything-3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install requests
```

If you want to stream directly from a USB camera, also install OpenCV:

```bash
pip install opencv-python
```

## 2. Start backend on the GPU server

Run this on the server side first:

```bash
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE --host 0.0.0.0 --port 8008
```

Then confirm your edge device can reach it:

```bash
curl http://<GPU_SERVER_IP>:8008/v1/healthz
```

## 3. Send frames from an image directory (easiest)

```bash
source .venv/bin/activate
python scripts/raspi_stream_uploader.py \
  --server http://<GPU_SERVER_IP>:8008 \
  --robot-id raspi-01 \
  --image-dir /path/to/frames \
  --fps 3 \
  --max-frames 120 \
  --chunk-size 8 \
  --process-res 320 \
  --export-format mini_npz \
  --session-out workspace/raspi_session_id.txt
```

## 4. Send frames from a live USB camera

```bash
source .venv/bin/activate
python scripts/raspi_stream_uploader.py \
  --server http://<GPU_SERVER_IP>:8008 \
  --robot-id raspi-01 \
  --camera 0 \
  --fps 5 \
  --max-frames 300 \
  --resize-width 640 \
  --resize-height 360 \
  --jpeg-quality 85 \
  --chunk-size 8 \
  --process-res 320 \
  --export-format mini_npz
```

## 5. Reuse an existing session

If you already have a session ID, append more frames without creating a new session:

```bash
python scripts/raspi_stream_uploader.py \
  --server http://<GPU_SERVER_IP>:8008 \
  --session-id <SESSION_ID> \
  --image-dir /path/to/more_frames
```

## Notes

- `--process-res` is the most important knob for GPU memory. Start from `320` when unstable.
- `--chunk-size` and `--max-inflight-chunks` control throughput vs memory pressure.
- By default the script flushes remaining frames on exit (`--no-flush-on-exit` to disable).
