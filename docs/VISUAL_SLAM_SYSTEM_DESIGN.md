# Visual SLAM System Design (Mac Edge + Remote GPU)

## 1. Goal
Build a near-real-time Visual SLAM pipeline where:
- a robot-mounted webcam publishes frames
- macOS edge process forwards data to a GPU server
- GPU server performs depth/pose inference and map fusion
- map and pose outputs are consumed by navigation

This document defines an MVP architecture and implementation roadmap.

## 2. Non-Goals (MVP)
- Full global optimization comparable to mature production SLAM stacks
- Hard real-time guarantees under all network conditions
- Full multi-sensor fusion (LiDAR/IMU/GNSS) in first iteration

## 3. High-Level Architecture

### Edge (Robot / Mac)
- `camera_source`: webcam or ROS2 `sensor_msgs/Image`
- `frame_gate`: resize, frame-rate cap, keyframe selection
- `uplink_client`: sends keyframes to cloud with metadata
- `downlink_client`: receives incremental pose/map updates
- `nav_bridge`: publishes outputs to ROS2 topics for planners

### Cloud GPU Server
- `ingest_api`: authenticated frame/session ingestion
- `task_queue`: per-session ordered processing
- `inference_worker`: DA3 inference (batch windowing)
- `map_fusion_worker`: incremental point cloud + pose graph update
- `map_store`: session state, keyframes, map chunks
- `stream_api`: pushes updates to edge/web dashboard

### Consumers
- `visualizer`: live point cloud and trajectory
- `navigation`: local occupancy/costmap generation and path planning input

## 4. Data Flow
1. Edge starts a `session_id`.
2. Frames are sampled (e.g., 5-10 FPS keyframes) and uploaded.
3. Cloud worker performs chunked inference (`da3_streaming` style).
4. Predicted depth/pose is fused into session map incrementally.
5. Updated outputs are emitted:
   - camera pose stream
   - map chunk updates
   - health/latency metrics
6. Edge converts map output to nav-compatible representation.

## 5. API Boundary (MVP Draft)

### 5.1 Session Start
`POST /v1/sessions`

Request (example):
```json
{
  "robot_id": "robot-01",
  "camera": {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0},
  "config": {"target_fps": 8, "process_res": 504}
}
```

Response:
```json
{
  "session_id": "sess_20260208_abc123",
  "upload_url": "/v1/sessions/sess_20260208_abc123/frames",
  "stream_url": "/v1/sessions/sess_20260208_abc123/stream"
}
```

### 5.2 Frame Upload
`POST /v1/sessions/{session_id}/frames`
- multipart upload: image bytes + metadata (`frame_id`, `timestamp_ns`, optional prior pose)

### 5.3 Stream Updates
`GET /v1/sessions/{session_id}/stream` (SSE or WebSocket)
- emits events:
  - `pose_update`
  - `map_chunk`
  - `health`

### 5.4 Session Snapshot
`GET /v1/sessions/{session_id}/map/latest`
- returns latest fused map pointer and summary stats

### 5.5 Operator Dashboard (MVP)
`GET /dashboard`
- single-page operations view for:
  - model health (load state, uptime, device)
  - active/completed inference tasks
  - active streaming sessions (`session_id`, robot id, frame/event counts)
- keyboard-accessible refresh controls and auto-refresh toggle

## 6. Mapping and Navigation Outputs

### Mapping Output
- global point cloud chunks (`.ply` or compact binary chunks)
- keyframe trajectory (`T_wc` or `T_cw`, must be fixed and documented)
- confidence mask per frame/chunk

### Navigation Output
- local occupancy projection from 3D map
- filtered traversability layer
- ROS2 topics (draft):
  - `/slam/odom` (`nav_msgs/Odometry`)
  - `/slam/path` (`nav_msgs/Path`)
  - `/slam/pointcloud` (`sensor_msgs/PointCloud2`)
  - `/slam/costmap` (custom or `nav2_msgs` compatible bridge)

## 7. Performance and Reliability Targets (MVP)
- End-to-end latency (frame upload -> pose update): <= 800 ms p50
- Stable processing throughput: >= 5 FPS effective keyframes
- Session recovery: reconnect within 10 seconds without losing map state

## 8. Security and Ops Baseline
- TLS for all edge-cloud traffic
- token auth per robot/session
- server-side rate limiting per session
- object storage lifecycle for old map chunks
- structured logs with request/session IDs

## 9. Implementation Roadmap

## Phase 0: Coordination and Contracts
- introduce AGENTS protocol and file locking
- freeze API contract and coordinate ownership

## Phase 1: Remote Ingestion
- add session and frame-upload APIs
- write edge uploader client for macOS/ROS2 webcam pipeline

## Phase 2: Incremental Mapping
- add queue-driven worker from uploaded frames
- fuse chunk outputs and publish pose/map updates

## Phase 3: Nav Bridge
- generate nav-consumable outputs
- ROS2 publisher integration and local planner dry-run

## Phase 4: Hardening
- latency profiling
- retry/reconnect behavior
- long-session memory control

## 10. Ownership Proposal
- `SP`: API, queue, worker lifecycle, ROS2 bridge base
- `DP3-DP`: live visualization dashboard and map inspector
- `RF`: refactor interfaces between ingestion/inference/fusion
- `TT`: scenario matrix (network jitter, dropped frames, long runs)

## 11. Immediate Next Tasks
1. Implement `/v1/sessions` and `/v1/sessions/{id}/frames` skeleton endpoints.
2. Add edge uploader CLI that sends webcam frames with timestamps.
3. Create streaming update channel with mock pose/map events.
4. Add E2E test that validates one session with at least 100 uploaded frames.
