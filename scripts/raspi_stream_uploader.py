#!/usr/bin/env python3
"""Simple edge uploader for DA3 streaming sessions.

Designed for Raspberry Pi or similar edge devices:
- create streaming session
- capture from webcam OR read images from directory
- upload frames with retry
- flush pending frames at the end
"""

from __future__ import annotations

import argparse
import pathlib
import time
import mimetypes
from typing import Iterable, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raspi-friendly DA3 stream uploader")

    p.add_argument("--server", required=True, help="Backend base URL (e.g., http://10.0.0.5:8008)")
    p.add_argument("--robot-id", default="raspi-01", help="Robot identifier")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="OpenCV camera index (e.g., 0)")
    src.add_argument("--image-dir", help="Directory of images to upload")

    p.add_argument("--fps", type=float, default=5.0, help="Capture/upload FPS target")
    p.add_argument("--max-frames", type=int, default=0, help="Max frames (0 = unlimited)")
    p.add_argument("--chunk-size", type=int, default=8, help="Server chunk size")
    p.add_argument("--max-inflight-chunks", type=int, default=1, help="Server inflight chunk limit")
    p.add_argument("--process-res", type=int, default=320, help="Server process resolution")
    p.add_argument("--export-format", default="mini_npz", help="Server export format")

    p.add_argument("--resize-width", type=int, default=0, help="Resize width before upload (0 = keep)")
    p.add_argument("--resize-height", type=int, default=0, help="Resize height before upload (0 = keep)")
    p.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for camera mode")

    p.add_argument("--flush-on-exit", action="store_true", default=True, help="Flush pending frames when done")
    p.add_argument("--no-flush-on-exit", action="store_false", dest="flush_on_exit")

    p.add_argument("--request-timeout", type=float, default=20.0, help="HTTP timeout (seconds)")
    p.add_argument("--retry", type=int, default=3, help="Upload retries per frame")
    p.add_argument("--retry-backoff", type=float, default=0.7, help="Retry backoff base seconds")
    p.add_argument("--status-every", type=int, default=10, help="Print server session status every N frames")

    p.add_argument("--session-id", default="", help="Reuse existing session ID (skip creation)")
    p.add_argument("--session-out", default="", help="Write session ID to this file")

    return p.parse_args()


def normalize_server(url: str) -> str:
    return url.rstrip("/")


def http_json(method: str, url: str, timeout: float, **kwargs):
    import requests

    resp = requests.request(method=method, url=url, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp.json()


def create_session(server: str, args: argparse.Namespace) -> str:
    payload = {
        "robot_id": args.robot_id,
        "config": {
            "chunk_size": args.chunk_size,
            "max_inflight_chunks": args.max_inflight_chunks,
            "process_res": args.process_res,
            "export_format": args.export_format,
            "auto_flush": False,
        },
    }
    data = http_json("POST", f"{server}/v1/sessions", timeout=args.request_timeout, json=payload)
    return data["session_id"]


def flush_session(server: str, session_id: str, timeout: float) -> None:
    data = http_json("POST", f"{server}/v1/sessions/{session_id}/flush", timeout=timeout)
    print(
        "[flush] queued_tasks={queued} pending_before={before} pending_after={after}".format(
            queued=data.get("queued_tasks"),
            before=data.get("pending_before"),
            after=data.get("pending_after"),
        )
    )


def session_status(server: str, session_id: str, timeout: float) -> dict:
    return http_json("GET", f"{server}/v1/sessions/{session_id}", timeout=timeout)


def iter_image_files(image_dir: str) -> Iterable[pathlib.Path]:
    p = pathlib.Path(image_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"no image files found in: {image_dir}")
    return files


def encode_camera_frame(frame, resize_hw: Tuple[int, int], jpeg_quality: int) -> bytes:
    import cv2

    w, h = resize_hw
    if w > 0 and h > 0:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("failed to encode frame as jpeg")
    return enc.tobytes()


def upload_frame(
    server: str,
    session_id: str,
    frame_name: str,
    frame_bytes: bytes,
    frame_idx: int,
    timeout: float,
    retries: int,
    backoff: float,
    content_type: str = "image/jpeg",
) -> None:
    import requests

    url = f"{server}/v1/sessions/{session_id}/frames"
    data = {
        "frame_id": f"f{frame_idx:06d}",
        "timestamp_ns": str(time.time_ns()),
    }
    files = {
        "frame": (frame_name, frame_bytes, content_type),
    }

    err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, data=data, files=files, timeout=timeout)
            resp.raise_for_status()
            return
        except Exception as exc:  # noqa: BLE001
            err = exc
            if attempt == retries:
                break
            sleep_s = backoff * attempt
            print(f"[retry] frame={frame_idx} attempt={attempt}/{retries} sleep={sleep_s:.2f}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"failed to upload frame {frame_idx}: {err}")


def run_camera_mode(server: str, session_id: str, args: argparse.Namespace) -> None:
    import cv2

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {args.camera}")

    resize_hw = (int(args.resize_width), int(args.resize_height))
    frame_idx = 0
    period = 1.0 / args.fps if args.fps > 0 else 0.0

    try:
        while True:
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")

            frame_bytes = encode_camera_frame(frame, resize_hw, args.jpeg_quality)
            upload_frame(
                server=server,
                session_id=session_id,
                frame_name=f"frame_{frame_idx:06d}.jpg",
                frame_bytes=frame_bytes,
                frame_idx=frame_idx,
                timeout=args.request_timeout,
                retries=args.retry,
                backoff=args.retry_backoff,
                content_type="image/jpeg",
            )

            frame_idx += 1
            print(f"[upload] frame={frame_idx}")

            if args.status_every > 0 and frame_idx % args.status_every == 0:
                st = session_status(server, session_id, args.request_timeout)
                print(
                    "[status] total={frame_count} pending={pending_frames} inflight={inflight_chunks} "
                    "completed={completed_chunks} failed={failed_chunks}".format(**st)
                )

            if period > 0:
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
    finally:
        cap.release()


def run_image_dir_mode(server: str, session_id: str, args: argparse.Namespace) -> None:
    files = iter_image_files(args.image_dir)
    frame_idx = 0
    period = 1.0 / args.fps if args.fps > 0 else 0.0

    for path in files:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        t0 = time.time()
        frame_bytes = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        if not content_type:
            content_type = "application/octet-stream"
        upload_frame(
            server=server,
            session_id=session_id,
            frame_name=path.name,
            frame_bytes=frame_bytes,
            frame_idx=frame_idx,
            timeout=args.request_timeout,
            retries=args.retry,
            backoff=args.retry_backoff,
            content_type=content_type,
        )

        frame_idx += 1
        print(f"[upload] frame={frame_idx} file={path.name}")

        if args.status_every > 0 and frame_idx % args.status_every == 0:
            st = session_status(server, session_id, args.request_timeout)
            print(
                "[status] total={frame_count} pending={pending_frames} inflight={inflight_chunks} "
                "completed={completed_chunks} failed={failed_chunks}".format(**st)
            )

        if period > 0:
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)


def main() -> int:
    args = parse_args()
    server = normalize_server(args.server)
    session_id: Optional[str] = None

    try:
        if args.session_id:
            session_id = args.session_id
            print(f"[session] reuse: {session_id}")
        else:
            session_id = create_session(server, args)
            print(f"[session] created: {session_id}")

        if args.session_out:
            pathlib.Path(args.session_out).write_text(session_id + "\n", encoding="utf-8")

        if args.camera is not None:
            run_camera_mode(server, session_id, args)
        else:
            run_image_dir_mode(server, session_id, args)

    except KeyboardInterrupt:
        print("[stop] interrupted")
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        return 1
    finally:
        if args.flush_on_exit and session_id:
            try:
                flush_session(server, session_id, args.request_timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] flush failed: {exc}")

    # Final status
    if not session_id:
        return 1

    try:
        st = session_status(server, session_id, args.request_timeout)
        print(
            "[done] session={sid} total={total} pending={pending} inflight={inflight} completed={done} failed={failed}".format(
                sid=session_id,
                total=st.get("frame_count"),
                pending=st.get("pending_frames"),
                inflight=st.get("inflight_chunks"),
                done=st.get("completed_chunks"),
                failed=st.get("failed_chunks"),
            )
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
