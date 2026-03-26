"""
extract_clips.py - Pull episode clips from LeRobot v3 HF datasets.

Requirements: pip install huggingface_hub pandas pyarrow imageio-ffmpeg av
Run:  py extract_clips.py
"""

import io, os, subprocess, sys, tempfile, requests
from pathlib import Path

import av
import pandas as pd
from huggingface_hub import hf_hub_url, list_repo_files
from huggingface_hub.utils import build_hf_headers
import imageio_ffmpeg

FFMPEG     = imageio_ffmpeg.get_ffmpeg_exe()
OUTPUT_DIR = Path(__file__).parent
FPS        = 30
CAMERA     = "observation.images.top_cam"

# (output_filename, repo_id, episode_index)
CLIP_MAP = [
    ("car_success_k100.mp4",  "jonathm126/eval_so101_car_pick_and_place-96_episodes_v0-real_v0", 12),
    ("pen_success.mp4",       "jonathm126/eval_so101_pick_pen_v4_real_v0",                       75),
    ("fail_oscillation.mp4",  "jonathm126/eval_so101_pick_pen_v5_real_v0",                       15),
    ("k25_failure.mp4",       "jonathm126/eval_so101_car_pick_and_place-96_episodes_v0-real_v0", 77),
    ("fail_mode_collapse.mp4","jonathm126/eval_so101_car_pick_and_place-96_episodes_v0-real_v0", 53),
    ("fail_drift.mp4",        "jonathm126/eval_so101_car_pick_and_place-96_episodes_v0-real_v0", 94),
    ("fail_grasp.mp4",        "jonathm126/eval_so101_pick_pen_v5_real_v0",                       14),
    ("fail_recovery.mp4",     "jonathm126/eval_so101_car_pick_and_place-96_episodes_v0-real_v0", 44),
    # fill in when you have episode indices:
    # ("oos_failure_car.mp4",  "jonathm126/...", ???),
    # ("yolo_car.mp4",         "jonathm126/eval_so101_car_pick_and_place-bbox_yolo_v3-real_v0", ???),
    # ("yolo_pen.mp4",         "jonathm126/eval_so101_pick_pen-bbox_yolo_v0-real_v0", ???),
]

# ── Helpers ────────────────────────────────────────────────────────────────

HEADERS = build_hf_headers()

def hf_get(repo_id, filename, stream=False, timeout=300):
    url = hf_hub_url(repo_id, filename, repo_type="dataset")
    r = requests.get(url, headers=HEADERS, timeout=timeout, stream=stream)
    r.raise_for_status()
    return r


def load_parquet(repo_id: str) -> pd.DataFrame:
    """Return combined frame-level DataFrame for a repo."""
    files = sorted(f for f in list_repo_files(repo_id, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet"))
    dfs = [pd.read_parquet(io.BytesIO(hf_get(repo_id, f).content)) for f in files]
    return pd.concat(dfs, ignore_index=True)


def probe_video_files(repo_id: str) -> list[dict]:
    """
    Return list of {filename, first_global_frame, n_frames} for all
    video chunk files in CAMERA, in order.
    """
    files = sorted(
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(f"videos/{CAMERA}/") and f.endswith(".mp4")
    )
    result = []
    cumulative = 0
    for filename in files:
        print(f"    Probing {filename} ...")
        tmp = Path(tempfile.mkdtemp()) / "probe.mp4"
        r = hf_get(repo_id, filename, stream=True)
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        container = av.open(str(tmp))
        vs = container.streams.video[0]
        n = int(float(vs.duration * vs.time_base) * FPS + 0.5)
        container.close()
        tmp.unlink(missing_ok=True)
        result.append({"filename": filename, "first_global_frame": cumulative, "n_frames": n})
        cumulative += n
    return result


def resolve_file_and_offset(video_files: list[dict], global_frame: int):
    """Return (filename, t_start_in_file) for a given global frame index."""
    for vf in video_files:
        end = vf["first_global_frame"] + vf["n_frames"]
        if global_frame < end:
            offset = global_frame - vf["first_global_frame"]
            return vf["filename"], offset / FPS
    raise ValueError(f"Frame {global_frame} beyond all video files")


def download_video(repo_id: str, filename: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / Path(filename).name
    print(f"    Downloading {filename} ...")
    r = hf_get(repo_id, filename, stream=True)
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    return tmp


def cut_clip(src: Path, t_start: float, duration: float, out: Path):
    cmd = [
        FFMPEG, "-y",
        "-ss", f"{t_start:.4f}",
        "-i", str(src),
        "-t", f"{duration:.4f}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-an", "-movflags", "+faststart",
        str(out),
    ]
    print(f"    Cutting t={t_start:.2f}s + {duration:.2f}s -> {out.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:\n", result.stderr[-2000:])
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parquet_cache: dict[str, pd.DataFrame] = {}
    video_file_cache: dict[str, list] = {}
    # cache downloaded video files to avoid re-downloading within same repo
    downloaded_video_cache: dict[tuple, Path] = {}

    for out_name, repo_id, episode_index in CLIP_MAP:
        out = OUTPUT_DIR / out_name
        if out.exists():
            print(f"[skip] {out_name} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"  {out_name}  (ep {episode_index} from {repo_id})")

        # Load parquet
        if repo_id not in parquet_cache:
            print(f"  Loading parquet ...")
            parquet_cache[repo_id] = load_parquet(repo_id)
        df = parquet_cache[repo_id]

        ep = df[df["episode_index"] == episode_index]
        if ep.empty:
            print(f"  ERROR: episode {episode_index} not found"); continue
        first_global = int(ep["index"].min())
        last_global  = int(ep["index"].max())
        duration_s   = (last_global - first_global + 1) / FPS
        print(f"  Global frames {first_global}-{last_global}  ({duration_s:.1f}s)")

        # Probe video files once per repo
        if repo_id not in video_file_cache:
            print(f"  Probing video files ...")
            video_file_cache[repo_id] = probe_video_files(repo_id)
        vfiles = video_file_cache[repo_id]

        filename, t_start = resolve_file_and_offset(vfiles, first_global)
        print(f"  -> {filename}  t_start={t_start:.2f}s")

        # Download video file (cache per (repo, filename))
        key = (repo_id, filename)
        if key not in downloaded_video_cache:
            downloaded_video_cache[key] = download_video(repo_id, filename)
        src = downloaded_video_cache[key]

        cut_clip(src, t_start, duration_s, out)
        print(f"  OK -> {out.name}")

    # Clean up downloaded video files
    for tmp in downloaded_video_cache.values():
        tmp.unlink(missing_ok=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
