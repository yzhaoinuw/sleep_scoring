# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:11:39 2025

@author: yzhao
"""

import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe


def make_mp4_clip(
    video_path, start_time, end_time, save_path=None, save_dir=Path("./assets/videos/")
):
    video_path = Path(video_path)
    save_dir = Path(save_dir)
    duration = end_time - start_time

    if save_path is None:
        video_name = video_path.stem  # filename without extension
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir if needed
        mp4_file = f"{video_name}_time_range_{start_time}-{end_time}.mp4"
        save_path = save_dir / mp4_file
    else:
        save_path = Path(save_path)

    ff = get_ffmpeg_exe()
    cmd = [
        ff,
        "-y",
        "-ss",
        str(start_time),
        "-i",
        str(video_path),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "22",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(save_path),
    ]
    subprocess.run(cmd, check=True)


def avi_to_mp4(src_path: Path, out_dir: Path) -> Path:
    """Given a source video path, return a path to an MP4 version.
    If src is already .mp4, just return it.
    If not, convert once to MP4 (if needed) and return the mp4 path."""
    src_path = Path(src_path)
    if src_path.suffix.lower() == ".mp4":
        return src_path

    video_name = src_path.stem
    out_path = Path(out_dir) / (video_name + ".mp4")

    ff = get_ffmpeg_exe()
    cmd = [
        ff,
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",  # Video quality settings
        "-preset",
        "ultrafast",
        "-crf",
        "22",
        "-c:a",
        "aac",  # Audio settings
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    subprocess.run(cmd, check=True)

    return out_path


if __name__ == "__main__":
    import time

    start_time = time.time()
    video_path = "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/35_ymaze_ymaze_Cam2.avi"
    make_mp4_clip(video_path, start_time=800, end_time=1300)
    # out_path = avi_to_mp4(video_path, "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
