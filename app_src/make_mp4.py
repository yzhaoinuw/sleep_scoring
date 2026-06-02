# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:11:39 2025

@author: yzhao
"""

import math
import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe, read_frames


CLIP_TIME_TOLERANCE_SECONDS = 0.05


def get_video_duration(video_path):
    try:
        video_path = Path(video_path)
    except TypeError as error:
        raise ValueError(
            "Could not determine video duration because no video is selected."
        ) from error

    reader = read_frames(str(video_path))
    try:
        try:
            metadata = next(reader)
        except (OSError, RuntimeError, StopIteration) as error:
            raise ValueError(f"Could not determine video duration for {video_path}.") from error
    finally:
        reader.close()

    try:
        duration = float(metadata.get("duration"))
    except (TypeError, ValueError):
        raise ValueError(f"Could not determine video duration for {video_path}.")

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Could not determine video duration for {video_path}.")

    return duration


def validate_clip_range(
    start_time,
    end_time,
    video_duration,
    tolerance=CLIP_TIME_TOLERANCE_SECONDS,
):
    try:
        start_time = float(start_time)
        end_time = float(end_time)
        video_duration = float(video_duration)
    except (TypeError, ValueError):
        return None, "Selected video range has invalid timing metadata."

    if not all(math.isfinite(value) for value in (start_time, end_time, video_duration)):
        return None, "Selected video range has invalid timing metadata."

    if end_time <= start_time:
        return None, "Selected time range is empty. Please select a wider range."

    if start_time < -tolerance:
        return (
            None,
            f"Video clip unavailable: selection starts {abs(start_time):.1f} s "
            "before the video begins. Select a later EEG range.",
        )

    if end_time > video_duration + tolerance:
        return (
            None,
            f"Video clip unavailable: selection ends {end_time - video_duration:.1f} s "
            "after the video ends. Select an earlier EEG range.",
        )

    return (max(0, start_time), min(video_duration, end_time)), ""


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
    video_path = (
        "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/35_ymaze_ymaze_Cam2.avi"
    )
    make_mp4_clip(video_path, start_time=800, end_time=1300)
    # out_path = avi_to_mp4(video_path, "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
