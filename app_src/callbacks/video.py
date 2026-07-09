# -*- coding: utf-8 -*-
"""Serverside callbacks for preparing, selecting, and showing video clips."""

from pathlib import Path

import dash
import dash_player
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app_src.dialogs import open_file_dialog
from app_src.make_mp4 import get_video_duration, make_mp4_clip, validate_clip_range
from app_src.server import VIDEO_DIR, app, cache, components
from app_src.session import coerce_video_start_time


@app.callback(
    Output("video-modal", "is_open"),
    Output("video-path-store", "data", allow_duplicate=True),
    Output("video-container", "children", allow_duplicate=True),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-button", "n_clicks"),
    State("video-modal", "is_open"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)
def prepare_video(n_clicks, is_open, metadata):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    file_unseen = True
    filename = cache.get("filename")
    recent_files_with_video = cache.get("recent_files_with_video")
    file_video_record = cache.get("file_video_record")
    if filename in recent_files_with_video:
        recent_files_with_video.remove(filename)
        video_info = file_video_record.get(filename)
        if video_info is not None and Path(video_info["video_path"]).is_file():
            file_unseen = False

    recent_files_with_video.append(filename)
    cache.set("recent_files_with_video", recent_files_with_video)
    if not file_unseen:
        video_path = video_info["video_path"]
        message = "Preparing clip..."
        return (not is_open), video_path, "", message

    # if original avi has not been uploaded, ask for it
    # video_path = cache.get("video_path")
    video_path = metadata.get("video_path")
    message = "Please select the video above."
    if video_path:
        message += f" You may find it at {video_path}."
    return (not is_open), dash.no_update, components.video_upload_button, message


@app.callback(
    Output("video-path-store", "data"),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-upload-button", "n_clicks"),
    prevent_initial_call=True,
)
def choose_video(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    selected_file_path = open_file_dialog(file_type="video")
    if selected_file_path is None:
        raise PreventUpdate  # user canceled dialog

    avi_path = Path(selected_file_path)  # need to turn WindowsPath to str for the output
    filename = cache.get("filename")
    recent_files_with_video = cache.get("recent_files_with_video")
    file_video_record = cache.get("file_video_record")
    file_video_record[filename] = {
        "video_path": str(avi_path),
        "video_name": avi_path.name,
    }
    if len(recent_files_with_video) > 3:
        filename_to_remove = recent_files_with_video.pop(0)
        if filename_to_remove in file_video_record:
            # avi_file_to_remove = Path(
            #    file_video_record[filename_to_remove]["video_path"]
            # )
            file_video_record.pop(filename_to_remove)
            # avi_file_to_remove.unlink(missing_ok=False)

    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)

    return str(avi_path), "Preparing clip..."


@app.callback(
    Output("video-path-store", "data", allow_duplicate=True),
    Output("video-title", "children"),
    Output("video-container", "children", allow_duplicate=True),
    Output("video-message", "children", allow_duplicate=True),
    Input("reselect-video-button", "n_clicks"),
    prevent_initial_call=True,
)
def reselect_video(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    message = "Please select video above."
    return dash.no_update, "", components.video_upload_button, message


@app.callback(
    Output("clip-name-store", "data"),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-path-store", "data"),
    State("box-select-store", "data"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)
def make_clip(video_path, box_select_range, metadata):
    if not box_select_range:
        return dash.no_update, ""

    start, end = box_select_range
    video_start_time = coerce_video_start_time((metadata or {}).get("video_start_time", 0))
    start = start + video_start_time
    end = end + video_start_time
    try:
        video_duration = get_video_duration(video_path)
    except ValueError as error_message:
        return None, str(error_message)

    clip_range, message = validate_clip_range(start, end, video_duration)
    if clip_range is None:
        return None, message
    start, end = clip_range

    video_name = Path(video_path).stem
    clip_name = video_name + f"_time_range_{start:.1f}-{end:.1f}" + ".mp4"
    save_path = VIDEO_DIR / clip_name
    if save_path.is_file():
        return clip_name, ""

    for file in VIDEO_DIR.iterdir():
        if file.is_file() and file.suffix == ".mp4":
            file.unlink()

    try:
        make_mp4_clip(
            video_path,
            start_time=start,
            end_time=end,
            save_path=save_path,
        )
    except ValueError as error_message:
        return dash.no_update, repr(error_message)

    return clip_name, ""


@app.callback(
    Output("video-title", "children", allow_duplicate=True),
    Output("video-container", "children"),
    Output("video-message", "children"),
    Input("clip-name-store", "data"),
    prevent_initial_call=True,
)
def show_clip(clip_name):
    if not clip_name:
        return "", "", dash.no_update

    if not (VIDEO_DIR / clip_name).is_file():
        return "", "", "Video not ready yet. Please check again in a second."

    clip_path = Path("/assets/videos") / VIDEO_DIR.name / clip_name
    player = dash_player.DashPlayer(
        id="player",
        url=str(clip_path),
        controls=True,
        width="100%",
        height="100%",
    )

    return clip_name, player, components.reselect_video_button
