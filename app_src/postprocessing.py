# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:14:49 2024

@author: yzhao
"""

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def get_sleep_segments(sleep_scores):
    transition_indices = np.flatnonzero(np.diff(sleep_scores))
    transition_indices = np.append(transition_indices, len(sleep_scores) - 1)

    REM_end_indices = np.flatnonzero(sleep_scores[transition_indices] == 2)
    REM_start_indices = REM_end_indices - 1
    REM_end_indices = transition_indices[REM_end_indices]
    REM_start_indices = transition_indices[REM_start_indices] + 1

    wake_end_indices = np.flatnonzero(sleep_scores[transition_indices] == 0)
    wake_start_indices = wake_end_indices - 1
    wake_end_indices = transition_indices[wake_end_indices]
    wake_start_indices = transition_indices[wake_start_indices] + 1

    SWS_end_indices = np.flatnonzero(sleep_scores[transition_indices] == 1)
    SWS_start_indices = SWS_end_indices - 1
    SWS_end_indices = transition_indices[SWS_end_indices]
    SWS_start_indices = transition_indices[SWS_start_indices] + 1

    df_rem = pd.DataFrame()
    df_rem["sleep_scores"] = pd.Series(np.array([2] * REM_end_indices.size))
    df_rem["start"] = pd.Series(REM_start_indices)
    df_rem["end"] = pd.Series(REM_end_indices)

    df_wake = pd.DataFrame()
    df_wake["sleep_scores"] = pd.Series(np.array([0] * wake_end_indices.size))
    df_wake["start"] = pd.Series(wake_start_indices)
    df_wake["end"] = pd.Series(wake_end_indices)

    df_SWS = pd.DataFrame()
    df_SWS["sleep_scores"] = pd.Series(np.array([1] * SWS_end_indices.size))
    df_SWS["start"] = pd.Series(SWS_start_indices)
    df_SWS["end"] = pd.Series(SWS_end_indices)

    frames = [df_rem, df_wake, df_SWS]
    df = pd.concat(frames)
    df = df.sort_values(by=["end"], ignore_index=True)
    df.at[0, "start"] = 0
    df["duration"] = df["end"] - df["start"] + 1
    return df


def merge_consecutive_sleep_scores(df):
    df["group"] = (df["sleep_scores"] != df["sleep_scores"].shift()).cumsum()
    # Group by 'id' and 'group' and then aggregate
    df_merged = (
        df.groupby(["sleep_scores", "group"])
        .agg(
            sleep_scores=("sleep_scores", "first"),
            start=("start", "min"),
            end=("end", "max"),
            duration=("duration", "sum"),
        )
        .reset_index(drop=True)
    )
    df_merged = df_merged.sort_values(by=["end"], ignore_index=True)
    return df_merged


def evaluate_Wake(
    emg, emg_frequency, start, end, prev_start, prev_end, next_start, next_end
):
    emg_seg = abs(emg[int(start * emg_frequency) : int((end + 1) * emg_frequency)])
    prev_emg_seg = abs(
        emg[int(prev_start * emg_frequency) : int((prev_end + 1) * emg_frequency)]
    )
    next_emg_seg = abs(
        emg[int(next_start * emg_frequency) : int((next_end + 1) * emg_frequency)]
    )
    # check 1: EMG increases
    high_emg = (
        np.percentile(emg_seg, q=99) > 100 * np.percentile(prev_emg_seg, q=60)
    ) and (np.percentile(emg_seg, q=99) > 100 * np.percentile(next_emg_seg, q=60))

    return high_emg


def modify_Wake(df, emg, emg_frequency):
    """change short Wake (<= 15s) if needed"""
    df_short_Wake = df[(df["sleep_scores"] == 0) & (df["duration"] <= 15)]
    for row in df_short_Wake.itertuples():
        index, start, end = row[0], row[2], row[3]
        if index < 1 or index >= len(df) - 1:
            continue

        prev_start, prev_end = df.loc[index - 1]["start"], df.loc[index - 1]["end"]
        next_start, next_end = df.loc[index + 1]["start"], df.loc[index + 1]["end"]
        if evaluate_Wake(
            emg, emg_frequency, start, end, prev_start, prev_end, next_start, next_end
        ):
            continue

        label = 0
        nearest_seg_duration = row[4]
        if index >= 1 and df.loc[index - 1]["duration"] > nearest_seg_duration:
            nearest_seg_duration = df.loc[index - 1]["duration"]
            label = df.loc[index - 1]["sleep_scores"]
        if index < len(df) - 1 and df.loc[index + 1]["duration"] > nearest_seg_duration:
            label = df.loc[index + 1]["sleep_scores"]

        df.at[index, "sleep_scores"] = label

    return df


def modify_SWS(df):
    """eliminate short SWS (<= 5s) between two Wake"""
    df_short_SWS = df[(df["sleep_scores"] == 1) & (df["duration"] <= 5)]
    for row in df_short_SWS.itertuples():
        index = row[0]
        change = 0
        if index >= 1:
            if df.loc[index - 1]["sleep_scores"] == 0:
                change += 1
        else:
            change += 1

        if index < len(df) - 1:
            if df.loc[index + 1]["sleep_scores"] == 0:
                change += 1
        else:
            change += 1

        if change == 2:
            df.at[index, "sleep_scores"] = 0

    return df


def check_REM_transitions(df, ne, ne_frequency=None):
    """check for wrong transitions"""
    df_rem = df[df["sleep_scores"] == 2]
    for row in df_rem.itertuples():
        index, start, end = row[0], row[2], row[3]
        rem = True
        if index < 1:
            continue

        prev_start = df.loc[index - 1]["start"]
        duration = row[4]
        valid_REM = True
        if ne_frequency is not None:
            valid_REM = validate_REM(df, ne, ne_frequency, start, end)

        if not valid_REM:
            df.at[index, "sleep_scores"] = 1
            continue

        # if preceded by Wake, make changes
        if df.loc[index - 1]["sleep_scores"] == 0:
            if valid_REM:
                df.at[index - 1, "sleep_scores"] = 1
            else:
                df.at[index, "sleep_scores"] = 1
                rem = False

        # if proceeded by a SWS, make changes
        if rem and index < len(df) - 1:
            if df.loc[index + 1]["sleep_scores"] == 1:
                if valid_REM:
                    df.at[index + 1, "sleep_scores"] = 0
                else:
                    df.at[index, "sleep_scores"] = 1
                    # df.at[index - 1, "sleep_scores"] = 1

    return df


def check_REM_duration(df):
    """eliminate short REM (< 7s)"""
    df_rem_short = df[(df["sleep_scores"] == 2) & (df["duration"] < 7)]
    for row in df_rem_short.itertuples():
        index = row[0]
        nearby_seg_duration = 0
        label = 2
        if index >= 1:
            nearby_seg_duration = df.loc[index - 1]["duration"]
            label = df.loc[index - 1]["sleep_scores"]
        if index < len(df) - 1 and df.loc[index + 1]["duration"] > nearby_seg_duration:
            label = df.loc[index + 1]["sleep_scores"]

        df.at[index, "sleep_scores"] = label

    return df


def validate_REM(df, ne, ne_frequency, REM_start, REM_end):
    if REM_end > len(ne) - 21 and REM_end > 21:
        return False
    ne_segment = ne[
        int((REM_end - 21) * ne_frequency) : int((REM_end + 1) * ne_frequency)
    ]
    next_ne_seg = ne[
        int((REM_end - 5) * ne_frequency) : int((REM_end + 21) * ne_frequency)
    ]
    next_ne_seg = next_ne_seg - min(ne_segment)
    ne_segment = ne_segment - min(ne_segment)

    NE_increase = np.percentile(next_ne_seg, 99) > 10 * np.percentile(ne_segment, q=25)
    return NE_increase


def evaluate_REM(df, ne, ne_frequency):
    df_rem = df[df["sleep_scores"] == 2]
    for row in df_rem.itertuples():
        index, start, end = row[0], row[2], row[3]

        if end > len(ne) - 20:
            df.at[index, "sleep_scores"] = 1
            return df
        ne_segment = ne[int((end - 20) * ne_frequency) : int((end + 1) * ne_frequency)]
        next_ne_seg = ne[int((end + 1) * ne_frequency) : int((end + 21) * ne_frequency)]

        next_ne_seg = next_ne_seg - min(ne_segment)
        ne_segment = ne_segment - min(ne_segment)
        # check 1: NE increases
        try:  # if REM is identified at the start or the end, discard REM
            NE_increase = np.percentile(next_ne_seg, 99) > 3 * np.percentile(
                ne_segment, q=50
            )
        except IndexError:
            NE_increase = False
        if NE_increase:
            continue

        df.at[index, "sleep_scores"] = 1

    return df


def edit_sleep_scores(sleep_scores, df):
    sleep_scores_post = sleep_scores.copy()
    for row in df.itertuples():
        start, end = row[2], row[3]
        label = row[1]
        sleep_scores_post[start : end + 1] = label
    return sleep_scores_post


def postprocess_sleep_scores(mat, return_table=False):
    sleep_scores = mat.get("sleep_scores")
    emg_frequency = mat.get("eeg_frequency")  # eeg and emg have the same frequency
    emg = mat.get("emg")
    ne = mat.get("ne")
    ne_frequency = None
    if ne is not None:
        if len(ne) > 1:
            ne_frequency = mat.get("ne_frequency")

    df = get_sleep_segments(sleep_scores)
    df = modify_Wake(df, emg, emg_frequency)
    df = merge_consecutive_sleep_scores(df)
    df = modify_SWS(df)
    df = merge_consecutive_sleep_scores(df)
    df = check_REM_duration(df)
    df = merge_consecutive_sleep_scores(df)
    df = check_REM_transitions(df, ne, ne_frequency)
    df = merge_consecutive_sleep_scores(df)
    # df = check_REM_transitions(df, ne, ne_frequency)
    # df = merge_consecutive_sleep_scores(df)

    sleep_scores_post = edit_sleep_scores(sleep_scores, df)
    if return_table:
        return sleep_scores_post, df
    return sleep_scores_post


def get_pred_label_stats(df_sleep_segments: pd.DataFrame):
    MA_indices = np.flatnonzero(
        (df_sleep_segments["sleep_scores"] == 0) & (df_sleep_segments["duration"] < 15)
    )
    df_sleep_segments.loc[MA_indices, "sleep_scores"] = 3

    wake_indices = np.flatnonzero(df_sleep_segments["sleep_scores"] == 0)
    SWS_indices = np.flatnonzero(df_sleep_segments["sleep_scores"] == 1)
    REM_indices = np.flatnonzero(df_sleep_segments["sleep_scores"] == 2)
    MA_indices = np.flatnonzero(df_sleep_segments["sleep_scores"] == 3)

    total_time = df_sleep_segments["duration"].sum()
    wake_time = df_sleep_segments.loc[wake_indices]["duration"].sum()
    SWS_time = df_sleep_segments.loc[SWS_indices]["duration"].sum()
    REM_time = df_sleep_segments.loc[REM_indices]["duration"].sum()
    MA_time = df_sleep_segments.loc[MA_indices]["duration"].sum()

    wake_time_percent = round(wake_time / total_time * 100, 2)
    SWS_time_percent = round(SWS_time / total_time * 100, 2)
    REM_time_percent = round(REM_time / total_time * 100, 2)
    MA_time_percent = round(MA_time / total_time * 100, 2)

    wake_seg_count = wake_indices.size
    SWS_seg_count = SWS_indices.size
    REM_seg_count = REM_indices.size
    MA_seg_count = MA_indices.size

    # count transitions
    df2 = pd.DataFrame(
        [[-1, np.nan, np.nan, np.nan]], columns=df_sleep_segments.columns
    )
    df2 = pd.concat([df_sleep_segments, df2], ignore_index=True)

    df_wake_transition = df2.loc[wake_indices + 1]
    wake_SWS_transition_count = df_wake_transition[
        df_wake_transition["sleep_scores"] == 1
    ].shape[0]

    df_SWS_transition = df2.loc[SWS_indices + 1]
    SWS_wake_transition_count = np.flatnonzero(
        df_SWS_transition["sleep_scores"] == 0
    ).size
    SWS_REM_transition_count = np.flatnonzero(
        df_SWS_transition["sleep_scores"] == 2
    ).size
    SWS_MA_transition_count = np.flatnonzero(
        df_SWS_transition["sleep_scores"] == 3
    ).size

    df_REM_transition = df2.loc[REM_indices + 1]
    REM_wake_transition_count = np.flatnonzero(
        df_REM_transition["sleep_scores"] == 0
    ).size
    REM_MA_transition_count = np.flatnonzero(
        df_REM_transition["sleep_scores"] == 3
    ).size

    df_MA_transition = df2.loc[MA_indices + 1]
    MA_SWS_transition_count = np.flatnonzero(df_MA_transition["sleep_scores"] == 1).size

    stats = {
        "Wake": [
            wake_time,
            wake_time_percent,
            wake_seg_count,
            np.nan,
            wake_SWS_transition_count,
            np.nan,
            np.nan,
        ],
        "SWS": [
            SWS_time,
            SWS_time_percent,
            SWS_seg_count,
            SWS_wake_transition_count,
            np.nan,
            SWS_REM_transition_count,
            SWS_MA_transition_count,
        ],
        "REM": [
            REM_time,
            REM_time_percent,
            REM_seg_count,
            REM_wake_transition_count,
            np.nan,
            np.nan,
            REM_MA_transition_count,
        ],
        "MA": [
            MA_time,
            MA_time_percent,
            MA_seg_count,
            np.nan,
            MA_SWS_transition_count,
            np.nan,
            np.nan,
        ],
    }

    df_stats = pd.DataFrame(data=stats)
    df_stats.index = [
        "Time (s)",
        "Time (%)",
        "Count",
        "Wake Transition Count",
        "SWS Transition Count",
        "REM Transition Count",
        "MA Transition Count",
    ]
    return df_stats


# %%
if __name__ == "__main__":
    data_path = ".\\user_test_files\\"
    mat_file = "aud_403_sdreamer_3class.mat"
    filename = os.path.splitext(os.path.basename(mat_file))[0]
    mat = loadmat(os.path.join(data_path, mat_file))
    sleep_scores_post, df = postprocess_sleep_scores(mat=mat, return_table=True)
    # df.to_excel(os.path.join(data_path, f"{filename}_table.xlsx"))

    # write two sheets in one xlsx file
    df_stats = get_pred_label_stats(df)
    with pd.ExcelWriter(os.path.join(data_path, f"{filename}_table.xlsx")) as writer:
        df.to_excel(writer, sheet_name="Sleep_bouts")
        df_stats.to_excel(writer, sheet_name="Sleep_stats")
        worksheet = writer.sheets["Sleep_stats"]
        worksheet.set_column(0, 0, 20)
