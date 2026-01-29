# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:25:38 2024

@author: yzhao
"""

import math
from fractions import Fraction

import numpy as np
from scipy import signal, stats
from scipy.io import loadmat


def trim_missing_labels(filt, trim="b"):
    first = 0
    trim = trim.upper()
    if "F" in trim:
        for i in filt:
            if i == -1 or np.isnan(i):
                first = first + 1
            else:
                break
    last = len(filt)
    if "B" in trim:
        for i in filt[::-1]:
            if i == -1 or np.isnan(i):
                last = last - 1
            else:
                break
    return filt[first:last]


def reshape_sleep_data_ne(
    mat, segment_size=512, segment_size_ne=10, standardize=False, has_labels=True
):
    eeg = mat["eeg"]
    emg = mat["emg"]
    if standardize:
        eeg = stats.zscore(eeg)
        emg = stats.zscore(emg)

    eeg_freq = mat["eeg_frequency"]

    # if sampling rate is much higher than 512, downsample using poly resample
    if math.ceil(eeg_freq) != segment_size and math.floor(eeg_freq) != segment_size:
        down, up = Fraction(eeg_freq / segment_size).limit_denominator(100).as_integer_ratio()
        eeg = signal.resample_poly(eeg, up, down)
        emg = signal.resample_poly(emg, up, down)
        eeg_freq = segment_size

    # recalculate end time after upsampling ne
    resampled_end_time_eeg = math.floor(len(eeg) / eeg_freq)

    ne = mat.get("ne")
    if ne is not None and len(ne) != 0:
        ne = ne.flatten()
        if standardize:
            ne = stats.zscore(ne)
        ne_freq = mat["ne_frequency"]
        resampled_end_time_ne = math.floor(len(ne) / ne_freq)
        end_time = min(resampled_end_time_eeg, resampled_end_time_ne)
    else:
        ne_freq = segment_size_ne
        ne = np.zeros(resampled_end_time_eeg * ne_freq, dtype=np.float32)
        end_time = resampled_end_time_eeg

    time_sec = np.arange(end_time)
    start_indices = np.ceil(time_sec * eeg_freq).astype(int)
    start_indices = start_indices[
        :, np.newaxis
    ]  # Reshape start_indices to be a column vector (N, 1)
    segment_array = np.arange(segment_size)
    indices = (
        start_indices + segment_array
    )  # Use broadcasting to add the range_array to each start index

    start_indices_ne = np.ceil(time_sec * ne_freq).astype(int)
    start_indices_ne = start_indices_ne[:, np.newaxis]
    segment_array_ne = np.arange(segment_size_ne)
    ne_indices = start_indices_ne + segment_array_ne

    eeg_reshaped = eeg[indices]
    emg_reshaped = emg[indices]
    ne_reshaped = ne[ne_indices]

    if has_labels:
        sleep_scores = mat["sleep_scores"]
        sleep_scores = trim_missing_labels(sleep_scores, trim="b")  # trim trailing zeros
        return eeg_reshaped, emg_reshaped, ne_reshaped, sleep_scores

    return eeg_reshaped, emg_reshaped, ne_reshaped


def reshape_sleep_data(mat, segment_size=512, standardize=False, has_labels=True):
    eeg = mat["eeg"]
    emg = mat["emg"]

    if standardize:
        eeg = stats.zscore(eeg)
        emg = stats.zscore(emg)

    eeg_freq = mat["eeg_frequency"]

    # clip the last non-full second and take the shorter duration of the two
    end_time = math.floor(eeg.size / eeg_freq)

    # if sampling rate is much higher than 512, downsample using poly resample
    if math.ceil(eeg_freq) != segment_size and math.floor(eeg_freq) != segment_size:
        down, up = Fraction(eeg_freq / segment_size).limit_denominator(100).as_integer_ratio()
        print(f"file has sampling frequency of {eeg_freq}.")
        eeg = signal.resample_poly(eeg, up, down)
        emg = signal.resample_poly(emg, up, down)
        eeg_freq = segment_size

    time_sec = np.arange(end_time)
    start_indices = np.ceil(time_sec * eeg_freq).astype(int)

    # Reshape start_indices to be a column vector (N, 1)
    start_indices = start_indices[:, np.newaxis]
    segment_array = np.arange(segment_size)
    # Use broadcasting to add the range_array to each start index
    indices = start_indices + segment_array

    eeg_reshaped = eeg[indices]
    emg_reshaped = emg[indices]

    if has_labels:
        sleep_scores = mat["sleep_scores"]
        sleep_scores = trim_missing_labels(sleep_scores, trim="b")  # trim trailing zeros
        return eeg_reshaped, emg_reshaped, sleep_scores

    return eeg_reshaped, emg_reshaped


if __name__ == "__main__":
    path = "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/"
    mat_file = path + "115_gs.mat"
    mat = loadmat(mat_file, squeeze_me=True)
    eeg_reshaped, emg_reshaped, ne_reshaped, sleep_scores = reshape_sleep_data_ne(mat)
