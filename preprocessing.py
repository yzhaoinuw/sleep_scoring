# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:25:38 2024

@author: yzhao
"""

import math
from fractions import Fraction

import numpy as np
from scipy import stats
from scipy import signal
from scipy.io import loadmat


def reshape_sleep_data(mat, segment_size=512, standardize=False):
    eeg = mat["eeg"].flatten()
    emg = mat["emg"].flatten()

    if standardize:
        eeg = stats.zscore(eeg)
        emg = stats.zscore(emg)

    eeg_freq = mat["eeg_frequency"].item()

    # clip the last non-full second and take the shorter duration of the two
    end_time = math.floor(eeg.size / eeg_freq)

    # if sampling rate is much higher than 512, downsample using poly resample
    if math.ceil(eeg_freq) != segment_size and math.floor(eeg_freq) != segment_size:
        down, up = (
            Fraction(eeg_freq / segment_size).limit_denominator(100).as_integer_ratio()
        )
        print(f"file {mat_file} has sampling frequency of {eeg_freq}.")
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
    return eeg_reshaped, emg_reshaped


if __name__ == "__main__":
    path = "./user_test_files/"
    mat_file = path + "preprocessed_240_BL_v2.mat"
    mat = loadmat(mat_file)
    eeg_reshaped, emg_reshaped = reshape_sleep_data(mat)
