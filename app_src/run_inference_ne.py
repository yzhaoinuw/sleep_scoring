# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:02:18 2024

@author: yzhao
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app_src.preprocessing import reshape_sleep_data_ne
from models.sdreamer import n2nBaseLineNE


class SequenceDataset(Dataset):
    def __init__(self, eeg_emg_standardized, normalized_ne_data):
        self.eeg_emg_traces = eeg_emg_standardized
        self.ne_trace = normalized_ne_data

    def __len__(self):
        return self.eeg_emg_traces.shape[0]

    def __getitem__(self, idx):
        eeg_emg_trace = self.eeg_emg_traces[idx]
        ne_trace = self.ne_trace[idx]
        return eeg_emg_trace, ne_trace


def make_dataset(data: dict, n_sequences: int = 64):
    eeg_standardized, emg_standardized, ne_standardized = reshape_sleep_data_ne(
        data, standardize=True, has_labels=False
    )
    eeg_emg_standardized = np.stack((eeg_standardized, emg_standardized), axis=1)
    eeg_emg_standardized = np.expand_dims(
        eeg_emg_standardized, axis=2
    )  # shape [n_seconds, 2, 1, seq_len]
    eeg_emg_standardized = torch.from_numpy(eeg_emg_standardized)
    # ne_standardized = np.expand_dims(ne_standardized, axis=2)
    ne_standardized = torch.from_numpy(ne_standardized)

    n_seconds = eeg_emg_standardized.shape[0]
    n_to_crop = n_seconds % n_sequences
    if n_to_crop != 0:
        eeg_emg_standardized = torch.cat(
            [eeg_emg_standardized[:-n_to_crop], eeg_emg_standardized[-n_sequences:]],
            dim=0,
        )
        ne_standardized = torch.cat(
            [ne_standardized[:-n_to_crop], ne_standardized[-n_sequences:]], dim=0
        )

    eeg_emg_standardized = eeg_emg_standardized.reshape(
        (
            -1,
            n_sequences,
            eeg_emg_standardized.shape[1],
            eeg_emg_standardized.shape[2],
            eeg_emg_standardized.shape[3],
        )
    )
    ne_standardized = ne_standardized.reshape(
        (
            -1,
            n_sequences,
            1,
            ne_standardized.shape[1],
        )
    )
    dataset = SequenceDataset(eeg_emg_standardized, ne_standardized)
    return dataset, n_seconds, n_to_crop


# %%

config = dict(
    features="ALL",
    n_sequences=256,
    useNorm=True,
    seq_len=512,
    patch_len=16,
    ne_patch_len=10,
    stride=8,
    padding_patch="end",
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    individual=0,
    mix_type=0,
    c_out=3,
    d_model=128,
    n_heads=8,
    e_layers=2,
    ca_layers=1,
    seq_layers=3,
    d_ff=512,
    dropout=0.1,
    path_drop=0.0,
    activation="glu",
    norm_type="layernorm",
    output_attentions=False,
)


def build_args(**kwargs):
    parser = argparse.ArgumentParser(description="Transformer family for sleep scoring")
    args = parser.parse_args()
    parser_dict = vars(args)

    for k, v in config.items():
        parser_dict[k] = v
    for k, v in kwargs.items():
        parser_dict[k] = v
    return args


# %%
def infer(data, model_path, batch_size=32):
    """
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    model_path : pathlib Path object
        DESCRIPTION.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 32.

    Returns
    -------
    all_pred : TYPE
        DESCRIPTION.
    all_prob : TYPE
        DESCRIPTION.

    """
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = n2nBaseLineNE.Model(args)
    model = model.to(device)
    state_dicts = list(model_path.glob("*ne_256.pt"))
    if state_dicts:
        state_dict_path = state_dicts[0]
    else:
        state_dict_path = list(model_path.glob("*.pt"))[0]
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    n_sequences = config["n_sequences"]
    dataset, n_seconds, n_to_crop = make_dataset(data, n_sequences=n_sequences)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        all_pred = []
        all_prob = []

        with tqdm(total=n_seconds, unit=" seconds of signal") as pbar:
            for batch, (traces, ne_trace) in enumerate(data_loader, 1):
                traces = traces.to(device)  # [batch_size, 64, 2, 1, 512]
                ne_trace = ne_trace.to(device)
                out_dict = model(traces, ne_trace, label=None)
                out = out_dict["out"]
                prob = torch.max(torch.softmax(out, dim=1), dim=1).values
                all_prob.append(prob.detach().cpu())
                pred = np.argmax(out.detach().cpu(), axis=1)
                all_pred.append(pred)
                pbar.update(batch_size * n_sequences)
            pbar.set_postfix({"Number of batches": batch})

        if n_to_crop != 0:
            all_pred[-1] = torch.cat(
                (
                    all_pred[-1][: -args.n_sequences],
                    all_pred[-1][-args.n_sequences :][-n_to_crop:],
                )
            )
            all_prob[-1] = torch.cat(
                (
                    all_prob[-1][: -args.n_sequences],
                    all_prob[-1][-args.n_sequences :][-n_to_crop:],
                )
            )

        all_pred = np.concatenate(all_pred)
        all_prob = np.concatenate(all_prob)

    return all_pred, all_prob


# %%
if __name__ == "__main__":
    from pathlib import Path

    from scipy.io import loadmat

    model_path = Path("C:/Users/yzhao/python_projects/sleep_scoring/models/sdreamer/checkpoints/")
    mat_file = "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/115_gs.mat"
    mat = loadmat(mat_file, squeeze_me=True)
    all_pred, all_prob = infer(mat, model_path)
    sleep_scores = mat.get("sleep_scores")
    if sleep_scores is not None:
        clip_len = min(len(all_pred), len(sleep_scores))
        acc = np.sum(all_pred[:clip_len] == sleep_scores[:clip_len]) / clip_len
