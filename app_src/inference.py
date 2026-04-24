# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:09:21 2023

@author: yzhao
"""

from pathlib import Path

from scipy.io import loadmat, savemat

from app_src.config import SLEEP_SCORING_MODEL
import app_src.run_inference_stats_model as run_inference_stats_model
import app_src.run_inference_ne as run_inference_ne
import app_src.run_inference_sdreamer as run_inference_sdreamer
from app_src.postprocessing import postprocess_sleep_scores

MODEL_PATH = Path(__file__).parents[1] / "models" / "sdreamer" / "checkpoints"


def run_inference(
    mat,
    model_path=MODEL_PATH,
    num_class=3,
    postprocess=False,
    output_path=None,
    save_inference=False,
):
    # num_class = 3
    should_postprocess = postprocess and SLEEP_SCORING_MODEL == "sdreamer"

    if SLEEP_SCORING_MODEL == "stats_model":
        predictions, confidence = run_inference_stats_model.infer(mat, model_path=None)
    elif SLEEP_SCORING_MODEL == "sdreamer":
        ne = mat.get("ne")
        if ne is not None and len(ne) != 0:
            predictions, confidence = run_inference_ne.infer(mat, model_path)
        else:
            predictions, confidence = run_inference_sdreamer.infer(mat, model_path)
    else:
        raise ValueError(
            "SLEEP_SCORING_MODEL must be 'sdreamer' or 'stats_model', "
            f"got {SLEEP_SCORING_MODEL!r}."
        )

    mat["sleep_scores"] = predictions
    mat["confidence"] = confidence
    if should_postprocess:
        predictions = postprocess_sleep_scores(mat)
        mat["sleep_scores"] = predictions

    if output_path is not None:
        # output_path = os.path.splitext(output_path)[0] + ".mat"
        # Filter out the default keys
        mat_filtered = {}
        for key, value in mat.items():
            if not key.startswith("_"):
                mat_filtered[key] = value
        savemat(output_path, mat_filtered)
    return mat, output_path


if __name__ == "__main__":
    data_path = Path("../user_test_files/")
    mat_file = data_path / "F268_FP-Data.mat"
    mat = loadmat(mat_file, squeeze_me=True)
    mat, output_path = run_inference(mat, postprocess=False)
