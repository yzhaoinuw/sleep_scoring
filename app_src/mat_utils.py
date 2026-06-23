# -*- coding: utf-8 -*-
"""Helpers for reading fields from a loaded ``.mat`` recording.

These helpers are read-only: they never mutate the loaded dict, so field
aliases are not written back into the user's ``.mat`` file on save.
"""


def get_ne_frequency(mat):
    """Return the NE (fiber-photometry) sampling rate from a loaded ``.mat``.

    Prefers the canonical ``ne_frequency`` field and falls back to the
    ``fp_frequency`` alias emitted by some upstream preprocessing pipelines,
    so a recording that carries an ``ne`` signal but names its sampling rate
    ``fp_frequency`` is handled the same as one using ``ne_frequency``.
    Returns ``None`` when neither field is present.
    """
    ne_frequency = mat.get("ne_frequency")
    if ne_frequency is None:
        ne_frequency = mat.get("fp_frequency")
    return ne_frequency
