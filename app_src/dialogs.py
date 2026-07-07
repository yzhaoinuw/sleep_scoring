# -*- coding: utf-8 -*-
"""Native pywebview file dialogs for choosing and saving files."""

import webview


def open_file_dialog(file_type):
    """
    Open a native file dialog (pywebview) with given file type filters.
    Returns a single file path as a string, or None if canceled.

    Parameters
    ----------
    file_type : tuple[str]
        Example: "mat" or "video"

    """
    if not webview.windows:
        return None

    window = webview.windows[0]

    if file_type == "mat":
        file_types = ("MAT files (*.mat)",)
    elif file_type == "video":
        file_types = ("Videos (*.avi;*.mp4)",)
    else:
        raise ValueError("Hey, it's either mat or video.")

    result = window.create_file_dialog(
        webview.FileDialog.OPEN,
        allow_multiple=False,
        file_types=file_types,
    )

    if result is None:
        return None

    if isinstance(result, (tuple, list)):
        # expected behavior on both Windows and macOS - returns tuple
        open_path = result[0] if result else None
    else:
        # In case macOS returns objc.pyobjc_unicode (string-like object)
        # See save_file_dialog
        open_path = str(result)

    return open_path


def save_file_dialog(file_type, filename):
    """
    Open a native save file dialog (pywebview) with given file type filters.
    Returns a single file path as a string, or None if canceled.

    Parameters
    ----------
    file_type : str
        Example: "mat", "xlsx", or "video"
    filename : str
        Default filename to suggest to the user

    Returns
    -------
    str or None
        The selected file path, or None if canceled
    """
    if not webview.windows:
        return None

    window = webview.windows[0]

    if file_type == "mat":
        file_types = ("MAT files (*.mat)",)
    elif file_type == "xlsx":
        file_types = ("Excel files (*.xlsx)",)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Use 'mat', 'xlsx'.")

    result = window.create_file_dialog(
        webview.FileDialog.SAVE,
        save_filename=filename,
        file_types=file_types,
    )
    # print(result)
    if result is None:
        return None

    # IMPORTANT: On macOS, SAVE dialog returns objc.pyobjc_unicode (string-like)
    # On Windows, it returns a tuple
    # Don't use result[0] - that gets the first CHARACTER, not first element!

    if isinstance(result, (tuple, list)):
        # Windows behavior - returns tuple
        save_path = result[0] if result else None
    else:
        # macOS behavior - returns objc.pyobjc_unicode (string-like object)
        save_path = str(result)  # Convert to regular Python string

    return save_path
