
"""Run-length encoding decode utilities for iMaterialist annotations."""

import numpy as np


def rle_decode(rle_string: str, shape: tuple) -> np.ndarray:
    """Decodes a single RLE-encoded mask string into a 2D binary array.

    iMaterialist RLE is column-major (Fortran order) — this is a dataset-
    specific detail that differs from some other COCO-style RLE formats,
    so getting the `order='F'` reshape right here is critical; getting it
    wrong silently produces transposed/garbled masks rather than an error.

    Args:
        rle_string: space-separated "start length start length ..." string.
        shape: (height, width) of the target mask.

    Returns:
        Binary mask array of shape `shape`, dtype uint8.
    """
    s = rle_string.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape(shape, order="F")
