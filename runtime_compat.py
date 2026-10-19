"""Runtime compatibility helpers shared by backend modules."""

from PIL import Image


_resampling = getattr(Image, "Resampling", Image)
LANCZOS = _resampling.LANCZOS
BILINEAR = _resampling.BILINEAR


def popcount(value):
    """Return the number of set bits on Python 3.8 and newer."""

    value = int(value)
    bit_count = getattr(value, "bit_count", None)
    return bit_count() if bit_count is not None else bin(value).count("1")
