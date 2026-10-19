"""Cued Speech Package.

Generate cued speech videos from a video of someone speaking, and decode
cued speech videos into French subtitles.
"""

__version__ = "0.5.0"
__author__ = "Boubacar Sow"
__email__ = "boubasow.pro@gmail.com"

# Suppress protobuf deprecation warnings globally
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

from .generator import CuedSpeechGenerator, generate_cue
from .data_manager import ensure_data_files, download_and_extract_data, get_default_paths

__all__ = [
    "CuedSpeechGenerator",
    "generate_cue",
    "decode_video",
    "ensure_data_files",
    "download_and_extract_data",
    "get_default_paths",
]


def __getattr__(name):
    # The decoder pulls in heavy native dependencies (KenLM, flashlight-text,
    # TFLite), so it is only imported when it is actually used.
    if name == "decode_video":
        from .decoder_tflite import decode_video
        return decode_video
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
