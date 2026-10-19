"""Cued Speech Data Package.

This package contains data files and utilities for the cued speech system.
"""

from .cue_mappings import (
    CONSONANTS,
    VOWELS,
    map_syllable_to_cue,
    text_to_ipa,
)

from .download_models import (
    get_default_model_paths,
    check_file_exists,
    find_model_file,
    ensure_model_files_available,
    download_file,
    download_model_files,
    setup_model_files,
    get_github_release_info,
    list_available_model_files,
    create_model_files_manifest,
)

__all__ = [
    "CONSONANTS",
    "VOWELS", 
    "map_syllable_to_cue",
    "text_to_ipa",
    "get_default_model_paths",
    "check_file_exists",
    "find_model_file",
    "ensure_model_files_available",
    "download_file",
    "download_model_files",
    "setup_model_files",
    "get_github_release_info",
    "list_available_model_files",
    "create_model_files_manifest",
] 