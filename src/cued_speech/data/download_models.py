"""Model and data file download utilities.

This module provides functions to download default model files and data
when they are not available locally.
"""

import os
import shutil
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, List
import json

# GitHub release URLs for model files
# Update these with your actual GitHub repository and release version
GITHUB_REPO = "bsow/cued-speech-models"  # Your GitHub repo for models
GITHUB_RELEASE_VERSION = "v1.0.0"  # Update this when you create new releases

REMOTE_MODEL_URLS = {
    "model_path": f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_VERSION}/acsr_ctc_tiret_2_0.001_128_2.pt",
    "vocab_path": f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_VERSION}/phonelist.csv", 
    "lexicon_path": f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_VERSION}/lexicon.txt",
    "kenlm_model_path": f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_VERSION}/language_model_fr_2.1.bin",
    "homophones_path": f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_VERSION}/french_homophones_updated_.jsonl",
    "lm_path": f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_VERSION}/french_ipa.binary"
}

def get_default_model_paths() -> dict:
    """Get the default paths for all model files."""
    from ..data_manager import get_data_dir
    data_dir = get_data_dir()
    return {
        "model_path": str(data_dir / "cuedspeech-model.pt"),
        "vocab_path": str(data_dir / "phonelist.csv"),
        "lexicon_path": str(data_dir / "lexicon.txt"),
        "kenlm_model_path": str(data_dir / "kenlm_fr.bin"),
        "homophones_path": str(data_dir / "homophones_dico.jsonl"),
        "lm_path": str(data_dir / "kenlm_ipa.binary")
    }

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists and is readable."""
    return os.path.exists(file_path) and os.path.isfile(file_path)

def download_file(url: str, destination: str, show_progress: bool = True) -> bool:
    """Download a file from URL to destination.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        show_progress: Whether to show download progress
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        if show_progress:
            print(f"Downloading {os.path.basename(destination)}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress
        def show_progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rProgress: {percent}%", end="", flush=True)
        
        if show_progress:
            urllib.request.urlretrieve(url, destination, show_progress_hook)
            print()  # New line after progress
        else:
            urllib.request.urlretrieve(url, destination)
            
        print(f"Successfully downloaded: {destination}")
        return True
        
    except urllib.error.URLError as e:
        print(f"Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_model_files(download_dir: Optional[str] = None, 
                        file_types: Optional[List[str]] = None,
                        force_download: bool = False) -> Dict[str, str]:
    """Download model files from remote URLs.
    
    Args:
        download_dir: Directory to download files to. If None, uses package data directory.
        file_types: List of file types to download. If None, downloads all.
        force_download: Whether to re-download existing files.
        
    Returns:
        dict: Mapping of file types to their downloaded paths
    """
    if download_dir is None:
        download_dir = get_package_data_path()
    
    os.makedirs(download_dir, exist_ok=True)
    
    if file_types is None:
        file_types = list(REMOTE_MODEL_URLS.keys())
    
    downloaded_paths = {}
    
    for file_type in file_types:
        if file_type not in REMOTE_MODEL_URLS:
            print(f"Warning: No download URL configured for {file_type}")
            continue
            
        url = REMOTE_MODEL_URLS[file_type]
        filename = os.path.basename(url)
        destination = os.path.join(download_dir, filename)
        
        # Check if file already exists
        if os.path.exists(destination) and not force_download:
            print(f"File already exists: {destination}")
            downloaded_paths[file_type] = destination
            continue
        
        # Download the file
        if download_file(url, destination):
            downloaded_paths[file_type] = destination
        else:
            print(f"Failed to download {file_type}")
    
    return downloaded_paths

def copy_default_files_to_package_data(package_data_dir: Optional[str] = None) -> dict:
    """Copy default files to package data directory for distribution.
    
    Args:
        package_data_dir: Directory to copy files to. If None, uses a temp directory.
        
    Returns:
        dict: Mapping of file types to their new paths
    """
    if package_data_dir is None:
        package_data_dir = tempfile.mkdtemp(prefix="cued_speech_data_")
    
    os.makedirs(package_data_dir, exist_ok=True)
    
    default_paths = get_default_model_paths()
    new_paths = {}
    
    for file_type, source_path in default_paths.items():
        if check_file_exists(source_path):
            filename = os.path.basename(source_path)
            dest_path = os.path.join(package_data_dir, filename)
            try:
                shutil.copy2(source_path, dest_path)
                new_paths[file_type] = dest_path
                print(f"Copied {file_type}: {filename}")
            except Exception as e:
                print(f"Failed to copy {file_type}: {e}")
        else:
            print(f"Source file not found for {file_type}: {source_path}")
    
    return new_paths

def get_package_data_path() -> str:
    """Get the package data directory path."""
    package_dir = Path(__file__).parent
    return str(package_dir)

def find_model_file(file_type: str, custom_path: Optional[str] = None) -> Optional[str]:
    """Find a model file, checking custom path first, then package data, then default location.
    
    Args:
        file_type: Type of file ('model_path', 'vocab_path', etc.)
        custom_path: Custom path provided by user
        
    Returns:
        Path to the file if found, None otherwise
    """
    # 1. Check custom path first
    if custom_path and check_file_exists(custom_path):
        return custom_path
    
    # 2. Check package data directory
    package_data_dir = get_package_data_path()
    default_paths = get_default_model_paths()
    
    if file_type in default_paths:
        filename = os.path.basename(default_paths[file_type])
        package_file_path = os.path.join(package_data_dir, filename)
        if check_file_exists(package_file_path):
            return package_file_path
    
    # 3. Check default location
    if file_type in default_paths:
        default_path = default_paths[file_type]
        if check_file_exists(default_path):
            return default_path
    
    return None

def ensure_model_files_available() -> dict:
    """Ensure all required model files are available.
    
    Returns:
        dict: Mapping of file types to their available paths
    """
    default_paths = get_default_model_paths()
    available_paths = {}
    
    for file_type, default_path in default_paths.items():
        found_path = find_model_file(file_type)
        if found_path:
            available_paths[file_type] = found_path
        else:
            print(f"Warning: {file_type} not found. Please ensure {default_path} exists.")
    
    return available_paths

def setup_model_files(download_missing: bool = True, 
                     download_dir: Optional[str] = None) -> Dict[str, str]:
    """Set up all model files, downloading missing ones if requested.
    
    Args:
        download_missing: Whether to download missing files
        download_dir: Directory to store downloaded files
        
    Returns:
        dict: Mapping of file types to their available paths
    """
    print("Setting up model files...")
    
    # First check what's already available
    available_paths = ensure_model_files_available()
    missing_files = []
    
    default_paths = get_default_model_paths()
    for file_type in default_paths.keys():
        if file_type not in available_paths:
            missing_files.append(file_type)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        
        if download_missing:
            print("Downloading missing files from GitHub...")
            downloaded_paths = download_model_files(download_dir, missing_files)
            available_paths.update(downloaded_paths)
        else:
            print("Some files are missing. Set download_missing=True to download them.")
    else:
        print("All model files are available!")
    
    return available_paths

def get_github_release_info() -> Dict[str, str]:
    """Get information about the current GitHub release configuration."""
    return {
        "repository": GITHUB_REPO,
        "version": GITHUB_RELEASE_VERSION,
        "base_url": f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_VERSION}"
    }

def list_available_model_files() -> Dict[str, str]:
    """List all available model files and their download URLs."""
    return REMOTE_MODEL_URLS.copy()

def create_model_files_manifest() -> Dict[str, str]:
    """Create a manifest of all model files with their expected filenames."""
    manifest = {}
    for file_type, url in REMOTE_MODEL_URLS.items():
        filename = os.path.basename(url)
        manifest[file_type] = filename
    return manifest 