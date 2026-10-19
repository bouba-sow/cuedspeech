"""Data management utilities for Cued Speech.

This module handles downloading and managing the required data files
from GitHub releases.
"""

import os
import zipfile
import urllib.request
import urllib.error
import ssl
import requests
from pathlib import Path
from typing import Optional, Dict, List
import shutil
import tempfile

# GitHub release configuration
GITHUB_REPO = "bouba-sow/cuedspeech"
GITHUB_RELEASE_TAG = "data"
DOWNLOAD_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/download.zip"

# Expected files in the download
EXPECTED_FILES = {
    "model": "cuedspeech-model.pt",
    "vocab": "phonelist.csv", 
    "lexicon": "lexicon.txt",
    "kenlm_fr": "kenlm_fr.bin",
    "homophones": "homophones_dico.jsonl",
    "kenlm_ipa": "kenlm_ipa.binary",
    "ipa_to_french": "ipa_to_french.csv",
    "test_decode": "test_decode.mp4",
    "test_generate": "test_generate.mp4",
    # TFLite/Tasks models saved into download/ as part of data setup
    "face_tflite": "face_landmarker.task",
    "hand_tflite": "hand_landmarker.task",
    "pose_tflite": "pose_landmarker_full.task"
}

# TFLite model download URLs (official MediaPipe models)
TFLITE_MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
}

# Expected directories in the download (handshapes now ship with the package,
# and MFA downloads its own models)
EXPECTED_DIRECTORIES: List[str] = []

def get_data_dir() -> Path:
    """Get the data directory path."""
    # Always use the current working directory for user convenience
    cwd_data_dir = Path.cwd() / "download"
    return cwd_data_dir

def download_tflite_models(data_dir: Path, show_progress: bool = True) -> None:
    """Download MediaPipe TFLite models.
    
    Args:
        data_dir: Directory to save models to
        show_progress: Whether to show download progress
    """
    print("\n📥 Downloading MediaPipe TFLite models (float16, latest)...")
    
    for filename, url in TFLITE_MODEL_URLS.items():
        file_path = data_dir / filename
        
        # Skip if already exists
        if file_path.exists():
            print(f"  ✓ {filename} already exists")
            continue
        
        print(f"  Downloading {filename}...")
        try:
            if show_progress:
                download_with_progress_requests(url, str(file_path))
            else:
                download_with_requests(url, str(file_path))
            print(f"  ✓ {filename} downloaded successfully")
        except Exception as e:
            print(f"  ⚠️  Failed to download {filename}: {e}")
            print(f"  You can download manually from: {url}")

def download_and_extract_data(force_download: bool = False, 
                            show_progress: bool = True) -> Path:
    """Download and extract the data files from GitHub release.
    
    Args:
        force_download: Whether to re-download even if files exist
        show_progress: Whether to show download progress
        
    Returns:
        Path to the extracted data directory
    """
    data_dir = get_data_dir()
    
    # Check if data already exists
    if data_dir.exists() and not force_download:
        # Verify all expected files are present (excluding TFLite models for now)
        missing_files = []
        for file_type, filename in EXPECTED_FILES.items():
            # Skip TFLite models in initial check - they'll be downloaded separately
            if file_type in ['face_tflite', 'hand_tflite', 'pose_tflite']:
                continue
            file_path = data_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if not missing_files:
            print(f"✅ Data files already available at: {data_dir}")
            # Download TFLite models if missing
            download_tflite_models(data_dir, show_progress)
            return data_dir
        else:
            print(f"⚠️  Some files missing: {missing_files}")
            print("Re-downloading data...")
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        zip_path = temp_dir_path / "download.zip"
        
        # Download the zip file
        print(f"📥 Downloading data from: {DOWNLOAD_URL}")
        try:
            if show_progress:
                download_with_progress_requests(DOWNLOAD_URL, str(zip_path))
            else:
                download_with_requests(DOWNLOAD_URL, str(zip_path))
        except Exception as e:
            print(f"❌ Failed to download with requests, trying urllib...")
            try:
                if show_progress:
                    download_with_progress_urllib(DOWNLOAD_URL, str(zip_path))
                else:
                    download_with_urllib(DOWNLOAD_URL, str(zip_path))
            except Exception as e2:
                raise RuntimeError(f"Failed to download data: {e2}")
        
        # Extract the zip file
        print("📦 Extracting data files...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir_path)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to extract zip file: {e}")
        
        # Find the extracted directory (should be 'download')
        extracted_dir = None
        for item in temp_dir_path.iterdir():
            if item.is_dir() and item.name == "download":
                extracted_dir = item
                break
        
        if not extracted_dir:
            raise RuntimeError("Could not find 'download' directory in extracted files")
        
        # Move to final location (current working directory)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        
        shutil.move(str(extracted_dir), str(data_dir))
        
        print(f"✅ Data files extracted to: {data_dir}")
        
        # Download TFLite models
        download_tflite_models(data_dir, show_progress)
        
        return data_dir

def download_with_requests(url: str, destination: str):
    """Download a file using requests library."""
    response = requests.get(url, stream=True, verify=False)  # Disable SSL verification
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_with_progress_requests(url: str, destination: str):
    """Download a file using requests library with progress."""
    response = requests.get(url, stream=True, verify=False)  # Disable SSL verification
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\r📥 Download progress: {percent}%", end="", flush=True)
    
    print()  # New line after progress

def download_with_urllib(url: str, destination: str):
    """Download a file using urllib with SSL context."""
    # Create SSL context that ignores certificate verification
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    urllib.request.urlretrieve(url, destination)

def download_with_progress_urllib(url: str, destination: str):
    """Download a file using urllib with progress and SSL context."""
    # Create SSL context that ignores certificate verification
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    def show_progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            print(f"\r📥 Download progress: {percent}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, destination, show_progress_hook)
    print()  # New line after progress

def ensure_data_files() -> Dict[str, Path]:
    """Ensure all required data files are available.
    
    Returns:
        Dictionary mapping file types to their paths
    """
    data_dir = get_data_dir()
    
    # Check if data directory exists and has all files
    missing_files = []
    available_files = {}
    
    for file_type, filename in EXPECTED_FILES.items():
        file_path = data_dir / filename
        if file_path.exists():
            available_files[file_type] = file_path
        else:
            missing_files.append(filename)
    
    # Check for expected directories
    missing_dirs = []
    for dir_name in EXPECTED_DIRECTORIES:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_files or missing_dirs:
        print(f"⚠️  Missing data files: {missing_files}")
        if missing_dirs:
            print(f"⚠️  Missing directories: {missing_dirs}")
        print("🔄 Downloading missing data...")
        data_dir = download_and_extract_data(force_download=True)
        
        # Re-check after download
        available_files = {}
        for file_type, filename in EXPECTED_FILES.items():
            file_path = data_dir / filename
            if file_path.exists():
                available_files[file_type] = file_path
            else:
                raise RuntimeError(f"Failed to download: {filename}")
    
    return available_files

def get_data_file_path(file_type: str) -> Optional[Path]:
    """Get the path to a specific data file.
    
    Args:
        file_type: Type of file (e.g., 'model', 'vocab', etc.)
        
    Returns:
        Path to the file if found, None otherwise
    """
    if file_type not in EXPECTED_FILES:
        return None
    
    data_dir = get_data_dir()
    file_path = data_dir / EXPECTED_FILES[file_type]
    
    if file_path.exists():
        return file_path
    
    return None

def list_data_files() -> Dict[str, Path]:
    """List all available data files.
    
    Returns:
        Dictionary mapping file types to their paths
    """
    return ensure_data_files()

def cleanup_data_files():
    """Remove all downloaded data files."""
    data_dir = get_data_dir()
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"🗑️  Removed data directory: {data_dir}")

def get_default_paths() -> Dict[str, str]:
    """Get default paths for all data files.
    
    Returns:
        Dictionary mapping file types to their default paths
    """
    data_dir = get_data_dir()
    paths = {}
    
    for file_type, filename in EXPECTED_FILES.items():
        paths[file_type] = str(data_dir / filename)
    
    return paths 