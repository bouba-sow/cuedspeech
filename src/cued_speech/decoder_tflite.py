"""Cued Speech Decoder Module with TFLite Support.

This module provides functionality for decoding cued speech videos using separate
TFLite models for face, hand, and pose detection instead of MediaPipe Holistic.
This enables easier integration with Flutter mobile applications.
"""

import csv
import json
import math
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from itertools import groupby

import absl.logging
import cv2
import kenlm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import ctc_decoder
import argparse

# Turn off Abseil's preinit-warning
absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity(absl.logging.ERROR)

# Try to import TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. TFLite models cannot be loaded.")
    TFLITE_AVAILABLE = False

# Try to import MediaPipe Tasks API for .task file support
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    MEDIAPIPE_TASKS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MediaPipe Tasks API not available. .task files cannot be loaded. ({e})")
    MEDIAPIPE_TASKS_AVAILABLE = False

# Constants
NUM_WORKERS = 4
QUEUE_MAXSIZE = 10
INFERENCE_BUFFER_SIZE = 5
INFERENCE_INTERVAL = 15

# Overlap-save windowing parameters
WINDOW_SIZE = 100  # Total frames processed at once
COMMIT_SIZE = 50   # Frames we keep from each window (after first two chunks)
LEFT_CONTEXT = 25  # Left context frames
RIGHT_CONTEXT = 25 # Right context frames

# IPA to LIAPHON mapping for French phoneme correction
IPA_TO_LIAPHON = {
    "a": "a",
    "ə": "x",
    "ɛ": "e^",
    "œ": "x^",
    "i": "i",
    "y": "y",
    "e": "e",
    "y": "y",
    "u": "u",
    "ɔ": "o",
    "o": "o^",
    "ɑ̃": "a~",
    "ɛ̃": "e~",
    "ɔ̃": "o~",
    "œ̃": "x~",
    " ": "_",
    "b": "b",
    "c": "k",
    "d": "d",
    "f": "f",
    "ɡ": "g",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "s": "s",
    "t": "t",
    "v": "v",
    "w": "w",
    "z": "z",
    "ɥ": "h",
    "ʁ": "r",
    "ʃ": "s^",
    "ʒ": "z^",
    "ɲ": "gn",
    "ŋ": "ng",
}
LIAPHON_TO_IPA = {v: k for k, v in IPA_TO_LIAPHON.items()}

# Landmark indices
LIP_INDICES = [17, 314, 405, 321, 375, 291, 84, 181, 91, 146,
               0, 267, 269, 270, 409, 40, 37, 39, 40, 185,
               61, 78, 95, 88, 87, 14, 317, 402, 324, 308,
               80, 81, 82, 13, 312, 311, 319, 308]
HAND_INDICES = list(range(21))
FACE_INDICES = [234, 200, 214, 280, 454]


def load_vocabulary(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load phoneme-to-index mapping with special tokens."""
    with open(vocab_path, "r") as file:
        reader = csv.reader(file)
        vocabulary_list = [row[0] for row in reader]

    # Remove duplicates while preserving order
    seen = set()
    unique_vocab = [x for x in vocabulary_list if not (x in seen or seen.add(x))]

    # Add special tokens if not present - BLANK should be at index 0 for CTC
    special_tokens = ["<BLANK>", "<UNK>", "<SOS>", "<EOS>", "<PAD>"]
    for token in reversed(special_tokens):
        if token not in unique_vocab:
            unique_vocab.insert(0, token)

    # Ensure BLANK is at index 0
    if unique_vocab[0] != "<BLANK>":
        unique_vocab.remove("<BLANK>")
        unique_vocab.insert(0, "<BLANK>")

    phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(unique_vocab)}
    index_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_index.items()}

    return phoneme_to_index, index_to_phoneme


class ThreeStreamFusionEncoder(nn.Module):
    """Three-stream fusion encoder for hand shape, hand position, and lips features."""

    def __init__(
        self,
        hand_shape_dim: int,
        hand_pos_dim: int,
        lips_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hand_shape_gru = nn.GRU(
            hand_shape_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True
        )
        self.hand_pos_gru = nn.GRU(
            hand_pos_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True
        )
        self.lips_gru = nn.GRU(
            lips_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True
        )
        self.fusion_gru = nn.GRU(
            hidden_dim * 6,
            hidden_dim * 3,
            n_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(
        self, hand_shape: torch.Tensor, hand_pos: torch.Tensor, lips: torch.Tensor
    ) -> torch.Tensor:
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)
        hand_pos_out, _ = self.hand_pos_gru(hand_pos)
        lips_out, _ = self.lips_gru(lips)
        combined_features = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)
        fusion_out, _ = self.fusion_gru(combined_features)
        return fusion_out


class CTCModel(nn.Module):
    """CTC-only model for cued speech recognition."""

    def __init__(
        self,
        hand_shape_dim: int,
        hand_pos_dim: int,
        lips_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = ThreeStreamFusionEncoder(
            hand_shape_dim, hand_pos_dim, lips_dim, hidden_dim, n_layers
        )
        encoder_output_dim = hidden_dim * 6
        self.ctc_fc = nn.Linear(encoder_output_dim, output_dim)

    def forward(
        self, hand_shape: torch.Tensor, hand_pos: torch.Tensor, lips: torch.Tensor
    ) -> torch.Tensor:
        encoder_out = self.encoder(hand_shape, hand_pos, lips)
        ctc_logits = self.ctc_fc(encoder_out)
        return ctc_logits


class TFLiteModelWrapper:
    """Wrapper for TFLite model inference."""
    
    def __init__(self, model_path: str, model_type: str):
        """
        Initialize TFLite model wrapper.
        
        Args:
            model_path: Path to the TFLite model file
            model_type: Type of model ('face', 'hand', or 'pose')
        """
        if not TFLITE_AVAILABLE:
            raise RuntimeError("TensorFlow is not available. Cannot load TFLite models.")
        
        # Validate file before attempting to load
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Guard against MediaPipe Task files which are NOT raw .tflite models
        _, ext = os.path.splitext(model_path)
        try:
            with open(model_path, "rb") as f:
                header = f.read(16)
        except Exception as e:
            raise RuntimeError(f"Failed to read model file: {model_path} ({e})")
        # TFLite flatbuffer models typically start with 'TFL3'
        is_tflite_flatbuffer = header.startswith(b"TFL3")
        looks_like_task = (ext.lower() == ".task") or (b"mediapipe" in header.lower())
        if looks_like_task and not is_tflite_flatbuffer:
            raise ValueError(
                "Model provided appears to be a MediaPipe .task file, which cannot be loaded "
                "by the TFLite Interpreter. Provide a raw .tflite model instead, or run the "
                "MediaPipe Tasks APIs for .task files.\n"
                f"Given path: {model_path}"
            )
        
        self.model_type = model_type
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Loaded {model_type} TFLite model from {model_path}")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Output details: {len(self.output_details)} outputs")
        for i, output in enumerate(self.output_details):
            print(f"    Output {i}: {output['shape']}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image in BGR format (from OpenCV)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get input shape from model
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize image to model input size
        resized = cv2.resize(image_rgb, (width, height))
        
        # Normalize to [0, 1] if model expects float32
        if self.input_details[0]['dtype'] == np.float32:
            resized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(resized, axis=0)
        
        return input_data
    
    def inference(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on the input image.
        
        Args:
            image: Input image in BGR format (from OpenCV)
            
        Returns:
            Dictionary with model outputs
        """
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        outputs = {}
        for i, output_detail in enumerate(self.output_details):
            output_data = self.interpreter.get_tensor(output_detail['index'])
            outputs[f'output_{i}'] = output_data
        
        return outputs


class MediaPipeTasksWrapper:
    """Wrapper for MediaPipe Tasks API (.task files)."""
    
    def __init__(self, model_path: str, model_type: str):
        """
        Initialize MediaPipe Tasks wrapper for .task files.
        
        Args:
            model_path: Path to the .task model file
            model_type: Type of model ('face', 'hand', or 'pose')
        """
        if not MEDIAPIPE_TASKS_AVAILABLE:
            raise RuntimeError("MediaPipe Tasks API is not available. Cannot load .task files.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_type = model_type
        self.model_path = model_path
        
        # Initialize the appropriate MediaPipe Tasks model
        base_options = BaseOptions(model_asset_path=model_path)
        
        if model_type == 'face':
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_faces=1,  # Optimize for single face
                min_face_detection_confidence=0.3,  # Match MediaPipe Holistic
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        elif model_type == 'hand':
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_hands=2,  # Detect both hands
                min_hand_detection_confidence=0.3,  # Match MediaPipe Holistic
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
        elif model_type == 'pose':
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                min_pose_detection_confidence=0.3,  # Match MediaPipe Holistic
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Loaded {model_type} MediaPipe Tasks model from {model_path}")
    
    def inference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on the input image using MediaPipe Tasks API.
        
        Args:
            image: Input image in BGR format (from OpenCV)
            
        Returns:
            Dictionary with landmarks in MediaPipe format
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Run inference
        if self.model_type == 'face':
            result = self.landmarker.detect(mp_image)
            return {'face_landmarks': result.face_landmarks}
        elif self.model_type == 'hand':
            result = self.landmarker.detect(mp_image)
            return {'hand_landmarks': result.hand_landmarks, 'handedness': result.handedness}
        elif self.model_type == 'pose':
            result = self.landmarker.detect(mp_image)
            return {'pose_landmarks': result.pose_landmarks}
        
        return {}
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


class MediaPipeStyleLandmarkExtractor:
    """
    Extract landmarks in MediaPipe-compatible format using separate TFLite models.
    This class mimics the interface of MediaPipe Holistic but uses TFLite models.
    """
    
    def __init__(
        self,
        face_model_path: Optional[str] = None,
        hand_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
    ):
        """
        Initialize landmark extractor with TFLite or MediaPipe Tasks models.
        
        Automatically detects the model type based on file extension:
        - .task files -> MediaPipe Tasks API
        - .tflite files -> TFLite Interpreter
        
        Args:
            face_model_path: Path to face mesh model (.tflite or .task)
            hand_model_path: Path to hand landmark model (.tflite or .task)
            pose_model_path: Path to pose landmark model (.tflite or .task)
        """
        self.face_model = None
        self.hand_model = None
        self.pose_model = None
        self.use_mediapipe_tasks = False
        
        if face_model_path and os.path.exists(face_model_path):
            self.face_model = self._create_model_wrapper(face_model_path, 'face')
        
        if hand_model_path and os.path.exists(hand_model_path):
            self.hand_model = self._create_model_wrapper(hand_model_path, 'hand')
        
        if pose_model_path and os.path.exists(pose_model_path):
            self.pose_model = self._create_model_wrapper(pose_model_path, 'pose')
    
    def _create_model_wrapper(self, model_path: str, model_type: str):
        """
        Create appropriate model wrapper based on file extension.
        
        Args:
            model_path: Path to model file
            model_type: Type of model ('face', 'hand', or 'pose')
            
        Returns:
            Model wrapper instance (TFLiteModelWrapper or MediaPipeTasksWrapper)
        """
        _, ext = os.path.splitext(model_path)
        
        if ext.lower() == '.task':
            # Use MediaPipe Tasks API
            if not MEDIAPIPE_TASKS_AVAILABLE:
                raise RuntimeError(
                    f"MediaPipe Tasks API is required for .task files but is not available. "
                    f"Install with: pip install mediapipe"
                )
            self.use_mediapipe_tasks = True
            return MediaPipeTasksWrapper(model_path, model_type)
        else:
            # Use TFLite Interpreter (for .tflite or other extensions)
            if not TFLITE_AVAILABLE:
                raise RuntimeError(
                    f"TFLite runtime is required for .tflite files but is not available. "
                    f"Install with: pip install tflite-runtime"
                )
            return TFLiteModelWrapper(model_path, model_type)
    
    def process(self, image: np.ndarray) -> 'LandmarkResults':
        """
        Process an image and extract landmarks.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            LandmarkResults object containing face, hand, and pose landmarks
        """
        results = LandmarkResults()
        
        # Face landmarks
        if self.face_model:
            face_outputs = self.face_model.inference(image)
            results.face_landmarks = self._parse_face_landmarks(face_outputs)
        
        # Hand landmarks (right hand)
        if self.hand_model:
            hand_outputs = self.hand_model.inference(image)
            results.right_hand_landmarks = self._parse_hand_landmarks(hand_outputs)
        
        # Pose landmarks
        if self.pose_model:
            pose_outputs = self.pose_model.inference(image)
            results.pose_landmarks = self._parse_pose_landmarks(pose_outputs)
        
        return results
    
    def _parse_face_landmarks(self, outputs: Dict[str, Any]) -> Optional['Landmarks']:
        """
        Parse face landmarks from model outputs (MediaPipe Tasks or TFLite).
        
        The face mesh model typically outputs 468 landmarks with (x, y, z) coordinates.
        """
        # Check if this is MediaPipe Tasks output (native landmarks)
        if 'face_landmarks' in outputs:
            mp_landmarks_list = outputs['face_landmarks']
            if mp_landmarks_list and len(mp_landmarks_list) > 0:
                # Convert MediaPipe landmarks to our Landmarks format
                return self._convert_mediapipe_landmarks_to_custom(mp_landmarks_list[0])
            return None
        
        # Otherwise, parse TFLite output
        if 'output_0' not in outputs:
            return None
        
        landmarks_array = outputs['output_0']
        
        # Handle different output formats
        if landmarks_array.ndim == 3:
            # Shape: (1, num_landmarks, 3)
            landmarks_array = landmarks_array[0]  # Remove batch dimension
        elif landmarks_array.ndim == 2:
            # Shape: (num_landmarks, 3)
            pass
        else:
            print(f"Warning: Unexpected face landmarks shape: {landmarks_array.shape}")
            return None
        
        # Create Landmarks object
        landmarks = Landmarks()
        for i in range(landmarks_array.shape[0]):
            if landmarks_array.shape[1] >= 3:
                x, y, z = landmarks_array[i, 0], landmarks_array[i, 1], landmarks_array[i, 2]
            else:
                x, y = landmarks_array[i, 0], landmarks_array[i, 1]
                z = 0.0
            
            landmark = Landmark(x=float(x), y=float(y), z=float(z))
            landmarks.landmark.append(landmark)
        
        return landmarks
    
    def _parse_hand_landmarks(self, outputs: Dict[str, Any]) -> Optional['Landmarks']:
        """
        Parse hand landmarks from model outputs (MediaPipe Tasks or TFLite).
        
        The hand landmark model outputs 21 landmarks with (x, y, z) coordinates.
        """
        # Check if this is MediaPipe Tasks output (native landmarks)
        if 'hand_landmarks' in outputs:
            mp_landmarks_list = outputs['hand_landmarks']
            handedness_list = outputs.get('handedness', [])
            
            if mp_landmarks_list and len(mp_landmarks_list) > 0:
                # Find the right hand (prefer right hand for cued speech)
                right_hand_idx = None
                for i, handedness in enumerate(handedness_list):
                    if handedness and len(handedness) > 0:
                        # MediaPipe returns 'Right' or 'Left' (from camera perspective)
                        if handedness[0].category_name == 'Right':
                            right_hand_idx = i
                            break
                
                # If no right hand found, use the first detected hand
                if right_hand_idx is None and len(mp_landmarks_list) > 0:
                    right_hand_idx = 0
                
                if right_hand_idx is not None:
                    return self._convert_mediapipe_landmarks_to_custom(mp_landmarks_list[right_hand_idx])
            return None
        
        # Otherwise, parse TFLite output
        if 'output_0' not in outputs:
            return None
        
        landmarks_array = outputs['output_0']
        
        # Handle different output formats
        if landmarks_array.ndim == 3:
            # Shape: (1, num_landmarks, 3)
            landmarks_array = landmarks_array[0]  # Remove batch dimension
        elif landmarks_array.ndim == 2:
            # Shape: (num_landmarks, 3)
            pass
        else:
            print(f"Warning: Unexpected hand landmarks shape: {landmarks_array.shape}")
            return None
        
        # Check if we have enough landmarks
        if landmarks_array.shape[0] < 21:
            return None
        
        # Create Landmarks object
        landmarks = Landmarks()
        for i in range(min(21, landmarks_array.shape[0])):
            if landmarks_array.shape[1] >= 3:
                x, y, z = landmarks_array[i, 0], landmarks_array[i, 1], landmarks_array[i, 2]
            else:
                x, y = landmarks_array[i, 0], landmarks_array[i, 1]
                z = 0.0
            
            landmark = Landmark(x=float(x), y=float(y), z=float(z))
            landmarks.landmark.append(landmark)
        
        return landmarks
    
    def _parse_pose_landmarks(self, outputs: Dict[str, Any]) -> Optional['Landmarks']:
        """
        Parse pose landmarks from model outputs (MediaPipe Tasks or TFLite).
        
        The pose landmark model outputs 33 landmarks with (x, y, z) coordinates.
        """
        # Check if this is MediaPipe Tasks output (native landmarks)
        if 'pose_landmarks' in outputs:
            mp_landmarks_list = outputs['pose_landmarks']
            if mp_landmarks_list and len(mp_landmarks_list) > 0:
                # Convert MediaPipe landmarks to our Landmarks format
                return self._convert_mediapipe_landmarks_to_custom(mp_landmarks_list[0])
            return None
        
        # Otherwise, parse TFLite output
        if 'output_0' not in outputs:
            return None
        
        landmarks_array = outputs['output_0']
        
        # Handle different output formats
        if landmarks_array.ndim == 3:
            # Shape: (1, num_landmarks, 3)
            landmarks_array = landmarks_array[0]  # Remove batch dimension
        elif landmarks_array.ndim == 2:
            # Shape: (num_landmarks, 3)
            pass
        else:
            print(f"Warning: Unexpected pose landmarks shape: {landmarks_array.shape}")
            return None
        
        # Create Landmarks object
        landmarks = Landmarks()
        for i in range(landmarks_array.shape[0]):
            if landmarks_array.shape[1] >= 3:
                x, y, z = landmarks_array[i, 0], landmarks_array[i, 1], landmarks_array[i, 2]
            else:
                x, y = landmarks_array[i, 0], landmarks_array[i, 1]
                z = 0.0
            
            landmark = Landmark(x=float(x), y=float(y), z=float(z))
            landmarks.landmark.append(landmark)
        
        return landmarks
    
    def _convert_mediapipe_landmarks_to_custom(self, mp_landmarks) -> 'Landmarks':
        """
        Convert MediaPipe Tasks API landmarks to our custom Landmarks format.
        
        Args:
            mp_landmarks: MediaPipe NormalizedLandmarkList
            
        Returns:
            Landmarks object in our custom format
        """
        landmarks = Landmarks()
        for mp_landmark in mp_landmarks:
            landmark = Landmark(
                x=float(mp_landmark.x),
                y=float(mp_landmark.y),
                z=float(mp_landmark.z) if hasattr(mp_landmark, 'z') else 0.0
            )
            landmarks.landmark.append(landmark)
        return landmarks
    
    def close(self):
        """Clean up resources."""
        # Close MediaPipe Tasks landmarkers if using them
        if self.use_mediapipe_tasks:
            if self.face_model and hasattr(self.face_model, 'close'):
                self.face_model.close()
            if self.hand_model and hasattr(self.hand_model, 'close'):
                self.hand_model.close()
            if self.pose_model and hasattr(self.pose_model, 'close'):
                self.pose_model.close()
        # TFLite interpreter cleanup is handled automatically


class Landmark:
    """Represents a single landmark point."""
    
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class Landmarks:
    """Represents a collection of landmarks."""
    
    def __init__(self):
        self.landmark = []


class LandmarkResults:
    """Represents the results of landmark detection."""
    
    def __init__(self):
        self.face_landmarks = None
        self.right_hand_landmarks = None
        self.left_hand_landmarks = None
        self.pose_landmarks = None


def polygon_area(xs: List[float], ys: List[float]) -> float:
    """Calculate polygon area using shoelace formula."""
    xs = np.array(xs)
    ys = np.array(ys)
    x_next = np.roll(xs, -1)
    y_next = np.roll(ys, -1)
    area = 0.5 * np.abs(np.dot(xs, y_next) - np.dot(ys, x_next))
    return area


def mean_contour_curvature(points: List[Tuple[float, float]]) -> float:
    """Calculate mean curvature of a contour."""
    pts = np.array(points)
    N = pts.shape[0]
    angles = []
    for i in range(N):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[(i + 1) % N]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        cosang = np.dot(v1, v2) / (norm1 * norm2)
        cosang = np.clip(cosang, -1.0, 1.0)
        angles.append(np.arccos(cosang))
    return np.mean(angles) if angles else 0.0


def get_index_pairs(property_type: str) -> List[Tuple[int, int]]:
    """Get index pairs for hand-face or hand-hand distances."""
    index_pairs = []
    if property_type == 'shape':
        index_pairs.extend([
            (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),  # Wrist to finger tips
        ])
    elif property_type == 'position':
        hand_indices = [8, 9, 12]  # Index and middle fingers
        face_indices = [234, 200, 214, 454, 280]  # Specific face landmarks
        for hand_index in hand_indices:
            for face_index in face_indices:
                index_pairs.append((hand_index, face_index))
    return index_pairs


def scalar_distance(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> float:
    """Calculate scalar distance between two 3D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def extract_features_single_row(
    row: pd.Series, prev: Optional[pd.Series] = None, prev2: Optional[pd.Series] = None
) -> Dict[str, List[float]]:
    """Extract all features from a single coordinate dict (row)."""
    features = {}
    
    # Normalization factors
    f1x, f1y, f1z = row.get("face_x454"), row.get("face_y454"), row.get("face_z454")
    f2x, f2y, f2z = row.get("face_x234"), row.get("face_y234"), row.get("face_z234")
    if None in (f1x, f1y, f1z, f2x, f2y, f2z):
        return features

    face_width = scalar_distance(f1x, f1y, f1z, f2x, f2y, f2z)
    
    # Hand span
    h0x, h0y, h0z = row.get("hand_x0"), row.get("hand_y0"), row.get("hand_z0")
    h9x, h9y, h9z = row.get("hand_x9"), row.get("hand_y9"), row.get("hand_z9")
    if None in (h0x, h0y, h9x, h9y, h0z, h9z):
        hand_span = face_width
    else:
        hand_span = scalar_distance(h0x, h0y, h0z, h9x, h9y, h9z)
        if hand_span == 0:
            hand_span = face_width

    # Hand-face distances & angles
    for h, f in get_index_pairs("position"):
        hx, hy, hz = row.get(f"hand_x{h}"), row.get(f"hand_y{h}"), row.get(f"hand_z{h}")
        fx, fy, fz = row.get(f"face_x{f}"), row.get(f"face_y{f}"), row.get(f"face_z{f}")
        if None not in (hx, hy, hz, fx, fy, fz):
            d = scalar_distance(hx, hy, hz, fx, fy, fz) / face_width
            features[f"distance_face{f}_hand{h}"] = d
            if f == 200:
                dx = (fx - hx) / face_width
                dy = (fy - hy) / face_width
                features[f"angle_face{f}_hand{h}"] = np.arctan2(dy, dx)

    # Hand-hand distances
    for h1, h2 in get_index_pairs("shape"):
        x1, y1, z1 = row.get(f"hand_x{h1}"), row.get(f"hand_y{h1}"), row.get(f"hand_z{h1}")
        x2, y2, z2 = row.get(f"hand_x{h2}"), row.get(f"hand_y{h2}"), row.get(f"hand_z{h2}")
        if None not in (x1, y1, z1, x2, y2, z2):
            d = scalar_distance(x1, y1, z1, x2, y2, z2) / hand_span
            features[f"distance_hand{h1}_hand{h2}"] = d
            
    # Thumb-index angle
    coords = [row.get(k) for k in ("hand_x4", "hand_y4", "hand_z4", "hand_x0", "hand_y0", "hand_z0", "hand_x8", "hand_y8", "hand_z8")]
    if None not in coords:
        x1, y1, z1, x2, y2, z2, x3, y3, z3 = coords
        def get_angle_single(x1, y1, z1, x2, y2, z2, x3, y3, z3):
            v1 = [x1-x2, y1-y2, z1-z2]
            v2 = [x3-x2, y3-y2, z3-z2]
            dot = sum(a*b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a*a for a in v1))
            norm2 = math.sqrt(sum(a*a for a in v2))
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = dot / (norm1 * norm2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
                return math.acos(cos_angle)
            return 0.0
        features["thumb_index_angle"] = get_angle_single(x1, y1, z1, x2, y2, z2, x3, y3, z3)
        
    # Lip metrics
    lx61, ly61, lz61 = row.get("lip_x61"), row.get("lip_y61"), row.get("lip_z61")
    lx291, ly291, lz291 = row.get("lip_x291"), row.get("lip_y291"), row.get("lip_z291")
    if None not in (lx61, ly61, lz61, lx291, ly291, lz291):
        features["lip_width"] = scalar_distance(lx61, ly61, lz61, lx291, ly291, lz291) / face_width
    
    lx0, ly0, lz0 = row.get("lip_x0"), row.get("lip_y0"), row.get("lip_z0")
    lx17, ly17, lz17 = row.get("lip_x17"), row.get("lip_y17"), row.get("lip_z17")
    if None not in (lx0, ly0, lz0, lx17, ly17, lz17):
        features["lip_height"] = scalar_distance(lx0, ly0, lz0, lx17, ly17, lz17) / face_width
    
    # Lip area and curvature
    outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    pts = [(row.get(f"lip_x{i}"), row.get(f"lip_y{i}")) for i in outer]
    if all(p[0] is not None and p[1] is not None for p in pts):
        area = polygon_area([p[0] for p in pts], [p[1] for p in pts])
        features["lip_area"] = area / (face_width ** 2)
        features["lip_curvature"] = mean_contour_curvature(pts)

    # Motion features
    if prev is not None:
        plx0, ply0 = prev.get("lip_x0"), prev.get("lip_y0")
        if None not in (lx0, ly0, plx0, ply0):
            features["lip_velocity_x"] = (lx0 - plx0) / face_width
            features["lip_velocity_y"] = (ly0 - ply0) / face_width
        
        hx8, hy8 = row.get("hand_x8"), row.get("hand_y8")
        phx8, phy8 = prev.get("hand_x8"), prev.get("hand_y8")
        if None not in (hx8, hy8, phx8, phy8):
            features["hand8_velocity_x"] = (hx8 - phx8) / hand_span
            features["hand8_velocity_y"] = (hy8 - phy8) / hand_span
        
        # Acceleration
        if prev2 is not None:
            pplx0, pply0 = prev2.get("lip_x0"), prev2.get("lip_y0")
            if None not in (plx0, pplx0):
                prev_vel_x = (plx0 - pplx0) / face_width
                features["lip_acceleration_x"] = features.get("lip_velocity_x", 0) - prev_vel_x
            if None not in (ply0, pply0):
                prev_vel_y = (ply0 - pply0) / face_width
                features["lip_acceleration_y"] = features.get("lip_velocity_y", 0) - prev_vel_y

    return features


def load_model(
    model_path: str, vocab_path: str
) -> Tuple[Any, Dict[str, int], Dict[int, str]]:
    """Load the TFLite CTC model and vocabulary."""
    # Load vocabulary
    phoneme_to_index, index_to_phoneme = load_vocabulary(vocab_path)

    if not TFLITE_AVAILABLE:
        raise RuntimeError("TFLite runtime is required but not available. Install with: pip install tflite-runtime")
    
    if not model_path.endswith('.tflite'):
        raise ValueError(f"Model must be a .tflite file, got: {model_path}")
    
    print(f"Loading TFLite CTC model from {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  TFLite model loaded successfully")
    
    # Return interpreter wrapped with metadata
    model_wrapper = {
        'type': 'tflite',
        'interpreter': interpreter,
        'input_details': input_details,
        'output_details': output_details
    }
    return model_wrapper, phoneme_to_index, index_to_phoneme


def run_model_inference(
    model: Any,
    Xhs: torch.Tensor,
    Xhp: torch.Tensor,
    Xlp: torch.Tensor
) -> torch.Tensor:
    """Run TFLite model inference.
    
    Args:
        model: TFLite model wrapper dict
        Xhs: Hand shape features (batch_size, seq_len, hand_shape_dim)
        Xhp: Hand position features (batch_size, seq_len, hand_pos_dim)
        Xlp: Lip features (batch_size, seq_len, lips_dim)
        
    Returns:
        Logits tensor (seq_len, vocab_size)
    """
    # TFLite inference with three separate inputs
    interpreter = model['interpreter']
    input_details = model['input_details']
    output_details = model['output_details']
    
    # Convert to numpy arrays (float32)
    hand_shape_np = Xhs.cpu().numpy().astype(np.float32)
    hand_pos_np = Xhp.cpu().numpy().astype(np.float32)
    lips_np = Xlp.cpu().numpy().astype(np.float32)

    # Set input tensors in order: [0]=lips (8), [1]=hand_shape (7), [2]=hand_pos (18)
    if len(input_details) != 3:
        raise RuntimeError(f"Expected TFLite model with 3 inputs, got {len(input_details)}")
    
    interpreter.set_tensor(input_details[0]['index'], lips_np)
    interpreter.set_tensor(input_details[1]['index'], hand_shape_np)
    interpreter.set_tensor(input_details[2]['index'], hand_pos_np)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    logits_np = interpreter.get_tensor(output_details[0]['index'])
    
    # Convert back to torch tensor: (batch_size, seq_len, vocab_size)
    logits = torch.from_numpy(logits_np)
    
    # Return first batch element: (seq_len, vocab_size)
    return logits[0]


def beam_search(
    homophone_lists: List[List[str]], lm_model: kenlm.Model, beam_width: int = 20
) -> List[str]:
    """Perform beam search using KenLM language model."""
    # Initialize KenLM state at begin-sentence
    state_in = kenlm.State()
    lm_model.BeginSentenceWrite(state_in)

    # beams: list of tuples (cum_log10_score, state, word_seq)
    beams = [(0.0, state_in, [])]

    for homos in homophone_lists:
        new_beams = []
        for cum_score, state, seq in beams:
            for w in homos:
                out_state = kenlm.State()
                # BaseScore returns the log10-prob P(w | state)
                inc_score = lm_model.BaseScore(state, w, out_state)
                new_beams.append((cum_score + inc_score, out_state, seq + [w]))
        # keep only top beam_width hypotheses by log10 score
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    # return the best word sequence
    return beams[0][2]


def process_decoded_sequence(
    decoded_sequence: str, liaphon_dict: Dict[str, str]
) -> str:
    """Process decoded sequence to convert LIAPHON to French words."""
    words = decoded_sequence.split()
    french_words = []

    for word in words:
        if word in liaphon_dict:
            french_words.append(liaphon_dict[word])
        else:
            french_words.append(word)

    return " ".join(french_words)


def correct_french_sentences(
    liaphon_decoded: List[List[str]], homophones_path: str, lm_path: str
) -> List[str]:
    """Correct French sentences using homophones and language model."""
    # Convert LIAPHON tokens to IPA and group into IPA words using '_' as word boundary
    ipa_word_sequences: List[List[str]] = []
    for decoded in liaphon_decoded:
        words: List[str] = []
        current_word_chars: List[str] = []
        for phone in decoded:
            if phone == "_":
                if current_word_chars:
                    words.append("".join(current_word_chars))
                    current_word_chars = []
                continue
            ipa_char = LIAPHON_TO_IPA.get(phone, phone)
            current_word_chars.append(ipa_char)
        if current_word_chars:
            words.append("".join(current_word_chars))
        ipa_word_sequences.append(words)

    # Load homophones mapping
    ipa_to_homophones = {}
    with open(homophones_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ipa = entry["ipa"]
            words = entry.get("words", [])
            # if no homophones, keep the IPA token itself as a placeholder
            ipa_to_homophones[ipa] = words if words else [ipa]

    # Load KenLM model
    lm = kenlm.Model(lm_path)

    # Apply beam search correction
    all_beam_decoded = []
    for tokens in ipa_word_sequences:
        homophone_lists = [
            ipa_to_homophones.get(tok, [tok])  # safe fallback to the token itself
            for tok in tokens
        ]
        best_seq = beam_search(homophone_lists, lm, beam_width=20)
        all_beam_decoded.append(best_seq)

    # Join into French sentences
    french_beam_sentences = [" ".join(seq).capitalize() for seq in all_beam_decoded]
    # add a full stop at the end of the sentence if it doesn't have one
    french_beam_sentences = [sentence + "." if not sentence.endswith(".") else sentence for sentence in french_beam_sentences]
    return french_beam_sentences


def write_subtitled_video(
    input_path: str, recognition_results: deque, output_path: str, fps: float
) -> None:
    """Write subtitled video with decoded French sentences at the bottom."""
    import unicodedata

    def remove_accents(text: str) -> str:
        """Remove accents from text for OpenCV compatibility."""
        nfd = unicodedata.normalize("NFD", text)
        without_accents = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        return without_accents

    # Load phoneme mapping for IPA to French conversion
    liaphon_to_french = {}
    try:
        # Try to load from download folder first
        from .data_manager import get_data_file_path
        ipa_file_path = get_data_file_path("ipa_to_french")
        if ipa_file_path:
            df_liaphon = pd.read_csv(ipa_file_path)
        else:
            # Fallback to hardcoded path
            df_liaphon = pd.read_csv("download/ipa_to_french.csv")
        df_liaphon.dropna(inplace=True)
        liaphon_to_french = dict(zip(df_liaphon["liaphon"], df_liaphon["french"]))
    except FileNotFoundError:
        print("Warning: Could not load IPA to French mapping file")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) & ~1  # make even
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) & ~1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Sort results by frame
    results = sorted(list(recognition_results), key=lambda r: r["frame"])
    idx = 0
    current_text = ""
    next_frame_update = results[0]["frame"] if results else float("inf")

    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Update decoded subtitle when reaching next inference frame
        if frame_num >= next_frame_update and idx < len(results):
            entry = results[idx]
            corrected = entry.get("french_sentence")
            if isinstance(corrected, str) and corrected.strip():
                current_text = remove_accents(corrected)
            else:
                decoded_seq = entry["phonemes"]
                if isinstance(decoded_seq, list):
                    decoded_seq = " ".join(decoded_seq)
                decoded_seq = process_decoded_sequence(decoded_seq, liaphon_to_french)
                current_text = remove_accents(decoded_seq)
            idx += 1
            next_frame_update = (
                results[idx]["frame"] if idx < len(results) else float("inf")
            )

        # Draw decoded subtitle at bottom
        text_size = cv2.getTextSize(current_text, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        y = int(height * 0.9)

        # Draw text with black outline and white fill
        cv2.putText(
            frame,
            current_text,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            current_text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        writer.write(frame)

    cap.release()
    writer.release()

    # Re-attach original audio
    temp_output = output_path.replace(".mp4", "_noaudio.mp4")
    os.rename(output_path, temp_output)

    final_command = (
        f'ffmpeg -y -i "{temp_output}" -i "{input_path}" '
        f'-c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}"'
    )
    result = os.system(final_command)
    if result == 0:
        os.remove(temp_output)
        print(f"Subtitled video with audio written to {output_path}")
    else:
        # If ffmpeg fails, keep the video without audio
        os.rename(temp_output, output_path)
        print(f"Subtitled video (without audio) written to {output_path}")
        print("Note: ffmpeg not available, audio could not be attached")


def decode_video_tflite(
    video_path: str,
    right_speaker: str,
    model_path: str,
    output_path: str,
    vocab_path: str,
    lexicon_path: str,
    kenlm_model_path: str,
    homophones_path: str,
    lm_path: str,
    face_tflite_path: Optional[str] = None,
    hand_tflite_path: Optional[str] = None,
    pose_tflite_path: Optional[str] = None,
) -> None:
    """
    Decode a cued speech video using TFLite models for landmark detection.
    
    This function uses separate TFLite models for face, hand, and pose detection
    instead of MediaPipe Holistic, making it easier to replicate in Flutter.

    Args:
        video_path: Path to input cued-speech video
        right_speaker: Which side is the speaker's hand overlay ("speaker1" or "speaker2")
        model_path: Path to the pretrained CTC model file
        output_path: Path to save subtitled video
        vocab_path: Path to vocabulary file
        lexicon_path: Path to lexicon file
        kenlm_model_path: Path to KenLM model file
        homophones_path: Path to homophones JSONL file
        lm_path: Path to language model file
        face_tflite_path: Path to face mesh TFLite model (optional)
        hand_tflite_path: Path to hand landmark TFLite model (optional)
        pose_tflite_path: Path to pose landmark TFLite model (optional)
    """
    # Load CTC model and vocabulary
    model, phoneme_to_index, index_to_phoneme = load_model(model_path, vocab_path)
    
    # Determine device
    if isinstance(model, dict) and model.get('type') == 'tflite':
        device = torch.device("cpu")  # TFLite runs on CPU
        print("Using TFLite model (CPU)")
    else:
        raise ValueError("Invalid model type. Expected TFLite model.")
    
    # Initialize CTC beam decoder
    tokens = list(phoneme_to_index.keys())
    blank_idx = phoneme_to_index["<BLANK>"]
    blank_token = index_to_phoneme[blank_idx]
    if blank_token not in tokens:
        tokens.append(blank_token)
    
    beam_decoder = ctc_decoder(
        lexicon=lexicon_path,
        tokens=tokens,
        lm=lm_path,
        nbest=1,
        beam_size=40,
        lm_weight=3.23,
        word_score=0,
        blank_token='<BLANK>',
        sil_score=0,
        beam_threshold=50,
        sil_token='_',
        unk_word='<UNK>'
    )

    # Initialize TFLite landmark extractor
    print("\n" + "="*70)
    print("Initializing TFLite landmark detection models")
    print("="*70)
    
    landmark_extractor = MediaPipeStyleLandmarkExtractor(
        face_model_path=face_tflite_path,
        hand_model_path=hand_tflite_path,
        pose_model_path=pose_tflite_path,
    )
    
    print("\nTFLite models loaded successfully!")
    print("="*70 + "\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    recognition_results = deque()

    # Process video frames with real-time overlap-save windowing
    frame_count = 0
    valid_features = []
    coordinate_buffer = deque(maxlen=INFERENCE_BUFFER_SIZE)
    
    # Required feature columns for validation
    required_hs_cols = 7
    required_hp_cols = 18
    required_lp_cols = 8
    
    # Windowing state
    all_logits = []
    chunk_idx = 0
    next_window_needed = INFERENCE_INTERVAL  # Start with incremental decoding for first window
    last_incremental_decode = 0  # Track last incremental decode position
    
    print("Processing video with incremental first-window + overlap-save windowing...")
    print(f"Video FPS: {fps}")
    print(f"Window size: {WINDOW_SIZE}, Commit size: {COMMIT_SIZE}, Inference interval: {INFERENCE_INTERVAL}")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame with TFLite models
        results = landmark_extractor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Extract landmarks
        landmarks_data = {"frame_number": frame_count}

        # Hand landmarks
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                landmarks_data[f"hand_x{i}"] = landmark.x
                landmarks_data[f"hand_y{i}"] = landmark.y
                landmarks_data[f"hand_z{i}"] = landmark.z

        # Face landmarks
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                landmarks_data[f"face_x{i}"] = landmark.x
                landmarks_data[f"face_y{i}"] = landmark.y
                landmarks_data[f"face_z{i}"] = landmark.z
                
        # Lip landmarks (same as face but with lip_ prefix for compatibility)
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                landmarks_data[f"lip_x{i}"] = landmark.x
                landmarks_data[f"lip_y{i}"] = landmark.y
                landmarks_data[f"lip_z{i}"] = landmark.z

        # Convert to DataFrame row
        row = pd.Series(landmarks_data)
        coordinate_buffer.append(row)

        # Extract features
        prev = coordinate_buffer[-2] if len(coordinate_buffer) >= 2 else None
        prev2 = coordinate_buffer[-3] if len(coordinate_buffer) >= 3 else None
        features = extract_features_single_row(row, prev, prev2)

        # Validate features
        if features:
            # carry original frame number for alignment in subtitles
            features["_frame"] = landmarks_data.get("frame_number", frame_count)
            hs_count = sum(1 for k in features.keys() if 'hand' in k and 'face' not in k)
            hp_count = sum(1 for k in features.keys() if 'face' in k)
            lp_count = sum(1 for k in features.keys() if 'lip' in k)
            
            if hs_count == required_hs_cols and hp_count == required_hp_cols and lp_count == required_lp_cols:
                valid_features.append(features)
            else:
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: Dropped (incomplete features: hs={hs_count}, hp={hp_count}, lp={lp_count})")
        
        # Check if we have enough valid frames to process
        num_valid = len(valid_features)
        
        # FIRST WINDOW: Incremental decoding every INFERENCE_INTERVAL frames up to frame 75
        if chunk_idx == 0 and num_valid >= next_window_needed and num_valid < WINDOW_SIZE - RIGHT_CONTEXT:
            # Incremental decode from 0 to current position
            decode_end = num_valid
            print(f"\n[First window incremental] Valid frames: {num_valid}, decoding [0, {decode_end}]")
            
            # Get features from start to current position
            incremental_features = valid_features[0:decode_end]
            df = pd.DataFrame(incremental_features)
            
            # Prepare inputs
            hs_cols = [c for c in df.columns if 'hand' in c and 'face' not in c and c != '_frame']
            hp_cols = [c for c in df.columns if 'face' in c and c != '_frame']
            lp_cols = [c for c in df.columns if 'lip' in c and c != '_frame']
            
            Xhs = torch.tensor(df[hs_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            Xhp = torch.tensor(df[hp_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            Xlp = torch.tensor(df[lp_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Forward pass
            incremental_logits = run_model_inference(model, Xhs, Xhp, Xlp)
            
            # Replace or append logits for this segment
            if len(all_logits) == 0:
                all_logits.append(incremental_logits)
            else:
                # Replace the entire accumulated logits with new ones (re-decode with more context)
                all_logits = [incremental_logits]
            
            last_incremental_decode = decode_end
            next_window_needed += INFERENCE_INTERVAL
            
            # Decode current accumulated logits
            full_logits = torch.cat(all_logits, dim=0) if len(all_logits) > 1 else all_logits[0]
            print(f"  Incremental logits shape: {full_logits.shape}")
            
            log_probs = F.log_softmax(full_logits, dim=1)
            beam_results = beam_decoder(log_probs.unsqueeze(0))
            if beam_results and beam_results[0]:
                best = beam_results[0][0]
                pred_tokens = beam_decoder.idxs_to_tokens(best.tokens)[1:-1]
                if len(pred_tokens) > 0:
                    if pred_tokens[-1] == '_':
                        pred_tokens = pred_tokens[:-1]
            else:
                argmax = log_probs.argmax(dim=1).tolist()
                pred_tokens = [index_to_phoneme[i] for i, _ in groupby(argmax) if i != phoneme_to_index['<BLANK>']]
            
            print(f"  Decoded (incremental): {pred_tokens}")
            
            if pred_tokens:
                # Use last frame of decoded segment to reflect real-time processing
                last_frame = incremental_features[-1].get("_frame", frame_count)
                recognition_results.append({
                    'frame': last_frame,
                    'phonemes': pred_tokens,
                })
            
            continue  # Continue accumulating frames for first window
        
        # Transition from incremental to overlap-save when we reach WINDOW_SIZE - RIGHT_CONTEXT
        if chunk_idx == 0 and num_valid >= (WINDOW_SIZE - RIGHT_CONTEXT):
            print(f"\n[Transitioning to overlap-save] Finalizing first window at {num_valid} frames")
            chunk_idx = 1
            next_window_needed = LEFT_CONTEXT + WINDOW_SIZE  # 125 for next window
            continue
        
        if num_valid >= next_window_needed and chunk_idx > 0:
            # Determine window boundaries based on chunk index
            if chunk_idx == 1:
                # First overlap-save chunk: frames 25-124, commit 50-74
                # This refines frames 50-74 that were decoded incrementally
                window_start = LEFT_CONTEXT
                window_end = min(window_start + WINDOW_SIZE - 1, num_valid - 1)
                commit_start = COMMIT_SIZE
                commit_end = min(commit_start + LEFT_CONTEXT - 1, num_valid - 1)
                next_window_needed = COMMIT_SIZE + WINDOW_SIZE  # 150
            else:
                # Regular chunks (2+): window starts at 50, 100, 150, etc.
                window_start = COMMIT_SIZE * (chunk_idx - 1)
                window_end = min(window_start + WINDOW_SIZE - 1, num_valid - 1)
                commit_start = window_start + LEFT_CONTEXT
                commit_end = min(commit_start + COMMIT_SIZE - 1, num_valid - 1)
                next_window_needed = COMMIT_SIZE * chunk_idx + WINDOW_SIZE  # +50 each time
            
            print(f"\n[Valid frames: {num_valid}] Chunk {chunk_idx}: window=[{window_start}, {window_end}], commit=[{commit_start}, {commit_end}]")
            
            # Extract features for this window
            window_features = valid_features[window_start:window_end + 1]
            
            # Convert to DataFrame
            df = pd.DataFrame(window_features)

            # Prepare inputs
            hs_cols = [c for c in df.columns if 'hand' in c and 'face' not in c and c != '_frame']
            hp_cols = [c for c in df.columns if 'face' in c and c != '_frame']
            lp_cols = [c for c in df.columns if 'lip' in c and c != '_frame']

            Xhs = torch.tensor(df[hs_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            Xhp = torch.tensor(df[hp_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            Xlp = torch.tensor(df[lp_cols].values, dtype=torch.float32).unsqueeze(0).to(device)

            # Forward pass over the window
            window_logits = run_model_inference(model, Xhs, Xhp, Xlp)
            
            # Extract logits for commit region
            commit_start_rel = commit_start - window_start
            commit_end_rel = commit_end - window_start
            committed_logits = window_logits[commit_start_rel:commit_end_rel + 1]
            
            print(f"  Window logits shape: {window_logits.shape}, committed: {committed_logits.shape}")
            
            # For chunk 1: replace logits [50-74] from incremental decode
            # For chunk 2+: append new logits
            if chunk_idx == 1:
                # We have logits [0-last_incremental_decode] from incremental phase
                # Keep [0-49], replace [50-74] with better quality from full window
                if len(all_logits) > 0 and last_incremental_decode >= COMMIT_SIZE:
                    # Extract the part to keep from incremental (0-49)
                    kept_logits = all_logits[0][:COMMIT_SIZE]  # First 50 frames
                    # Combine with new commit region
                    all_logits = [kept_logits, committed_logits]
                else:
                    # Fallback: just use the new committed logits
                    all_logits = [committed_logits]
            else:
                # Regular overlap-save: append new committed logits
                all_logits.append(committed_logits)
            
            # Decode using full accumulated logits
            if all_logits:
                full_logits = torch.cat(all_logits, dim=0)
                print(f"  Full accumulated logits shape: {full_logits.shape}")
                
                log_probs = F.log_softmax(full_logits, dim=1)
                
                beam_results = beam_decoder(log_probs.unsqueeze(0))
                if beam_results and beam_results[0]:
                    best = beam_results[0][0]
                    pred_tokens = beam_decoder.idxs_to_tokens(best.tokens)[1:-1]
                    if len(pred_tokens) > 0: 
                        if pred_tokens[-1] == '_': 
                            pred_tokens = pred_tokens[:-1] 
                else:
                    argmax = log_probs.argmax(dim=1).tolist()
                    pred_tokens = [index_to_phoneme[i] for i, _ in groupby(argmax) if i != phoneme_to_index['<BLANK>']]

                print(f"  Decoded sentence after chunk {chunk_idx}: {pred_tokens}")
                    
                if pred_tokens:
                    # Use last frame of commit region to reflect real-time processing
                    commit_last_frame = window_features[commit_end_rel].get("_frame", frame_count)
                    recognition_results.append({
                        'frame': commit_last_frame,
                        'phonemes': pred_tokens,
                    })
                
                chunk_idx += 1
    
    # Process final chunk
    num_valid = len(valid_features)
    if num_valid > 0:
        # Calculate how many frames have been committed so far
        if chunk_idx == 0:
            # Still in incremental phase - use last_incremental_decode
            frames_committed = last_incremental_decode
        elif chunk_idx == 1:
            frames_committed = COMMIT_SIZE + LEFT_CONTEXT  # 75 (kept 0-49 from incremental, added 50-74 from chunk 1)
        else:
            frames_committed = COMMIT_SIZE + LEFT_CONTEXT + (chunk_idx - 2) * COMMIT_SIZE  # 75 + 50*(chunk_idx-2)
        
        if frames_committed < num_valid:
            print(f"\n[Video ended] Processing final chunk with {num_valid - frames_committed} uncommitted frames")
            
            if chunk_idx == 0:
                window_start = 0
                window_end = num_valid - 1
                commit_start = 0
                commit_end = num_valid - 1
            elif chunk_idx == 1:
                window_start = LEFT_CONTEXT
                window_end = num_valid - 1
                commit_start = COMMIT_SIZE
                commit_end = num_valid - 1
            else:
                window_start = COMMIT_SIZE * (chunk_idx - 1)
                window_end = num_valid - 1
                commit_start = window_start + LEFT_CONTEXT
                commit_end = num_valid - 1
            
            if window_end - window_start + 1 >= LEFT_CONTEXT:
                print(f"Final chunk {chunk_idx}: window=[{window_start}, {window_end}], commit=[{commit_start}, {commit_end}]")
                
                window_features = valid_features[window_start:window_end + 1]
                window_size_actual = len(window_features)
                
                if window_size_actual < WINDOW_SIZE:
                    padding_needed = WINDOW_SIZE - window_size_actual
                    zero_feature = {k: 0.0 for k in window_features[0].keys()}
                    window_features.extend([zero_feature] * padding_needed)
                    print(f"  Padded final window with {padding_needed} zero frames")
                
                df = pd.DataFrame(window_features)
                
                hs_cols = [c for c in df.columns if 'hand' in c and 'face' not in c and c != '_frame']
                hp_cols = [c for c in df.columns if 'face' in c and c != '_frame']
                lp_cols = [c for c in df.columns if 'lip' in c and c != '_frame']

                Xhs = torch.tensor(df[hs_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
                Xhp = torch.tensor(df[hp_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
                Xlp = torch.tensor(df[lp_cols].values, dtype=torch.float32).unsqueeze(0).to(device)

                # Forward pass over the window
                window_logits = run_model_inference(model, Xhs, Xhp, Xlp)
                
                commit_start_rel = commit_start - window_start
                commit_end_rel = min(commit_end - window_start, window_size_actual - 1)
                committed_logits = window_logits[commit_start_rel:commit_end_rel + 1]
                
                print(f"  Final window logits shape: {window_logits.shape}, committed: {committed_logits.shape}")
                
                all_logits.append(committed_logits)
                
                if all_logits:
                    full_logits = torch.cat(all_logits, dim=0)
                    print(f"  Final full accumulated logits shape: {full_logits.shape}")
                    
                    log_probs = F.log_softmax(full_logits, dim=1)
                    
                    beam_results = beam_decoder(log_probs.unsqueeze(0))
                    if beam_results and beam_results[0]:
                        best = beam_results[0][0]
                        pred_tokens = beam_decoder.idxs_to_tokens(best.tokens)[1:-1]
                        if len(pred_tokens) > 0: 
                            if pred_tokens[-1] == '_': 
                                pred_tokens = pred_tokens[:-1] 
                    else:
                        argmax = log_probs.argmax(dim=1).tolist()
                        pred_tokens = [index_to_phoneme[i] for i, _ in groupby(argmax) if i != phoneme_to_index['<BLANK>']]

                    print(f"  Final decoded sentence: {pred_tokens}")

                    if pred_tokens:
                        # Use last frame of commit region to reflect real-time processing
                        commit_last_frame = window_features[commit_end_rel].get("_frame", frame_count)
                        recognition_results.append({
                            'frame': commit_last_frame,
                            'phonemes': pred_tokens,
                        })
    
    print(f"\nTotal valid frames: {len(valid_features)} (out of {frame_count} total frames)")
    print(f"Total chunks processed: {len(all_logits)}")
    
    if not all_logits:
        print("No valid frames to decode")

    cap.release()
    landmark_extractor.close()

    # Apply French sentence correction
    if recognition_results:
        liaphon_sequences = [result["phonemes"] for result in recognition_results]
        corrected_sentences = correct_french_sentences(
            liaphon_sequences, homophones_path, kenlm_model_path
        )

        for i, result in enumerate(recognition_results):
            if i < len(corrected_sentences):
                result["french_sentence"] = corrected_sentences[i]
                print(f"Frame {result['frame']}: French sentence: {corrected_sentences[i]}")

    # Write subtitled video
    write_subtitled_video(video_path, recognition_results, output_path, fps)

    print(f"Decoding complete. Output saved to: {output_path}")


def decode_video(
    video_path: str = "download/test_decode.mp4",
    right_speaker: bool = True,
    model_path: str = "download/cuedspeech_model_fixed_temporal.tflite",
    output_path: str = "output/decoder/decoded_video.mp4",
    vocab_path: str = "download/phonelist.csv",
    lexicon_path: str = "download/lexicon.txt",
    kenlm_model_path: str = "download/kenlm_fr.bin",
    homophones_path: str = "download/homophones_dico.jsonl",
    lm_path: str = "download/kenlm_ipa.binary",
    face_tflite_path: Optional[str] = "download/face_landmarker.task",
    hand_tflite_path: Optional[str] = "download/hand_landmarker.task",
    pose_tflite_path: Optional[str] = "download/pose_landmarker_full.task",
) -> None:
    """Decode a cued speech video with the TFLite pipeline, using the CLI's default paths."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    decode_video_tflite(
        video_path=video_path,
        right_speaker=right_speaker,
        model_path=model_path,
        output_path=output_path,
        vocab_path=vocab_path,
        lexicon_path=lexicon_path,
        kenlm_model_path=kenlm_model_path,
        homophones_path=homophones_path,
        lm_path=lm_path,
        face_tflite_path=face_tflite_path,
        hand_tflite_path=hand_tflite_path,
        pose_tflite_path=pose_tflite_path,
    )


def _build_argparser():
    parser = argparse.ArgumentParser(description="Decode cued speech using TFLite landmark models")
    parser.add_argument("--video_path", required=True, help="Path to input cued-speech video")
    parser.add_argument("--output_path", required=True, help="Path to save subtitled video")
    parser.add_argument("--right_speaker", default=True, type=bool, help="Left or right speaker")
    parser.add_argument("--model_path", required=True, help="Path to pretrained CTC model (.pth)")
    parser.add_argument("--vocab_path", required=True, help="Path to vocabulary CSV")
    parser.add_argument("--lexicon_path", required=True, help="Path to lexicon file")
    parser.add_argument("--kenlm_model_path", required=True, help="Path to KenLM model (FR)")
    parser.add_argument("--homophones_path", required=True, help="Path to homophones JSONL")
    parser.add_argument("--lm_path", required=True, help="Path to language model (IPA)")
    parser.add_argument("--face_tflite_path", required=True, help="Path to face landmark TFLite model")
    parser.add_argument("--hand_tflite_path", required=True, help="Path to hand landmark TFLite model")
    parser.add_argument("--pose_tflite_path", default=None, help="Path to pose landmark TFLite model (optional)")
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    decode_video_tflite(
        video_path=args.video_path,
        right_speaker=args.right_speaker,
        model_path=args.model_path,
        output_path=args.output_path,
        vocab_path=args.vocab_path,
        lexicon_path=args.lexicon_path,
        kenlm_model_path=args.kenlm_model_path,
        homophones_path=args.homophones_path,
        lm_path=args.lm_path,
        face_tflite_path=args.face_tflite_path,
        hand_tflite_path=args.hand_tflite_path,
        pose_tflite_path=args.pose_tflite_path,
    )

