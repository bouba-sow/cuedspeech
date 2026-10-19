"""Cued Speech Generation Module.

Generates a cued speech video from a video of someone speaking:

1. extract the audio, and get the text (given, or transcribed with Whisper);
2. align the text on the audio with the Montreal Forced Aligner (MFA);
3. group the phones into Cued Speech keys and predict when the hand moves
   and when the fingers change shape (``cue_timing``);
4. render each frame: track the face, place the hand at the vowel position
   with the right angle and size (``cue_placement``) and draw the handshape
   (``hand_assets``), then encode the video with the original audio.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .cue_placement import HAND_TO_FACE_RATIO, POSITION_ANGLES, FaceTracker
from .cue_timing import NEUTRAL, Phone, Track, build_keys, predict_transitions
from .hand_assets import HandShapeSet

logger = logging.getLogger(__name__)

# Avatar handshapes shipped with the package.
DEFAULT_HANDSHAPES_DIR = str(Path(__file__).parent / "data" / "handshapes")


class CuedSpeechGenerator:
    """Main class for generating cued speech videos from text."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the cued speech generator."""
        defaults = self._get_default_config()
        self.config = dict(defaults)
        for key, value in (config or {}).items():
            if key in defaults:
                self.config[key] = value
            else:
                logger.warning(f"Unknown config key '{key}' ignored. Valid keys: {list(defaults.keys())}")
        self.keys = []
        self._timings: Dict[str, float] = {}

    def _get_default_config(self) -> Dict:
        """Get default configuration for generation."""
        return {
            "language": "french",
            "handshapes_dir": None,  # default: the handshapes shipped with the package
            "mfa_args": ["--beam", "200", "--retry_beam", "400", "--fine_tune"],
            # Hand timing: "rules" (Auto-CS default), "revised", "attina" or "base"
            "timing_model": "rules",
            "pause_threshold": 0.25,  # silence (s) after which the hand goes back to rest
            "hand_scale": 1.0,  # multiplier on the face-proportional hand size
            "correct_head_tilt": True,  # rotate the hand with the head
            "enable_morphing": True,  # crossfade between handshapes instead of switching
            "enable_transparency": True,  # fade the hand in/out when leaving/reaching rest
            "enable_debug_landmarks": False,  # draw the vowel targets on the video
            "skip_whisper": False,
            "model": None,  # preloaded Whisper model
            "whisper_model": "medium",
            "crf": 18,  # x264 quality (lower is better)
        }

    def generate_cue(
        self,
        text: Optional[str],
        video_path: str,
        output_path: str,
        audio_path: Optional[str] = None,
    ) -> str:
        """Generate cued speech video from text input using Whisper + MFA workflow."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if text is None and self.config["skip_whisper"]:
            raise ValueError("skip_whisper requires text to be provided")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="cued_speech_") as temp_dir:
            if audio_path is None:
                t0 = time.perf_counter()
                audio_path = self._extract_audio(video_path, temp_dir)
                self._timings["extract_audio"] = time.perf_counter() - t0

            if text is None:
                logger.info("No text provided, transcribing the audio with Whisper...")
                t0 = time.perf_counter()
                text = self._transcribe_audio(audio_path)
                self._timings["whisper"] = time.perf_counter() - t0
                logger.info(f"Whisper transcription: {text}")
            else:
                logger.info(f"Using provided text: '{text}'")

            t0 = time.perf_counter()
            phones = self._run_mfa_alignment(audio_path, text, temp_dir)
            self._timings["mfa_align"] = time.perf_counter() - t0

            segments = build_keys(phones, self.config["pause_threshold"])
            if not segments:
                raise RuntimeError("No cueable phones found in the alignment")
            position_transitions, shape_transitions = predict_transitions(
                segments, self.config["timing_model"])
            self.keys = [key for segment in segments for key in segment]
            for key in self.keys:
                logger.info(f"  key '{key.label}' ({key.structure}, rank {key.rank}): "
                            f"shape {key.shape} at {key.position}, A1={key.a1:.3f} A3={key.a3:.3f}, "
                            f"M=[{key.m1:.3f}, {key.m2:.3f}] D=[{key.d1:.3f}, {key.d2:.3f}]")

            t0 = time.perf_counter()
            self._render(video_path, audio_path, output_path, position_transitions, shape_transitions)
            self._timings["render"] = time.perf_counter() - t0

        logger.info("⏱️ Timing summary (seconds): " +
                    ", ".join(f"{k}: {v:.1f}" for k, v in self._timings.items()))
        logger.info(f"Cued speech generation complete: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Audio, transcription and alignment
    # ------------------------------------------------------------------

    def _extract_audio(self, video_path: str, temp_dir: str) -> str:
        """Extract audio from video file."""
        from moviepy.editor import VideoFileClip

        audio_path = os.path.join(temp_dir, "audio.wav")
        with VideoFileClip(video_path) as video:
            if video.audio is None:
                raise ValueError(f"Video has no audio track: {video_path}")
            video.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
        return audio_path

    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper."""
        import torch
        import whisper

        model = self.config["model"]
        if model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model(self.config["whisper_model"], device=device)
        result = model.transcribe(audio_path, language=self.config["language"])
        return result["text"].strip()

    def _run_mfa_alignment(self, audio_path: str, text: str, temp_dir: str) -> List[Phone]:
        """Align the text on the audio with MFA and return the phones."""
        from praatio import textgrid as tgio

        mfa_path = self._find_mfa_executable()
        if not mfa_path:
            raise RuntimeError(
                "Montreal Forced Aligner (MFA) not found. Install it with conda "
                "(conda install -c conda-forge montreal-forced-aligner) or run inside "
                "the pixi environment. See "
                "https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html")

        mfa_input_dir = os.path.join(temp_dir, "mfa_input")
        mfa_output_dir = os.path.join(temp_dir, "mfa_output")
        os.makedirs(mfa_input_dir, exist_ok=True)
        shutil.copy(audio_path, os.path.join(mfa_input_dir, "utterance.wav"))
        with open(os.path.join(mfa_input_dir, "utterance.lab"), "w") as f:
            f.write(text)

        model = f"{self.config['language']}_mfa"
        # MFA calls OpenFST/Kaldi binaries that live next to it.
        env = dict(os.environ)
        env["PATH"] = os.path.dirname(os.path.abspath(mfa_path)) + os.pathsep + env.get("PATH", "")
        self._ensure_mfa_models(mfa_path, model, env)

        cmd = [mfa_path, "align", mfa_input_dir, model, model, mfa_output_dir, "--clean"]
        cmd += self.config["mfa_args"]
        logger.info(f"Running MFA: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"MFA alignment failed:\n{result.stderr[-2000:]}")

        textgrid_path = next(Path(mfa_output_dir).rglob("utterance.TextGrid"), None)
        if textgrid_path is None:
            raise RuntimeError(f"MFA produced no TextGrid. Output:\n{result.stdout[-2000:]}")
        tg = tgio.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
        return [Phone(label, start, end) for start, end, label in tg.getTier("phones").entries]

    @staticmethod
    def _ensure_mfa_models(mfa_path: str, model: str, env: Dict[str, str]) -> None:
        """Download the MFA acoustic model and dictionary on first use."""
        root = Path(env.get("MFA_ROOT_DIR") or Path.home() / "Documents" / "MFA")
        expected = {
            "acoustic": root / "pretrained_models" / "acoustic" / f"{model}.zip",
            "dictionary": root / "pretrained_models" / "dictionary" / f"{model}.dict",
        }
        for kind, path in expected.items():
            if path.exists():
                continue
            logger.info(f"Downloading MFA {kind} model '{model}'...")
            result = subprocess.run([mfa_path, "model", "download", kind, model],
                                    capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"Could not download MFA {kind} model '{model}':\n"
                                   f"{result.stderr[-2000:]}")

    def _find_mfa_executable(self) -> Optional[str]:
        """Find the MFA executable: PATH, next to this Python, or a local pixi env."""
        candidates = [
            shutil.which("mfa"),
            os.path.join(os.path.dirname(sys.executable), "mfa"),
            os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "mfa"),
            *[str(p) for p in Path.cwd().glob(".pixi/envs/*/bin/mfa")],
        ]
        for path in candidates:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, video_path: str, audio_path: str, output_path: str,
                position_transitions, shape_transitions) -> None:
        """Draw the hand on every frame and encode the video with the audio."""
        import imageio_ffmpeg
        import mediapipe as mp

        handshapes_dir = self.config["handshapes_dir"] or DEFAULT_HANDSHAPES_DIR
        hands = HandShapeSet.load(handshapes_dir)
        # Straight paths at constant speed, as in Auto-CS.
        positions = Track(NEUTRAL, position_transitions)
        shapes = Track(shape_transitions[0][2], shape_transitions)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracker = FaceTracker(fps)
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.3, min_tracking_confidence=0.3)

        ffmpeg = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", str(self.config["crf"]), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", "-movflags", "+faststart", output_path,
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        frame_idx = frames_without_face = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = result.multi_face_landmarks[0] if result.multi_face_landmarks else None
                if landmarks is None:
                    frames_without_face += 1
                face = tracker.update(landmarks, width, height)
                if face is not None:
                    self._draw_hand(frame, frame_idx / fps, face, hands, positions, shapes)
                ffmpeg.stdin.write(frame.tobytes())
                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info(f"Rendered {frame_idx} frames")
        finally:
            cap.release()
            face_mesh.close()
            ffmpeg.stdin.close()
            stderr = ffmpeg.stderr.read().decode(errors="replace")
            ffmpeg.wait()
        if ffmpeg.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr}")
        if frames_without_face:
            logger.warning(f"No face detected on {frames_without_face}/{frame_idx} frames "
                           "(the last known face position was used)")

    def _draw_hand(self, frame, t, face, hands: HandShapeSet, positions: Track, shapes: Track) -> None:
        targets = face.vowel_positions()
        weights = positions.weights(t)
        target = sum(w * targets[code] for code, w in weights.items())
        angle = sum(w * POSITION_ANGLES[code] for code, w in weights.items())

        opacity = 1.0
        if self.config["enable_transparency"]:
            opacity = float(np.clip(1.0 - weights.get(NEUTRAL, 0.0), 0.0, 1.0))

        if self.config["correct_head_tilt"]:
            angle += face.angle - 90.0

        shape_weights = shapes.weights(t)
        if not self.config["enable_morphing"]:
            shape_weights = {max(shape_weights, key=shape_weights.get): 1.0}

        scale = hands.scale_for_face(face.height, HAND_TO_FACE_RATIO) * self.config["hand_scale"]
        hands.composite(frame, shape_weights, target, angle, scale, opacity)

        if self.config["enable_debug_landmarks"]:
            for code, p in targets.items():
                p = tuple(int(v) for v in p)
                cv2.circle(frame, p, 5, (0, 255, 255), -1)
                cv2.putText(frame, code, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.circle(frame, tuple(int(v) for v in target), 7, (255, 0, 255), 2)


# Public API function for backward compatibility
def generate_cue(
    text: Optional[str],
    video_path: str,
    output_path: str,
    audio_path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> str:
    """
    Generate cued speech video from text input or extract text from video using Whisper.

    This is a convenience function that creates a CuedSpeechGenerator instance.
    For more control, use CuedSpeechGenerator directly.
    """
    generator = CuedSpeechGenerator(config)
    return generator.generate_cue(text, video_path, output_path, audio_path)
