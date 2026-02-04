"""Audio Event Detection (laughter/cheer/shout).

Uses a lightweight audio event classifier (YAMNet-style) to detect semantic
audio events at regular hop intervals. Outputs per-hop probabilities for
events like laughter, cheering, applause, screaming, etc.

The event_combo_z score can be integrated into highlight scoring as a first-class
signal alongside audio RMS, motion, and chat spikes.
"""

from __future__ import annotations

import os
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Suppress console windows from GPU libraries (ROCm/CUDA/DirectML) on Windows
# This must be done BEFORE importing torch/tensorflow/onnxruntime
if sys.platform == "win32":
    # Tell HIP/ROCm not to show console windows
    os.environ.setdefault("HIP_LAUNCH_BLOCKING", "0")
    # Suppress TensorFlow console spam
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Suppress ONNX Runtime console output
    os.environ.setdefault("ORT_DISABLE_ALL_LOGS", "1")
    
    # Use ctypes to set the default process creation flags to hide console windows
    # This affects any subprocess spawned by child libraries (torch, etc.)
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # Try to hide the console window if we're in one
        # GetConsoleWindow returns 0 if there's no console
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            # SW_HIDE = 0, but we don't want to hide our own console if running from terminal
            pass
    except Exception:
        pass

# Fix protobuf 6.x compatibility with TensorFlow - MessageFactory.GetPrototype error
# Force upb (C++) implementation to avoid the deprecated Python MessageFactory API
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "upb")

from .ffmpeg import ffprobe_duration_seconds, AudioStreamParams, stream_audio_blocks_f32le
from .peaks import moving_average, pick_top_peaks, robust_z
from .project import Project, save_npz, update_project


# YAMNet class indices for events of interest
# These are based on AudioSet ontology class indices
# https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
YAMNET_CLASSES = {
    "laughter": [17, 18, 19],       # Laughter, Baby laughter, Giggling
    "cheering": [368],               # Cheering
    "applause": [369],               # Applause
    "screaming": [20, 21],           # Screaming, Shout
    "shouting": [21],                # Shout
    "crowd": [370, 371],             # Crowd, Hubbub/babble
    "music": [137, 138, 139],        # Music (generic)
    "speech": [0, 1],                # Speech, Narration
}


@dataclass
class AudioEventsConfig:
    """Configuration for audio event detection."""
    enabled: bool = True
    hop_seconds: float = 0.5
    window_seconds: float = 1.0  # Classification window (YAMNet works best with ~1s)
    smooth_seconds: float = 2.0
    sample_rate: int = 16000  # YAMNet expects 16kHz
    backend: str = "auto"  # auto | onnx_directml | onnx_cpu | tensorflow | heuristic
    
    # Candidate extraction parameters (matching audio RMS)
    top: int = 20
    min_gap_seconds: float = 30.0
    pre_seconds: float = 15.0
    post_seconds: float = 30.0
    skip_start_seconds: float = 60.0
    min_score_z: float = 1.0
    min_clip_seconds: float = 3.0
    
    # Event weights for combined score
    events: Dict[str, float] = field(default_factory=lambda: {
        "laughter": 1.0,
        "cheering": 0.7,
        "applause": 0.5,
        "screaming": 0.8,
        "shouting": 0.6,
    })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AudioEventsConfig":
        events = d.get("events", {})
        return cls(
            enabled=bool(d.get("enabled", True)),
            hop_seconds=float(d.get("hop_seconds", 0.5)),
            window_seconds=float(d.get("window_seconds", 1.0)),
            smooth_seconds=float(d.get("smooth_seconds", 2.0)),
            sample_rate=int(d.get("sample_rate", 16000)),
            backend=str(d.get("backend", "auto")),
            top=int(d.get("top", 20)),
            min_gap_seconds=float(d.get("min_gap_seconds", 30.0)),
            pre_seconds=float(d.get("pre_seconds", 15.0)),
            post_seconds=float(d.get("post_seconds", 30.0)),
            skip_start_seconds=float(d.get("skip_start_seconds", 60.0)),
            min_score_z=float(d.get("min_score_z", 1.0)),
            min_clip_seconds=float(d.get("min_clip_seconds", 3.0)),
            events=events if events else cls().events,
        )


def _suppress_protobuf_warnings():
    """Context manager to suppress protobuf 6.x MessageFactory warnings during TensorFlow import.
    
    TensorFlow 2.19.x with protobuf 6.x shows 'MessageFactory.GetPrototype' AttributeError
    warnings, but these are non-fatal and TensorFlow still works correctly.
    """
    import contextlib
    import io
    import sys
    
    @contextlib.contextmanager
    def suppress():
        old_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            yield
        finally:
            # Restore stderr, but check for any real errors (not MessageFactory warnings)
            captured = sys.stderr.getvalue()
            sys.stderr = old_stderr
            # Filter out the MessageFactory warnings, print anything else
            for line in captured.splitlines():
                if 'MessageFactory' not in line and line.strip():
                    print(line, file=sys.stderr)
    return suppress()


def _try_load_yamnet():
    """Try to load YAMNet model. Returns (model, None) or (None, error_msg)."""
    try:
        with _suppress_protobuf_warnings():
            import tensorflow_hub as hub
            model = hub.load("https://tfhub.dev/google/yamnet/1")
        return model, None
    except ImportError:
        return None, "tensorflow_hub not installed"
    except Exception as e:
        return None, f"Failed to load YAMNet: {e}"


def _try_load_pytorch_classifier():
    """Try to load PyTorch audio classifier (uses torchaudio's pretrained models).
    
    Returns (model, device, sample_rate, None) on success or (None, None, None, error_msg) on failure.
    """
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import WAV2VEC2_BASE
        
        # Check for GPU (ROCm for AMD, CUDA for NVIDIA)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Load Wav2Vec2 model - produces embeddings useful for audio analysis
        bundle = WAV2VEC2_BASE
        model = bundle.get_model().to(device)
        model.eval()
        
        # Wav2Vec2 expects specific sample rate
        model_sample_rate = bundle.sample_rate
        
        return model, device, model_sample_rate, None
    except ImportError as e:
        return None, None, None, f"torch/torchaudio not installed: {e}"
    except Exception as e:
        return None, None, None, f"Failed to load PyTorch model: {e}"


def _try_load_onnx_directml():
    """Try to load YAMNet ONNX model with DirectML (AMD/Intel GPU on Windows).
    
    Returns (session, None) on success or (None, error_msg) on failure.
    """
    try:
        import onnxruntime as ort
        # Check if DirectML is available
        available_providers = ort.get_available_providers()
        if 'DmlExecutionProvider' not in available_providers:
            return None, "DirectML provider not available (need onnxruntime-directml)"
        
        # Look for local ONNX model
        model_paths = [
            Path(__file__).parent / "models" / "yamnet.onnx",
            Path.home() / ".cache" / "videopipeline" / "yamnet.onnx",
        ]
        for mp in model_paths:
            if mp.exists():
                session = ort.InferenceSession(
                    str(mp), 
                    providers=['DmlExecutionProvider', 'CPUExecutionProvider']
                )
                return session, None
        return None, "yamnet.onnx not found"
    except ImportError:
        return None, "onnxruntime not installed"
    except Exception as e:
        return None, f"Failed to load ONNX DirectML: {e}"


def _try_load_yamnet_onnx():
    """Try to load YAMNet ONNX model (CPU). Returns (session, None) or (None, error_msg)."""
    try:
        import onnxruntime as ort
        # Look for local ONNX model first
        model_paths = [
            Path(__file__).parent / "models" / "yamnet.onnx",
            Path.home() / ".cache" / "videopipeline" / "yamnet.onnx",
        ]
        for mp in model_paths:
            if mp.exists():
                session = ort.InferenceSession(str(mp), providers=['CPUExecutionProvider'])
                return session, None
        return None, "yamnet.onnx not found"
    except ImportError:
        return None, "onnxruntime not installed"
    except Exception as e:
        return None, f"Failed to load ONNX model: {e}"


class AudioEventClassifier:
    """Lightweight audio event classifier using YAMNet or ONNX fallback.
    
    Backend priority (auto mode):
    1. PyTorch (GPU via ROCm/CUDA, or CPU)
    2. ONNX DirectML (GPU on Windows AMD/Intel)
    3. ONNX CPU 
    4. TensorFlow Hub
    5. Heuristic (always works)
    """
    
    def __init__(self, sample_rate: int = 16000, backend: str = "auto"):
        """Initialize classifier.
        
        Args:
            sample_rate: Audio sample rate (16kHz recommended)
            backend: One of "auto", "pytorch", "onnx_directml", "onnx_cpu", "tensorflow", "heuristic"
        """
        self.sample_rate = sample_rate
        self._model = None
        self._onnx_session = None
        self._pytorch_model = None
        self._pytorch_device = None
        self._pytorch_sample_rate = None  # Model's expected sample rate
        self._backend = "none"
        self._backend_errors: Dict[str, str] = {}
        
        if backend == "auto":
            self._init_auto()
        elif backend == "pytorch":
            self._init_pytorch()
        elif backend == "onnx_directml":
            self._init_onnx_directml()
        elif backend == "onnx_cpu":
            self._init_onnx_cpu()
        elif backend == "tensorflow":
            self._init_tensorflow()
        elif backend == "heuristic":
            self._backend = "heuristic"
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_auto(self):
        """Auto-select best available backend."""
        # Priority: PyTorch (GPU) > ONNX DirectML > PyTorch (CPU) > ONNX CPU > TensorFlow > Heuristic
        
        # Try PyTorch first (best option if ROCm/CUDA available)
        model, device, model_sr, err = _try_load_pytorch_classifier()
        if model is not None:
            self._pytorch_model = model
            self._pytorch_device = device
            self._pytorch_sample_rate = model_sr
            self._backend = "pytorch"
            return
        self._backend_errors["pytorch"] = err or "unknown error"
        
        # Try ONNX CPU
        session, err = _try_load_yamnet_onnx()
        if session is not None:
            self._onnx_session = session
            self._backend = "onnx_cpu"
            return
        self._backend_errors["onnx_cpu"] = err or "unknown error"
        
        # Try TensorFlow Hub (often slow to load)
        model, err = _try_load_yamnet()
        if model is not None:
            self._model = model
            self._backend = "tensorflow"
            return
        self._backend_errors["tensorflow"] = err or "unknown error"
        
        # Fall back to heuristic
        self._backend = "heuristic"
    
    def _init_onnx_directml(self):
        session, err = _try_load_onnx_directml()
        if session is not None:
            self._onnx_session = session
            self._backend = "onnx_directml"
        else:
            self._backend_errors["onnx_directml"] = err or "unknown error"
            self._backend = "heuristic"
    
    def _init_onnx_cpu(self):
        session, err = _try_load_yamnet_onnx()
        if session is not None:
            self._onnx_session = session
            self._backend = "onnx_cpu"
        else:
            self._backend_errors["onnx_cpu"] = err or "unknown error"
            self._backend = "heuristic"
    
    def _init_tensorflow(self):
        model, err = _try_load_yamnet()
        if model is not None:
            self._model = model
            self._backend = "tensorflow"
        else:
            self._backend_errors["tensorflow"] = err or "unknown error"
            self._backend = "heuristic"
    
    def _init_pytorch(self):
        model, device, model_sr, err = _try_load_pytorch_classifier()
        if model is not None:
            self._pytorch_model = model
            self._pytorch_device = device
            self._pytorch_sample_rate = model_sr
            self._backend = "pytorch"
        else:
            self._backend_errors["pytorch"] = err or "unknown error"
            self._backend = "heuristic"
    
    @property
    def is_ml_available(self) -> bool:
        return self._backend in ("tensorflow", "onnx_cpu", "onnx_directml", "pytorch")
    
    def classify_chunk(self, audio_f32: np.ndarray) -> Dict[str, float]:
        """Classify a chunk of audio and return event probabilities.
        
        Args:
            audio_f32: Audio samples as float32, mono, at self.sample_rate
            
        Returns:
            Dict mapping event names to probabilities [0, 1]
        """
        if self._backend == "pytorch":
            return self._classify_pytorch(audio_f32)
        elif self._backend == "tensorflow":
            return self._classify_tensorflow(audio_f32)
        elif self._backend in ("onnx_cpu", "onnx_directml"):
            return self._classify_onnx(audio_f32)
        else:
            return self._classify_heuristic(audio_f32)
    
    def _classify_tensorflow(self, audio_f32: np.ndarray) -> Dict[str, float]:
        """Classify using TensorFlow Hub YAMNet."""
        import tensorflow as tf
        
        # YAMNet expects float32 waveform
        waveform = tf.convert_to_tensor(audio_f32.astype(np.float32))
        
        # Run inference
        scores, embeddings, spectrogram = self._model(waveform)
        scores = scores.numpy()
        
        # Average scores across frames
        if len(scores.shape) > 1:
            avg_scores = np.mean(scores, axis=0)
        else:
            avg_scores = scores
        
        # Extract probabilities for events of interest
        results = {}
        for event_name, class_indices in YAMNET_CLASSES.items():
            if event_name in ("laughter", "cheering", "applause", "screaming", "shouting", "crowd"):
                prob = float(np.max([avg_scores[i] for i in class_indices if i < len(avg_scores)]))
                results[event_name] = min(1.0, max(0.0, prob))
        
        return results
    
    def _classify_onnx(self, audio_f32: np.ndarray) -> Dict[str, float]:
        """Classify using ONNX YAMNet."""
        # Prepare input
        input_name = self._onnx_session.get_inputs()[0].name
        waveform = audio_f32.astype(np.float32).reshape(1, -1)
        
        # Run inference
        outputs = self._onnx_session.run(None, {input_name: waveform})
        scores = np.asarray(outputs[0])
        
        # Handle various output shapes:
        # - (classes,): single frame output
        # - (frames, classes): multiple frames, no batch dim
        # - (1, frames, classes): batch dim present
        scores = np.squeeze(scores)  # Remove batch dim if present: (1, F, C) -> (F, C)
        
        if scores.ndim == 2:
            # (frames, classes) -> average over frames to get (classes,)
            avg_scores = np.mean(scores, axis=0)
        elif scores.ndim == 1:
            # Already (classes,)
            avg_scores = scores
        else:
            # Unexpected shape, flatten and hope for the best
            avg_scores = scores.flatten()
        
        # Extract probabilities for events of interest
        results = {}
        for event_name, class_indices in YAMNET_CLASSES.items():
            if event_name in ("laughter", "cheering", "applause", "screaming", "shouting", "crowd"):
                prob = float(np.max([avg_scores[i] for i in class_indices if i < len(avg_scores)]))
                results[event_name] = min(1.0, max(0.0, prob))
        
        return results
    
    def _classify_heuristic(self, audio_f32: np.ndarray) -> Dict[str, float]:
        """Fallback heuristic classification using acoustic features.
        
        This uses spectral characteristics to approximate event detection:
        - Laughter: High spectral flux + specific frequency patterns
        - Cheering/Crowd: High energy in mid frequencies + broadband noise
        - Screaming/Shouting: High energy in upper frequencies + high amplitude
        """
        # Compute basic acoustic features
        audio = audio_f32.astype(np.float64)
        n = len(audio)
        
        if n < 256:
            return {
                "laughter": 0.0,
                "cheering": 0.0,
                "applause": 0.0,
                "screaming": 0.0,
                "shouting": 0.0,
                "crowd": 0.0,
            }
        
        # RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))
        
        # Spectral features using FFT
        fft = np.fft.rfft(audio * np.hanning(n))
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        
        # Frequency band energies
        low_mask = (freqs >= 100) & (freqs < 500)
        mid_mask = (freqs >= 500) & (freqs < 2000)
        high_mask = (freqs >= 2000) & (freqs < 4000)
        
        total_energy = np.sum(mag ** 2) + 1e-10
        low_energy = np.sum(mag[low_mask] ** 2) / total_energy
        mid_energy = np.sum(mag[mid_mask] ** 2) / total_energy
        high_energy = np.sum(mag[high_mask] ** 2) / total_energy
        
        # Spectral centroid (higher = brighter sound)
        centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-10)
        
        # Spectral flatness (higher = more noise-like)
        geo_mean = np.exp(np.mean(np.log(mag + 1e-10)))
        arith_mean = np.mean(mag)
        flatness = geo_mean / (arith_mean + 1e-10)
        
        # Zero crossing rate (higher = more noisy/speech-like)
        zcr = float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * n))
        
        # Heuristic scores (rough approximations)
        # Laughter: mid-frequency emphasis, moderate energy, variable amplitude
        laughter_score = min(1.0, mid_energy * 2.0 * min(1.0, rms * 5))
        
        # Cheering: broadband noise, high energy
        cheering_score = min(1.0, flatness * 2.0 * min(1.0, rms * 3))
        
        # Applause: very high flatness (noise-like), broadband
        applause_score = min(1.0, flatness * 3.0 * (1.0 - abs(mid_energy - 0.4)))
        
        # Screaming: high frequency emphasis, high energy
        screaming_score = min(1.0, high_energy * 3.0 * min(1.0, rms * 4))
        
        # Shouting: high energy, speech-like frequencies
        shouting_score = min(1.0, (mid_energy + low_energy) * min(1.0, rms * 5))
        
        # Crowd: moderate flatness, broadband mid-frequencies
        crowd_score = min(1.0, flatness * mid_energy * 4.0)
        
        return {
            "laughter": float(laughter_score),
            "cheering": float(cheering_score),
            "applause": float(applause_score),
            "screaming": float(screaming_score),
            "shouting": float(shouting_score),
            "crowd": float(crowd_score),
        }
    
    def _classify_pytorch(self, audio_f32: np.ndarray) -> Dict[str, float]:
        """Classify using PyTorch Wav2Vec2 model.
        
        Wav2Vec2 produces embeddings. We use embedding statistics combined
        with spectral features to estimate audio event probabilities.
        """
        import torch
        import torchaudio.functional as F
        
        # Resample if needed (Wav2Vec2 expects 16kHz)
        waveform = torch.from_numpy(audio_f32.astype(np.float32)).unsqueeze(0)
        
        if self._pytorch_sample_rate and self._pytorch_sample_rate != self.sample_rate:
            waveform = F.resample(waveform, self.sample_rate, self._pytorch_sample_rate)
        
        waveform = waveform.to(self._pytorch_device)
        
        with torch.no_grad():
            # Wav2Vec2 outputs features of shape (batch, frames, hidden_size)
            # We extract features, not transcriptions
            features, _ = self._pytorch_model.extract_features(waveform)
            
            # Get last layer features
            if isinstance(features, list):
                embeddings = features[-1]  # Last layer
            else:
                embeddings = features
            
            # Average over frames to get (batch, hidden_size)
            embeddings = embeddings.mean(dim=1)
            embeddings = embeddings.squeeze().cpu().numpy()
        
        # Use embedding statistics as features for heuristic classification
        emb_mean = float(np.mean(embeddings))
        emb_std = float(np.std(embeddings))
        emb_energy = float(np.sum(embeddings ** 2) / len(embeddings))
        
        # Normalize embedding stats
        emb_complexity = min(1.0, emb_std / 2.0)  # Adjusted for Wav2Vec2 scale
        
        # Also compute spectral features from raw audio for better estimates
        audio = audio_f32.astype(np.float64)
        n = len(audio)
        
        if n < 256:
            return {
                "laughter": 0.0, "cheering": 0.0, "applause": 0.0,
                "screaming": 0.0, "shouting": 0.0, "crowd": 0.0,
            }
        
        rms = float(np.sqrt(np.mean(audio ** 2)))
        fft = np.fft.rfft(audio * np.hanning(n))
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        
        mid_mask = (freqs >= 500) & (freqs < 2000)
        high_mask = (freqs >= 2000) & (freqs < 4000)
        total_energy = np.sum(mag ** 2) + 1e-10
        mid_energy = np.sum(mag[mid_mask] ** 2) / total_energy
        high_energy = np.sum(mag[high_mask] ** 2) / total_energy
        
        geo_mean = np.exp(np.mean(np.log(mag + 1e-10)))
        arith_mean = np.mean(mag)
        flatness = geo_mean / (arith_mean + 1e-10)
        
        # Heuristic scores boosted by embedding complexity
        # Higher complexity suggests more interesting audio content
        boost = 1.0 + emb_complexity * 0.5
        
        laughter_score = min(1.0, mid_energy * 2.0 * min(1.0, rms * 5) * boost)
        cheering_score = min(1.0, flatness * 2.0 * min(1.0, rms * 3) * boost)
        applause_score = min(1.0, flatness * 3.0 * (1.0 - abs(mid_energy - 0.4)) * boost)
        screaming_score = min(1.0, high_energy * 3.0 * min(1.0, rms * 4) * boost)
        shouting_score = min(1.0, mid_energy * min(1.0, rms * 5) * boost)
        crowd_score = min(1.0, flatness * mid_energy * 4.0 * boost)
        
        return {
            "laughter": float(laughter_score),
            "cheering": float(cheering_score),
            "applause": float(applause_score),
            "screaming": float(screaming_score),
            "shouting": float(shouting_score),
            "crowd": float(crowd_score),
        }


# ============================================================================
# Standalone Audio Events Analysis (for early processing during download)
# ============================================================================

def compute_audio_events_from_file(
    audio_path: Path,
    *,
    cfg: AudioEventsConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute audio events from an audio file (no Project required).
    
    This is used during URL download to run audio event detection in parallel
    with video download and transcription.
    
    Args:
        audio_path: Path to audio file (WAV, MP3, M4A, etc.)
        cfg: AudioEventsConfig with detection parameters
        on_progress: Optional progress callback
        
    Returns:
        Dict with event timelines and metadata
        Can be saved to a project later using save_audio_events_to_project()
    """
    from datetime import datetime, timezone
    
    if not cfg.enabled:
        return {"enabled": False, "skipped": True}
    
    start_time = _time.time()
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    duration_s = ffprobe_duration_seconds(audio_path)
    
    hop_samples = int(cfg.sample_rate * cfg.hop_seconds)
    window_samples = int(cfg.sample_rate * cfg.window_seconds)
    if hop_samples <= 0:
        raise ValueError("hop_seconds too small")
    if window_samples <= 0:
        raise ValueError("window_seconds too small")
    
    # Initialize classifier
    import logging
    log = logging.getLogger(__name__)
    log.info(f"[audio_events] Initializing classifier for {duration_s:.1f}s audio...")
    classifier = AudioEventClassifier(sample_rate=cfg.sample_rate, backend=cfg.backend)
    log.info(f"[audio_events] Using backend: {classifier._backend}")
    
    params = AudioStreamParams(sample_rate=cfg.sample_rate, channels=1)
    total_hops = max(1, int(duration_s / cfg.hop_seconds))
    
    # Initialize arrays
    times_start: List[float] = []
    times_center: List[float] = []
    laughter_probs: List[float] = []
    cheering_probs: List[float] = []
    applause_probs: List[float] = []
    screaming_probs: List[float] = []
    shouting_probs: List[float] = []
    crowd_probs: List[float] = []
    
    processed = 0
    current_time = 0.0
    
    if on_progress:
        on_progress(0.01)
    
    audio_buffer = np.array([], dtype=np.float32)
    
    for block in stream_audio_blocks_f32le(
        audio_path,
        params=params,
        block_samples=hop_samples,
        yield_partial=True,
    ):
        audio_buffer = np.concatenate([audio_buffer, block])
        
        while len(audio_buffer) >= window_samples:
            window = audio_buffer[:window_samples]
            probs = classifier.classify_chunk(window)
            
            times_start.append(current_time)
            times_center.append(current_time + cfg.window_seconds / 2.0)
            laughter_probs.append(probs.get("laughter", 0.0))
            cheering_probs.append(probs.get("cheering", 0.0))
            applause_probs.append(probs.get("applause", 0.0))
            screaming_probs.append(probs.get("screaming", 0.0))
            shouting_probs.append(probs.get("shouting", 0.0))
            crowd_probs.append(probs.get("crowd", 0.0))
            
            audio_buffer = audio_buffer[hop_samples:]
            current_time += cfg.hop_seconds
            processed += 1
            
            if on_progress and processed % 50 == 0:
                frac = min(0.9, processed / total_hops)
                on_progress(frac)
    
    # Handle remaining samples
    if len(audio_buffer) > 0:
        if len(audio_buffer) < window_samples:
            padding = np.zeros(window_samples - len(audio_buffer), dtype=np.float32)
            window = np.concatenate([audio_buffer, padding])
        else:
            window = audio_buffer[:window_samples]
        
        probs = classifier.classify_chunk(window)
        times_start.append(current_time)
        times_center.append(current_time + cfg.window_seconds / 2.0)
        laughter_probs.append(probs.get("laughter", 0.0))
        cheering_probs.append(probs.get("cheering", 0.0))
        applause_probs.append(probs.get("applause", 0.0))
        screaming_probs.append(probs.get("screaming", 0.0))
        shouting_probs.append(probs.get("shouting", 0.0))
        crowd_probs.append(probs.get("crowd", 0.0))
    
    # Convert to arrays and compute scores
    times_arr = np.array(times_start, dtype=np.float64)
    times_center_arr = np.array(times_center, dtype=np.float64)
    laughter = np.array(laughter_probs, dtype=np.float64)
    cheering = np.array(cheering_probs, dtype=np.float64)
    applause = np.array(applause_probs, dtype=np.float64)
    screaming = np.array(screaming_probs, dtype=np.float64)
    shouting = np.array(shouting_probs, dtype=np.float64)
    crowd = np.array(crowd_probs, dtype=np.float64)
    
    # Compute weighted combined score
    event_weights = cfg.events
    event_combo = np.zeros_like(laughter)
    weight_sum = 0.0
    
    event_arrays = {
        "laughter": laughter,
        "cheering": cheering,
        "applause": applause,
        "screaming": screaming,
        "shouting": shouting,
        "crowd": crowd,
    }
    
    for event_name, weight in event_weights.items():
        if weight <= 0:
            continue
        if event_name in event_arrays:
            event_combo += weight * event_arrays[event_name]
            weight_sum += weight
    
    if weight_sum > 0:
        event_combo /= weight_sum
    
    # Smooth and compute z-scores
    smooth_frames = max(1, int(round(cfg.smooth_seconds / cfg.hop_seconds)))
    event_combo_smoothed = moving_average(event_combo, smooth_frames)
    event_combo_z = robust_z(event_combo_smoothed)
    laughter_z = robust_z(moving_average(laughter, smooth_frames))
    cheering_z = robust_z(moving_average(cheering, smooth_frames))
    screaming_z = robust_z(moving_average(screaming, smooth_frames))
    
    if on_progress:
        try:
            on_progress(0.95, "Building results")
        except TypeError:
            on_progress(0.95)
    
    elapsed_seconds = _time.time() - start_time
    
    result = {
        "times": times_arr.tolist(),
        "times_center": times_center_arr.tolist(),
        "laughter": laughter.tolist(),
        "cheering": cheering.tolist(),
        "applause": applause.tolist(),
        "screaming": screaming.tolist(),
        "shouting": shouting.tolist(),
        "crowd": crowd.tolist(),
        "event_combo": event_combo.tolist(),
        "event_combo_smoothed": event_combo_smoothed.tolist(),
        "event_combo_z": event_combo_z.tolist(),
        "laughter_z": laughter_z.tolist(),
        "cheering_z": cheering_z.tolist(),
        "screaming_z": screaming_z.tolist(),
        "hop_seconds": cfg.hop_seconds,
        "window_seconds": cfg.window_seconds,
        "smooth_seconds": cfg.smooth_seconds,
        "sample_rate": cfg.sample_rate,
        "duration_seconds": duration_s,
        "backend": classifier._backend,
        "ml_available": classifier.is_ml_available,
        "config": {
            "hop_seconds": cfg.hop_seconds,
            "window_seconds": cfg.window_seconds,
            "smooth_seconds": cfg.smooth_seconds,
            "sample_rate": cfg.sample_rate,
            "events": cfg.events,
        },
        "elapsed_seconds": elapsed_seconds,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    if on_progress:
        on_progress(1.0)
    
    return result


def save_audio_events_to_project(
    proj: Project,
    events_data: Dict[str, Any],
) -> None:
    """Save pre-computed audio events analysis to a project.
    
    Args:
        proj: Project instance
        events_data: Audio events data from compute_audio_events_from_file()
    """
    if events_data.get("skipped"):
        return
    
    # Convert lists back to numpy arrays
    times = np.array(events_data["times"], dtype=np.float64)
    times_center = np.array(events_data["times_center"], dtype=np.float64)
    laughter = np.array(events_data["laughter"], dtype=np.float64)
    cheering = np.array(events_data["cheering"], dtype=np.float64)
    applause = np.array(events_data["applause"], dtype=np.float64)
    screaming = np.array(events_data["screaming"], dtype=np.float64)
    shouting = np.array(events_data["shouting"], dtype=np.float64)
    crowd = np.array(events_data["crowd"], dtype=np.float64)
    event_combo = np.array(events_data["event_combo"], dtype=np.float64)
    event_combo_smoothed = np.array(events_data["event_combo_smoothed"], dtype=np.float64)
    event_combo_z = np.array(events_data["event_combo_z"], dtype=np.float64)
    laughter_z = np.array(events_data["laughter_z"], dtype=np.float64)
    cheering_z = np.array(events_data["cheering_z"], dtype=np.float64)
    screaming_z = np.array(events_data["screaming_z"], dtype=np.float64)
    
    audio_events_path = proj.audio_events_features_path
    save_npz(
        audio_events_path,
        times=times,
        times_start=times,
        times_center=times_center,
        window_seconds=np.array([events_data["window_seconds"]], dtype=np.float64),
        laughter=laughter,
        cheering=cheering,
        applause=applause,
        screaming=screaming,
        shouting=shouting,
        crowd=crowd,
        event_combo=event_combo,
        event_combo_smoothed=event_combo_smoothed,
        event_combo_z=event_combo_z,
        laughter_z=laughter_z,
        cheering_z=cheering_z,
        screaming_z=screaming_z,
        hop_seconds=np.array([events_data["hop_seconds"]], dtype=np.float64),
    )
    
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["audio_events"] = {
            "video": str(proj.audio_source),  # May be audio file during early analysis
            "duration_seconds": events_data["duration_seconds"],
            "method": "audio_event_classifier",
            "backend": events_data["backend"],
            "ml_available": events_data["ml_available"],
            "config": events_data["config"],
            "features_npz": str(audio_events_path.relative_to(proj.project_dir)),
            "elapsed_seconds": events_data["elapsed_seconds"],
            "generated_at": events_data["generated_at"],
        }
    
    update_project(proj, _upd)


# ============================================================================
# Project-based Audio Events Analysis
# ============================================================================

def compute_audio_events_analysis(
    proj: Project,
    *,
    cfg: AudioEventsConfig,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Compute audio event detection timeline.
    
    Persists:
      - analysis/audio_events_features.npz (times, event probs, combo scores)
      - project.json -> analysis.audio_events section
      
    Args:
        proj: Project instance
        cfg: AudioEventsConfig with detection parameters
        on_progress: Optional progress callback
        
    Returns:
        Analysis result dict with config and metadata
    """
    if not cfg.enabled:
        return {"enabled": False}

    start_time = _time.time()
    video_path = Path(proj.audio_source)  # Use audio_source for fallback during early analysis
    duration_s = ffprobe_duration_seconds(video_path)

    def _clamp_window(peak_time: float) -> tuple[float, float]:
        """Clamp a (peak_time - pre_s, peak_time + post_s) window into [0, duration].

        If the window hits an edge, we *shift* it to preserve the requested
        duration as much as possible.
        """
        desired = max(0.0, float(cfg.pre_seconds + cfg.post_seconds))
        start = float(peak_time - cfg.pre_seconds)
        end = float(peak_time + cfg.post_seconds)

        # Shift right if we underflow.
        if start < 0.0:
            end = min(duration_s, end - start)
            start = 0.0

        # Shift left if we overflow.
        if end > duration_s:
            start = max(0.0, start - (end - duration_s))
            end = duration_s

        # If video is shorter than desired, just clamp.
        if desired > 0 and (end - start) > desired + 1e-6:
            end = min(duration_s, start + desired)
        return start, end
    
    hop_samples = int(cfg.sample_rate * cfg.hop_seconds)
    window_samples = int(cfg.sample_rate * cfg.window_seconds)
    if hop_samples <= 0:
        raise ValueError("hop_seconds too small")
    if window_samples <= 0:
        raise ValueError("window_seconds too small")
    
    # Initialize classifier with configured backend
    import logging
    log = logging.getLogger(__name__)
    log.info(f"[audio_events] Initializing classifier for {duration_s:.1f}s video...")
    classifier = AudioEventClassifier(sample_rate=cfg.sample_rate, backend=cfg.backend)
    log.info(f"[audio_events] Using backend: {classifier._backend}")
    if classifier._backend_errors:
        log.info(f"[audio_events] Backend errors tried: {classifier._backend_errors}")
    
    params = AudioStreamParams(sample_rate=cfg.sample_rate, channels=1)
    
    total_hops = max(1, int(duration_s / cfg.hop_seconds))
    
    # Initialize arrays for each event type
    # `times_start` refers to the window start. `times_center` is often a better
    # timestamp to align the probability with other signals.
    times_start: List[float] = []
    times_center: List[float] = []
    laughter_probs: List[float] = []
    cheering_probs: List[float] = []
    applause_probs: List[float] = []
    screaming_probs: List[float] = []
    shouting_probs: List[float] = []
    crowd_probs: List[float] = []
    
    processed = 0
    current_time = 0.0
    
    # Report early progress so UI shows something
    def _report(frac: float, msg: str = "") -> None:
        if on_progress:
            try:
                on_progress(frac, msg)
            except TypeError:
                on_progress(frac)
    
    _report(0.01, f"Initializing ML classifier ({classifier._backend})")
    
    log.info(f"[audio_events] Starting classification of {total_hops} hops ({duration_s:.1f}s at {cfg.hop_seconds}s hop)")
    
    # Rolling buffer approach: accumulate audio, classify when buffer >= window_samples
    # This decouples window_seconds (classification) from hop_seconds (output timeline)
    audio_buffer = np.array([], dtype=np.float32)
    
    for block in stream_audio_blocks_f32le(
        video_path,
        params=params,
        block_samples=hop_samples,
        yield_partial=True,
    ):
        audio_buffer = np.concatenate([audio_buffer, block])
        
        # While we have enough samples for a full window, classify and step forward
        while len(audio_buffer) >= window_samples:
            # Classify the current window
            window = audio_buffer[:window_samples]
            probs = classifier.classify_chunk(window)
            
            times_start.append(current_time)
            times_center.append(current_time + cfg.window_seconds / 2.0)
            laughter_probs.append(probs.get("laughter", 0.0))
            cheering_probs.append(probs.get("cheering", 0.0))
            applause_probs.append(probs.get("applause", 0.0))
            screaming_probs.append(probs.get("screaming", 0.0))
            shouting_probs.append(probs.get("shouting", 0.0))
            crowd_probs.append(probs.get("crowd", 0.0))
            
            # Step forward by hop_samples
            audio_buffer = audio_buffer[hop_samples:]
            current_time += cfg.hop_seconds
            processed += 1
            
            if processed % 100 == 0:
                pct = 100 * processed / total_hops
                _report(min(0.85, processed / total_hops), f"Classifying audio: {processed}/{total_hops} ({pct:.0f}%)")
                if processed % 500 == 0:
                    log.info(f"[audio_events] Progress: {processed}/{total_hops} ({pct:.1f}%)")
    
    # Handle remaining samples at the end (pad with zeros if needed)
    if len(audio_buffer) > 0:
        # Pad to window_samples
        if len(audio_buffer) < window_samples:
            padding = np.zeros(window_samples - len(audio_buffer), dtype=np.float32)
            window = np.concatenate([audio_buffer, padding])
        else:
            window = audio_buffer[:window_samples]
        
        probs = classifier.classify_chunk(window)
        times_start.append(current_time)
        times_center.append(current_time + cfg.window_seconds / 2.0)
        laughter_probs.append(probs.get("laughter", 0.0))
        cheering_probs.append(probs.get("cheering", 0.0))
        applause_probs.append(probs.get("applause", 0.0))
        screaming_probs.append(probs.get("screaming", 0.0))
        shouting_probs.append(probs.get("shouting", 0.0))
        crowd_probs.append(probs.get("crowd", 0.0))
    
    # Convert to arrays
    times_arr = np.array(times_start, dtype=np.float64)
    times_center_arr = np.array(times_center, dtype=np.float64)
    laughter = np.array(laughter_probs, dtype=np.float64)
    cheering = np.array(cheering_probs, dtype=np.float64)
    applause = np.array(applause_probs, dtype=np.float64)
    screaming = np.array(screaming_probs, dtype=np.float64)
    shouting = np.array(shouting_probs, dtype=np.float64)
    crowd = np.array(crowd_probs, dtype=np.float64)
    
    # Compute weighted combined score
    event_weights = cfg.events
    event_combo = np.zeros_like(laughter)
    weight_sum = 0.0
    
    # Map of known event names to their arrays
    event_arrays = {
        "laughter": laughter,
        "cheering": cheering,
        "applause": applause,
        "screaming": screaming,
        "shouting": shouting,
        "crowd": crowd,
    }
    
    for event_name, weight in event_weights.items():
        if weight <= 0:
            continue
        if event_name in event_arrays:
            event_combo += weight * event_arrays[event_name]
            weight_sum += weight
        # Unknown event names are now silently ignored (no weight_sum increment)
    
    if weight_sum > 0:
        event_combo /= weight_sum
    
    # Smooth and compute z-scores
    smooth_frames = max(1, int(round(cfg.smooth_seconds / cfg.hop_seconds)))
    event_combo_smoothed = moving_average(event_combo, smooth_frames)
    event_combo_z = robust_z(event_combo_smoothed)
    
    # Also compute individual z-scores for breakdown
    laughter_z = robust_z(moving_average(laughter, smooth_frames))
    cheering_z = robust_z(moving_average(cheering, smooth_frames))
    screaming_z = robust_z(moving_average(screaming, smooth_frames))

    # ------------------------------------------------------------------
    # Candidate clips directly from semantic event peaks (event_combo_z)
    # ------------------------------------------------------------------
    scores_for_peaks = np.array(event_combo_z, dtype=np.float64, copy=True)
    skip_frames = int(round(cfg.skip_start_seconds / cfg.hop_seconds))
    if skip_frames > 0 and skip_frames < len(scores_for_peaks):
        scores_for_peaks[:skip_frames] = -np.inf

    min_gap_frames = max(1, int(round(cfg.min_gap_seconds / cfg.hop_seconds)))
    peak_idxs = pick_top_peaks(
        scores_for_peaks,
        top_k=int(cfg.top),
        min_gap_frames=min_gap_frames,
        min_score=float(cfg.min_score_z),
    )

    candidates: List[Dict[str, Any]] = []
    rank = 0
    for idx in peak_idxs:
        peak_time = float(times_center_arr[idx]) if idx < len(times_center_arr) else float(idx * cfg.hop_seconds)
        start, end = _clamp_window(peak_time)
        if end - start < float(cfg.min_clip_seconds):
            continue
        rank += 1
        candidates.append(
            {
                "rank": rank,
                "peak_time_s": peak_time,
                "start_s": start,
                "end_s": end,
                "score": float(event_combo_z[idx]),
                "laughter_z": float(laughter_z[idx]) if idx < len(laughter_z) else 0.0,
                "cheering_z": float(cheering_z[idx]) if idx < len(cheering_z) else 0.0,
                "screaming_z": float(screaming_z[idx]) if idx < len(screaming_z) else 0.0,
            }
        )
    
    _report(0.95, "Saving audio_events_features.npz")
    
    # Save features
    audio_events_path = proj.audio_events_features_path
    save_npz(
        audio_events_path,
        times=times_arr,
        times_start=times_arr,
        times_center=times_center_arr,
        window_seconds=np.array([cfg.window_seconds], dtype=np.float64),
        laughter=laughter,
        cheering=cheering,
        applause=applause,
        screaming=screaming,
        shouting=shouting,
        crowd=crowd,
        event_combo=event_combo,
        event_combo_smoothed=event_combo_smoothed,
        event_combo_z=event_combo_z,
        laughter_z=laughter_z,
        cheering_z=cheering_z,
        screaming_z=screaming_z,
        hop_seconds=np.array([cfg.hop_seconds], dtype=np.float64),
    )
    
    # Compute peak events for metadata
    peak_laughter_idx = int(np.argmax(laughter_z)) if len(laughter_z) > 0 else 0
    peak_cheering_idx = int(np.argmax(cheering_z)) if len(cheering_z) > 0 else 0
    peak_combo_idx = int(np.argmax(event_combo_z)) if len(event_combo_z) > 0 else 0

    def _safe_time(idx: int) -> float:
        if idx < len(times_center_arr):
            return float(times_center_arr[idx])
        return float(idx * cfg.hop_seconds)
    
    payload = {
        "video": str(video_path),
        "duration_seconds": duration_s,
        "method": "audio_event_classifier",
        "backend": classifier._backend,
        "ml_available": classifier.is_ml_available,
        "backend_errors": getattr(classifier, "_backend_errors", {}),
        "config": {
            "hop_seconds": cfg.hop_seconds,
            "window_seconds": cfg.window_seconds,
            "smooth_seconds": cfg.smooth_seconds,
            "sample_rate": cfg.sample_rate,
            "top": cfg.top,
            "min_gap_seconds": cfg.min_gap_seconds,
            "pre_seconds": cfg.pre_seconds,
            "post_seconds": cfg.post_seconds,
            "skip_start_seconds": cfg.skip_start_seconds,
            "min_score_z": cfg.min_score_z,
            "min_clip_seconds": cfg.min_clip_seconds,
            "events": cfg.events,
        },
        "peaks": {
            "laughter_time_s": _safe_time(peak_laughter_idx),
            "laughter_z": float(laughter_z[peak_laughter_idx]) if len(laughter_z) > 0 else 0.0,
            "cheering_time_s": _safe_time(peak_cheering_idx),
            "cheering_z": float(cheering_z[peak_cheering_idx]) if len(cheering_z) > 0 else 0.0,
            "combo_time_s": _safe_time(peak_combo_idx),
            "combo_z": float(event_combo_z[peak_combo_idx]) if len(event_combo_z) > 0 else 0.0,
            "candidate_count": len(candidates),
        },
        "candidates": candidates,
        "elapsed_seconds": _time.time() - start_time,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    
    def _upd(d: Dict[str, Any]) -> None:
        d.setdefault("analysis", {})
        d["analysis"]["audio_events"] = {
            **payload,
            "features_npz": str(audio_events_path.relative_to(proj.project_dir)),
        }
    
    _report(0.98, "Updating project metadata")
    update_project(proj, _upd)
    
    _report(1.0, "Done")
    
    return payload


def load_audio_events_features(proj: Project) -> Optional[Dict[str, np.ndarray]]:
    """Load audio events features if available."""
    audio_events_path = proj.audio_events_features_path
    if not audio_events_path.exists():
        return None
    from .project import load_npz
    return load_npz(audio_events_path)
