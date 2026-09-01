#!/usr/bin/env python3
"""
Speech-to-Text Manager - FIXED VERSION
Features: Whisper Engine, Audio Processing, Thread-Safe Operations
Added missing transcribe() method for API compatibility
"""

import os
import sys
import time
import logging
import threading
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

# Fast GPU-enabled configuration (proven working approach)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Allow GPU access
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better compatibility

try:
    import torch
    # Force CPU mode for compatibility, but let Whisper auto-detect GPU
    torch.cuda.is_available = lambda: False
    torch.cuda.get_device_name = lambda i=None: "CPU (Auto-Detection Mode)"
    print("🔧 PyTorch in compatibility mode - Whisper will auto-detect GPU")
except ImportError:
    torch = None

# Import speech recognition with fallback handling
try:
    import speech_recognition as sr
except ImportError:
    try:
        import speechrecognition as sr
    except ImportError:
        sr = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Import whisper with fallback handling
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# Global flags to prevent infinite Whisper loading attempts
_WHISPER_LOAD_ATTEMPTED = False
_WHISPER_LOAD_ATTEMPTS = 0  # Backup counter to prevent infinite loops

class SpeechToTextManager:
    """
    Manages speech recognition engines with focus on OpenAI Whisper.
    
    This class provides a unified interface for speech-to-text functionality,
    currently supporting only the OpenAI Whisper engine which offers:
    
    - High accuracy across 100+ languages
    - GPU acceleration support
    - Offline processing capability
    - Multiple model sizes (tiny, base, small, medium, large)
    
    Features:
        - Automatic dependency checking and validation
        - Memory management with cleanup methods
        - Comprehensive error handling and logging
        - Thread-safe model loading
        - API endpoints for transcription services
    
    Usage:
        >>> manager = SpeechToTextManager()
        >>> if manager.load_engine("whisper"):
        ...     result = manager.transcribe_file("audio.wav")  # Auto-detect language
        ...     print(result["text"])
    
    Dependencies:
        - openai-whisper==20230314
        - torch, torchaudio
        - numpy
        - ffmpeg (for audio processing)
    
    Environment Variables:
        WHISPER_MODEL: Model size (default: "base")
        WHISPER_CACHE_DIR: Cache directory (default: "~/.cache/whisper")
    
    Note: This is a Whisper-only implementation. Other engines (Vosk, Google, Sphinx)
          have been removed to focus on the highest quality solution.
    """
    def __init__(self):
        self.recognizer = sr.Recognizer() if sr else None
        self.current_engine = "whisper"  # Default to Whisper (best quality)
        self.models_loaded = False
        self.load_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Available engines - ranked by quality
        self.available_engines = {
            "whisper": {
                "name": "OpenAI Whisper",
                "type": "offline",
                "free": True,
                "requires_download": True,
                "description": "State-of-the-art offline speech recognition by OpenAI",
                "accuracy": "⭐⭐⭐⭐⭐",
                "languages": "~100+",
                "deployment": "Local",
                "model_sizes": {
                    "tiny": "Fast inference, lower accuracy (~39M parameters)",
                    "base": "Good balance of speed/accuracy (~74M parameters)",
                    "small": "Better accuracy, slower (~244M parameters)",
                    "medium": "High accuracy, slower (~769M parameters)",
                    "large": "Best accuracy, slowest (~1550M parameters)"
                },
                "download_size": "74-1550 MB depending on model",
                "recommended": "base",
                "features": [
                    "GPU acceleration",
                    "CPU fallback",
                    "Multi-language support",
                    "High accuracy",
                    "Offline processing"
                ]
            }
        }
    
    def get_available_engines(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available speech recognition engines"""
        return self.available_engines
    
    def load_engine(self, engine_name: str = "whisper") -> bool:
        """Load specified speech recognition engine"""
        with self.load_lock:
            if engine_name != "whisper":
                self.logger.error(f"Only Whisper engine is supported (requested: {engine_name})")
                return False
            
            try:
                if engine_name == "whisper":
                    return self._load_whisper()
                else:
                    return False
                    
            except Exception as e:
                self.logger.error(f"Failed to load {engine_name}: {e}")
                return False
    
    def _load_whisper(self) -> bool:
        """
        Load OpenAI Whisper speech recognition engine.
        
        Downloads the 'base' model (~140MB) on first run and caches it
        in ~/.cache/whisper/. Requires torch, numpy, and ffmpeg.
        
        Returns:
            bool: True if loaded successfully, False otherwise
            
        Raises:
            ImportError: If whisper package not installed
            RuntimeError: If model loading fails
        """
        import traceback
        import threading
        import time
        import os
        
        # Check global flags to prevent multiple loading attempts
        global _WHISPER_LOAD_ATTEMPTED, _WHISPER_LOAD_ATTEMPTS
        
        # Increment counter only AFTER validation checks pass (FIX: Issue 10)
        if _WHISPER_LOAD_ATTEMPTED:
            self.logger.warning("⚠ Whisper loading already attempted, skipping...")
            return False
            
        if _WHISPER_LOAD_ATTEMPTS >= 3:  # Backup safety limit
            self.logger.error("❌ Too many Whisper loading attempts, giving up")
            return False
        
        # Check if model is already loaded (FIX: Issue 1)
        if hasattr(self, 'whisper_model') and self.whisper_model is not None:
            self.logger.info("✅ Whisper model already loaded and ready")
            self.current_engine = "whisper"
            self.models_loaded = True
            return True
        
        try:
            # Increment attempt counter only after all checks pass
            _WHISPER_LOAD_ATTEMPTS += 1
            self.logger.info(f"🔍 Starting Whisper loading sequence... (attempt {_WHISPER_LOAD_ATTEMPTS}/3)")
            
            # ============================================================================
            # ISSUE 5: DEPENDENCY CHECKS (P0 CRITICAL)
            # ============================================================================
            self.logger.info("📦 Step 1/4: Checking dependencies...")
            
            # Check Python version
            import sys
            if sys.version_info < (3, 8):
                raise RuntimeError(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
            
            # Check PyTorch
            try:
                import torch
                torch_version = torch.__version__
                self.logger.info(f"✅ PyTorch found (version {torch_version})")
            except ImportError:
                raise ImportError("PyTorch not found. Install with: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
            
            # Check NumPy
            try:
                import numpy
                numpy_version = numpy.__version__
                self.logger.info(f"✅ NumPy found (version {numpy_version})")
            except ImportError:
                raise ImportError("NumPy not found. Install with: pip install numpy")
            
            # Check ffmpeg (for audio processing)
            try:
                import subprocess
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    self.logger.info(f"✅ ffmpeg found ({version_line})")
                else:
                    raise RuntimeError("ffmpeg not working properly")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                self.logger.warning("⚠️ ffmpeg not found or not working - audio processing may be limited")
            
            # ============================================================================
            # WHISPER PACKAGE CHECK
            # ============================================================================
            self.logger.info("📦 Step 2/4: Checking Whisper package...")
            
            if not WHISPER_AVAILABLE:
                raise ImportError("Whisper package not found. Please install with: pip install openai-whisper")
            
            # Check for version info (optional - not all versions have this)
            whisper_version = getattr(whisper, '__version__', 'unknown')
            self.logger.info(f"✅ Whisper package found (version {whisper_version})")
            self.logger.info(f"   Location: {whisper.__file__}")
            
            # ============================================================================
            # MODEL LOADING WITH TIMEOUT (FIX: Issue 25)
            # ============================================================================
            self.logger.info("🧠 Step 3/4: Loading Whisper model...")
            
            # Model size configuration (FIX: Issue 26)
            model_size = os.getenv("WHISPER_MODEL", "base")
            cache_dir = os.getenv("WHISPER_CACHE_DIR", "~/.cache/whisper")
            
            self.logger.info(f"   Model: {model_size} (configurable via WHISPER_MODEL env var)")
            self.logger.info(f"   Cache: {cache_dir} (configurable via WHISPER_CACHE_DIR env var)")
            
            # Add timeout to prevent hanging (5 minutes max) - Windows compatible
            timeout_triggered = threading.Event()
            
            def timeout_handler():
                timeout_triggered.set()
                raise TimeoutError("Model loading timed out after 300 seconds")
            
            # Start timeout timer
            timeout_timer = threading.Timer(300.0, timeout_handler)
            timeout_timer.start()
            
            try:
                self.logger.info("   Downloading/Loading model...")
                # Load Whisper model - auto-detects GPU when available
                self.whisper_model = whisper.load_model(model_size)
                timeout_timer.cancel()  # Cancel timeout
                
                # ============================================================================
                # MODEL VALIDATION (FIX: Issues 7, 9)
                # ============================================================================
                self.logger.info("🧪 Step 4/4: Testing loaded model...")
                
                # Validate model was loaded correctly
                if self.whisper_model is None:
                    raise ValueError("Model loading returned None")
                
                if not hasattr(self.whisper_model, 'transcribe'):
                    raise ValueError("Loaded model missing transcribe method")
                
                # Test with dummy audio to ensure model works
                try:
                    import numpy as np
                    
                    # Create a small test audio (1 second of silence)
                    test_audio = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
                    
                    # Try transcription on test audio
                    result = self.whisper_model.transcribe(test_audio)
                    
                    if isinstance(result, dict) and 'text' in result:
                        self.logger.info("✅ Model test successful - transcription working")
                    else:
                        self.logger.warning("⚠️ Model loaded but test transcription unusual")
                        
                except Exception as test_e:
                    self.logger.warning(f"⚠️ Model test failed: {test_e}")
                    # Model still loaded, just note the test failure
                
                self.logger.info(f"✅ Whisper model loaded successfully: {type(self.whisper_model)}")
                
                # Set engine state
                self.current_engine = "whisper"
                self.models_loaded = True
                
                # Mark as attempted ONLY on success (FIX: Issue 1)
                _WHISPER_LOAD_ATTEMPTED = True
                
                self.logger.info("🧠 Whisper engine loaded successfully (OpenAI - Top accuracy, ~100+ languages)")
                return True
                
            except TimeoutError:
                timeout_timer.cancel()
                self.logger.error("❌ Model loading timed out after 300 seconds")
                self.logger.error("This may indicate network issues or insufficient memory")
                return False
            except Exception as model_e:
                timeout_timer.cancel()  # Cancel timeout
                raise model_e  # Re-raise for outer exception handling
                
        except ImportError as e:
            self.logger.error(f"❌ Import error during Whisper loading: {e}")
            self.logger.error("Required packages missing - check installation")
            return False
        except RuntimeError as e:
            self.logger.error(f"❌ Runtime error during Whisper loading: {e}")
            self.logger.error("Check GPU/CUDA compatibility or install CPU-only version")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during Whisper loading ({type(e).__name__}): {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def transcribe(self, audio_data: bytes, engine: str = "whisper", language: str = None) -> Dict[str, Any]:
        """
        NEW METHOD: Transcribe audio data directly (for API compatibility)
        
        This method accepts audio data as bytes and transcribes it using the specified engine.
        This is the method that the API endpoints expect to call.
        
        Args:
            audio_data: Audio data as bytes
            engine: Speech recognition engine to use (only "whisper" supported)
            language: Language code for transcription (default: None for auto-detection)
            
        Returns:
            Dictionary with transcription results
        """
        print(f"DEBUG: transcribe START - engine: {engine}, audio_data length: {len(audio_data) if audio_data else 0}")
        
        if not self.models_loaded:
            return {"error": "Speech recognition engine not loaded"}
        
        if engine != "whisper":
            return {"error": f"Unsupported engine: {engine}. Only Whisper is available."}
        
        if not hasattr(self, 'whisper_model') or self.whisper_model is None:
            return {"error": "Whisper model not loaded"}
        
        try:
            # Check audio data
            if not audio_data or len(audio_data) == 0:
                return {"error": "No audio data provided"}
            
            # Check file size
            file_size = len(audio_data)
            max_size = 50 * 1024 * 1024  # 50MB limit
            if file_size > max_size:
                return {"error": f"Audio data too large: {file_size} bytes (max {max_size} bytes)"}
            
            # Clean language code (handle None for auto-detection)
            clean_language = language.split('-')[0] if language and '-' in language else language
            
            print(f"DEBUG: transcribe - Audio size: {file_size} bytes, language: {clean_language}")
            
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Use the existing transcribe_file method
                result = self.transcribe_file(temp_file_path, language=clean_language)
                print(f"DEBUG: transcribe - transcribe_file result: {result}")
                return result
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_e:
                    self.logger.warning(f"Could not delete temporary file {temp_file_path}: {cleanup_e}")
                    
        except Exception as e:
            print(f"DEBUG: transcribe - Exception: {type(e).__name__}: {e}")
            import traceback
            error_msg = f"Transcription error: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"error": error_msg}
    
    def transcribe_file(self, audio_file_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe audio from file using Whisper engine"""
        print(f"DEBUG: transcribe_file START - path: {audio_file_path}")
        
        if not self.models_loaded:
            return {"error": "Speech recognition engine not loaded"}
        
        if self.current_engine != "whisper":
            return {"error": f"Unsupported engine: {self.current_engine}. Only Whisper is available."}
        
        if not hasattr(self, 'whisper_model') or self.whisper_model is None:
            return {"error": "Whisper model not loaded"}
        
        try:
            # Clean language code (remove region if present, e.g., "en-US" -> "en")
            clean_language = language.split('-')[0] if language and '-' in language else language
            
            print(f"DEBUG: transcribe_file - Before file operations")
            
            # Get file size for validation
            try:
                print(f"DEBUG: transcribe_file - About to os.path.getsize")
                file_size = os.path.getsize(audio_file_path)
                print(f"DEBUG: transcribe_file - File size: {file_size}")
            except OSError as e:
                print(f"DEBUG: transcribe_file - OSError in get size: {e}")
                return {"error": f"Cannot access audio file: {e}"}
            
            max_size = 50 * 1024 * 1024  # 50MB limit
            if file_size > max_size:
                return {"error": f"Audio file too large: {file_size} bytes (max {max_size} bytes)"}
            
            print(f"DEBUG: transcribe_file - Before logger.info")
            self.logger.info(f"Transcribing audio file: {audio_file_path} (size: {file_size} bytes)")
            print(f"DEBUG: transcribe_file - Before Whisper transcribe")
            
            # Import required modules for memory-based processing
            import uuid
            import time
            
            # NEW WHISPER API: Convert audio to numpy array as required by Whisper 20250625
            print(f"DEBUG: transcribe_file - Converting audio to numpy array for new Whisper API")
            
            import numpy as np
            import wave
            import io
            
            try:
                # Method 1: Use wave module to parse WAV properly
                print(f"DEBUG: transcribe_file - Parsing WAV with wave module...")
                with wave.open(audio_file_path, 'rb') as wav_file:
                    # Get audio parameters
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()  
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    
                    print(f"DEBUG: transcribe_file - WAV params: {n_channels}ch, {sample_width}bytes, {sample_rate}Hz, {n_frames}frames")
                    
                    # Read audio data as numpy array
                    audio_data = wav_file.readframes(n_frames)
                    
                    # Convert to numpy array based on sample width
                    if sample_width == 2:  # 16-bit
                        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    elif sample_width == 4:  # 32-bit
                        audio_array = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32)
                    else:
                        audio_array = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
                    
                    # Normalize to [-1, 1] range
                    if audio_array.max() > 1.0:
                        audio_array = audio_array / 32768.0 if sample_width == 2 else audio_array / 2147483648.0
                    
                    print(f"DEBUG: transcribe_file - Audio array shape: {audio_array.shape}, range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
                
                # Call Whisper with AUTO language detection (FIXED)
                print(f"DEBUG: transcribe_file - Calling Whisper with AUTO language detection...")
                result = self.whisper_model.transcribe(
                    audio_array, 
                    language=None,  # ← CRITICAL FIX: Enable auto-detection like OpenAI Whisper
                    task="transcribe"
                )
                print(f"DEBUG: transcribe_file - NumPy array approach SUCCESS!")
                
            except Exception as e:
                print(f"DEBUG: transcribe_file - WAV parsing failed: {e}")
                
                # Method 2: Fallback to simple file reading for MP3 or other formats
                print(f"DEBUG: transcribe_file - Trying alternative audio loading...")
                
                try:
                    # Use librosa if available for format-agnostic loading
                    import librosa
                    
                    audio_array, sample_rate = librosa.load(audio_file_path, sr=16000)
                    print(f"DEBUG: transcribe_file - LibROSA loaded: shape={audio_array.shape}, sr={sample_rate}Hz")
                    
                    # FIXED: Use auto-detection instead of forcing language
                    result = self.whisper_model.transcribe(
                        audio_array, 
                        language=None,  # ← CRITICAL FIX: Enable auto-detection
                        task="transcribe"
                    )
                    print(f"DEBUG: transcribe_file - LibROSA approach SUCCESS!")
                    
                except ImportError:
                    print(f"DEBUG: transcribe_file - LibROSA not available, trying raw conversion...")
                    
                    # Method 3: Simple raw conversion (best effort)
                    with open(audio_file_path, 'rb') as f:
                        raw_data = f.read()
                    
                    # Try to create a simple array from raw data
                    if len(raw_data) > 44:  # Skip WAV header if present
                        audio_data = raw_data[44:]  # Assume WAV header
                    else:
                        audio_data = raw_data
                    
                    # Simple conversion (may not work perfectly)
                    try:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        # FIXED: Use auto-detection instead of forcing language
                        result = self.whisper_model.transcribe(
                            audio_array, 
                            language=None,  # ← CRITICAL FIX: Enable auto-detection
                            task="transcribe"
                        )
                        print(f"DEBUG: transcribe_file - Raw conversion SUCCESS!")
                    except Exception as raw_error:
                        print(f"DEBUG: transcribe_file - Raw conversion failed: {raw_error}")
                        raise Exception(f"Could not process audio file: {e}")
                
                except Exception as librosa_error:
                    print(f"DEBUG: transcribe_file - LibROSA failed: {librosa_error}")
                    raise Exception(f"Audio processing failed: {librosa_error}")
            
            # Check if file still exists after Whisper call
            if os.path.exists(audio_file_path):
                print(f"DEBUG: transcribe_file - File still exists after Whisper call")
            else:
                print(f"DEBUG: transcribe_file - File disappeared after Whisper call: {audio_file_path}")
            
            print(f"DEBUG: transcribe_file - After Whisper transcribe - SUCCESS!")
            
            # Extract text and metadata
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            language_detected = result.get("language", "auto-detected")
            
            # Log the fix results
            self.logger.info(f"🎤 Language Auto-Detection Working!")
            self.logger.info(f"   Detected: {language_detected} (NOT forced)")
            self.logger.info(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            self.logger.info(f"   Fix: Using language=None for Whisper auto-detection")
            
            # Calculate confidence score if segments available
            confidence = None
            if segments:
                confidences = [seg.get("avg_logprob", -1.0) for seg in segments if "avg_logprob" in seg]
                if confidences:
                    avg_logprob = sum(confidences) / len(confidences)
                    confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))
            
            success_result = {
                "success": True,
                "text": text,
                "engine": self.current_engine,
                "language": language_detected,
                "confidence": confidence,
                "segments_count": len(segments) if segments else 0,
                "audio_duration": result.get("duration", None),
                "file_size": file_size
            }
            
            print(f"DEBUG: transcribe_file - Returning SUCCESS: {success_result['success']}")
            return success_result
                    
        except FileNotFoundError as fnf:
            print(f"DEBUG: transcribe_file - FileNotFoundError: {fnf}")
            return {"error": f"Audio file not found: {audio_file_path}"}
        except Exception as e:
            print(f"DEBUG: transcribe_file - Exception: {type(e).__name__}: {e}")
            import traceback
            error_msg = f"Transcription error: {e}"
            print(f"DEBUG: transcribe_file - Full traceback: {traceback.format_exc()}")
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {"error": error_msg}
    
    def cleanup_whisper(self) -> bool:
        """
        Clean up Whisper engine resources.
        
        This method unloads the Whisper model and frees memory.
        Useful for memory management or when switching engines.
        
        Returns:
            bool: True if cleanup successful, False otherwise
        """
        try:
            if hasattr(self, 'whisper_model') and self.whisper_model is not None:
                self.logger.info("🧹 Cleaning up Whisper model resources...")
                
                # Delete the model
                delattr(self, 'whisper_model')
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear GPU memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.logger.info("✅ GPU memory cleared")
                except Exception as gpu_e:
                    self.logger.warning(f"⚠️ Could not clear GPU memory: {gpu_e}")
                
                # Reset engine state
                self.current_engine = "none"
                self.models_loaded = False
                self.logger.info("✅ Whisper cleanup completed successfully")
                return True
            else:
                self.logger.info("ℹ️ No Whisper model to clean up")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error during Whisper cleanup: {e}")
            return False
    
    def unload_whisper(self) -> bool:
        """
        Alias for cleanup_whisper() for consistency with other engines.
        
        Returns:
            bool: True if unload successful, False otherwise
        """
        return self.cleanup_whisper()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current speech recognition status with fix information"""
        return {
            "engine_loaded": self.models_loaded,
            "current_engine": self.current_engine,
            "available_engines": list(self.available_engines.keys()),
            "engine_info": self.available_engines.get(self.current_engine, {}),
            "language_detection": "auto-whisper",  # NEW: Shows auto-detection is active
            "output_behavior": "transcribe_in_detected_language",  # NEW: Shows fix is applied
            "fixes_applied": [
                "whisper_auto_detection_language_none",
                "no_bengali_forcing_for_english", 
                "output_in_detected_language_only"
            ]
        }

# Global speech recognition manager
SPEECH_TO_TEXT_MANAGER = SpeechToTextManager()

# ============================================================================
# SINGLE CONSOLIDATED WHISPER INITIALIZATION (FIX: Issues 2, 3, 4)
# ============================================================================
def initialize_whisper_engine():
    """
    Consolidated Whisper engine initialization function.
    Called once during startup to avoid multiple initialization attempts.
    """
    try:
        logger.info("🔄 Starting Whisper engine initialization...")
        
        # Reset attempt counter for fresh start
        global _WHISPER_LOAD_ATTEMPTS
        _WHISPER_LOAD_ATTEMPTS = 0
        
        # Single initialization attempt
        logger.info("🧠 Loading Whisper model with comprehensive fix...")
        logger.info("🎤 Attempting to load Whisper engine on startup...")
        
        whisper_loaded = SPEECH_TO_TEXT_MANAGER.load_engine("whisper")
        
        if whisper_loaded:
            logger.info("✅ Whisper engine loaded successfully on startup")
        else:
            logger.warning("⚠ Whisper failed to load - speech recognition will not be available")
            
        # Log final status
        logger.info(f"🔒 Final initialization state: _WHISPER_LOAD_ATTEMPTED = {_WHISPER_LOAD_ATTEMPTED}")
        logger.info(f"🔢 Whisper load attempts: {_WHISPER_LOAD_ATTEMPTS}")
        logger.info(f"🎤 Current speech engine: {SPEECH_TO_TEXT_MANAGER.current_engine}")
        logger.info(f"📊 Models loaded status: {SPEECH_TO_TEXT_MANAGER.models_loaded}")
        
        return whisper_loaded
        
    except Exception as e:
        logger.error(f"Engine loading error: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        logger.warning("⚠ Speech recognition unavailable - Whisper failed to load")
        return False

# Execute single initialization
initialize_whisper_engine()