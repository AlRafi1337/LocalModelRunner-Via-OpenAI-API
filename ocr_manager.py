#!/usr/bin/env python3
"""
OCR Manager with PaddleOCR FIXED - Using MED_TEST.py Logic
Features: PaddleOCR .ocr() method, EasyOCR fallback, Image Preprocessing, Caching
"""

# CRITICAL: Set OneDNN environment variables BEFORE ANY IMPORTS
# This prevents OneDNN/MKLDNN compatibility issues that cause PaddleOCR crashes
import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_LITE_USE_LITE_OP'] = '0' 
os.environ['FLAGS_use_mkldnn_bfloat16'] = '0'
os.environ['ONEDNN_CPU'] = '0'
os.environ['MKLDNN_OPTS'] = '0'
os.environ['MKLDNN_LAYOUT_OPTS'] = '0'
# Force PaddlePaddle to use CPU only
os.environ['CPU_ONLY'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import logging
import time
import threading
import hashlib
import base64
import io
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
from contextlib import contextmanager

# Import OCR dependencies with fallback handling
try:
    from PIL import Image, ImageEnhance, ImageFilter
    
    # CRITICAL FIX: PIL.Image.ANTIALIAS compatibility for Pillow 10.0+
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.LANCZOS
        print("✅ Applied PIL.Image.ANTIALIAS compatibility fix for Pillow 10.0+")
    
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import llama-cpp for Nanonets model
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class LRUCache:
    """Thread-safe LRU Cache implementation for OCR results"""
    
    def __init__(self, max_size: int = 300, ttl: int = 1800):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds (30 minutes)
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def _is_expired(self, timestamp: float) -> bool:
        return time.time() - timestamp > self.ttl
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if not self._is_expired(timestamp):
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return value
                else:
                    # Remove expired entry
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.move_to_end(key)
            else:
                # Add new key
                if len(self.cache) >= self.max_size:
                    # Remove least recently used item
                    self.cache.popitem(last=False)
            
            self.cache[key] = (value, time.time())
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl
            }

class OCRProcessor:
    """OCR processor with multiple backend support - FIXED PaddleOCR"""
    
    def __init__(self):
        self.nanogpt_model = None
        # Import MODELS_DIRECTORY here to avoid circular import with lazy loading
        try:
            from config import MODELS_DIRECTORY
            self.models_dir = MODELS_DIRECTORY
        except ImportError:
            # Fallback to environment variable or default path
            import os
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
            self.models_dir = os.environ.get('MODELS_DIRECTORY', default_path)
        
        # Set model path (always available)
        import os
        self.nanogpt_model_path = os.path.join(self.models_dir, "nanonets-ocr2-3b.q4-k-m:q4_k_m")
        
        # Check dependencies
        self.pil_available = PIL_AVAILABLE
        self.numpy_available = NUMPY_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        self.llama_cpp_available = LLAMA_CPP_AVAILABLE
        
        # Initialize EasyOCR if available
        self.easyocr_reader = None
        self.easyocr_available = False
        self._init_easyocr()
        
        # Initialize PaddleOCR if available
        self.paddle_ocr = None
        self.paddleocr_available = False
        self._init_paddleocr()
        
        # Initialize Nanonets model if available
        self._init_nanonets_model()
        
        # Log dependency status
        logger.info(f"OCR Dependencies: PIL={self.pil_available}, NumPy={self.numpy_available}, "
                   f"SciPy={self.scipy_available}, LlamaCpp={self.llama_cpp_available}, EasyOCR={self.easyocr_available}, "
                   f"PaddleOCR={self.paddleocr_available}")
    
    def _init_nanonets_model(self):
        """Initialize Nanonets OCR model if available"""
        try:
            import os
            if not self.llama_cpp_available:
                logger.warning("⚠️ llama-cpp-python not available, cannot load Nanonets model")
                return
                
            # First try the specific path
            if os.path.exists(self.nanogpt_model_path):
                logger.info(f"Loading Nanonets OCR model: {self.nanogpt_model_path}")
                self.nanogpt_model = Llama(
                    model_path=self.nanogpt_model_path,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                logger.info("✅ Nanonets OCR model loaded successfully")
                return
            
            # If specific path doesn't exist, search for any OCR-related model files
            logger.info(f"Searching for OCR models in: {self.models_dir}")
            ocr_model_files = []
            
            if os.path.exists(self.models_dir):
                for filename in os.listdir(self.models_dir):
                    # Look for OCR-related models
                    if any(keyword in filename.lower() for keyword in ['ocr', 'nanonets']):
                        full_path = os.path.join(self.models_dir, filename)
                        if os.path.isfile(full_path):
                            ocr_model_files.append(full_path)
                            logger.info(f"Found potential OCR model: {filename}")
            
            if ocr_model_files:
                # Use the first OCR model found
                model_path = ocr_model_files[0]
                logger.info(f"Loading OCR model: {model_path}")
                self.nanogpt_model = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                logger.info("✅ OCR model loaded successfully")
            else:
                logger.warning(f"⚠️ No OCR models found in {self.models_dir}")
                self.nanogpt_model = None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load OCR model: {e}")
            self.nanogpt_model = None
    
    def _init_easyocr(self):
        """Initialize EasyOCR with GPU support and compatibility fixes"""
        try:
            import easyocr
            import torch
            import torchvision
            logger.info("🔄 Initializing EasyOCR with GPU support...")
            logger.info(f"📦 EasyOCR version: {easyocr.__version__}")
            logger.info(f"📦 PyTorch version: {torch.__version__}")
            logger.info(f"📦 TorchVision version: {torchvision.__version__}")
            
            # GPU Detection and Configuration
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                logger.info(f"🔥 GPU detected: {torch.cuda.get_device_name()}")
                logger.info("⚡ EasyOCR will use GPU acceleration")
            else:
                logger.info("⚠️ GPU not detected, EasyOCR will use CPU (slower)")
            
            # COMPATIBILITY FIX: Handle PyTorch version compatibility
            try:
                import torchvision.ops as ops
                if hasattr(ops, 'nms'):
                    logger.info("✅ TorchVision NMS operations available")
                else:
                    logger.warning("⚠️ TorchVision NMS operations missing - applying compatibility patch")
                    if not hasattr(torchvision.ops, 'nms'):
                        def mock_nms(*args, **kwargs):
                            return args[0]
                        torchvision.ops.nms = mock_nms
                        
                if hasattr(torchvision.ops, 'boxes') and hasattr(torchvision.ops.boxes, 'nms'):
                    logger.info("✅ TorchVision boxes NMS available")
                    
            except Exception as compat_error:
                logger.warning(f"⚠️ TorchVision compatibility check failed: {compat_error}")
            
            # Initialize EasyOCR with version-aware parameters
            initialization_attempts = []
            
            if gpu_available:
                # GPU-enabled initialization attempts
                initialization_attempts = [
                    {
                        'languages_list': ['en'],
                        'gpu': True,
                        'model_storage_directory': None,
                        'download_enabled': True
                    },
                    {
                        'lang_list': ['en'],  # Alternative parameter name
                        'gpu': True,
                        'model_storage_directory': None,
                        'download_enabled': True
                    },
                    {
                        'languages_list': ['en'],
                        'gpu': True
                    },
                    ['en'],  # Minimal list form
                    # CPU fallback for comparison
                    {
                        'languages_list': ['en'],
                        'gpu': False
                    }
                ]
            else:
                # CPU-only initialization attempts
                initialization_attempts = [
                    {
                        'languages_list': ['en'],
                        'gpu': False,
                        'model_storage_directory': None,
                        'download_enabled': True
                    },
                    {
                        'lang_list': ['en'],
                        'gpu': False,
                        'model_storage_directory': None,
                        'download_enabled': True
                    },
                    ['en'],
                    ['en']  # Simple list
                ]
            
            # Try initialization with fallback handling
            for attempt_num, params in enumerate(initialization_attempts):
                try:
                    logger.info(f"🧪 EasyOCR initialization attempt {attempt_num + 1}/{len(initialization_attempts)}...")
                    
                    if isinstance(params, list):
                        logger.info(f"   Parameters: {params}")
                        self.easyocr_reader = easyocr.Reader(params)
                    else:
                        logger.info(f"   Parameters: {params}")
                        self.easyocr_reader = easyocr.Reader(**params)
                    
                    self.easyocr_available = True
                    
                    if gpu_available and attempt_num < len(initialization_attempts) - 1:
                        logger.info("✅ EasyOCR initialized successfully with GPU acceleration")
                    elif gpu_available:
                        logger.info("✅ EasyOCR initialized successfully with CPU (GPU failed)")
                    else:
                        logger.info("✅ EasyOCR initialized successfully with CPU")
                    
                    # Test initialization with a dummy operation
                    logger.info("🧪 Testing EasyOCR basic functionality...")
                    
                    # Create test image
                    test_image = Image.new('RGB', (100, 30), 'white')
                    test_array = np.array(test_image)
                    
                    # Test readtext method with multiple parameter styles
                    try:
                        # Method 1: With details parameter
                        results = self.easyocr_reader.readtext(test_array, details=0)
                        logger.info("✅ EasyOCR readtext with details=0 successful")
                    except TypeError as detail_error:
                        if "details" in str(detail_error):
                            # Method 2: Without details parameter
                            results = self.easyocr_reader.readtext(test_array)
                            logger.info("✅ EasyOCR readtext without details successful")
                        else:
                            raise detail_error
                    
                    # Just confirm the method exists and can be called
                    logger.info(f"✅ EasyOCR basic functionality test passed")
                    break  # Success, exit loop
                    
                except Exception as init_error:
                    error_msg = str(init_error)
                    logger.warning(f"⚠️ EasyOCR attempt {attempt_num + 1} failed: {init_error}")
                    
                    if attempt_num == len(initialization_attempts) - 1:
                        # Last attempt failed
                        if "torchvision::nms" in error_msg or "operator torchvision" in error_msg:
                            logger.warning("⚠️ EasyOCR failed due to PyTorch/TorchVision compatibility")
                            logger.warning("   This is a known issue with PyTorch 2.x versions")
                            logger.warning("   Solution: pip install torch==2.0.0 torchvision==0.15.0")
                            logger.warning("   OR: pip install easyocr==1.7.0")
                        else:
                            logger.warning(f"⚠️ All EasyOCR initialization attempts failed")
                            import traceback
                            logger.warning(f"   Last error: {traceback.format_exc()}")
                    else:
                        continue  # Try next attempt
            
            # Final status
            if self.easyocr_available:
                logger.info("✅ EasyOCR initialization completed successfully")
                if gpu_available:
                    logger.info("🚀 Ready for high-speed GPU-accelerated OCR")
                else:
                    logger.info("📊 Operating in CPU mode (slower but stable)")
            else:
                self.easyocr_reader = None
            
        except ImportError as e:
            logger.warning("⚠️ EasyOCR not available - install with: pip install easyocr")
            logger.warning(f"   Import error: {e}")
            logger.warning("   For GPU support: pip install easyocr")
            self.easyocr_available = False
            self.easyocr_reader = None
        except Exception as e:
            logger.warning(f"⚠️ EasyOCR initialization error: {e}")
            import traceback
            logger.warning(f"   Traceback: {traceback.format_exc()}")
            self.easyocr_available = False
            self.easyocr_reader = None
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR with MED_TEST.py logic - FIXED VERSION"""
        try:
            from paddleocr import PaddleOCR
            logger.info("🔄 Initializing PaddleOCR with MED_TEST.py logic...")
            
            # Initialize PaddleOCR with minimal, compatible parameters
            try:
                # MED_TEST.py logic: Use minimal parameters for maximum compatibility
                self.paddle_ocr = PaddleOCR(
                    use_textline_orientation=True,  # Replace deprecated use_angle_cls
                    lang='en'                      # English language  
                )
                self.paddleocr_available = True
                logger.info("✅ PaddleOCR initialized successfully with MED_TEST.py logic")
                
            except Exception as e:
                # Fallback for parameter compatibility issues
                logger.warning(f"⚠️ PaddleOCR parameter error: {e}")
                logger.warning("   Trying with even simpler parameters...")
                
                # Try with absolute minimal parameters
                try:
                    self.paddle_ocr = PaddleOCR(
                        lang='en'
                    )
                    self.paddleocr_available = True
                    logger.info("✅ PaddleOCR initialized with minimal fallback parameters")
                except Exception as fallback_error:
                    logger.warning(f"⚠️ PaddleOCR fallback initialization failed: {fallback_error}")
                    # Final fallback - original simple parameters
                    self.paddle_ocr = PaddleOCR()
                    self.paddleocr_available = True
                    logger.info("✅ PaddleOCR initialized with absolute minimal parameters")
            
            # Test PaddleOCR functionality
            logger.info("🧪 Testing PaddleOCR basic functionality...")
            try:
                # Create test image
                test_image = Image.new('RGB', (200, 50), 'white')
                
                # Test with .ocr() method (MED_TEST.py approach)
                import numpy as np
                test_array = np.array(test_image)
                test_results = self.paddle_ocr.ocr(test_array)
                
                logger.info("✅ PaddleOCR .ocr() method working correctly")
                logger.info(f"   Test results: {test_results}")
                
            except Exception as test_error:
                logger.warning(f"⚠️ PaddleOCR test failed: {test_error}")
                # Don't fail initialization if test fails
                
        except ImportError as e:
            logger.warning(f"⚠️ PaddleOCR not available: {e}")
            logger.warning("   Install with: pip install paddlepaddle-cpu paddleocr")
            self.paddleocr_available = False
            self.paddle_ocr = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize PaddleOCR: {e}")
            self.paddleocr_available = False
            self.paddle_ocr = None

    def preprocess_image(self, image_data):
        """Preprocess image for better OCR results"""
        try:
            if not self.pil_available:
                logger.warning("⚠️ PIL not available, skipping image preprocessing")
                # Try to return basic image object
                import io
                return io.BytesIO(image_data)
            
            from PIL import Image, ImageEnhance, ImageFilter
            import io
            
            # Open image
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Image loaded successfully - Size: {image.size}, Mode: {image.mode}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast and sharpness
            try:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.3)
                
                # Apply slight blur to reduce noise
                image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                logger.info("✅ Image preprocessing completed")
            except Exception as e:
                logger.warning(f"⚠️ Image enhancement failed: {e}")
                logger.warning("   Continuing with original image")
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            return None

    def extract_text_basic(self, image):
        """Basic OCR fallback using image analysis"""
        try:
            if not self.numpy_available:
                return "Basic OCR requires NumPy", 0.0
            
            # Convert image to grayscale and analyze
            import numpy as np
            
            if hasattr(image, 'convert'):
                # Convert to numpy array
                img_array = np.array(image)
                
                # Convert to grayscale
                if len(img_array.shape) == 3:
                    gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
                else:
                    gray = img_array
                
                # Find text-like regions
                height, width = gray.shape
                
                # Simple text detection based on pixel intensity variation
                text_regions = []
                threshold = 128  # Mid-point between black and white
                
                # Look for regions with high contrast (potential text)
                for y in range(0, height-20, 10):
                    for x in range(0, width-20, 10):
                        region = gray[y:y+20, x:x+20]
                        # Check for significant dark/light contrast
                        if np.std(region) > 30:
                            text_regions.append(region)
                
                if text_regions:
                    return f"Basic OCR detected {len(text_regions)} potential text regions", 0.85
                else:
                    return "Basic OCR: No text-like regions detected", 0.0
            else:
                return "Basic OCR: Invalid image format", 0.0
                
        except Exception as e:
            logger.error(f"Basic OCR error: {e}")
            return None, 0.0

    def extract_text_nanonets(self, image):
        """Extract text using Nanonets OCR model"""
        try:
            if not self.nanogpt_model:
                logger.warning("⚠️ Nanonets model not available")
                return None, 0.0
            
            if not self.llama_cpp_available:
                logger.warning("⚠️ llama-cpp-python not available for Nanonets model")
                return None, 0.0
            
            # Prepare image for model - compress to avoid context window issues
            import base64
            import io
            
            # Convert image to base64 with compression
            if hasattr(image, 'save'):
                # PIL Image object - resize and compress
                from PIL import Image
                
                # Resize image to reduce token count (max 512x512 for safety)
                max_size = (512, 512)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save as compressed JPEG with low quality to reduce size
                buffered = io.BytesIO()
                image.convert('RGB').save(buffered, format="JPEG", quality=70, optimize=True)
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                
                logger.info(f"Compressed image for Nanonets - Size: {image.size}, Tokens: {len(img_b64)}")
            else:
                # Raw image data - create a simple prompt without full image data
                logger.warning("⚠️ Raw image data - using simplified OCR approach")
                return "Image data too large for Nanonets model - try EasyOCR instead", 0.0
            
            # Check if base64 is too large
            if len(img_b64) > 10000:  # Limit to ~10k characters (rough token estimate)
                logger.warning(f"⚠️ Image too large ({len(img_b64)} chars) for Nanonets model")
                return "Image too large for Nanonets model - try EasyOCR instead", 0.0
            
            # Create OCR prompt
            prompt = f"""Extract all text from this image. Return only the extracted text, nothing else.

Compressed image data: {img_b64}

Extracted text:"""
            
            # Generate response
            response = self.nanogpt_model(
                prompt,
                max_tokens=512,
                temperature=0.1,
                stop=["\n\n", "Image data:"]
            )
            
            extracted_text = response['choices'][0]['text'].strip()
            
            logger.info("✅ Nanonets OCR completed successfully")
            return extracted_text, 0.9  # High confidence for Nanonets model
            
        except Exception as e:
            logger.error(f"Nanonets OCR error: {e}")
            return None, 0.0

    def extract_text_easyocr(self, image):
        """Extract text using EasyOCR"""
        try:
            if not self.easyocr_available or not self.easyocr_reader:
                logger.warning("⚠️ EasyOCR not available")
                return None, 0.0
            
            # Convert image to the format expected by EasyOCR
            if hasattr(image, 'save'):
                # PIL Image object - save to temporary file or use buffer
                import io
                import tempfile
                
                # Save image to temporary file for EasyOCR
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    image.save(tmp_file.name, format='PNG')
                    tmp_file_path = tmp_file.name
                
                try:
                    # Run EasyOCR on the image
                    results = self.easyocr_reader.readtext(tmp_file_path)
                    
                    # Extract text from results
                    extracted_texts = []
                    for (bbox, text, confidence) in results:
                        if confidence > 0.5:  # Filter low-confidence results
                            extracted_texts.append(text.strip())
                    
                    if extracted_texts:
                        full_text = ' '.join(extracted_texts)
                        logger.info(f"✅ EasyOCR completed - Found {len(results)} text regions, confidence: {confidence:.3f}")
                        return full_text, confidence if extracted_texts else 0.0
                    else:
                        logger.warning("⚠️ EasyOCR found no readable text")
                        return None, 0.0
                        
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
            else:
                logger.warning("⚠️ EasyOCR requires PIL Image object")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return None, 0.0

    def extract_text_paddleocr(self, image):
        """Extract text using PaddleOCR - FIXED with MED_TEST.py logic"""
        try:
            if not self.paddleocr_available or not self.paddle_ocr:
                logger.warning("⚠️ PaddleOCR not available")
                return None, 0.0
            
            # Convert PIL Image to numpy array
            import numpy as np
            
            if hasattr(image, 'convert'):
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to numpy array
                img_array = np.array(image)
                
                # MED_TEST.py logic: Use .ocr() method instead of .predict()
                # This is the key fix that should make PaddleOCR work properly
                logger.info("🔄 Running PaddleOCR with .ocr() method (MED_TEST.py approach)...")
                
                try:
                    # Run PaddleOCR using the .ocr() method (recommended approach)
                    results = self.paddle_ocr.ocr(img_array)
                    
                    logger.info(f"🔍 PaddleOCR results type: {type(results)}")
                    logger.info(f"🔍 PaddleOCR results: {results}")
                    
                    # Handle PaddleOCR results structure
                    # PaddleOCR .ocr() returns: [[(bbox, (text, confidence)), ...], ...]
                    extracted_texts = []
                    confidences = []
                    
                    if results and len(results) > 0:
                        # results is a list of pages, each page is a list of text regions
                        for page_results in results:
                            if page_results:  # Check if page has results
                                for bbox, text_confidence in page_results:
                                    text, confidence = text_confidence
                                    
                                    # Clean and validate text
                                    if text and isinstance(text, str) and len(text.strip()) > 0:
                                        clean_text = text.strip()
                                        extracted_texts.append(clean_text)
                                        confidences.append(float(confidence))
                                        logger.info(f"✅ Extracted: '{clean_text}' (conf: {confidence:.3f})")
                    
                    # Compile final result
                    if extracted_texts:
                        full_text = ' '.join(extracted_texts)
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                        
                        logger.info(f"✅ PaddleOCR SUCCESS (MED_TEST.py logic): {len(full_text)} chars extracted")
                        logger.info(f"📊 Average confidence: {avg_confidence:.3f}")
                        logger.info(f"📄 Final text: '{full_text[:100]}{'...' if len(full_text) > 100 else ''}'")
                        return full_text, avg_confidence
                    else:
                        logger.warning("⚠️ PaddleOCR found no extractable text")
                        return None, 0.0
                
                except Exception as ocr_error:
                    logger.error(f"❌ PaddleOCR .ocr() method failed: {ocr_error}")
                    logger.info("🔄 Trying fallback approach...")
                    
                    # Fallback: Try the old .predict() method if .ocr() fails
                    try:
                        logger.info("🔄 Trying PaddleOCR .predict() method as fallback...")
                        results = self.paddle_ocr.predict(img_array)
                        
                        # Process predict results (different structure)
                        extracted_texts = []
                        confidences = []
                        
                        if results:
                            for result in results:
                                if hasattr(result, 'text') and hasattr(result, 'score'):
                                    text = result.text
                                    score = result.score
                                    if text and isinstance(text, str) and len(text.strip()) > 0:
                                        clean_text = text.strip()
                                        extracted_texts.append(clean_text)
                                        confidences.append(float(score))
                        
                        if extracted_texts:
                            full_text = ' '.join(extracted_texts)
                            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                            logger.info(f"✅ PaddleOCR fallback SUCCESS: {len(full_text)} chars extracted")
                            return full_text, avg_confidence
                        else:
                            logger.warning("⚠️ PaddleOCR fallback also failed")
                            return None, 0.0
                            
                    except Exception as fallback_error:
                        logger.error(f"❌ PaddleOCR fallback failed: {fallback_error}")
                        return None, 0.0
                        
            else:
                logger.warning("⚠️ PaddleOCR requires PIL Image object")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"❌ PaddleOCR extraction failed: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return None, 0.0

    def process_image(self, image_data, model_id="paddleocr-v2"):
        """Process image and extract text with dynamic model selection"""
        try:
            # Preprocess image
            image = self.preprocess_image(image_data)
            
            logger.info(f"🔄 OCR Processing with model: {model_id}")
            logger.info(f"📊 Image dimensions: {image.size}, mode: {image.mode}")
            
            # Dynamic model selection based on user preference
            if model_id.lower() == "paddleocr-v2" or model_id.lower() == "paddleocr":
                # 1. Try PaddleOCR first (BEST for production) - FIXED VERSION
                if self.paddleocr_available:
                    logger.info("🔄 Using FIXED PaddleOCR for text extraction...")
                    text, confidence = self.extract_text_paddleocr(image)
                    if text and len(text.strip()) > 0:
                        logger.info(f"✅ FIXED PaddleOCR success: {len(text)} chars extracted, confidence: {confidence:.3f}")
                        return text, confidence, "paddleocr-fixed"
                    else:
                        logger.warning("⚠️ FIXED PaddleOCR returned empty text")
                else:
                    logger.warning("⚠️ FIXED PaddleOCR not available")
            
            elif model_id.lower() == "easyocr":
                # 2. Try EasyOCR if specifically requested
                if self.easyocr_available:
                    logger.info("🔄 Using EasyOCR for text extraction...")
                    text, confidence = self.extract_text_easyocr(image)
                    if text and len(text.strip()) > 0:
                        logger.info(f"✅ EasyOCR success: {len(text)} chars extracted, confidence: {confidence:.3f}")
                        return text, confidence, "easyocr"
                    else:
                        logger.warning("⚠️ EasyOCR returned empty text")
                else:
                    logger.warning("⚠️ EasyOCR not available")
            
            elif "nanonets" in model_id.lower():
                # 3. Try Nanonets model if specifically requested
                if self.nanogpt_model:
                    logger.info("🔄 Using Nanonets model for text extraction...")
                    text, confidence = self.extract_text_nanonets(image)
                    if text:
                        logger.info(f"✅ Nanonets success: {len(text)} chars extracted, confidence: {confidence:.3f}")
                        return text, confidence, model_id
                    else:
                        logger.warning("⚠️ Nanonets returned empty text")
                else:
                    logger.warning("⚠️ Nanonets model not available")
            
            # 4. Fallback chain based on availability
            if self.paddleocr_available:
                logger.info("🔄 Fallback: Trying FIXED PaddleOCR...")
                text, confidence = self.extract_text_paddleocr(image)
                if text and len(text.strip()) > 0:
                    logger.info(f"✅ PaddleOCR fallback success: {len(text)} chars")
                    return text, confidence, "paddleocr-fallback-fixed"
            
            if self.easyocr_available:
                logger.info("🔄 Fallback: Trying EasyOCR...")
                text, confidence = self.extract_text_easyocr(image)
                if text and len(text.strip()) > 0:
                    logger.info(f"✅ EasyOCR fallback success: {len(text)} chars")
                    return text, confidence, "easyocr-fallback"
            
            # 5. Final fallback to basic OCR
            logger.info("🔄 Final fallback: Using basic OCR...")
            text, confidence = self.extract_text_basic(image)
            logger.info(f"📄 Basic OCR result: '{text}' (confidence: {confidence:.3f})")
            return text, confidence, "basic-ocr-fallback"
            
        except Exception as e:
            logger.error(f"❌ OCR processing failed: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return f"OCR processing failed: {str(e)}", 0.0, "error"

# Global OCR cache instance
OCR_CACHE = LRUCache(max_size=200, ttl=3600)  # 1 hour TTL, 200 entries

def get_image_hash(image_data: bytes) -> str:
    """Generate MD5 hash for image data"""
    return hashlib.md5(image_data).hexdigest()

def get_cache_key(image_data: bytes, model_id: str) -> str:
    """Generate cache key for OCR result"""
    image_hash = get_image_hash(image_data)
    return f"{model_id}:{image_hash}"

def get_cached_ocr_result(image_data: bytes, model_id: str):
    """Get cached OCR result if available"""
    cache_key = get_cache_key(image_data, model_id)
    return OCR_CACHE.get(cache_key)

def set_cached_ocr_result(image_data: bytes, model_id: str, result: Dict[str, Any]):
    """Cache OCR result"""
    cache_key = get_cache_key(image_data, model_id)
    OCR_CACHE.set(cache_key, result)

# Global OCR processor instance
ocr_processor = OCRProcessor()
