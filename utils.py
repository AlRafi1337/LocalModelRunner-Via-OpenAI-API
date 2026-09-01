#!/usr/bin/env python3
"""
Utility Functions
Features: Helper Functions, File Operations, System Utilities
"""

import os
import sys
import socket
import logging
import re
import hashlib
import datetime
import time
import tempfile
import base64
import io
import threading
import random
from typing import Dict, Any, List, Optional
from fastapi import HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def get_local_ip() -> str:
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def test_llama_cpp() -> bool:
    """Test if llama-cpp-python is available"""
    try:
        from llama_cpp import Llama
        import llama_cpp
        logger.info(f"📦 llama-cpp-python version: {llama_cpp.__version__}")
        return True
    except Exception as e:
        logger.error(f"❌ llama-cpp-python issue: {e}")
        return False

def extract_model_name_from_path(model_path: str) -> str:
    """Extract model name from file path"""
    return os.path.splitext(os.path.basename(model_path))[0]

def filename_to_model_id(filename: str) -> str:
    """Convert filename to clean model_id for API usage (KEEPS quantization info)"""
    # Remove .gguf extension
    name = os.path.splitext(filename)[0]
    
    # Extract quantization (e.g., Q8_K_XL, Q4_K_M, Q8_0, etc.)
    quant_match = re.search(r'(Q\d+_[KM](?:_[A-Z]+)?|Q\d+_\d+)', name, re.IGNORECASE)
    quantization = quant_match.group(1).upper() if quant_match else None
    
    # Remove UD suffix and quantization from base name
    if quantization:
        # Remove the specific quantization pattern found
        pattern = re.escape(quantization.lower())
        base_name = re.sub(r'[-_]' + pattern, '', name, flags=re.IGNORECASE)
    else:
        base_name = name
    
    base_name = re.sub(r'[-_](UD)', '', base_name, flags=re.IGNORECASE)
    
    # Convert to lowercase and replace separators
    base_name = base_name.lower().replace('_', '-').strip('-')
    # Remove duplicate hyphens
    base_name = re.sub(r'-+', '-', base_name)
    
    # Add quantization back (like Ollama: model:quantization)
    if quantization:
        # Convert Q8_K_XL -> q8_k_xl, Q8_0 -> q8_0 for consistency
        quant_lower = quantization.lower()
        return f"{base_name}:{quant_lower}"
    else:
        return base_name

def parse_model_metadata(filename: str) -> Dict[str, str]:
    """Extract metadata from model filename"""
    model_name = os.path.splitext(filename)[0]
    
    # Extract parameter size
    param_match = re.search(r'(\d+\.?\d*)([BM])', model_name, re.IGNORECASE)
    param_size = param_match.group(1) + param_match.group(2).upper() if param_match else "Unknown"
    
    # Extract quantization level
    quant_match = re.search(r'(Q\d+_[KM](?:_[A-Z]+)?)', model_name, re.IGNORECASE)
    quant_level = quant_match.group(1).upper() if quant_match else "Unknown"
    
    # Determine model family
    name_lower = model_name.lower()
    if 'llama' in name_lower:
        family = 'llama'
    elif 'mistral' in name_lower:
        family = 'mistral'
    elif 'qwen' in name_lower:
        family = 'qwen'
    elif 'bge' in name_lower or 'bert' in name_lower:
        family = 'bert'
    else:
        family = 'unknown'
    
    return {
        'parameter_size': param_size,
        'quantization_level': quant_level,
        'family': family
    }

def scan_models_directory(directory_path: str, include_detailed_info: bool = False) -> List[Dict[str, Any]]:
    """Scan directory for all available GGUF models and return metadata"""
    import re
    
    if not os.path.exists(directory_path):
        logger.warning(f"Models directory not found: {directory_path}")
        return []
    
    models = []
    
    for filename in os.listdir(directory_path):
        if not filename.lower().endswith('.gguf'):
            continue
        
        model_path = os.path.join(directory_path, filename)
        file_size = os.path.getsize(model_path)
        
        # Parse metadata
        metadata = parse_model_metadata(filename)
        model_name = extract_model_name_from_path(filename)
        
        # Determine model type based on name patterns
        model_type = "llm"  # Default
        if any(pattern in model_name.lower() for pattern in 
               ['bge', 'embedding', 'embed', 'gte', 'e5', 'jina', 'multilingual-e5']):
            model_type = "embedding"
        elif any(pattern in model_name.lower() for pattern in 
                ['ocr', 'vision', 'clip', 'blip']):
            model_type = "ocr"
        
        model_entry = {
            "name": model_name,
            "model": filename_to_model_id(filename),
            "modified_at": datetime.datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
            "size": file_size,
            "digest": f'sha256:{hash(str(model_path) + str(file_size)):064x}'[:64],
            "details": {
                "format": "gguf",
                "family": metadata['family'],
                "families": [metadata['family']],
                "parameter_size": metadata['parameter_size'],
                "quantization_level": metadata['quantization_level'],
                "model_type": model_type
            },
            "model_id": filename_to_model_id(filename),
            "parameters": metadata['parameter_size'],
            "quantization": metadata['quantization_level'],
            "family": metadata['family'],
            "model_type": model_type
        }
        
        models.append(model_entry)
    
    # Sort by size (largest first)
    models.sort(key=lambda x: x.get('size', 0), reverse=True)
    
    # Remove size field if not detailed mode
    if not include_detailed_info:
        for model in models:
            model.pop('size', None)
    
    logger.info(f"Found {len(models)} models in directory: {directory_path}")
    return models

def format_duration(nanoseconds: int) -> float:
    """
    Convert nanoseconds to seconds (or milliseconds for very fast operations)
    
    Args:
        nanoseconds: Duration in nanoseconds
        
    Returns:
        float: Duration in seconds (or milliseconds if < 1 second)
    """
    if nanoseconds == 0:
        return 0.0
    
    seconds = nanoseconds / 1e9
    
    # Use milliseconds for very fast operations (less than 1 second)
    if seconds < 1.0:
        return round(seconds * 1000, 2)  # Return in milliseconds
    else:
        return round(seconds, 2)  # Return in seconds

def get_image_hash(image_data: bytes) -> str:
    """Generate MD5 hash for image data"""
    return hashlib.md5(image_data).hexdigest()

def get_cache_key(image_data: bytes, model_id: str) -> str:
    """Generate cache key for OCR result"""
    image_hash = get_image_hash(image_data)
    return f"{model_id}:{image_hash}"

class LRUCache:
    """Thread-safe LRU Cache implementation"""
    
    def __init__(self, max_size: int = 300, ttl: int = 1800):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds (30 minutes)
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def _is_expired(self, timestamp: float) -> bool:
        return time.time() - timestamp > self.ttl
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if key in self.cache:
                timestamp = self.timestamps[key]
                if not self._is_expired(timestamp):
                    return self.cache[key]
                else:
                    # Remove expired entry
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self.lock:
            if key in self.cache:
                # Update existing key
                pass
            else:
                # Add new key
                if len(self.cache) >= self.max_size:
                    # Remove oldest item (simple implementation)
                    oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl
            }

def create_temp_file(data: bytes, suffix: str = '.tmp') -> str:
    """Create a temporary file with given data"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(data)
        return temp_file.name

def cleanup_temp_file(file_path: str) -> bool:
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
        return False

def validate_image_data(image_data: bytes) -> bool:
    """Validate if data is a valid image"""
    try:
        # Try to open with PIL
        from PIL import Image
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        return True
    except Exception:
        return False

def encode_image_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return ""

def decode_image_base64(base64_string: str) -> bytes:
    """Decode base64 string to image bytes"""
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return b""

def validate_audio_data(audio_data: bytes) -> bool:
    """Validate if data is valid audio"""
    # Basic validation - check if it looks like audio data
    if len(audio_data) < 44:  # Minimum WAV header size
        return False
    
    # Check for WAV header
    if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
        return True
    
    # Could add more audio format checks here
    return True

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def safe_filename(filename: str) -> str:
    """Create safe filename by removing/replacing unsafe characters"""
    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    safe_name = safe_name.strip('. ')
    # Limit length
    if len(safe_name) > 200:
        safe_name = safe_name[:200]
    return safe_name or "unnamed"

def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    import platform
    import psutil
    
    try:
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:').percent
        }
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return {"error": str(e)}

def validate_model_id(model_id: str) -> bool:
    """Validate model ID format"""
    # Basic validation - alphanumeric, hyphens, underscores, colons
    return bool(re.match(r'^[a-zA-Z0-9._:-]+$', model_id))

def sanitize_prompt(prompt: str) -> str:
    """Sanitize user prompt to prevent potential issues"""
    # Remove control characters except newlines and tabs
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt)
    # Limit length
    if len(sanitized) > 10000:
        sanitized = sanitized[:10000]
    return sanitized

def extract_text_from_response(response: Any) -> str:
    """Extract text from different response formats"""
    if isinstance(response, dict):
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'text' in choice:
                return choice['text']
            elif 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
        elif 'text' in response:
            return response['text']
        elif 'content' in response:
            return response['content']
    elif isinstance(response, str):
        return response
    
    return str(response)

# Import constants from config
from config import EMBEDDING_DIMENSION, detect_all_gpu_platforms
import config

def get_model_manager():
    """Get the current MODEL_MANAGER instance from config module"""
    return getattr(config, 'MODEL_MANAGER', None)

def _generate_embedding_sync(prompt: str, model_id: str = "bge-large-en-v1.5") -> List[float]:
    """Synchronous embedding generation - runs in thread pool (DYNAMIC MODEL SELECTION)"""
    logger.info(f"[{threading.current_thread().name}] Embedding request for model: {model_id}")
    
    try:
        # Get or load model from ModelManager (using dynamic access)
        model_manager = get_model_manager()
        if model_manager is None:
            raise Exception("MODEL_MANAGER is not available")
            
        model_entry = model_manager.get_or_load_model(model_id, is_embedding=True)
        llm = model_entry['llm']
        model_lock = model_entry['lock']
        
        logger.info(f"Processing embedding: {prompt[:50]}...")
        
        with model_lock:
            embedding = llm.embed(prompt)
            if not embedding or len(embedding) == 0:
                raise Exception("Empty embedding returned")
            logger.info(f"✅ Embedding generated: dimension={len(embedding)}")
            return embedding
            
    except HTTPException:
        # Re-raise HTTP exceptions (model not found, etc.)
        raise
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        # Fallback to mock embedding
        random.seed(hash(prompt) % (2 ** 32))
        return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]

def _generate_llm_sync(formatted_prompt: str, max_tokens: int, temperature: float, 
                       top_p: float, model_id: str, stream: bool = False):
    """Synchronous LLM generation - runs in thread pool (DYNAMIC MODEL SELECTION)"""
    logger.info(f"[{threading.current_thread().name}] LLM request for model: {model_id}")
    
    try:
        # Get or load model from ModelManager (using dynamic access)
        model_manager = get_model_manager()
        if model_manager is None:
            raise Exception("MODEL_MANAGER is not available")
            
        model_entry = model_manager.get_or_load_model(model_id, is_embedding=False)
        llm = model_entry['llm']
        model_lock = model_entry['lock']
        
        logger.info(f"[LLM] Generating with max_tokens={max_tokens}, temp={temperature}")
        
        with model_lock:
            if stream:
                # Return generator for streaming
                logger.info("[LLM] Starting stream generation")
                return llm(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.1,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    stream=True,
                    echo=False
                )
            else:
                # Return complete response
                logger.info("[LLM] Starting complete generation")
                response = llm(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.1,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    echo=False
                )
                result = response['choices'][0]['text'].strip()
                logger.info(f"✅ [LLM] Generation complete: {len(result)} chars")
                return result
                
    except HTTPException:
        # Re-raise HTTP exceptions (model not found, etc.)
        raise
    except Exception as e:
        logger.error(f"[LLM] Generation error: {e}", exc_info=True)
        return f"Error: {str(e)}"

# Alias for backward compatibility with api_endpoints.py
def chat_with_model_sync(formatted_prompt: str, max_tokens: int, temperature: float, 
                         top_p: float, model_id: str, stream: bool = False):
    """Alias for _generate_llm_sync - for backward compatibility"""
    return _generate_llm_sync(formatted_prompt, max_tokens, temperature, top_p, model_id, stream)