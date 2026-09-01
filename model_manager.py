#!/usr/bin/env python3
"""
Model Manager
Features: Dynamic Model Loading, GPU Strategies, Thread-Safe Operations
"""

import os
import sys
import re
import logging
import datetime
import threading
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

# Import llama-cpp for Nanonets model
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Import config for MODEL_MANAGER access
import config

def get_model_manager():
    """Get the current MODEL_MANAGER instance from config module"""
    return getattr(config, 'MODEL_MANAGER', None)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Import config globals
from config import (
    MODELS_DIRECTORY, AUTO_UNLOAD_TIMEOUT, LLM_MAX_WORKERS, EMBEDDING_MAX_WORKERS,
    EMBEDDING_DIMENSION, GPU_PLATFORM_INFO
)

# Thread pools
LLM_EXECUTOR = None
EMBEDDING_EXECUTOR = None

# Global Model Manager instance (initialized in lifespan)
MODEL_MANAGER = None

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
    if quant_match:
        quant = quant_match.group(1)
        # Find the part before quantization
        base_name = name[:quant_match.start()].rstrip('-_')
        return f"{base_name}:{quant}"
    else:
        return name

def parse_model_metadata(filename: str) -> Dict[str, Any]:
    """Parse model metadata from filename"""
    import re
    
    name = os.path.splitext(filename)[0]
    
    # Extract family (e.g., llama, mistral, qwen, etc.)
    family_match = re.search(r'(llama|llama2|llama3|mistral|mixtral|qwen|gemma|phi|neural|command|toolformer)', name, re.IGNORECASE)
    family = family_match.group(1) if family_match else "unknown"
    
    # Extract parameter size (e.g., 7b, 13b, 70b, etc.)
    param_match = re.search(r'(\d+)[bB]', name)
    if param_match:
        param_size = int(param_match.group(1))
        if param_size >= 1000:
            parameters = f"{param_size//1000}B"
        else:
            parameters = f"{param_size}B"
    else:
        parameters = "unknown"
    
    # Extract quantization level
    quant_match = re.search(r'(Q\d+_[KM](?:_[A-Z]+)?|Q\d+_\d+)', name, re.IGNORECASE)
    quantization = quant_match.group(1) if quant_match else "unknown"
    
    return {
        'family': family.lower(),
        'parameter_size': parameters,
        'quantization_level': quantization
    }

def scan_models_directory(directory_path: str, include_detailed_info: bool = False) -> List[Dict[str, Any]]:
    """Scan directory for available models"""
    try:
        if not os.path.exists(directory_path):
            logger.error(f"Models directory not found: {directory_path}")
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
        
    except Exception as e:
        logger.error(f"Error scanning models directory: {e}")
        return []

class ModelManager:
    """Manages LLM and embedding models separately without caching"""
    
    def __init__(self, models_directory: str):
        self.models_directory = models_directory
        
        # Track LLM and embedding models separately (no cache)
        self.current_llm_model_id = None
        self.current_llm_model = None
        self.current_embedding_model_id = None
        self.current_embedding_model = None
        self.current_model_lock = Lock()  # Single lock for both model types
        
        logger.info(f"🎯 ModelManager initialized: Dual model tracking (LLM + Embeddings)")
    
    def get_model_file_path(self, model_id: str) -> Optional[str]:
        """Find model file path by model_id (flexible matching)"""
        if not os.path.exists(self.models_directory):
            logger.error(f"Models directory not found: {self.models_directory}")
            return None
        
        # Normalize user input
        normalized_input = model_id.lower().strip()
        
        # Generate variants for flexible matching
        input_variants = [normalized_input]
        
        # Variant 1: Replace last dash before Q with colon (e.g., "model-q8" -> "model:q8")
        if '-q' in normalized_input and ':' not in normalized_input:
            parts = normalized_input.rsplit('-q', 1)
            if len(parts) == 2:
                input_variants.append(f"{parts[0]}:q{parts[1]}")
        
        # Variant 2: Replace colon with dash (e.g., "model:q8" -> "model-q8")
        if ':' in normalized_input:
            input_variants.append(normalized_input.replace(':', '-'))
        
        # Variant 3: Case-insensitive underscore matching (Q8_K vs q8-k)
        for variant in list(input_variants):
            # Try with underscores converted to hyphens
            if '_' in variant or '-' in variant:
                input_variants.append(variant.replace('-', '_'))
                input_variants.append(variant.replace('_', '-'))
        
        # Remove duplicates while preserving order
        input_variants = list(dict.fromkeys(input_variants))
        
        logger.info(f"🔍 Searching for model '{model_id}' using variants: {input_variants[:3]}...")
        
        # Scan directory for matching model
        for filename in os.listdir(self.models_directory):
            if not filename.lower().endswith('.gguf'):
                continue
            
            file_model_id = filename_to_model_id(filename)
            
            # Try exact match first
            if file_model_id in input_variants:
                logger.info(f"✅ Exact match: '{model_id}' -> '{filename}'")
                return os.path.join(self.models_directory, filename)
            
            # Try fuzzy match (case-insensitive, ignore separators)
            file_normalized = file_model_id.lower().replace('_', '').replace('-', '').replace(':', '')
            for variant in input_variants:
                variant_normalized = variant.lower().replace('_', '').replace('-', '').replace(':', '')
                if file_normalized == variant_normalized:
                    logger.info(f"✅ Fuzzy match: '{model_id}' -> '{filename}'")
                    return os.path.join(self.models_directory, filename)
        
        logger.error(f"❌ Model file not found for: '{model_id}' (tried {len(input_variants)} variants)")
        logger.info(f"Available models: {[filename_to_model_id(f) for f in os.listdir(self.models_directory) if f.endswith('.gguf')][:5]}")
        return None
    
    def _load_model_internal(self, model_path: str, is_embedding: bool = False) -> Tuple[Any, Tuple, Dict]:
        """Internal method to load a model from file"""
        from llama_cpp import Llama
        
        filename = os.path.basename(model_path)
        model_name = extract_model_name_from_path(model_path)
        metadata = parse_model_metadata(filename)
        
        # Check if it's an embedding model
        is_embedding = is_embedding or any(pattern in model_name.lower() for pattern in 
                                          ['bge', 'embedding', 'embed', 'gte', 'e5'])
        
        model_type = "Embedding" if is_embedding else "LLM"
        logger.info(f"🚀 Loading {model_type} model: {model_name}")
        
        # Load with multi-platform fallback strategies
        llm, strategy = load_model_with_fallback(model_path, is_embedding)
        
        # Build info dict
        model_info = {
            'model_name': model_name,
            'size': os.path.getsize(model_path),
            'format': 'gguf',
            'family': metadata['family'],
            'families': [metadata['family']],
            'parameter_size': metadata['parameter_size'],
            'quantization_level': metadata['quantization_level'],
            'digest': f'sha256:{hash(str(model_path) + str(os.path.getsize(model_path))):064x}'[:64],
            'is_embedding': is_embedding,
            'gpu_platform': GPU_PLATFORM_INFO['platform']
        }
        
        if is_embedding:
            # Test embedding dimension
            test_emb = llm.embed("test")
            embedding_dim = len(test_emb) if test_emb else EMBEDDING_DIMENSION
            model_info['embedding_dimension'] = embedding_dim
        
        logger.info(f"✅ {model_type} loaded successfully: {model_name} [{strategy[0]}]")
        return llm, strategy, model_info

    def get_or_load_model(self, model_id: str, is_embedding: bool = False) -> Dict[str, Any]:
        """Load model directly (no caching) - tracks LLM and embedding models separately"""
        model_id = model_id.lower().strip()
        model_type = "embedding" if is_embedding else "LLM"
        
        with self.current_model_lock:
            # Check if the requested model is already the current model of the correct type
            if is_embedding:
                if self.current_embedding_model_id == model_id:
                    logger.info(f"♻️ Using current {model_type} model: {model_id}")
                    return {
                        "llm": self.current_embedding_model['llm'],
                        "info": self.current_embedding_model['info'],
                        "strategy": self.current_embedding_model['strategy'],
                        "lock": self.current_model_lock
                    }
                
                # Only unload current embedding model if different
                if self.current_embedding_model is not None:
                    logger.info(f"🗑️ Unloading current {model_type} model: {self.current_embedding_model_id}")
                    self.current_embedding_model = None
                    self.current_embedding_model_id = None
            else:
                if self.current_llm_model_id == model_id:
                    logger.info(f"♻️ Using current {model_type} model: {model_id}")
                    return {
                        "llm": self.current_llm_model['llm'],
                        "info": self.current_llm_model['info'],
                        "strategy": self.current_llm_model['strategy'],
                        "lock": self.current_model_lock
                    }
                
                # Only unload current LLM model if different
                if self.current_llm_model is not None:
                    logger.info(f"🗑️ Unloading current {model_type} model: {self.current_llm_model_id}")
                    self.current_llm_model = None
                    self.current_llm_model_id = None
            
            # Load the new model
            logger.info(f"📥 Loading {model_type} model: {model_id}")
            
            # Find model file
            model_path = self.get_model_file_path(model_id)
            if not model_path:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_id}' not found in models directory"
                )
            
            logger.info(f"🔄 Loading {model_type} model {model_id} from: {os.path.basename(model_path)}")
            
            try:
                llm, strategy, info = self._load_model_internal(model_path, is_embedding)
            except Exception as e:
                logger.error(f"❌ Failed to load {model_type} model {model_id}: {e}")
                if is_embedding:
                    self.current_embedding_model_id = None
                    self.current_embedding_model = None
                else:
                    self.current_llm_model_id = None
                    self.current_llm_model = None
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load {model_type} model '{model_id}': {str(e)}"
                )
            
            # Store as current model of the appropriate type
            if is_embedding:
                self.current_embedding_model_id = model_id
                self.current_embedding_model = {
                    "llm": llm,
                    "info": info,
                    "strategy": strategy,
                    "lock": self.current_model_lock
                }
            else:
                self.current_llm_model_id = model_id
                self.current_llm_model = {
                    "llm": llm,
                    "info": info,
                    "strategy": strategy,
                    "lock": self.current_model_lock
                }
            
            logger.info(f"✅ {model_type} model {model_id} loaded successfully")
            
            # Return the appropriate model based on type
            if is_embedding:
                return self.current_embedding_model
            else:
                return self.current_llm_model
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """Get currently loaded LLM and embedding models"""
        with self.current_model_lock:
            loaded_models = []
            
            # Add LLM model if loaded
            if self.current_llm_model is not None:
                entry = self.current_llm_model
                loaded_models.append({
                    "model_id": self.current_llm_model_id,
                    "name": entry['info']['model_name'],
                    "family": entry['info']['family'],
                    "parameters": entry['info']['parameter_size'],
                    "quantization": entry['info']['quantization_level'],
                    "strategy": entry['strategy'][0],
                    "model_type": "LLM",
                    "loaded_at": datetime.datetime.now().isoformat(),
                    "last_used": datetime.datetime.now().isoformat(),
                    "idle_time": 0.0,
                    "gpu_platform": entry['info'].get('gpu_platform', 'cpu')
                })
            
            # Add embedding model if loaded
            if self.current_embedding_model is not None:
                entry = self.current_embedding_model
                loaded_models.append({
                    "model_id": self.current_embedding_model_id,
                    "name": entry['info']['model_name'],
                    "family": entry['info']['family'],
                    "parameters": entry['info']['parameter_size'],
                    "quantization": entry['info']['quantization_level'],
                    "strategy": entry['strategy'][0],
                    "model_type": "embedding",
                    "loaded_at": datetime.datetime.now().isoformat(),
                    "last_used": datetime.datetime.now().isoformat(),
                    "idle_time": 0.0,
                    "gpu_platform": entry['info'].get('gpu_platform', 'cpu')
                })
            
            return loaded_models
    
    def unload_model(self, model_id: str) -> bool:
        """Manually unload LLM or embedding model"""
        model_id = model_id.lower().strip()
        
        with self.current_model_lock:
            # Check if it's the current LLM model
            if self.current_llm_model_id == model_id:
                logger.info(f"🗑️ Manually unloading LLM model: {model_id}")
                self.current_llm_model = None
                self.current_llm_model_id = None
                logger.info(f"✅ LLM model {model_id} unloaded")
                return True
            
            # Check if it's the current embedding model
            if self.current_embedding_model_id == model_id:
                logger.info(f"🗑️ Manually unloading embedding model: {model_id}")
                self.current_embedding_model = None
                self.current_embedding_model_id = None
                logger.info(f"✅ Embedding model {model_id} unloaded")
                return True
            
            logger.warning(f"Cannot unload {model_id}: not currently loaded")
            return False

def load_model_with_fallback(model_path: str, is_embedding: bool = False):
    """Load model with multi-platform GPU fallback strategies"""
    from llama_cpp import Llama
    
    # Get detected GPU platform
    gpu_platform = GPU_PLATFORM_INFO['platform']
    gpu_available = gpu_platform != "cpu"
    
    # Configure platform-specific GPU strategies
    strategies = []
    
    if gpu_available:
        if gpu_platform == "cuda":
            # NVIDIA CUDA strategies
            if is_embedding:
                strategies.extend([
                    ("CUDA Full GPU", {'n_gpu_layers': -1, 'n_batch': 512, 'n_threads': 4}),
                    ("CUDA CPU Fallback", {'n_gpu_layers': 0, 'n_batch': 128, 'n_threads': 8}),
                ])
            else:
                strategies.extend([
                    ("CUDA Full GPU", {'n_gpu_layers': -1, 'n_batch': 512, 'n_threads': 4}),
                    ("CUDA High GPU (80%)", {'n_gpu_layers': 32, 'n_batch': 256, 'n_threads': 6}),
                    ("CUDA Medium GPU (60%)", {'n_gpu_layers': 24, 'n_batch': 128, 'n_threads': 8}),
                    ("CUDA Low GPU (40%)", {'n_gpu_layers': 16, 'n_batch': 64, 'n_threads': 8}),
                ])
        
        elif gpu_platform == "roc":
            # AMD ROCm strategies
            if is_embedding:
                strategies.extend([
                    ("ROCm Full GPU", {'n_gpu_layers': -1, 'n_batch': 512, 'n_threads': 4}),
                    ("ROCm CPU Fallback", {'n_gpu_layers': 0, 'n_batch': 128, 'n_threads': 8}),
                ])
            else:
                strategies.extend([
                    ("ROCm Full GPU", {'n_gpu_layers': -1, 'n_batch': 512, 'n_threads': 4}),
                    ("ROCm High GPU (80%)", {'n_gpu_layers': 32, 'n_batch': 256, 'n_threads': 6}),
                    ("ROCm Medium GPU (60%)", {'n_gpu_layers': 24, 'n_batch': 128, 'n_threads': 8}),
                    ("ROCm Low GPU (40%)", {'n_gpu_layers': 16, 'n_batch': 64, 'n_threads': 8}),
                ])
        
        elif gpu_platform == "vulkan":
            # Intel Vulkan strategies
            if is_embedding:
                strategies.extend([
                    ("Vulkan GPU", {'n_gpu_layers': -1, 'n_batch': 256, 'n_threads': 4}),
                    ("Vulkan CPU Fallback", {'n_gpu_layers': 0, 'n_batch': 128, 'n_threads': 8}),
                ])
            else:
                strategies.extend([
                    ("Vulkan GPU", {'n_gpu_layers': -1, 'n_batch': 256, 'n_threads': 4}),
                    ("Vulkan High GPU", {'n_gpu_layers': 32, 'n_batch': 128, 'n_threads': 6}),
                    ("Vulkan Medium GPU", {'n_gpu_layers': 24, 'n_batch': 64, 'n_threads': 8}),
                ])
        
        elif gpu_platform == "metal":
            # Apple Metal strategies
            if is_embedding:
                strategies.extend([
                    ("Metal GPU", {'n_gpu_layers': -1, 'n_batch': 512, 'n_threads': 4}),
                    ("Metal CPU Fallback", {'n_gpu_layers': 0, 'n_batch': 128, 'n_threads': 8}),
                ])
            else:
                strategies.extend([
                    ("Metal GPU", {'n_gpu_layers': -1, 'n_batch': 512, 'n_threads': 4}),
                    ("Metal High GPU", {'n_gpu_layers': 32, 'n_batch': 256, 'n_threads': 6}),
                    ("Metal Medium GPU", {'n_gpu_layers': 24, 'n_batch': 128, 'n_threads': 8}),
                ])
        
        elif gpu_platform == "oneapi":
            # Intel oneAPI strategies
            if is_embedding:
                strategies.extend([
                    ("oneAPI GPU", {'n_gpu_layers': -1, 'n_batch': 256, 'n_threads': 4}),
                    ("oneAPI CPU Fallback", {'n_gpu_layers': 0, 'n_batch': 128, 'n_threads': 8}),
                ])
            else:
                strategies.extend([
                    ("oneAPI GPU", {'n_gpu_layers': -1, 'n_batch': 256, 'n_threads': 4}),
                    ("oneAPI High GPU", {'n_gpu_layers': 32, 'n_batch': 128, 'n_threads': 6}),
                    ("oneAPI Medium GPU", {'n_gpu_layers': 24, 'n_batch': 64, 'n_threads': 8}),
                ])
    
    # Always add CPU fallback
    strategies.append(("CPU Only", {'n_gpu_layers': 0, 'n_batch': 64, 'n_threads': 8}))
    
    llm = None
    successful_strategy = None
    
    for strategy_name, config in strategies:
        try:
            model_type = "Embedding" if is_embedding else "LLM"
            logger.info(f"🚀 Trying {model_type} {strategy_name} - GPU Layers: {config['n_gpu_layers']}")
            
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=config['n_gpu_layers'],
                n_ctx=2048 if is_embedding else 4096,
                n_batch=config['n_batch'],
                n_threads=config['n_threads'],
                embedding=is_embedding,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                seed=-1
            )
            
            # Test the model
            logger.info(f"🧪 Testing {model_type} model...")
            if is_embedding:
                test_embedding = llm.embed("Hello")
                if test_embedding and len(test_embedding) > 0:
                    logger.info(f"✅ Embedding dimension: {len(test_embedding)}")
                else:
                    raise Exception("Embedding test failed")
            else:
                test_response = llm("Hello", max_tokens=5, echo=False)
            
            logger.info(f"✅ {model_type} {strategy_name} successful!")
            successful_strategy = (strategy_name, config)
            break
            
        except Exception as e:
            logger.warning(f"❌ {strategy_name} failed: {e}")
            if llm:
                del llm
                llm = None
            continue
    
    if not llm:
        raise Exception(f"All {gpu_platform.upper()}/CPU strategies failed for {'embedding' if is_embedding else 'LLM'} model")
    
    return llm, successful_strategy

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
        import random
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