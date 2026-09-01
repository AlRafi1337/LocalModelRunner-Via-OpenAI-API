#!/usr/bin/env python3
"""
Configuration and Global Variables - Multi-Platform GPU Detection
Features: Enhanced GPU Detection, Path Configuration, Threading Settings
Updated: Fixed Vulkan detection, improved ROCm detection, better platform prioritization
FIXED: GPU platform detection KeyError: 'info'
"""

# Import dotenv for environment variable loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not available, install with: pip install python-dotenv")

import os
import sys
import time
import logging
import subprocess
import platform
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Import global instances from other modules
try:
    from speech_manager import SPEECH_TO_TEXT_MANAGER
    from ocr_manager import ocr_processor as OCR_MANAGER
except ImportError:
    SPEECH_TO_TEXT_MANAGER = None
    OCR_MANAGER = None
    print("⚠️ Some modules not available, will be initialized later")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flags to prevent infinite loading attempts
_WHISPER_LOAD_ATTEMPTED = False
_WHISPER_LOAD_ATTEMPTS = 0

# ============================================================================
# Configuration
# ============================================================================

# Main models directory (AI model files - gguf, pth, etc.)
MODELS_DIRECTORY = os.getenv('MODELS_DIRECTORY', './models')
AUTO_UNLOAD_TIMEOUT = 300  # Seconds (5 minutes)

# Thread pool configuration
LLM_MAX_WORKERS = int(os.getenv('LLM_MAX_WORKERS', '2'))  # Max concurrent LLM requests (adjust based on VRAM)
EMBEDDING_MAX_WORKERS = int(os.getenv('EMBEDDING_MAX_WORKERS', '4'))  # Max concurrent embedding requests

# Embedding dimension constant
EMBEDDING_DIMENSION = 768

# Thread pools
LLM_EXECUTOR = None
EMBEDDING_EXECUTOR = None

# Server Configuration
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '11435'))

# Global Model Manager instance (initialized in lifespan)
MODEL_MANAGER = None

# Enhanced GPU Platform Detection Results
GPU_PLATFORM_INFO = {
    "platform": "cpu",
    "cuda": False,
    "roc": False,
    "vulkan": False,
    "metal": False,
    "oneapi": False,
    "primary_gpu": None,
    "all_gpus": [],
    "backend_info": {},
    "supported_backends": [],
    "detection_confidence": "low",
    "performance_score": 0,
    "recommendations": []
}

# ============================================================================
# Enhanced GPU Platform Detection & Auto-Configuration
# ============================================================================

def get_amd_gpu_info() -> Tuple[str, str, int]:
    """Get detailed AMD GPU information from lspci or alternative sources"""
    try:
        # Try lspci first (most reliable)
        result = subprocess.run(['lspci', '-v'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            in_vga_section = False
            gpu_name = None
            vram_mb = 0
            
            for line in lines:
                line = line.strip()
                if 'VGA' in line or 'Display' in line:
                    in_vga_section = True
                    if 'ATI' in line or 'AMD' in line or 'Radeon' in line:
                        gpu_name = line.split(':')[-1].strip()
                elif in_vga_section and 'Memory' in line and 'prefetchable' in line:
                    # Extract VRAM from memory size
                    if 'MB' in line:
                        try:
                            vram_str = line.split('MB')[0].strip().split()[-1]
                            vram_mb = int(vram_str) if vram_str.isdigit() else 0
                        except:
                            pass
                elif line == '' and in_vga_section:
                    break
                    
            if gpu_name:
                return gpu_name, "AMD", vram_mb
    except:
        pass
    
    # Fallback: Check /proc/cpuinfo for AMD
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'AuthenticAMD' in cpuinfo:
                return "AMD CPU/GPU System", "AMD", 0
    except:
        pass
    
    return None, None, 0

def detect_cuda() -> Dict[str, Any]:
    """Enhanced CUDA detection with multiple methods"""
    cuda_info = {
        "available": False,
        "device_count": 0,
        "device_name": None,
        "cuda_version": None,
        "driver_version": None,
        "memory_mb": 0,
        "compute_capability": None
    }
    
    # Method 1: Try PyTorch CUDA detection
    try:
        import torch
        if torch.cuda.is_available():
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            cuda_info["compute_capability"] = f"{torch.cuda.get_device_capability(0)}"
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["driver_version"] = torch.version.cuda
            cuda_info["available"] = True
            logger.info(f"🔥 NVIDIA CUDA GPU detected via PyTorch: {cuda_info['device_name']} ({cuda_info['device_count']} device(s))")
    except Exception as e:
        logger.info(f"⚠️ PyTorch CUDA detection failed: {e}")
    
    # Method 2: Try nvidia-smi
    if not cuda_info["available"]:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    if len(parts) >= 2:
                        cuda_info["device_name"] = parts[0]
                        cuda_info["memory_mb"] = int(parts[1])
                        cuda_info["driver_version"] = parts[2] if len(parts) > 2 else "Unknown"
                        cuda_info["available"] = True
                        cuda_info["device_count"] = len(lines)
                        logger.info(f"🔥 NVIDIA CUDA GPU detected via nvidia-smi: {cuda_info['device_name']}")
        except:
            pass
    
    return cuda_info

def detect_rocm() -> Dict[str, Any]:
    """Enhanced ROCm detection with multiple methods"""
    rocm_info = {
        "available": False,
        "rocm_version": None,
        "device_name": None,
        "device_count": 0,
        "memory_mb": 0
    }
    
    # Method 1: Check for ROCm installation
    rocm_paths = ['/opt/rocm', '/usr/local/rocm']
    rocm_found = any(os.path.exists(path) for path in rocm_paths)
    
    if rocm_found:
        # Method 2: Try rocminfo
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                rocm_info["available"] = True
                rocm_info["rocm_version"] = "Unknown"  # Would need to parse version from rocminfo
                
                # Count GPU devices
                gpu_count = result.stdout.count('gfx')
                rocm_info["device_count"] = gpu_count
                
                if gpu_count > 0:
                    rocm_info["device_name"] = f"AMD GPU (ROCm)"
                    logger.info(f"🔥 ROCm detected via rocminfo: {rocm_info['device_name']} ({gpu_count} device(s))")
        except:
            pass
    
    # Method 2: Check if running on AMD system and PyTorch ROCm
    if not rocm_info["available"]:
        try:
            import torch
            if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
                rocm_info["device_name"] = f"AMD GPU (PyTorch ROCm {torch.version.hip})"
                rocm_info["rocm_version"] = str(torch.version.hip)
                rocm_info["available"] = True
                rocm_info["device_count"] = 1
                logger.info(f"🔥 PyTorch ROCm detected: {rocm_info['device_name']}")
        except:
            pass
    
    return rocm_info

def detect_xpu() -> Dict[str, Any]:
    """Detect Intel XPU (oneAPI) support"""
    xpu_info = {
        "available": False,
        "device_name": None,
        "device_count": 0
    }
    
    try:
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            xpu_info["available"] = True
            xpu_info["device_name"] = "Intel XPU"
            xpu_info["device_count"] = torch.xpu.device_count()
            logger.info("🔥 Intel XPU detected via PyTorch")
    except:
        pass
    
    return xpu_info

def detect_windows_amd() -> Dict[str, Any]:
    """Detect Windows AMD GPU support"""
    windows_amd_info = {
        "available": False,
        "device_name": None,
        "vulkan_available": False
    }
    
    if platform.system() == "Windows":
        # Check for AMD GPU in Windows
        try:
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'AMD' in result.stdout:
                windows_amd_info["available"] = True
                lines = [line.strip() for line in result.stdout.split('\n') if 'AMD' in line]
                if lines:
                    windows_amd_info["device_name"] = lines[0]
                logger.info(f"🔥 Windows AMD GPU detected: {windows_amd_info['device_name']}")
        except:
            pass
    
    return windows_amd_info

def detect_vulkan() -> Dict[str, Any]:
    """Basic Vulkan detection"""
    vulkan_info = {
        "available": False,
        "device_name": None
    }
    
    try:
        # Check for Vulkan loader
        result = subprocess.run(['vulkaninfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'deviceName' in result.stdout:
            vulkan_info["available"] = True
            # Extract first device name
            for line in result.stdout.split('\n'):
                if 'deviceName' in line and ':' in line:
                    vulkan_info["device_name"] = line.split(':')[1].strip()
                    break
            logger.info(f"🔥 Vulkan detected: {vulkan_info['device_name']}")
    except:
        pass
    
    return vulkan_info

def get_platform_performance_score(platform_name: str, platform_info: Dict[str, Any]) -> int:
    """Calculate performance score for each platform"""
    base_scores = {
        'cuda': 95,
        'roc': 90,
        'xpu': 95,
        'vulkan': 75,
        'metal': 70,
        'windows_amd': 60,
        'cpu': 10
    }
    
    base_score = base_scores.get(platform_name, 50)
    
    # Adjust score based on available info
    if platform_info.get("available", False):
        # Higher score for more info
        if platform_info.get("device_name"):
            base_score += 5
        if platform_info.get("device_count", 0) > 1:
            base_score += 3
        if platform_info.get("memory_mb", 0) > 8000:
            base_score += 2
    
    return min(base_score, 100)

def generate_recommendations(gpu_info: Dict[str, Any]) -> List[str]:
    """Generate platform-specific recommendations"""
    recommendations = []
    
    platform = gpu_info.get("platform", "cpu")
    device_name = gpu_info.get("device_name", "")
    
    if platform == "cuda":
        recommendations.append("💡 NVIDIA GPU detected - optimal for PyTorch training")
        if "RTX" in device_name:
            recommendations.append("🎮 RTX GPU detected - excellent for AI workloads")
    elif platform == "roc":
        recommendations.append("💡 ROCm GPU detected - good for PyTorch training on AMD")
        recommendations.append("💡 Ensure ROCm is properly installed for optimal performance")
    elif platform == "xpu":
        recommendations.append("💡 Intel XPU detected - consider Intel Extension for PyTorch")
    elif platform == "cpu":
        recommendations.append("💡 CPU-only mode - consider installing a GPU for better performance")
        recommendations.append("💡 For NVIDIA: pip install torch torchvision torchaudio")
        recommendations.append("💡 For AMD (Linux): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.3")
        recommendations.append("💡 Consider Intel Extension for PyTorch")
    
    return recommendations

def detect_all_gpu_platforms() -> Dict[str, Any]:
    """Enhanced multi-platform GPU detection with automatic configuration"""
    
    logger.info("\n🔍 ENHANCED MULTI-PLATFORM GPU DETECTION")
    logger.info("=" * 60)
    
    # Detect all available platforms
    platforms = {
        "cuda": detect_cuda(),
        "roc": detect_rocm(), 
        "xpu": detect_xpu(),
        "vulkan": detect_vulkan(),
        "windows_amd": detect_windows_amd()
    }
    
    # Log detection results
    if any(p.get("available", False) for p in platforms.values()):
        logger.info("✅ Available GPU Platforms (sorted by performance):")
        for i, (name, info) in enumerate(platforms.items(), 1):
            if info.get("available", False):
                device_name = info.get("device_name", "Unknown")
                score = get_platform_performance_score(name, info)
                logger.info(f"   {i}. {name.upper()}: {device_name} (Score: {score})")
    else:
        logger.warning("⚠️ No GPU platforms detected, will use CPU only")
    
    # Determine primary platform
    available_platforms = []
    for name, info in platforms.items():
        if info.get("available", False):
            score = get_platform_performance_score(name, info)
            available_platforms.append({
                "name": name,
                "info": info,
                "score": score
            })
    
    # Sort by performance score
    available_platforms.sort(key=lambda x: x["score"], reverse=True)
    
    if not available_platforms:
        # CPU only
        GPU_PLATFORM_INFO.update({
            "platform": "cpu",
            "cuda": False, "roc": False, "vulkan": False, "metal": False, "oneapi": False,
            "primary_gpu": None,
            "all_gpus": [],
            "backend_info": {},
            "supported_backends": ["cpu"],
            "detection_confidence": "high",
            "performance_score": 10,
            "recommendations": generate_recommendations({"platform": "cpu"})
        })
    else:
        # Select best platform
        best_platform = available_platforms[0]
        
        # Build final GPU info - FIXED: Ensure proper structure
        primary_gpu_info = best_platform["info"]
        if not isinstance(primary_gpu_info, dict):
            primary_gpu_info = {"device_name": "Unknown GPU"}
        
        GPU_PLATFORM_INFO.update({
            "platform": best_platform["name"],
            "cuda": platforms["cuda"].get("available", False),
            "roc": platforms["roc"].get("available", False),
            "vulkan": platforms["vulkan"].get("available", False),
            "metal": False,  # macOS Metal (not implemented)
            "oneapi": platforms["xpu"].get("available", False),
            "primary_gpu": primary_gpu_info,
            "all_gpus": [p["info"] for p in available_platforms if isinstance(p["info"], dict)],
            "backend_info": {name: p for name, p in platforms.items() if p.get("available", False) and isinstance(p, dict)},
            "supported_backends": [p.get("name", "unknown") for p in available_platforms],
            "detection_confidence": primary_gpu_info.get("confidence", "medium"),
            "performance_score": best_platform["score"],
            "recommendations": generate_recommendations({
                "platform": best_platform["name"],
                **primary_gpu_info
            })
        })
    
    # Print final selection
    logger.info(f"\n🎯 PRIMARY GPU PLATFORM: {GPU_PLATFORM_INFO['platform'].upper()}")
    logger.info(f"📊 PERFORMANCE SCORE: {GPU_PLATFORM_INFO['performance_score']}/100")
    logger.info(f"🎯 CONFIDENCE LEVEL: {GPU_PLATFORM_INFO['detection_confidence']}")
    
    if GPU_PLATFORM_INFO['primary_gpu']:
        device_name = GPU_PLATFORM_INFO['primary_gpu'].get('device_name', 'Unknown')
        logger.info(f"🖥️  PRIMARY DEVICE: {device_name}")
    
    logger.info("=" * 60)
    
    return GPU_PLATFORM_INFO

# Initialize GPU detection on module load
try:
    detect_all_gpu_platforms()
except Exception as e:
    logger.warning(f"GPU detection failed during initialization: {e}")
    # Keep default CPU-only configuration

# Additional imports after configuration
try:
    from api_models import *
except ImportError as e:
    logger.warning(f"Could not import API models: {e}")

# Export for external use
__all__ = [
    'logger', 'SERVER_HOST', 'SERVER_PORT', 'MODELS_DIRECTORY',
    'GPU_PLATFORM_INFO', 'MODEL_MANAGER', 'LLM_EXECUTOR', 'EMBEDDING_EXECUTOR',
    'detect_all_gpu_platforms', 'get_platform_performance_score'
]