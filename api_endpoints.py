"""
API Endpoints for the Complete Multimodal Ollama-Compatible API Service - FIXED VERSION
FastAPI route handlers for all API endpoints
Fixed speech transcription compatibility issues
"""

import asyncio
import json
import time
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, status, Response, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from typing import Any
from contextlib import asynccontextmanager

# Import our modular components
from config import (
    MODELS_DIRECTORY, 
    GPU_PLATFORM_INFO, 
    LLM_MAX_WORKERS, 
    EMBEDDING_MAX_WORKERS,
    SPEECH_TO_TEXT_MANAGER,
    OCR_MANAGER,
    logger
)
from api_models import (
    ChatRequest, 
    GenerateRequest, 
    EmbeddingsRequest, 
    ShowRequest
)
from utils import (
    get_local_ip, 
    scan_models_directory, 
    format_duration,
    _generate_llm_sync,
    _generate_embedding_sync
)
from model_manager import ModelManager

# Global variables that will be set by the lifespan function
MODEL_MANAGER = None
LLM_EXECUTOR = None
EMBEDDING_EXECUTOR = None

# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup on startup/shutdown"""
    global LLM_EXECUTOR, EMBEDDING_EXECUTOR, MODEL_MANAGER
    
    # Startup
    logger.info("🚀 Starting Complete Multimodal Ollama-Compatible API Service")
    
    # Speech recognition engine already loaded by consolidated initialization function
    
    # Auto-detect GPU platforms
    logger.info("\n" + "="*60)
    logger.info("🔍 AUTO-DETECTING GPU PLATFORMS...")
    logger.info("="*60)
    
    # Detect and apply GPU platform configuration
    from config import detect_all_gpu_platforms
    GPU_PLATFORM_INFO.update(detect_all_gpu_platforms())
    
    # Initialize Model Manager
    logger.info(f"\n🎯 Initializing ModelManager...")
    logger.info(f"   • Models Directory: {MODELS_DIRECTORY}")
    logger.info(f"   • Loading Mode: Direct (no caching)")
    logger.info(f"   • GPU Platform: {GPU_PLATFORM_INFO['platform'].upper()}")
    
    MODEL_MANAGER = ModelManager(models_directory=MODELS_DIRECTORY)
    
    # Also set it in config for shared access
    try:
        import config
        config.MODEL_MANAGER = MODEL_MANAGER
        logger.info("✅ MODEL_MANAGER synchronized across modules")
    except Exception as e:
        logger.warning(f"Could not set MODEL_MANAGER in config: {e}")
    
    # Initialize thread pools
    logger.info(f"\n🔧 Initializing Thread Pools...")
    logger.info(f"   • LLM Workers: {LLM_MAX_WORKERS}")
    logger.info(f"   • Embedding Workers: {EMBEDDING_MAX_WORKERS}")
    
    LLM_EXECUTOR = ThreadPoolExecutor(
        max_workers=LLM_MAX_WORKERS,
        thread_name_prefix="llm_worker"
    )
    EMBEDDING_EXECUTOR = ThreadPoolExecutor(
        max_workers=EMBEDDING_MAX_WORKERS,
        thread_name_prefix="embedding_worker"
    )
    
    # Scan available models
    logger.info("\n" + "="*60)
    logger.info("📂 Scanning Available Models...")
    logger.info("="*60)
    
    available_models = scan_models_directory(MODELS_DIRECTORY, include_detailed_info=False)
    llm_models = [m for m in available_models if m['model_type'] == 'llm']
    embedding_models = [m for m in available_models if m['model_type'] == 'embedding']
    ocr_models = [m for m in available_models if m['model_type'] == 'ocr']
    
    logger.info(f"\n📊 Model Inventory:")
    logger.info(f"   • Total Models Found: {len(available_models)}")
    logger.info(f"   • LLM Models: {len(llm_models)}")
    logger.info(f"   • Embedding Models: {len(embedding_models)}")
    logger.info(f"   • OCR Models: {len(ocr_models)}")
    
    if llm_models:
        logger.info(f"\n🤖 Available LLM Models:")
        for model in llm_models[:5]:  # Show first 5
            logger.info(f"   • {model['model_id']} ({model['parameters']}, {model['quantization']})")
        if len(llm_models) > 5:
            logger.info(f"   ... and {len(llm_models) - 5} more")
    
    if embedding_models:
        logger.info(f"\n🔢 Available Embedding Models:")
        for model in embedding_models:
            logger.info(f"   • {model['model_id']} ({model['parameters']}, {model['quantization']})")
    
    if ocr_models:
        logger.info(f"\n👁️ Available OCR Models:")
        for model in ocr_models:
            logger.info(f"   • {model['model_id']} ({model['parameters']}, {model['quantization']})")
    
    local_ip = get_local_ip()
    port = 11435  # Match the port in main.py
    
    print(f"\n{'='*70}")
    print(f"🌐 Complete Multimodal Ollama-Compatible API Server Running!")
    print(f"{'='*70}")
    print(f"\n📍 Access Points:")
    print(f"   • Local:   http://127.0.0.1:{port}")
    print(f"   • Network: http://{local_ip}:{port}")
    print(f"   • Docs:    http://127.0.0.1:{port}/docs")
    
    print(f"\n🔥 Multi-Platform GPU Support: AUTO-DETECTED")
    print(f"   • Primary Platform: {GPU_PLATFORM_INFO['platform'].upper()}")
    print(f"   • Available Backends: {', '.join(GPU_PLATFORM_INFO['supported_backends'])}")
    print(f"   • Model Loading: Direct (no caching)")
    
    print(f"\n🎯 Model Inventory:")
    print(f"   • LLM Models: {len(llm_models)}")
    print(f"   • Embedding Models: {len(embedding_models)}")
    print(f"   • OCR Models: {len(ocr_models)}")
    
    # Dynamic speech engine status
    speech_engine = SPEECH_TO_TEXT_MANAGER.current_engine if hasattr(SPEECH_TO_TEXT_MANAGER, 'current_engine') else "unknown"
    models_status = "Loaded" if SPEECH_TO_TEXT_MANAGER.models_loaded else "Not loaded"
    print(f"   • Speech Recognition: Ready ({speech_engine.title()} engine - {models_status})")
    
    print(f"\n⚡ Multi-Threading Enabled:")
    print(f"   • LLM: {LLM_MAX_WORKERS} concurrent requests")
    print(f"   • Embeddings: {EMBEDDING_MAX_WORKERS} concurrent requests")
    
    print(f"\n🔌 Core Endpoints:")
    print(f"   • POST /api/generate              - Text generation")
    print(f"   • POST /api/chat                  - Chat completion")
    print(f"   • POST /api/embeddings            - Text embeddings")
    print(f"   • GET  /api/models/available      - List all available models")
    print(f"   • GET  /api/models/loaded         - List loaded models")
    
    print(f"\n🎤 Speech-to-Text Endpoints:")
    print(f"   • GET  /api/speech/engines        - List speech engines")
    print(f"   • POST /api/speech/load           - Load speech engine")
    print(f"   • POST /api/speech/transcribe     - Transcribe audio file")
    print(f"   • POST /api/speech/transcribe/base64 - Transcribe base64 audio")
    
    print(f"\n👁️ OCR Endpoints:")
    print(f"   • GET  /api/ocr/models            - List OCR models")
    print(f"   • POST /api/ocr/file              - OCR from image file")
    print(f"   • POST /api/ocr/extract           - OCR from base64 image")
    
    print(f"\n🔗 Multimodal Endpoints:")
    print(f"   • POST /api/chat/multimodal       - Chat with text+images+audio")
    print(f"\n💡 Press Ctrl+C to stop the server")
    print(f"{'='*70}\n")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down server...")
    logger.info("Closing thread pools...")
    LLM_EXECUTOR.shutdown(wait=True)
    EMBEDDING_EXECUTOR.shutdown(wait=True)
    logger.info("✅ Graceful shutdown complete")


def setup_api_endpoints(app: FastAPI):
    """Setup all API endpoints for the FastAPI app"""
    
    # Update FastAPI app to use lifespan
    app.router.lifespan_context = lifespan

    # ========================================================================
    # Core API Endpoints
    # ========================================================================

    @app.get("/")
    async def root():
        """Server information endpoint - MULTIMODAL"""
        loaded_models = MODEL_MANAGER.list_loaded_models() if MODEL_MANAGER else []
        
        return {
            "status": "Complete Multimodal Ollama-Compatible API Running",
            "version": "2.0.0-multimodal",
            "mode": "multimodal_with_speech_ocr",
            "gpu_platform": GPU_PLATFORM_INFO,
            "speech_recognition": {
                "engine_loaded": SPEECH_TO_TEXT_MANAGER.models_loaded,
                "current_engine": SPEECH_TO_TEXT_MANAGER.current_engine,
                "available_engines": list(SPEECH_TO_TEXT_MANAGER.available_engines.keys())
            },
            "multi_threading": {
                "enabled": True,
                "llm_workers": LLM_MAX_WORKERS,
                "embedding_workers": EMBEDDING_MAX_WORKERS
            },
            "model_manager": {
                "loading_mode": "direct",
                "current_llm_model": MODEL_MANAGER.current_llm_model_id,
                "current_embedding_model": MODEL_MANAGER.current_embedding_model_id,
                "models_loaded": len(loaded_models),
                "loaded_models": [m['model_id'] for m in loaded_models],
                "model_types": {m['model_id']: m['model_type'] for m in loaded_models}
            },
            "endpoints": {
                "core": {
                    "generate": "POST /api/generate",
                    "chat": "POST /api/chat",
                    "embeddings": "POST /api/embeddings",
                    "available_models": "GET /api/models/available",
                    "loaded_models": "GET /api/models/loaded"
                },
                "speech": {
                    "engines": "GET /api/speech/engines",
                    "load": "POST /api/speech/load",
                    "transcribe": "POST /api/speech/transcribe",
                    "transcribe_base64": "POST /api/speech/transcribe/base64"
                },
                "ocr": {
                    "models": "GET /api/ocr/models",
                    "file": "POST /api/ocr/file",
                    "extract": "POST /api/ocr/extract"
                },
                "multimodal": {
                    "chat": "POST /api/chat/multimodal"
                }
            }
        }

    @app.get("/api/version")
    async def version():
        """Get API version"""
        return {"version": "2.0.0-multimodal"}

    @app.get("/api/gpu/info")
    async def gpu_info():
        """Get GPU platform information"""
        return {
            "platform": GPU_PLATFORM_INFO['platform'],
            "supported_backends": GPU_PLATFORM_INFO['supported_backends'],
            "backend_details": GPU_PLATFORM_INFO['backend_info'],
            "cuda_available": GPU_PLATFORM_INFO['cuda'],
            "roc_available": GPU_PLATFORM_INFO['roc'],
            "vulkan_available": GPU_PLATFORM_INFO['vulkan'],
            "metal_available": GPU_PLATFORM_INFO['metal'],
            "oneapi_available": GPU_PLATFORM_INFO['oneapi']
        }

    @app.get("/api/tags")
    async def list_models():
        """List all available models (Ollama compatibility)"""
        available_models = scan_models_directory(MODELS_DIRECTORY, include_detailed_info=False)
        
        # Format for Ollama compatibility
        formatted_models = []
        for model in available_models:
            formatted_models.append({
                "name": model['model_id'],
                "model": model['model_id'],
                "modified_at": datetime.datetime.utcnow().isoformat() + "Z",
                "size": 0,  # Size not exposed in clean API
                "digest": f"sha256:{model['model_id']}",
                "details": {
                    "format": "gguf",
                    "family": model['family'],
                    "families": [model['family']],
                    "parameter_size": model['parameters'],
                    "quantization_level": model['quantization']
                }
            })
        
        return {"models": formatted_models}

    @app.get("/api/models/available")
    async def list_available_models():
        """List all GGUF models found in the models directory - with current model status"""
        available_models = scan_models_directory(MODELS_DIRECTORY, include_detailed_info=False)
        
        # Get currently loaded models (dual tracking)
        current_llm_id = None
        current_llm_info = None
        current_embedding_id = None
        current_embedding_info = None
        
        if MODEL_MANAGER:
            with MODEL_MANAGER.current_model_lock:
                # Get LLM model info
                if hasattr(MODEL_MANAGER, 'current_llm_model_id'):
                    current_llm_id = MODEL_MANAGER.current_llm_model_id
                    if MODEL_MANAGER.current_llm_model:
                        current_llm_info = {
                            'strategy': MODEL_MANAGER.current_llm_model.get('strategy', ['unknown'])[0],
                            'loaded_at': datetime.datetime.now().isoformat()
                        }
                
                # Get embedding model info
                if hasattr(MODEL_MANAGER, 'current_embedding_model_id'):
                    current_embedding_id = MODEL_MANAGER.current_embedding_model_id
                    if MODEL_MANAGER.current_embedding_model:
                        current_embedding_info = {
                            'strategy': MODEL_MANAGER.current_embedding_model.get('strategy', ['unknown'])[0],
                            'loaded_at': datetime.datetime.now().isoformat()
                        }
        
        # Categorize models
        llm_models = []
        embedding_models = []
        ocr_models = []
        
        for model in available_models:
            model_type = model['model_type']
            
            if model_type == 'llm':
                # Check LLM tracking
                if model['model_id'] == current_llm_id:
                    model['loaded'] = True
                    if current_llm_info:
                        model['load_strategy'] = current_llm_info['strategy']
                        model['loaded_at'] = current_llm_info['loaded_at']
                else:
                    model['loaded'] = False
                    model.pop('load_strategy', None)
                    model.pop('loaded_at', None)
                llm_models.append(model)
                
            elif model_type == 'embedding':
                # Check embedding tracking
                if model['model_id'] == current_embedding_id:
                    model['loaded'] = True
                    if current_embedding_info:
                        model['load_strategy'] = current_embedding_info['strategy']
                        model['loaded_at'] = current_embedding_info['loaded_at']
                else:
                    model['loaded'] = False
                    model.pop('load_strategy', None)
                    model.pop('loaded_at', None)
                embedding_models.append(model)
                
            elif model_type == 'ocr':
                # OCR models load on demand
                model['loaded'] = False
                model['load_strategy'] = 'Background Loading'
                ocr_models.append(model)
        
        # Add speech recognition status
        speech_status = SPEECH_TO_TEXT_MANAGER.get_status()
        
        return {
            "status": "success",
            "total_models": len(available_models),
            "llm_count": len(llm_models),
            "embedding_count": len(embedding_models),
            "ocr_count": len(ocr_models),
            "speech_recognition": {
                "engine_loaded": speech_status["engine_loaded"],
                "current_engine": speech_status["current_engine"],
                "available_engines": speech_status["available_engines"]
            },
            "models": {
                "llm": llm_models,
                "embedding": embedding_models,
                "ocr": ocr_models
            }
        }

    @app.get("/api/models/loaded")
    async def list_loaded_models():
        """List currently loaded model"""
        if not MODEL_MANAGER:
            return {"status": "error", "message": "ModelManager not initialized"}
        
        loaded = MODEL_MANAGER.list_loaded_models()
        
        return {
            "status": "success",
            "models_loaded": len(loaded),
            "models": loaded
        }

    @app.get("/api/ps")
    async def running_models():
        """List currently running model"""
        if not MODEL_MANAGER:
            return {"models": []}
        
        loaded = MODEL_MANAGER.list_loaded_models()
        
        models = []
        for entry in loaded:
            models.append({
                "name": entry['model_id'],
                "model": entry['model_id'],
                "size": 0,  # Size not exposed
                "digest": f"sha256:{entry['model_id']}",
                "details": {
                    "format": "gguf",
                    "family": entry['family'],
                    "parameter_size": entry['parameters'],
                    "quantization_level": entry['quantization']
                },
                "expires_at": 0,  # No timeout - model stays until changed
                "size_vram": 0
            })
        
        return {"models": models}

    @app.post("/api/show")
    async def show_model(request: ShowRequest):
        """Show model details - DYNAMIC"""
        model_id = request.name.lower().strip()
        
        # Check if this model is the current LLM or embedding model
        model_entry = None
        model_info = None
        
        if MODEL_MANAGER:
            with MODEL_MANAGER.current_model_lock:
                # Check if it's the current LLM model
                if MODEL_MANAGER.current_llm_model_id == model_id and MODEL_MANAGER.current_llm_model:
                    model_entry = MODEL_MANAGER.current_llm_model
                    model_info = "LLM"
                
                # Check if it's the current embedding model
                elif MODEL_MANAGER.current_embedding_model_id == model_id and MODEL_MANAGER.current_embedding_model:
                    model_entry = MODEL_MANAGER.current_embedding_model
                    model_info = "embedding"
            
            if model_entry:
                info = model_entry['info']
                return {
                    "modelfile": f"# {info['model_name']}\nFROM {info['model_name']}.gguf",
                    "parameters": "temperature 0.7\ntop_p 0.9\nrepeat_penalty 1.1",
                    "template": "<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "details": {
                        "format": info['format'],
                        "family": info['family'],
                        "families": [info['family']],
                        "parameter_size": info['parameter_size'],
                        "quantization_level": info['quantization_level']
                    }
                }
        
        # Model not loaded - return basic info from scan
        available_models = scan_models_directory(MODELS_DIRECTORY, include_detailed_info=False)
        for model in available_models:
            if model['model_id'] == model_id:
                return {
                    "modelfile": f"# {model['model_id']}\nFROM {model['model_id']}.gguf",
                    "details": {
                        "format": "gguf",
                        "family": model['family'],
                        "families": [model['family']],
                        "parameter_size": model['parameters'],
                        "quantization_level": model['quantization']
                    }
                }
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found"
        )

    # ========================================================================
    # Embeddings Endpoint
    # ========================================================================

    @app.post("/api/embeddings")
    async def embeddings(request: EmbeddingsRequest):
        """Generate embeddings for text - MULTI-THREADED WITH DYNAMIC MODEL SELECTION"""
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty"
            )
        
        # Extract model_id from request (keep quantization if present)
        model_id = request.model.lower().replace(":latest", "").strip()
        
        logger.info(f"📊 Embeddings request for model: {model_id}, text: {prompt[:50]}...")
        
        start_time = time.time()
        
        # Run in thread pool with model selection
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            EMBEDDING_EXECUTOR,
            _generate_embedding_sync,
            prompt,
            model_id
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Embedding generated in {elapsed_time:.2f}s")
        
        return {
            "model": request.model,
            "embedding": embedding,
            "embedding_dimension": len(embedding),
            "total_duration": format_duration(int(elapsed_time * 1e9)),
            "load_duration": 0
        }

    # ========================================================================
    # Generate Endpoints
    # ========================================================================

    @app.post("/api/generate")
    async def generate(request: GenerateRequest):
        """Generate text from prompt - DYNAMIC MODEL SELECTION"""
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty"
            )
        
        # Extract model_id from request (keep quantization: "llama-3.1-8b-instruct:q8_k_xl")
        model_id = request.model.lower().replace(":latest", "").strip()
        
        options = request.options or {}
        temperature = max(0.0, min(2.0, options.get('temperature', 0.7)))
        top_p = max(0.0, min(1.0, options.get('top_p', 0.9)))
        max_tokens = max(1, min(8192, options.get('num_predict', 512)))
        
        logger.info(f"🎯 Generate: model={model_id}, stream={request.stream}, tokens={max_tokens}")
        
        # llama-cpp-python automatically adds <|begin_of_text|>, so we don't include it
        formatted_prompt = f"""<|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        if request.stream:
            return StreamingResponse(
                generate_stream(formatted_prompt, model_id, temperature, top_p, max_tokens),
                media_type="application/x-ndjson"
            )
        else:
            return await generate_complete(formatted_prompt, model_id, temperature, top_p, max_tokens)

    async def generate_stream(formatted_prompt: str, model_id: str, temperature: float, top_p: float, max_tokens: int):
        """Generator for streaming responses - DYNAMIC MODEL SELECTION"""
        start_time = time.time()
        full_response = ""
        
        loop = asyncio.get_event_loop()
        
        # Get generator from thread pool with dynamic model selection
        generator = await loop.run_in_executor(
            LLM_EXECUTOR,
            _generate_llm_sync,
            formatted_prompt,
            max_tokens,
            temperature,
            top_p,
            model_id,  # Pass model_id
            True  # stream=True
        )
        
        if isinstance(generator, str):
            # Mock or error response
            for word in generator.split():
                chunk = {
                    "model": model_id,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "response": word + " ",
                    "done": False
                }
                yield json.dumps(chunk) + "\n"
                full_response += word + " "
                await asyncio.sleep(0.01)  # Small delay for realistic streaming
        else:
            # Real streaming
            try:
                for output in generator:
                    token = output['choices'][0]['text']
                    full_response += token
                    
                    chunk = {
                        "model": model_id,
                        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "response": token,
                        "done": False
                    }
                    yield json.dumps(chunk) + "\n"
                    await asyncio.sleep(0)  # Yield control to event loop
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                error_chunk = {
                    "model": model_id,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "response": f"\n[Error: {str(e)}]",
                    "done": False
                }
                yield json.dumps(error_chunk) + "\n"
        
        elapsed_time = time.time() - start_time
        final_chunk = {
            "model": model_id,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "response": "",
            "done": True,
            "context": [],
            "total_duration": format_duration(int(elapsed_time * 1e9)),
            "load_duration": 0,
            "prompt_eval_count": len(formatted_prompt.split()),
            "prompt_eval_duration": format_duration(int(elapsed_time * 0.1 * 1e9)),
            "eval_count": len(full_response.split()),
            "eval_duration": format_duration(int(elapsed_time * 0.9 * 1e9))
        }
        yield json.dumps(final_chunk) + "\n"

    async def generate_complete(formatted_prompt: str, model_id: str, temperature: float, top_p: float, max_tokens: int):
        """Generate complete response - DYNAMIC MODEL SELECTION"""
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        # Run blocking LLM generation in thread pool with dynamic model selection
        ai_response = await loop.run_in_executor(
            LLM_EXECUTOR,
            _generate_llm_sync,
            formatted_prompt,
            max_tokens,
            temperature,
            top_p,
            model_id,  # Pass model_id
            False  # stream=False
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Generation completed in {elapsed_time:.2f}s")
        
        # Get model info to show in response
        model_info = {}
        if MODEL_MANAGER:
            with MODEL_MANAGER.current_model_lock:
                # Check if it's the current LLM model
                if MODEL_MANAGER.current_llm_model_id == model_id and MODEL_MANAGER.current_llm_model:
                    entry = MODEL_MANAGER.current_llm_model
                    model_info = {
                        "name": entry['info']['model_name'],
                        "quantization": entry['info']['quantization_level'],
                        "parameters": entry['info']['parameter_size'],
                        "family": entry['info']['family']
                    }
        
        return {
            "model": model_id,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "response": ai_response,
            "done": True,
            "context": [],
            "total_duration": format_duration(int(elapsed_time * 1e9)),
            "load_duration": 0,
            "prompt_eval_count": len(formatted_prompt.split()),
            "prompt_eval_duration": format_duration(int(elapsed_time * 0.1 * 1e9)),
            "eval_count": len(ai_response.split()),
            "eval_duration": format_duration(int(elapsed_time * 0.9 * 1e9)),
            "model_info": model_info  # Show which exact model ran
        }

    # ========================================================================
    # Chat Endpoints
    # ========================================================================

    # Import chat-related models and functions
    from utils import chat_with_model_sync
    
    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        """Chat completion endpoint - DYNAMIC MODEL SELECTION"""
        # Extract model_id from request (keep quantization format: "llama-3.1-8b-instruct:q8_k_xl")
        model_id = request.model.lower().replace(":latest", "").strip()
        
        options = request.options or {}
        temperature = max(0.0, min(2.0, options.get('temperature', 0.7)))
        top_p = max(0.0, min(1.0, options.get('top_p', 0.9)))
        max_tokens = max(1, min(8192, options.get('num_predict', 512)))
        
        logger.info(f"💬 Chat: model={model_id}, stream={request.stream}, messages={len(request.messages)}")
        
        # Format conversation for model
        formatted_conversation = format_chat_messages(request.messages)
        
        if request.stream:
            return StreamingResponse(
                chat_stream(formatted_conversation, model_id, temperature, top_p, max_tokens),
                media_type="application/x-ndjson"
            )
        else:
            return await chat_complete(formatted_conversation, model_id, temperature, top_p, max_tokens)

    def format_chat_messages(messages):
        """Format chat messages for the model"""
        formatted = ""
        for message in messages:
            # Access Pydantic object attributes instead of dictionary keys
            role = message.role
            content = message.content
            if role == 'system':
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'user':
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'assistant':
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        # Add assistant header for response
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted

    async def chat_stream(formatted_conversation: str, model_id: str, temperature: float, top_p: float, max_tokens: int):
        """Stream chat responses"""
        start_time = time.time()
        full_response = ""
        
        loop = asyncio.get_event_loop()
        generator = await loop.run_in_executor(
            LLM_EXECUTOR,
            chat_with_model_sync,
            formatted_conversation,
            max_tokens,
            temperature,
            top_p,
            model_id,
            True
        )
        
        if isinstance(generator, str):
            # Mock response
            for word in generator.split():
                chunk = {
                    "model": model_id,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "message": {"role": "assistant", "content": word + " "},
                    "done": False
                }
                yield json.dumps(chunk) + "\n"
                full_response += word + " "
                await asyncio.sleep(0.01)
        else:
            # Real streaming
            try:
                for output in generator:
                    token = output['choices'][0]['text']
                    full_response += token
                    chunk = {
                        "model": model_id,
                        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "message": {"role": "assistant", "content": token},
                        "done": False
                    }
                    yield json.dumps(chunk) + "\n"
                    await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Chat streaming error: {e}", exc_info=True)
                error_chunk = {
                    "model": model_id,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "message": {"role": "assistant", "content": f"\n[Error: {str(e)}]"},
                    "done": False
                }
                yield json.dumps(error_chunk) + "\n"
        
        elapsed_time = time.time() - start_time
        final_chunk = {
            "model": model_id,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "total_duration": format_duration(int(elapsed_time * 1e9)),
            "load_duration": 0,
            "prompt_eval_count": len(formatted_conversation.split()),
            "eval_count": len(full_response.split())
        }
        yield json.dumps(final_chunk) + "\n"

    async def chat_complete(formatted_conversation: str, model_id: str, temperature: float, top_p: float, max_tokens: int):
        """Complete chat response"""
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        ai_response = await loop.run_in_executor(
            LLM_EXECUTOR,
            chat_with_model_sync,
            formatted_conversation,
            max_tokens,
            temperature,
            top_p,
            model_id,
            False
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Chat completed in {elapsed_time:.2f}s")
        
        return {
            "id": "chatcmpl-" + datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ai_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(formatted_conversation.split()),
                "completion_tokens": len(ai_response.split()),
                "total_tokens": len(formatted_conversation.split()) + len(ai_response.split())
            }
        }

    # ========================================================================
    # Speech Endpoints - FIXED VERSION
    # ========================================================================

    @app.get("/api/speech/engines")
    async def list_speech_engines():
        """List available speech engines"""
        return {
            "engines": list(SPEECH_TO_TEXT_MANAGER.available_engines.keys()),
            "current_engine": SPEECH_TO_TEXT_MANAGER.current_engine,
            "models_loaded": SPEECH_TO_TEXT_MANAGER.models_loaded
        }

    @app.get("/api/speech/status")
    async def speech_status():
        """Get speech recognition status"""
        return SPEECH_TO_TEXT_MANAGER.get_status()

    @app.post("/api/speech/load")
    async def load_speech_engine(engine_name: str = "whisper"):
        """Load a speech engine"""
        try:
            result = SPEECH_TO_TEXT_MANAGER.load_engine(engine_name)
            return {
                "status": "success",
                "engine": engine_name,
                "loaded": result
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/speech/transcribe")
    async def transcribe_audio(file: UploadFile = File(...), engine: str = "whisper"):
        """Transcribe audio file - FIXED VERSION"""
        try:
            # Read file content
            audio_data = await file.read()
            
            # Use the fixed transcribe method that accepts audio_data directly
            result = SPEECH_TO_TEXT_MANAGER.transcribe(audio_data, engine=engine)
            
            return {
                "text": result.get("text", ""),
                "language": result.get("language", "en"),
                "confidence": result.get("confidence", 0.0),
                "engine": engine,
                "filename": file.filename
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/speech/transcribe/file")
    async def transcribe_audio_file(file: UploadFile = File(...)):
        """Transcribe from uploaded audio file - FIXED VERSION"""
        try:
            # Read file content
            audio_data = await file.read()
            
            # Use the fixed transcribe method that accepts audio_data directly
            result = SPEECH_TO_TEXT_MANAGER.transcribe(audio_data)
            
            return {
                "text": result.get("text", ""),
                "language": result.get("language", "en"),
                "confidence": result.get("confidence", 0.0),
                "filename": file.filename
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/speech/transcribe/base64")
    async def transcribe_base64_audio(audio_data: dict):
        """Transcribe base64 encoded audio - FIXED VERSION"""
        try:
            import base64
            
            if "audio" not in audio_data:
                raise HTTPException(status_code=400, detail="No audio data provided")
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data["audio"])
            
            # Transcribe using speech manager
            result = SPEECH_TO_TEXT_MANAGER.transcribe(audio_bytes)
            
            return {
                "status": "success",
                "text": result.get("text", ""),
                "language": result.get("language", "en"),
                "confidence": result.get("confidence", 0.0),
                "engine": SPEECH_TO_TEXT_MANAGER.current_engine
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # OCR Endpoints
    # ========================================================================

    @app.get("/api/ocr/models")
    async def list_ocr_models():
        """List available OCR models"""
        # Get OCR models from the model scanner
        available_models = scan_models_directory(MODELS_DIRECTORY, include_detailed_info=False)
        ocr_models = [m for m in available_models if m['model_type'] == 'ocr']
        
        return {
            "models": ocr_models,
            "total_count": len(ocr_models)
        }

    @app.post("/api/ocr/extract")
    async def extract_text_from_image(image_data: dict):
        """Extract text from base64 encoded image using OCR"""
        try:
            import base64
            
            if "image" not in image_data:
                raise HTTPException(status_code=400, detail="No image data provided")
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data["image"])
            
            # Use the OCR manager to process the image
            # OCR_MANAGER.process_image returns tuple: (text, confidence, method)
            text, confidence, method = OCR_MANAGER.process_image(image_bytes)
            
            return {
                "status": "success",
                "text": text,
                "confidence": confidence,
                "method": method
            }
        except Exception as e:
            logger.error(f"OCR error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/ocr/file")
    async def ocr_from_file(file: UploadFile = File(...)):
        """OCR from uploaded image file"""
        try:
            # Read file content
            image_data = await file.read()
            
            # Use the OCR manager to process the image
            # OCR_MANAGER.process_image returns tuple: (text, confidence, method)
            text, confidence, method = OCR_MANAGER.process_image(image_data)
            
            return {
                "text": text,
                "confidence": confidence,
                "method": method,
                "filename": file.filename
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Multimodal Endpoints
    # ========================================================================

    @app.post("/api/chat/multimodal")
    async def multimodal_chat(request: dict):
        """Multimodal chat with text, images, and audio"""
        # This is a placeholder for multimodal functionality
        # In a full implementation, this would handle:
        # 1. Text processing through LLM
        # 2. Image analysis through OCR/vision models
        # 3. Audio processing through speech-to-text
        # 4. Combining all modalities for comprehensive responses
        
        raise HTTPException(status_code=501, detail="Multimodal chat not fully implemented in this version")

    # ========================================================================
    # Cache Management Endpoints
    # ========================================================================

    @app.get("/api/cache/stats")
    async def cache_stats():
        """Get cache statistics"""
        return {
            "status": "success",
            "message": "Cache statistics not available in direct loading mode"
        }

    @app.post("/api/cache/clear")
    async def clear_cache():
        """Clear model cache"""
        return {
            "status": "success",
            "message": "Cache cleared (no caching in direct loading mode)"
        }

    @app.post("/api/cache/warmup")
    async def warmup_cache():
        """Warm up model cache"""
        return {
            "status": "success",
            "message": "Cache warmed (no caching in direct loading mode)"
        }

    # ========================================================================
    # Ollama Compatibility Endpoints
    # ========================================================================

    @app.post("/api/pull")
    async def pull_model():
        """Pull model from registry - not supported"""
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Pull not supported")

    @app.post("/api/create")
    async def create_model():
        """Create model - not supported"""
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Create not supported")

    @app.post("/api/push")
    async def push_model():
        """Push model to registry - not supported"""
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Push not supported")

    @app.post("/api/copy")
    async def copy_model():
        """Copy model - not supported"""
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Copy not supported")

    @app.delete("/api/delete")
    async def delete_model():
        """Delete model - not supported"""
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Delete not supported")