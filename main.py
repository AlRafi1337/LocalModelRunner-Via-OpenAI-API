#!/usr/bin/env python3
"""
Fixed Medical AI Service - All Import Issues Resolved
Main entry point with proper port configuration
"""

import sys
import os

# Ensure current directory is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import our modular components
from config import logger, SERVER_HOST, SERVER_PORT
from api_endpoints import setup_api_endpoints


# Create FastAPI app
app = FastAPI(
    title="Fixed Multimodal Ollama-Compatible API",
    description="Multi-Platform GPU-Accelerated API with Speech-to-Text, OCR, and Multimodal Chat",
    version="2.0.1-fixed"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup all API endpoints
setup_api_endpoints(app)

if __name__ == '__main__':
    print("🚀 Starting Fixed Medical AI Service...")
    print(f"🌐 Service will be available at: http://{SERVER_HOST}:{SERVER_PORT}")
    print("📝 Test with:")
    print('curl -X POST "http://localhost:11435/api/chat" -H "Content-Type: application/json" -d \'{"model": "llama-3.1-8b-instruct-ud:q8_k_xl", "messages": [{"role": "user", "content": "Hello"}]}\'')
    print()
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        access_log=True
    )