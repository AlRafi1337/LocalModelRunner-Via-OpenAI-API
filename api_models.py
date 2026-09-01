#!/usr/bin/env python3
"""
Pydantic Models for Request/Response Validation
Features: API Data Models, Validation, Type Hints
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator

class Message(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ['system', 'user', 'assistant']:
            raise ValueError(f"Role must be 'system', 'user', or 'assistant', got '{v}'")
        return v

class GenerateRequest(BaseModel):
    model: str = Field(default="llama3.1:8b", description="Model name")
    prompt: str = Field(..., description="Prompt text")
    stream: bool = Field(default=False, description="Enable streaming")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    model: str = Field(default="llama3.1:8b", description="Model name")
    messages: List[Message] = Field(..., description="Conversation messages")
    stream: bool = Field(default=False, description="Enable streaming")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        if not v or len(v) == 0:
            raise ValueError("Messages list cannot be empty")
        return v

class EmbeddingsRequest(BaseModel):
    model: str = Field(default="bge-large-en:latest", description="Model name")
    prompt: str = Field(..., description="Text to embed")

class ShowRequest(BaseModel):
    name: str = Field(..., description="Model name")

class ModelEntry(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]

# ============================================================================
# Multimodal Models
# ============================================================================

class SpeechRecognitionRequest(BaseModel):
    audio_data: Optional[str] = None  # Base64 encoded audio
    language: str = "en-US"
    engine: str = "vosk"  # vosk, whisper_local, sphinx, google
    duration: Optional[int] = None  # For microphone input

class SpeechRecognitionResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    engine: Optional[str] = None
    language: Optional[str] = None
    error: Optional[str] = None

# Alias for backward compatibility
SpeechRequest = SpeechRecognitionRequest
SpeechResponse = SpeechRecognitionResponse

class OCRRequest(BaseModel):
    image_data: str  # Base64 encoded image
    model_id: str = "paddleocr-v2"

class OCRResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    model_id: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

class MultimodalMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]  # Support both text and multimodal content

class MultimodalChatRequest(BaseModel):
    model: str
    messages: List[MultimodalMessage]
    stream: bool = False
    options: Optional[Dict[str, Any]] = None

class MultimodalChatResponse(BaseModel):
    model: str
    created_at: str
    message: MultimodalMessage
    done: bool