"""
MCP Server for loading and using local Hugging Face models for sentiment analysis.
"""
import asyncio
import os
import sys
import logging
from typing import Any, Dict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from dotenv import load_dotenv

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
from log_util import logger

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
DEFAULT_MODEL_NAME = os.getenv("HF_MODEL_NAME", "TungCan/tuning-sentiment-abp-pos")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model cache
_model_cache = {}

# ============================================================================
# MODEL LOADING
# ============================================================================
def get_local_hf_model(model_name: str = None, device_map: str = "auto"):
    """
    Load a local Hugging Face model (with caching).
    
    Args:
        model_name: Hugging Face model identifier
        device_map: Device to load model on ("auto", "cpu", "cuda")
    
    Returns:
        tuple: (tokenizer, model, config)
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    
    # Check cache
    cache_key = f"{model_name}_{device_map}"
    if cache_key in _model_cache:
        logger.info(f"Using cached model: {model_name}")
        return _model_cache[cache_key]
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device_map}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Load config
        config = AutoConfig.from_pretrained(model_name)
        
        # Cache for reuse
        _model_cache[cache_key] = (tokenizer, model, config)
        
        logger.info(f"✅ Model loaded successfully: {model_name}")
        logger.info(f"   Labels: {config.id2label}")
        
        return tokenizer, model, config
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

def predict_sentiment(text: str, model_name: str = None, device_map: str = "auto") -> Dict[str, Any]:
    """
    Predict sentiment for given text using local HF model.
    
    Args:
        text: Input text to analyze
        model_name: Model to use (optional)
        device_map: Device map (optional)
    
    Returns:
        dict: Prediction results with label, confidence, and all scores
    """
    if not text or not text.strip():
        raise ValueError("Text không được rỗng")
    
    # Load model
    tokenizer, model, config = get_local_hf_model(model_name, device_map)
    
    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
        max_length=512
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=-1).cpu().numpy()[0]
        predicted_idx = np.argmax(scores)
    
    # Format results
    result = {
        "text": text,
        "predicted_label": config.id2label[predicted_idx],
        "confidence": float(scores[predicted_idx]),
        "all_scores": {
            config.id2label[i]: float(score) 
            for i, score in enumerate(scores)
        },
        "model": model_name or DEFAULT_MODEL_NAME
    }
    
    logger.info(f"Prediction: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
    
    return result

# ============================================================================
# MCP SERVER SETUP
# ============================================================================
server = Server("hf-local-server")

# ============================================================================
# TOOLS HANDLERS
# ============================================================================
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Danh sách các tools hỗ trợ"""
    return [
        types.Tool(
            name="predict_sentiment",
            description="Phân tích sentiment của văn bản sử dụng local Hugging Face model",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Văn bản cần phân tích sentiment",
                    },
                    "model_name": {
                        "type": "string",
                        "description": f"Tên model (mặc định: {DEFAULT_MODEL_NAME})",
                    },
                    "device_map": {
                        "type": "string",
                        "description": f"Device để chạy model (mặc định: {DEFAULT_DEVICE})",
                        "enum": ["auto", "cpu", "cuda", "cuda:0"]
                    }
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="get_model_info",
            description="Lấy thông tin về model đang được load",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": f"Tên model (mặc định: {DEFAULT_MODEL_NAME})",
                    }
                },
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """Xử lý các tool calls"""
    import json
    
    try:
        if name == "predict_sentiment":
            if not arguments or "text" not in arguments:
                raise ValueError("Thiếu tham số 'text'")
            
            text = arguments["text"]
            model_name = arguments.get("model_name")
            device_map = arguments.get("device_map", "auto")
            
            # Run prediction
            result = predict_sentiment(text, model_name, device_map)
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False),
                )
            ]
        
        elif name == "get_model_info":
            model_name = arguments.get("model_name") if arguments else None
            
            # Load model (will use cache if already loaded)
            tokenizer, model, config = get_local_hf_model(model_name)
            
            info = {
                "model_name": model_name or DEFAULT_MODEL_NAME,
                "num_labels": config.num_labels,
                "labels": config.id2label,
                "model_type": config.model_type,
                "vocab_size": tokenizer.vocab_size,
                "device": str(next(model.parameters()).device),
                "cached_models": list(_model_cache.keys())
            }
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(info, indent=2, ensure_ascii=False),
                )
            ]
        
        else:
            raise ValueError(f"Tool không xác định: {name}")
            
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, ensure_ascii=False)
            )
        ]

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
async def main():
    """Điểm vào chương trình: khởi động MCP server"""
    try:
        logger.info("Starting HF Local MCP Server...")
        logger.info(f"Default model: {DEFAULT_MODEL_NAME}")
        logger.info(f"Default device: {DEFAULT_DEVICE}")
        
        # Preload model (optional)
        if os.getenv("PRELOAD_MODEL", "false").lower() == "true":
            logger.info("Preloading model...")
            get_local_hf_model()
        
        # Run MCP server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="hf-local-server",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())