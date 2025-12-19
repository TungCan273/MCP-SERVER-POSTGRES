import os
import sys
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
import mcp.server.stdio
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import Any, Dict, List
from time import time
import re
import json

load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

from log_util import logger

# Cấu hình: Chọn provider (openrouter hoặc google)
PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").lower()  # openrouter hoặc google
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Model mặc định cho từng provider
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b")  # Có thể dùng: openai/gpt-3.5-turbo, google/gemini-pro, anthropic/claude-3-haiku, etc.
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-pro")

# Khởi tạo LLM dựa trên provider
if PROVIDER == "openrouter":
    from langchain_openai import ChatOpenAI
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY chưa được set! Vui lòng set biến môi trường OPENROUTER_API_KEY")
    
    # OpenRouter sử dụng OpenAI-compatible API
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=1.0,
    )
    logger.info(f"✅ Đang sử dụng OpenRouter với model: {OPENROUTER_MODEL}")
elif PROVIDER == "google":
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY chưa được set! Vui lòng set biến môi trường GOOGLE_API_KEY")
    
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=1.0,
    )
    logger.info(f"✅ Đang sử dụng Google Gemini với model: {GOOGLE_MODEL}")
else:
    raise ValueError(f"Provider không hợp lệ: {PROVIDER}. Chọn 'openrouter' hoặc 'google'")


# ============================================================================
# RATE LIMITER
# ============================================================================
class RateLimiter:
    """Rate limiter đơn giản cho LLM calls"""
    
    def __init__(self, max_calls: int = 10, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
    
    def can_call(self) -> bool:
        """Kiểm tra có thể gọi LLM không"""
        now = time()
        # Xóa các calls cũ ngoài time window
        self.calls = [t for t in self.calls if now - t < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            return False
        
        self.calls.append(now)
        return True
    
    def get_wait_time(self) -> float:
        """Tính thời gian phải đợi (seconds)"""
        if not self.calls:
            return 0.0
        oldest_call = min(self.calls)
        return max(0, self.time_window - (time() - oldest_call))

sentiment_limiter = RateLimiter(max_calls=20, time_window=60)

# ============================================================================
# LLM OPERATIONS
# ============================================================================
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Phân tích cảm xúc của đoạn văn bản sử dụng LLM.
    Returns: dict chứa sentiment, confidence, reason
    """
    # Rate limiting check
    if not sentiment_limiter.can_call():
        wait_time = sentiment_limiter.get_wait_time()
        raise ValueError(
            f"Quá nhiều requests. Vui lòng thử lại sau {int(wait_time)} giây"
        )
    
    if not text or not text.strip():
        raise ValueError("Text không được rỗng")
    
    # Giới hạn độ dài text
    if len(text) > 5000:
        text = text[:5000] + "..."
        logger.warning("Text quá dài, đã cắt xuống 5000 ký tự")
    
    try:
        prompt = f"""Bạn là hệ thống phân tích cảm xúc. Đánh giá cảm xúc của đoạn văn sau:

"{text}"

Nhiệm vụ:
- Phân loại sentiment: positive | negative | neutral
- Trả về JSON hợp lệ, KHÔNG giải thích thêm

Schema:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.85,
    "reason": "Lý do ngắn gọn (<= 30 từ)"
}}"""
        
        logger.info("Calling LLM for sentiment analysis...")
        response = llm([HumanMessage(content=prompt)])
        result_text = response.content if hasattr(response, "content") else str(response)
        
        try:
            # Loại bỏ markdown code blocks nếu có
            clean_text = result_text.strip()
            if clean_text.startswith("```"):
                clean_text = re.sub(r"```json\s*|\s*```", "", clean_text).strip()
            
            result = json.loads(clean_text)
            logger.info(f"Sentiment analysis completed: {result}")
            return {
                "success": True,
                "sentiment": result.get("sentiment", "neutral"),
                "confidence": result.get("confidence", 0.0),
                "reason": result.get("reason", ""),
                "raw_response": result_text
            }
        except json.JSONDecodeError:
            # Nếu không parse được JSON, trả về raw text
            logger.warning("Failed to parse LLM response as JSON")
            return {
                "success": True,
                "raw_response": result_text,
                "note": "LLM không trả về JSON format"
            }
            
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Khởi tạo instance server để dùng cho decorator
server = Server("sentiment-mcp-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    Trả về danh sách các tool hỗ trợ:
    - textDocument_sentiment: đánh giá cảm xúc của đoạn văn
    """
    logger.info("Use handle tool")
    return [
        types.Tool(
            name="predict_sentiment",
            description="Đánh giá cảm xúc của đoạn văn bản (positive/negative/neutral)",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Đoạn văn cần đánh giá cảm xúc",
                    },
                },
                "required": ["text"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Nhận tên công cụ được gọi và tham số. Xử lý các gọi tool:
    - Nếu tool là textDocument_sentiment: đánh giá cảm xúc của đoạn văn
    """
    if name != "predict_sentiment":
        raise ValueError(f"Tool không xác định: {name}")

    text = (arguments or {}).get("text", "")
    if not text:
        return [
            types.TextContent(type="text", text="Missing text"),
        ]
    
    try:
        # prompt = (
        #     "Đánh giá cảm xúc của đoạn văn sau "
        #     "(positive/negative/neutral, giải thích ngắn gọn):\n"
        #     f"{text}"
        # )
        # response = llm([HumanMessage(content=prompt)])
        # result = response.content if hasattr(response, "content") else str(response)
        # return [
        #     types.TextContent(
        #         type="text",
        #         text=result,
        #     )
        # ]
        if not arguments or "text" not in arguments:
            raise ValueError("Thiếu tham số 'text'")
        else:
            text = arguments["text"]
            result = await analyze_sentiment(text)
            
            return [
                types.TextContent(
                    type="text",
                    text=result["label"],
                )
            ]
    except Exception as e:
        logger.exception("Sentiment tool error")
        return [
            types.TextContent(
                type="text",
                text="INTERNAL_ERROR {}".format(e)
            )
        ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Trả về nội dung mẫu lời nhắc (prompt template, message) tuỳ từng loại phân tích.
    Dùng cho AI chatbot hoặc front-end truyền vào để sinh câu hỏi ra LLM.
    """
    if name == "sentiment_analysis":
        text = arguments.get("text")
        prompt = (
            f"Đánh giá cảm xúc của đoạn văn sau {text}"
            "Kết quả trả về là 1 trong 3 giá trị: positive, negative, neutral"
            "Giải thích ngắn gọn không quá 30 từ."
            "Nếu không chắc chắn về kết quả hãy trả về trung tính"
        )

        prompt_result = types.GetPromptResult(
            description="Đánh giá cảm xúc của đoạn văn",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt)
                )
            ],
        )
        logger.info(prompt_result)

        return prompt_result
    else:
        raise ValueError(f"Prompt không xác định: {name}")  

async def main():
    """
    Điểm vào chương trình: mở giao tiếp chuẩn stdin/stdout, khởi động MCP server, cung cấp các capabilities (tính năng) ra ngoài.
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sentiment-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
