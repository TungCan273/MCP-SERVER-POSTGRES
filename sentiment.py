import os
import sys
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
import mcp.server.stdio
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import Any

load_dotenv()

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
    print(f"✅ Đang sử dụng OpenRouter với model: {OPENROUTER_MODEL}", file=sys.stderr)
elif PROVIDER == "google":
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY chưa được set! Vui lòng set biến môi trường GOOGLE_API_KEY")
    
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=1.0,
    )
    print(f"✅ Đang sử dụng Google Gemini với model: {GOOGLE_MODEL}", file=sys.stderr)
else:
    raise ValueError(f"Provider không hợp lệ: {PROVIDER}. Chọn 'openrouter' hoặc 'google'")

# Khởi tạo instance server để dùng cho decorator
server = Server("sentiment-mcp-server")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    Trả về danh sách các tool hỗ trợ:
    - textDocument_sentiment: đánh giá cảm xúc của đoạn văn
    """
    return [
        types.Tool(
            name="textDocument_sentiment",
            description="""Đánh giá cảm xúc của đoạn văn""",

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
    if name != "textDocument_sentiment":
        raise ValueError(f"Tool không xác định: {name}")

    text = (arguments or {}).get("text", "")
    if not text:
        return [
            types.TextContent(type="text", text="Missing text"),
        ]
    
    try:
        prompt = (
            "Đánh giá cảm xúc của đoạn văn sau "
            "(positive/negative/neutral, giải thích ngắn gọn):\n"
            f"{text}"
        )
        response = llm([HumanMessage(content=prompt)])
        result = response.content if hasattr(response, "content") else str(response)
        return [
            types.TextContent(
                type="text",
                text=result,
            )
        ]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Lỗi: {str(e)}")]


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
        return types.GetPromptResult(
            description="Đánh giá cảm xúc của đoạn văn",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt)
                )
            ],
        )
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
