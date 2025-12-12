@echo off
REM Script để chạy sentiment.py MCP server
REM Hỗ trợ cả OpenRouter và Google Gemini
REM Mặc định sử dụng OpenRouter

echo Đang chạy Sentiment MCP Server...
echo.

REM Kiểm tra provider
set PROVIDER=%LLM_PROVIDER%
if "%PROVIDER%"=="" set PROVIDER=openrouter

if "%PROVIDER%"=="openrouter" (
    echo Sử dụng OpenRouter...
    if "%OPENROUTER_API_KEY%"=="" (
        echo CẢNH BÁO: OPENROUTER_API_KEY chưa được set!
        echo Vui lòng set biến môi trường OPENROUTER_API_KEY trước khi chạy.
        echo.
        echo Lấy API key tại: https://openrouter.ai/keys
        echo.
        echo Ví dụ: set OPENROUTER_API_KEY=your-api-key-here
        echo.
        pause
        exit /b 1
    )
) else if "%PROVIDER%"=="google" (
    echo Sử dụng Google Gemini...
    if "%GOOGLE_API_KEY%"=="" (
        echo CẢNH BÁO: GOOGLE_API_KEY chưa được set!
        echo Vui lòng set biến môi trường GOOGLE_API_KEY trước khi chạy.
        echo.
        echo Lấy API key miễn phí tại: https://aistudio.google.com/app/apikey
        echo.
        echo Ví dụ: set GOOGLE_API_KEY=your-api-key-here
        echo.
        pause
        exit /b 1
    )
) else (
    echo Lỗi: LLM_PROVIDER phải là 'openrouter' hoặc 'google'
    pause
    exit /b 1
)

REM Chạy server bằng uv
uv run python sentiment.py

