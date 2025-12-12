# Hướng dẫn sử dụng Sentiment MCP Server

## 🎯 Tổng quan

Server này hỗ trợ **2 cách sử dụng API**:

1. **OpenRouter** (Khuyến nghị) - Một API key để truy cập nhiều model
2. **Google Gemini** - API trực tiếp từ Google (miễn phí)

## 📋 OpenRouter vs API trực tiếp

### ✅ OpenRouter (Khuyến nghị)
- **Một API key** để truy cập **400+ models** từ nhiều nhà cung cấp
- Hỗ trợ: OpenAI, Google Gemini, Anthropic Claude, Meta, v.v.
- Giá cả hợp lý, thanh toán theo usage
- Không cần mua API riêng từ từng nhà cung cấp
- **Lấy API key**: https://openrouter.ai/keys

### Google Gemini trực tiếp
- Miễn phí nhưng có giới hạn
- Chỉ dùng được model của Google
- **Lấy API key**: https://aistudio.google.com/app/apikey

## 🚀 Cách sử dụng

### Cách 1: Sử dụng OpenRouter (Mặc định)

#### Bước 1: Lấy API key từ OpenRouter
1. Truy cập: https://openrouter.ai/keys
2. Đăng ký/Đăng nhập
3. Tạo API key mới
4. Copy API key

#### Bước 2: Set biến môi trường

**PowerShell:**
```powershell
$env:LLM_PROVIDER = "openrouter"
$env:OPENROUTER_API_KEY = "sk-or-v1-your-api-key-here"
# Tùy chọn: Chọn model (mặc định: google/gemini-pro)
$env:OPENROUTER_MODEL = "google/gemini-pro"  # hoặc "openai/gpt-3.5-turbo", "anthropic/claude-3-haiku", etc.
```

**Command Prompt:**
```cmd
set LLM_PROVIDER=openrouter
set OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
set OPENROUTER_MODEL=google/gemini-pro
```

#### Bước 3: Chạy server
```cmd
run_sentiment.bat
```

### Cách 2: Sử dụng Google Gemini trực tiếp

#### Bước 1: Lấy API key từ Google
1. Truy cập: https://aistudio.google.com/app/apikey
2. Đăng nhập bằng Google account
3. Tạo API key mới
4. Copy API key

#### Bước 2: Set biến môi trường

**PowerShell:**
```powershell
$env:LLM_PROVIDER = "google"
$env:GOOGLE_API_KEY = "your-google-api-key-here"
```

**Command Prompt:**
```cmd
set LLM_PROVIDER=google
set GOOGLE_API_KEY=your-google-api-key-here
```

#### Bước 3: Chạy server
```cmd
run_sentiment.bat
```

## 📝 Các model phổ biến trên OpenRouter

### Miễn phí / Rẻ
- `google/gemini-pro` - Google Gemini Pro (miễn phí)
- `google/gemini-flash-1.5` - Google Gemini Flash (nhanh, rẻ)
- `meta-llama/llama-3.2-3b-instruct:free` - Meta Llama 3.2 (miễn phí)

### Trả phí (giá hợp lý)
- `openai/gpt-3.5-turbo` - OpenAI GPT-3.5 Turbo
- `openai/gpt-4o-mini` - OpenAI GPT-4o Mini
- `anthropic/claude-3-haiku` - Anthropic Claude 3 Haiku
- `google/gemini-1.5-pro` - Google Gemini 1.5 Pro

Xem danh sách đầy đủ: https://openrouter.ai/models

## 🔧 Cấu hình nâng cao

### Chọn model khác với OpenRouter
```powershell
$env:OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

### Chọn model khác với Google
```powershell
$env:GOOGLE_MODEL = "gemini-1.5-flash"
```

## 💡 Lưu ý

1. **OpenRouter** là lựa chọn tốt nhất nếu bạn muốn:
   - Dùng nhiều model khác nhau
   - Linh hoạt chuyển đổi model
   - Quản lý chi phí tập trung

2. **Google Gemini trực tiếp** phù hợp nếu:
   - Chỉ cần dùng model của Google
   - Muốn tận dụng hạn mức miễn phí

3. Mặc định server sử dụng **OpenRouter** nếu không set `LLM_PROVIDER`

## 🐛 Troubleshooting

### Lỗi: "OPENROUTER_API_KEY chưa được set"
→ Set biến môi trường `OPENROUTER_API_KEY` trước khi chạy

### Lỗi: "Provider không hợp lệ"
→ `LLM_PROVIDER` phải là `"openrouter"` hoặc `"google"`

### Lỗi: Model không tìm thấy (OpenRouter)
→ Kiểm tra tên model tại https://openrouter.ai/models
→ Đảm bảo bạn có quyền truy cập model đó

