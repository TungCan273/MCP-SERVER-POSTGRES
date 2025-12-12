# Giải thích code file postgres.py - Tích hợp PostgreSQL và MCP server

# Import các thư viện cần thiết
import asyncio                       # Cho phép xử lý bất đồng bộ
import os                            # Đọc biến môi trường
from typing import Any, Dict, List, Optional
import psycopg2                      # Thư viện kết nối PostgreSQL
from psycopg2.extras import RealDictCursor   # Kiểu cursor trả về dict
from mcp.server.models import InitializationOptions    # Cấu hình MCP server
import mcp.types as types                         # Các kiểu dữ liệu MCP
from mcp.server import NotificationOptions, Server  # MCP Server và options
import mcp.server.stdio                           # Giao tiếp chuẩn I/O MCP

# 1. Đọc cấu hình database từ biến môi trường
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "postgres"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# 2. Khai báo cache lưu trữ schema các bảng để tăng tốc truy xuất metadata
schema_cache: Dict[str, Any] = {}

# 3. Hàm tạo connection PostgreSQL 
def get_db_connection():
    """
    Tạo kết nối đến PostgreSQL sử dụng các thông số cấu hình từ DB_CONFIG.
    Dùng RealDictCursor để các kết quả truy vấn trả về dạng dict dễ xử lý.
    """
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

# 4. Lấy schema của tất cả các bảng trong database
def fetch_table_schemas() -> Dict[str, Any]:
    """
    Truy vấn thông tin schema các bảng trong database (thông qua information_schema.columns).
    Trả về dict chứa thông tin từng bảng và danh sách các cột của bảng đó.
    """
    query = """
        SELECT 
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, ordinal_position;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        schemas = {}
        for row in rows:
            key = f"{row['table_schema']}.{row['table_name']}"
            if key not in schemas:
                schemas[key] = {
                    "schema": row['table_schema'],
                    "table": row['table_name'],
                    "columns": [],
                }
            schemas[key]["columns"].append({
                "name": row['column_name'],
                "type": row['data_type'],
                "nullable": row['is_nullable'] == "YES",
                "default": row['column_default'],
            })
        return schemas
    finally:
        conn.close()

# 5. Thực thi một câu lệnh SQL chỉ đọc (SELECT)
def execute_read_only_query(sql: str) -> Dict[str, Any]:
    """
    Kiểm tra Câu lệnh SQL có phải SELECT không. Nếu hợp lệ thì thực thi và trả về kết quả.
    Không cho phép các lệnh có thể thay đổi dữ liệu (INSERT, UPDATE, ...).
    Trả về: rows, số dòng, tên các cột.
    """
    normalized_sql = sql.strip().lower()
    write_operations = ['insert', 'update', 'delete', 'drop', 'create', 
                        'alter', 'truncate', 'grant', 'revoke']
    for op in write_operations:
        if normalized_sql.startswith(op):
            raise ValueError(
                f"Thao tác ghi '{op}' không được phép. "
                "Chỉ cho phép câu lệnh SELECT."
            )
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return {
                "rows": [dict(row) for row in rows],
                "row_count": len(rows),
                "columns": [desc[0] for desc in cur.description] if cur.description else [],
            }
    finally:
        conn.close()

# 6. Khởi tạo đối tượng MCP server cho Postgres backend
server = Server("postgres-mcp-server")

# 7. Định nghĩa endpoint trả về danh sách resource (các bảng) trong database
@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    Xử lý yêu cầu liệt kê resource: Lấy các bảng (schema) trong DB, trả về dạng MCP Resource.
    """
    global schema_cache
    schema_cache = fetch_table_schemas()
    resources = []
    for key, schema in schema_cache.items():
        resources.append(
            types.Resource(
                uri=f"postgres://schema/{key}",
                name=f"{schema['schema']}.{schema['table']}",
                description=f"Schema của bảng {schema['table']} trong schema {schema['schema']}",
                mimeType="application/json",
            )
        )
    return resources

# 8. Endpoint đọc chi tiết của một resource/table (xem cấu trúc bảng)
@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """
    Nhận vào một uri (postgres://schema/schema.table), trả về chi tiết metadata schema bảng đó ở dạng JSON.
    """
    if not uri.startswith("postgres://schema/"):
        raise ValueError(f"URI không hợp lệ: {uri}")
    table_key = uri.replace("postgres://schema/", "")
    if table_key not in schema_cache:
        schema_cache.update(fetch_table_schemas())
    if table_key not in schema_cache:
        raise ValueError(f"Không tìm thấy bảng: {table_key}")
    import json
    return json.dumps(schema_cache[table_key], indent=2, ensure_ascii=False)

# 9. Định nghĩa các công cụ (tools) hỗ trợ: Query SQL, liệt kê bảng, mô tả bảng chi tiết
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    Trả về danh sách các tool hỗ trợ thao tác với DB:
      - query_database: thực hiện SELECT SQL
      - list_tables: liệt kê bảng
      - describe_table: xem chi tiết cấu trúc bảng
    """
    return [
        types.Tool(
            name="query_database",
            description="Thực thi câu lệnh SQL SELECT để truy vấn dữ liệu từ database",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Câu lệnh SQL SELECT để thực thi",
                    },
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="list_tables",
            description="Liệt kê tất cả các bảng trong database",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="describe_table",
            description="Hiển thị cấu trúc chi tiết của một bảng",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Tên bảng (format: schema.table hoặc chỉ table)",
                    },
                },
                "required": ["table_name"],
            },
        ),
    ]

# 10. Xử lý gọi các công cụ nêu trên (tool call)
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Nhận tên công cụ được gọi và tham số. Xử lý các gọi tool:
    - Nếu tool là query_database: thực thi lệnh SQL SELECT.
    - Nếu tool là list_tables: trả về danh sách các bảng.
    - Nếu tool là describe_table: trả về chi tiết schema bảng đó.
    """
    import json
    if name == "query_database":
        sql = arguments.get("sql")
        if not sql:
            raise ValueError("Thiếu tham số 'sql'")
        try:
            result = execute_read_only_query(sql)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False),
                )
            ]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Lỗi: {str(e)}")]
    elif name == "list_tables":
        schemas = fetch_table_schemas()
        tables = [
            {"schema": s["schema"], "table": s["table"]} 
            for s in schemas.values()
        ]
        return [
            types.TextContent(
                type="text",
                text=json.dumps(tables, indent=2, ensure_ascii=False),
            )
        ]
    elif name == "describe_table":
        table_name = arguments.get("table_name")
        if not table_name:
            raise ValueError("Thiếu tham số 'table_name'")
        schemas = fetch_table_schemas()
        # Hỗ trợ nhận table_name là full (schema.table) hoặc chỉ tên table
        matching_key = None
        for key in schemas.keys():
            if key == table_name or key.endswith(f".{table_name}"):
                matching_key = key
                break
        if not matching_key:
            return [
                types.TextContent(
                    type="text",
                    text=f"Không tìm thấy bảng: {table_name}"
                )
            ]
        return [
            types.TextContent(
                type="text",
                text=json.dumps(schemas[matching_key], indent=2, ensure_ascii=False),
            )
        ]
    raise ValueError(f"Tool không xác định: {name}")

# 11. Định nghĩa các prompt (lời nhắc) thường dùng cho AI phân tích dữ liệu
@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    Trả về các prompt hữu ích cho việc phân tích và hỏi đáp về dữ liệu.
    Ví dụ: phân tích bảng, tìm giá trị trùng lặp, kiểm tra chất lượng dữ liệu, thống kê tổng quan bảng.
    """
    return [
        types.Prompt(
            name="analyze_table",
            description="Phân tích cấu trúc và dữ liệu mẫu của một bảng",
            arguments=[
                types.PromptArgument(
                    name="table_name",
                    description="Tên của bảng cần phân tích",
                    required=True,
                )
            ],
        ),
        types.Prompt(
            name="find_duplicates",
            description="Tìm các bản ghi trùng lặp trong một bảng dựa trên các cột chỉ định",
            arguments=[
                types.PromptArgument(
                    name="table_name",
                    description="Tên của bảng cần kiểm tra",
                    required=True,
                ),
                types.PromptArgument(
                    name="columns",
                    description="Các cột để kiểm tra trùng lặp (phân cách bằng dấu phẩy)",
                    required=True,
                ),
            ],
        ),
        types.Prompt(
            name="data_quality_check",
            description="Kiểm tra chất lượng dữ liệu: giá trị null, giá trị trống, outliers",
            arguments=[
                types.PromptArgument(
                    name="table_name",
                    description="Tên của bảng cần kiểm tra",
                    required=True,
                )
            ],
        ),
        types.Prompt(
            name="summarize_table",
            description="Tạo báo cáo tóm tắt về một bảng: số lượng records, phân bố dữ liệu",
            arguments=[
                types.PromptArgument(
                    name="table_name",
                    description="Tên của bảng cần tóm tắt",
                    required=True,
                )
            ],
        ),
    ]

# 12. Xử lý lấy mô tả (prompt) cụ thể dựa trên tên prompt và tham số (table_name, v.v.)
@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Trả về nội dung mẫu lời nhắc (prompt template, message) tuỳ từng loại phân tích.
    Dùng cho AI chatbot hoặc front-end truyền vào để sinh câu hỏi ra LLM.
    """
    if name == "analyze_table":
        table_name = arguments.get("table_name")
        return types.GetPromptResult(
            description=f"Phân tích bảng {table_name}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Vui lòng phân tích bảng '{table_name}':
1. Mô tả cấu trúc bảng (các cột, kiểu dữ liệu)
2. Lấy 10 dòng dữ liệu mẫu
3. Đếm tổng số records
4. Xác định các cột có giá trị null
5. Đưa ra các insights và recommendations"""
                    )
                )
            ],
        )
    elif name == "find_duplicates":
        table_name = arguments.get("table_name")
        columns = arguments.get("columns")
        return types.GetPromptResult(
            description=f"Tìm duplicates trong {table_name}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Tìm các bản ghi trùng lặp trong bảng '{table_name}' dựa trên cột(s): {columns}

Vui lòng:
1. Viết query để tìm các giá trị trùng lặp
2. Hiển thị số lượng bản ghi trùng lặp
3. Đưa ra examples của các bản ghi trùng lặp
4. Đề xuất cách xử lý"""
                    )
                )
            ],
        )
    elif name == "data_quality_check":
        table_name = arguments.get("table_name")
        return types.GetPromptResult(
            description=f"Kiểm tra chất lượng dữ liệu cho {table_name}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Thực hiện kiểm tra chất lượng dữ liệu cho bảng '{table_name}':

1. Kiểm tra các giá trị NULL trong từng cột
2. Tìm các giá trị trống hoặc khoảng trắng
3. Với các cột số: tìm outliers và phân bố giá trị
4. Với các cột text: kiểm tra độ dài và patterns
5. Tổng hợp báo cáo và đề xuất improvements"""
                    )
                )
            ],
        )
    elif name == "summarize_table":
        table_name = arguments.get("table_name")
        return types.GetPromptResult(
            description=f"Tóm tắt bảng {table_name}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Tạo báo cáo tóm tắt cho bảng '{table_name}':

1. Tổng số records
2. Danh sách các cột và kiểu dữ liệu
3. Phân bố dữ liệu cho các cột quan trọng
4. Identify primary key và foreign keys (nếu có)
5. Mô tả mục đích và cách sử dụng của bảng
6. Đề xuất các queries hữu ích"""
                    )
                )
            ],
        )
    raise ValueError(f"Prompt không xác định: {name}")

# 13. Hàm main bất đồng bộ – chạy server MCP giao tiếp qua stdin/stdout (CLI integration)
async def main():
    """
    Điểm vào chương trình: mở giao tiếp chuẩn stdin/stdout, khởi động MCP server, cung cấp các capabilities (tính năng) ra ngoài.
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="postgres-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

# 14. Chạy chương trình nếu là main module
if __name__ == "__main__":
    asyncio.run(main())
# Code này xây dựng một lớp giao diện tiêu chuẩn hóa giữa một database Postgres và hệ thống MCP để truy vấn, phân tích dữ liệu, sinh prompt AI tự động hóa.