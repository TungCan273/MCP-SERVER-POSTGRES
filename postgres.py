import asyncio
import os
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Cấu hình kết nối database từ biến môi trường
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "postgres"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# Cache cho schemas
schema_cache: Dict[str, Any] = {}

def get_db_connection():
    """Tạo kết nối đến PostgreSQL database"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def fetch_table_schemas() -> Dict[str, Any]:
    """Lấy schema của tất cả các bảng trong database"""
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

def execute_read_only_query(sql: str) -> Dict[str, Any]:
    """Thực thi câu lệnh SQL chỉ đọc"""
    # Kiểm tra câu lệnh chỉ được phép SELECT
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

# Khởi tạo MCP server
server = Server("postgres-mcp-server")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """Liệt kê tất cả table schemas như resources"""
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

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Đọc thông tin chi tiết của một table schema"""
    if not uri.startswith("postgres://schema/"):
        raise ValueError(f"URI không hợp lệ: {uri}")
    
    table_key = uri.replace("postgres://schema/", "")
    
    if table_key not in schema_cache:
        schema_cache.update(fetch_table_schemas())
    
    if table_key not in schema_cache:
        raise ValueError(f"Không tìm thấy bảng: {table_key}")
    
    import json
    return json.dumps(schema_cache[table_key], indent=2, ensure_ascii=False)

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Liệt kê các tools có sẵn"""
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

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Xử lý các tool calls"""
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
        
        # Tìm bảng (có thể có hoặc không có schema prefix)
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

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """Liệt kê các prompts cho phân tích dữ liệu thường dùng"""
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

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Xử lý prompt requests"""
    
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

async def main():
    """Main entry point"""
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

if __name__ == "__main__":
    asyncio.run(main())