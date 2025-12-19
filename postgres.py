import asyncio
import os
import sys
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from langchain_core.messages import HumanMessage
import mcp.server.stdio
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

from log_util import logger

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5433")),
    "database": os.getenv("POSTGRES_DB", "postgres"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# ============================================================================
# SCHEMA CACHE WITH TTL
# ============================================================================

def json_safe(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
class SchemaCache:
    """Cache với TTL (Time To Live) cho schema database"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Any] = {}
        self.last_refresh: Optional[datetime] = None
        self.ttl = timedelta(seconds=ttl_seconds)
        logger.info(f"Schema cache initialized with TTL: {ttl_seconds}s")
    
    def is_expired(self) -> bool:
        """Kiểm tra cache đã hết hạn chưa"""
        if not self.last_refresh:
            return True
        return datetime.now() - self.last_refresh > self.ttl
    
    def get(self) -> Dict[str, Any]:
        """Lấy cache, tự động refresh nếu hết hạn"""
        if self.is_expired():
            logger.info("Schema cache expired, refreshing...")
            self.refresh()
        return self.cache
    
    def refresh(self):
        """Refresh cache thủ công"""
        try:
            self.cache = fetch_table_schemas()
            self.last_refresh = datetime.now()
            logger.info(f"Schema cache refreshed: {len(self.cache)} tables")
        except Exception as e:
            logger.error(f"Failed to refresh schema cache: {e}")
            raise
    
    def clear(self):
        """Xóa cache"""
        self.cache = {}
        self.last_refresh = None
        logger.info("Schema cache cleared")

schema_cache = SchemaCache(ttl_seconds=300)

# ============================================================================
# DATABASE CONNECTION POOL
# ============================================================================
connection_pool: Optional[pool.SimpleConnectionPool] = None

def init_connection_pool():
    """Khởi tạo connection pool"""
    global connection_pool
    try:
        connection_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **DB_CONFIG,
            cursor_factory=RealDictCursor
        )
        logger.info("✅ Database connection pool initialized")
    except psycopg2.Error as e:
        logger.error(f"❌ Failed to initialize connection pool: {e}")
        connection_pool = None
        raise ConnectionError(f"Không thể khởi tạo connection pool: {e}")

def get_db_connection():
    """Lấy connection từ pool"""
    global connection_pool
    
    if connection_pool is None:
        init_connection_pool()
    
    try:
        conn = connection_pool.getconn()
        if conn.closed:
            connection_pool.putconn(conn)
            conn = connection_pool.getconn()
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to get connection: {e}")
        raise ConnectionError(f"Không thể lấy connection: {e}")

def return_connection(conn):
    """Trả connection về pool"""
    if connection_pool and conn:
        connection_pool.putconn(conn)

# ============================================================================
# SQL VALIDATION & SECURITY
# ============================================================================
def validate_sql_query(sql: str) -> bool:
    """
    Validate SQL query để đảm bảo an toàn:
    - Chỉ cho phép SELECT
    - Không cho phép multiple statements
    - Không cho phép các pattern nguy hiểm
    """
    if not sql or not sql.strip():
        raise ValueError("SQL query không được rỗng")
    
    # Giới hạn độ dài
    if len(sql) > 10000:
        raise ValueError("SQL query quá dài (max 10000 ký tự)")
    
    # Normalize
    normalized_sql = sql.strip().lower()
    
    # 1. Chỉ cho phép SELECT
    if not normalized_sql.startswith('select'):
        # Kiểm tra các lệnh ghi
        write_operations = [
            'insert', 'update', 'delete', 'drop', 'create',
            'alter', 'truncate', 'grant', 'revoke', 'exec',
            'execute', 'call'
        ]
        for op in write_operations:
            if normalized_sql.startswith(op):
                raise ValueError(
                    f"Thao tác '{op}' không được phép. Chỉ cho phép câu lệnh SELECT."
                )
    
    # 2. Không cho phép multiple statements
    # Loại bỏ các semicolon trong string literals trước khi check
    sql_without_strings = re.sub(r"'[^']*'", "", sql)
    if sql_without_strings.count(';') > 1:
        raise ValueError("Không cho phép nhiều câu lệnh SQL (multiple statements)")
    
    # 3. Kiểm tra các pattern nguy hiểm
    dangerous_patterns = [
        r'--',           # SQL comment
        r'/\*',          # Block comment start
        r'\*/',          # Block comment end
        r'xp_',          # SQL Server extended procedures
        r'sp_',          # SQL Server stored procedures
        r'\binto\s+outfile\b',  # MySQL file operations
        r'\bload_file\b',       # MySQL file operations
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, normalized_sql, re.IGNORECASE):
            raise ValueError(f"Pattern không an toàn được phát hiện: {pattern}")
    
    return True

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================
def fetch_table_schemas() -> Dict[str, Any]:
    """
    Truy vấn thông tin schema các bảng trong database.
    Returns: dict chứa thông tin từng bảng và danh sách các cột.
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
    
    conn = None
    try:
        conn = get_db_connection()
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
        
        logger.info(f"Fetched schema for {len(schemas)} tables")
        return schemas
        
    except psycopg2.Error as e:
        logger.error(f"Database error fetching schemas: {e}")
        raise
    finally:
        if conn:
            return_connection(conn)

def execute_read_only_query(sql: str) -> Dict[str, Any]:
    """
    Thực thi câu lệnh SQL SELECT (chỉ đọc).
    Validate query trước khi thực thi để đảm bảo an toàn.
    """
    # Validate SQL
    validate_sql_query(sql)

    def serialize_row(row: dict) -> dict:
        return {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in row.items()
        }
    
    conn = None
    try:
        conn = get_db_connection()
        logger.info(f"Executing query: {sql[:100]}...")
        
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            
            result = {
                "rows": [serialize_row(dict(row)) for row in rows],
                "row_count": len(rows),
                "columns": [desc[0] for desc in cur.description] if cur.description else [],
            }
            
            logger.info(f"Query returned {result['row_count']} rows")
            return result
            
    except psycopg2.Error as e:
        logger.error(f"Database error executing query: {e}")
        raise ValueError(f"Lỗi thực thi query: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        if conn:
            return_connection(conn)

# ============================================================================
# MCP SERVER SETUP
# ============================================================================
server = Server("postgres-mcp-server")

# ============================================================================
# RESOURCES HANDLERS
# ============================================================================
@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """Liệt kê các bảng trong database dưới dạng resources"""
    try:
        schemas = schema_cache.get()
        resources = []
        
        for key, schema in schemas.items():
            resources.append(
                types.Resource(
                    uri=f"postgres://schema/{key}",
                    name=f"{schema['schema']}.{schema['table']}",
                    description=f"Schema của bảng {schema['table']} trong schema {schema['schema']}",
                    mimeType="application/json",
                )
            )
        
        logger.info(f"Listed {len(resources)} resources")
        return resources
        
    except Exception as e:
        logger.error(f"Failed to list resources: {e}")
        raise

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Đọc chi tiết schema của một bảng"""
    try:
        if not uri.startswith("postgres://schema/"):
            raise ValueError(f"URI không hợp lệ: {uri}")
        
        table_key = uri.replace("postgres://schema/", "")
        schemas = schema_cache.get()
        
        if table_key not in schemas:
            raise ValueError(f"Không tìm thấy bảng: {table_key}")
        
        import json
        result = json.dumps(schemas[table_key], indent=2, ensure_ascii=False)
        logger.info(f"Read resource: {table_key}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to read resource: {e}")
        raise

# ============================================================================
# TOOLS HANDLERS
# ============================================================================
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Danh sách các tools hỗ trợ"""
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
    
    try:
        logger.info(f"Tool called: {name}")
        
        if name == "query_database":
            if not arguments or "sql" not in arguments:
                raise ValueError("Thiếu tham số 'sql'")
            
            elif "sql" in arguments:
                sql = arguments["sql"].strip()
            elif "query" in arguments:
                sql = arguments["query"].strip()
            elif "query_database" in arguments:
                sql = arguments["query_database"]

            result = execute_read_only_query(sql)
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, 
                        indent=2, 
                        ensure_ascii=False, 
                        default=json_safe
                    ),
                )
            ]
        
        elif name == "list_tables":
            schemas = schema_cache.get()
            tables = [
                {"schema": s["schema"], "table": s["table"]} 
                for s in schemas.values()
            ]
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(tables, 
                        indent=2, 
                        ensure_ascii=False, 
                        default=json_safe
                    ),
                )
            ]
        
        elif name == "describe_table":
            if not arguments or "table_name" not in arguments:
                raise ValueError("Thiếu tham số 'table_name'")
            
            table_name = arguments["table_name"].strip()
            schemas = schema_cache.get()
            
            # Tìm bảng matching
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
                    text=json.dumps(schemas[matching_key], 
                        indent=2, 
                        ensure_ascii=False, 
                        default=json_safe
                    ),
                )
            ]
        
        elif name == "update_sentiment":
            # return await update_sentiment()
            pass

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
# PROMPTS HANDLERS
# ============================================================================
@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """Danh sách các prompt templates"""
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
            description="Tìm các bản ghi trùng lặp trong một bảng",
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
            description="Kiểm tra chất lượng dữ liệu của một bảng",
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
            description="Tạo báo cáo tóm tắt về một bảng",
            arguments=[
                types.PromptArgument(
                    name="table_name",
                    description="Tên của bảng cần tóm tắt",
                    required=True,
                )
            ],
        ),
        types.Tool(
            name="update_sentiment",
            description="Update sentiment result back to database",
            inputSchema={
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "id": {"type": "integer"},
                    "sentiment": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["table", "id", "sentiment"]
            }
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Trả về nội dung của prompt template"""
    
    if not arguments:
        arguments = {}
    
    if name == "analyze_table":
        table_name = arguments.get("table_name", "")
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
        table_name = arguments.get("table_name", "")
        columns = arguments.get("columns", "")
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
        table_name = arguments.get("table_name", "")
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
        table_name = arguments.get("table_name", "")
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
    elif name == "update_sentiment":
        table_name = arguments.get("table_name", "")
        return types.GetPromptResult(
            description=f"Update sentiment result back to database",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Update dữ liệu lại cho bảng '{table_name} ở trường thông tin sentiment, lưu ý':

1. Giữ nguyên các trường khác, chỉ update sentiment
2. kiểu dữ liệu là số nguyên, với nguyên tắc 'sentiment: negative|positive|neutral, label: 0|1|2'
3. không update các bảng khác, chỉ được sử dụng bảng đang chỉ định và update trường sentiment
4. update theo primary key vs foreign keys (nếu có) - bắt buộc phải so sánh key id trước để tránh nhầm lẫn bản ghi
"""
                    )
                )
            ],
        )
    raise ValueError(f"Prompt không xác định: {name}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
async def main():
    """Điểm vào chương trình: khởi động MCP server"""
    try:
        logger.info("Starting postgres-mcp-server...")
        
        # Initialize connection pool
        init_connection_pool()
        
        # Warm up schema cache
        schema_cache.refresh()
        
        # Run MCP server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="postgres-mcp-server",
                    server_version="2.0.0",
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
    finally:
        # Cleanup
        if connection_pool:
            connection_pool.closeall()
            logger.info("Connection pool closed")

if __name__ == "__main__":
    asyncio.run(main())