import logging
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
logger.propagate = False 

# Format logger
formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)s] [trace_id=%(trace_id)s] [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
class TraceIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        return True

if not logger.handlers:
    file_handler = logging.FileHandler(
        f"{LOG_DIR}/app.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(TraceIdFilter())

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(TraceIdFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)