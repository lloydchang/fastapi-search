# File: backend/fastapi/utils/logger.py

import logging
import os
import sys
from colorlog import ColoredFormatter

# Create and configure the main logger
logger = logging.getLogger(__name__)

# Set up a basic log format with colorized levels using `colorlog`
formatter = ColoredFormatter(
    fmt="%(asctime)s - %(filename)s:%(lineno)d - %(log_color)s%(levelname)s%(reset)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Match timestamp format
)

# Console handler setup
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Set log level from environment variable, defaulting to 'DEBUG'
log_level = os.getenv("PYTHON_LOGGER_LEVEL", "DEBUG").upper()
logger.setLevel(getattr(logging, log_level, logging.CRITICAL))

# Log the configured log level for debugging
logger.debug(f"Logger initialized with level: {log_level}")
