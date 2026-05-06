import os
from os.path import basename
import logging

log_level = os.getenv(
    "LOG_LEVEL", "INFO"
).upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
