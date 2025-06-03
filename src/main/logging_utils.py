"""
Author: <WU Xinyan>
日志工具模块 - 提供全局日志功能
"""
import os
import logging
from datetime import datetime

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 默认日志级别
DEFAULT_LOG_LEVEL = 'INFO'

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logger(name=None, level=None, log_to_file=True, log_dir='logs'):

    if level is None:
        level = os.environ.get('LOG_LEVEL', DEFAULT_LOG_LEVEL)

    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_to_file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'turbine_fault_{timestamp}.log')

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        logger.info(f"日志文件已创建: {log_file}")
    return logger


def get_logger(name=None):
    logger = logging.getLogger(name)

    if not logger.handlers:
        return setup_logger(name)

    return logger
