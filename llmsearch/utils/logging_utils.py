"""
Logging Related Utils
"""
import os
import logging
from logging import (
    DEBUG,
    INFO,
    WARNING,
)
import warnings

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
log_level_strs = log_levels.keys()

default_logging_level = "warning"


def _get_lib_name() -> str:
    """Get Library Name"""
    return __name__.split(".", maxsplit=1)[0]


def _get_lib_default_logging_level() -> int:
    """
    Get Default logging level, overridden by env variable `LLMSEARCH_VERBOSITY`
    the env var can take one of the following values - ["debug", "info", "warning", "error", "critical"]
    """
    env_var = "LLMSEARCH_VERBOSITY"
    env_value = os.getenv(env_var, None)
    if env_value is not None:
        if env_value not in log_level_strs:
            warnings.warn(
                f"Env variable - {env_var} should be one of - {list(log_levels.keys())}"
            )
            return log_levels[default_logging_level]
        return log_levels[env_value]
    return log_levels[default_logging_level]


def _get_library_root_logger() -> logging.Logger:
    """Retrives library root logger"""
    return logging.getLogger(_get_lib_name())


def _configure_library_root_logger() -> logging.Logger:
    """Configure library root logger"""
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    library_root_logger = logging.getLogger(_get_lib_name())
    library_root_logger.addHandler(stream_handler)
    library_root_logger.setLevel(_get_lib_default_logging_level())
    library_root_logger.propagate = False
    return library_root_logger


def get_logger(name: str) -> logging.Logger:
    """Gets a logger if name is supplied or returns the library root logger"""
    if name:
        return logging.getLogger(name)
    return _get_library_root_logger()


def set_verbosity(verbosity: int):
    """Sets the logging verbosity of the library root logger"""
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Sets the logging verbosity of the library root logger to info"""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Sets the logging verbosity of the library root logger to warning"""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Sets the logging verbosity of the library root logger to debug"""
    return set_verbosity(DEBUG)


# setup library root logger
_configure_library_root_logger()
