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

default_logging_level = "warning"


def _get_lib_name():
    return __name__.split(".")[0]


def _get_lib_default_logging_level():
    env_var = "LLMSEARCH_VERBOSITY"
    env_value = os.getenv(env_var, None)
    if env_value is not None:
        if env_value not in log_levels.keys():
            warnings.warn(
                f"Env variable - {env_var} should be one of - {list(log_levels.keys())}"
            )
            return log_levels[default_logging_level]
        return log_levels[env_value]
    return log_levels[default_logging_level]


def _get_library_root_logger():
    return logging.getLogger(_get_lib_name())


def _configure_library_root_logger():
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


def get_logger(name: str):
    if name:
        return logging.getLogger(name)
    return _get_library_root_logger()


def set_verbosity(verbosity: int):
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    return set_verbosity(INFO)


def set_verbosity_warning():
    return set_verbosity(WARNING)


def set_verbosity_debug():
    return set_verbosity(DEBUG)


_configure_library_root_logger()
