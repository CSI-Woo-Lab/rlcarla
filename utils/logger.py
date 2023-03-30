import abc
import logging
import sys
from pathlib import Path
from typing import Dict, Union


class Logging(abc.ABC):
    _initialized_: bool = False
    _loggers_: Dict[str, logging.Logger] = {}

    @classmethod
    def setup(
        cls,
        filepath: Union[str, Path] = "outputs.log",
        level=logging.DEBUG,
        formatter="(%(asctime)s) [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ):
        cls.__filepath = Path(filepath) if isinstance(filepath, str) else filepath
        cls.__level = level
        cls.__format = formatter
        cls.__datefmt = datefmt

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

        for logger in cls._loggers_.values():
            cls.set_logger(logger)

        cls._initialized_ = True

    @classmethod
    def get_logger(cls, name: str):
        logger = logging.getLogger(name)
        if cls._initialized_:
            cls.set_logger(logger)

        cls._loggers_[name] = logger
        return logger

    @classmethod
    def set_logger(cls, logger: logging.Logger):
        logger.setLevel(cls.__level)

        for handler in logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stdout
            ):
                logger.removeHandler(handler)

        handler = logging.FileHandler(cls.__filepath)
        handler.setLevel(cls.__level)

        formatter = logging.Formatter(cls.__format, datefmt=cls.__datefmt)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
