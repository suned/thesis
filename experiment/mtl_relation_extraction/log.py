import logging

import colorlog

from ..io import arguments


class MultiLineColorFormatter(colorlog.ColoredFormatter):
    def format(self, record):
        indent = " " * 32
        message = colorlog.ColoredFormatter.format(self, record)
        if message.count("\n") == 0:
            return message
        lines = message.split("\n")
        all_but_first = lines[1:]
        all_but_first = "\n".join(
            indent + line for line in all_but_first
        )
        return lines[0] + "\n" + all_but_first


formatter = MultiLineColorFormatter(
    "%(log_color)s[%(levelname)s]%(reset)s\t[%(asctime)s]\t%(message)s",
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'cyan',
        'ERROR': 'red',
        'WARNING': 'yellow'
    },
    secondary_log_colors={},
    style='%',
    datefmt='%d/%m/%Y %H:%M:%S'
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)
logger = colorlog.getLogger()
logger.setLevel(
    arguments.log_level
    if arguments.log_level is not None
    else logging.DEBUG
)
logger.addHandler(handler)


def info(message, *args, **kwargs):
    logger.info(message, *args, **kwargs)


def debug(message, *args, **kwargs):
    logger.debug(message, *args, **kwargs)


def error(message, *args, **kwargs):
    logger.error(message, *args, **kwargs)


def warning(message, *args, **kwargs):
    logging.warning(message, *args, **kwargs)
