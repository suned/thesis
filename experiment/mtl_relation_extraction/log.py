import colorlog

from .io import arguments

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)s]%(reset)s\t[%(asctime)s]\t%(message)s",
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'cyan',
        'ERROR': 'red',
    },
    secondary_log_colors={},
    style='%',
    datefmt='%d/%m/%Y %H:%M:%S'
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)
logger = colorlog.getLogger()
logger.setLevel(arguments.log_level)
logger.addHandler(handler)


def info(message, *args, **kwargs):
    logger.info(message, *args, **kwargs)


def debug(message, *args, **kwargs):
    logger.debug(message, *args, **kwargs)


def error(message, *args, **kwargs):
    logger.error(message, *args, **kwargs)
