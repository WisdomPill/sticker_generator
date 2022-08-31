import logging

logger = logging.getLogger()


def set_up_logger() -> None:
    # use no handler, for stream logger to aws monitoring
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    log_format = "[%(levelname)s]%(module)s:%(lineno)d -> %(message)s"

    logging.basicConfig(level=logging.INFO, format=log_format)
