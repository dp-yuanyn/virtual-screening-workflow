
def get_logger():
    import logging
    import sys

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(levelname)s][%(asctime)s][%(filename)s %(lineno)d] %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger