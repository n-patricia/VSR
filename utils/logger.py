import datetime
import logging
import time

initialized_logger = {}


def get_logger(logger_name='vsr', log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    # logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    # logger.propagate = False
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(format_str))
    # logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    initialized_logger[logger_name] = True
    return logger


def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger
