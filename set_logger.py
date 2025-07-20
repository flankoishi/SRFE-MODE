import logging
import torch.distributed as dist
import os


def get_logger(log_name, log_level, log_file=None, file_mode='a'):
    '''
    获取由logging库提供的logger
    log_name:用于标识log的名称
    log_level:打印等级，例如logging.INFO  logging.DEBUG  logging.ERROR  logging.CRITICAL
    log_file:输出日志的文件地址
    file_mode:输出日志的文件模式，a为追加，w为覆盖等
    '''

    logger = logging.getLogger(log_name)
    logger.propagate = False  # 阻止日志消息传递给父级logger

    # 判断是否为是多卡运行
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    handlers = []

    # 流处理器
    stream_handler = logging.StreamHandler()  # 用于将日志消息输出到控制台或者标准输出流
    handlers.append(stream_handler)

    if rank == 0 and log_file is not None:
        # 文件处理器
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file, file_mode)  # file_mode为'a'则追加，为'w'则覆盖
        handlers.append(file_handler)

    # 格式化器
    plain_formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    )

    formatter = plain_formatter

    # 处理器加格式
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger


if __name__ == "__main__":
    log_name = "Debug"
    log_level = logging.INFO  # DEBUG INFO WARNING ERROR CRITICAL
    logging.DEBUG
    logging.ERROR
    logging.CRITICAL
    file_mode = 'w'
    log_file = './log/train.log'

    logger = get_logger(log_name, log_level, log_file, file_mode='w')
    logger.info("=> Loading config ...")
    logger.info("=> Start train!")
    pass