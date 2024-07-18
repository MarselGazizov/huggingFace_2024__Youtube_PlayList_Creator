import logging


# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def get_logger(name, level=logging.NOTSET):
    logging.basicConfig()
    log = logging.getLogger(name)
    log.setLevel(level)
    return log
