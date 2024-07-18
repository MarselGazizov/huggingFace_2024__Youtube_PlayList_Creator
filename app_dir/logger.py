import logging


# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def get_logger(name, level):
    log = logging.getLogger(name)
    log.setLevel(level)
    return log
