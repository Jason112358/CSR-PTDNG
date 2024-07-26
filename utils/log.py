import os
import logging

def get_logger(filename='output/log/temp.log', verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level_dict[verbosity])

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, "w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
 
    return logger

