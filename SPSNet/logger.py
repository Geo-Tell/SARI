import logging
import torch.distributed as dist

def setup_logger(logfile):
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank()!=0:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())
