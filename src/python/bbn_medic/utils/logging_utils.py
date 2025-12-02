import logging


def setup_logger():
    logging.basicConfig(format='%(asctime)s {P%(process)d:%(module)s:%(lineno)d} %(levelname)-8s %(message)s',
                        level=logging.INFO)