"""Speech performance logging without recording conversation text or audio."""

import logging


logger = logging.getLogger("airis.speech")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [speech] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
