from loguru import logger

logger.disable("stroke_segmentor")

from stroke_segmentor.inferer import Inferer  # noqa: F401
