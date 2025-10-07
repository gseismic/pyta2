from loguru import logger as _logger

__all__ = ['logger']

logger = _logger.bind(name='pyta2')