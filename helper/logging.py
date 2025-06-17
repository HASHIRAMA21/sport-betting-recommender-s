import logging
import warnings
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "./logs") -> logging.Logger:
    """Configure le système de logging."""

    Path(log_dir).mkdir(exist_ok=True)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logger = logging.getLogger("SportsRecommendation")
    logger.setLevel(getattr(logging, log_level.upper()))

    file_handler = logging.FileHandler(
        Path(log_dir) / f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()
warnings.filterwarnings('ignore')
