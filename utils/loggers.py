import sys
import logging
from pathlib import Path


def config_logger(log_file: str = None, level=logging.INFO) -> None:
    """
    Configure Python's logging với một StreamHandler và (tuỳ chọn) FileHandler.
    Remove existing handlers before adding new ones.

    Args:
        log_file (str, optional): Path to a log file. Defaults to None.
        level ([type], optional): Logging level. Defaults to logging.INFO.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(filename=log_file))

    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
