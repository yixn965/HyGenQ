import datetime
import logging
import os


def setup_logger(output_dir):
    """Create the application logger in a timestamped output directory."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outpath = os.path.join(output_dir, timestamp)
    os.makedirs(outpath, exist_ok=True)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_file = os.path.join(outpath, "main.log")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger, outpath


# Logs are stored in timestamped subdirectories under the existing output path.
output_dir = "output"
logger, outpath = setup_logger(output_dir)
