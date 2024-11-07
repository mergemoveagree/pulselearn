"""Logging handler for the quantumlearning project.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
from datetime import datetime
import getpass
import logging
import os
from pathlib import Path

ROOT_LOGGER_NAME = "quantumlearning"
LOG_FORMAT_STR = (
    "[%(asctime)s][%(name)s:%(funcName)s:%(lineno)d][%(levelname)s] %(message)s"
)
LOG_FORMATTER = logging.Formatter(LOG_FORMAT_STR, datefmt="%Y-%m-%d %H:%M:%S")

run_root: Path


def init_logging(relative_root: Path | None = None, log_to_stream: bool = True) -> None:
    """Set up handlers for logging to files and console.

    :param relative_root: Base directory for log files. Defaults to the run root.
    :param log_to_stream: Whether to log to the console. Defaults to True
    """
    make_run_root()

    project_root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    project_root_logger.handlers.clear()

    if relative_root is None:
        relative_root = run_root

    info_fh = logging.FileHandler(relative_root / "quantumlearning.log")
    info_fh.setFormatter(LOG_FORMATTER)
    info_fh.setLevel(logging.INFO)
    project_root_logger.addHandler(info_fh)

    debug_fh = logging.FileHandler(relative_root / "quantumlearning_debug.log")
    debug_fh.setFormatter(LOG_FORMATTER)
    debug_fh.setLevel(logging.DEBUG)
    project_root_logger.addHandler(debug_fh)

    if log_to_stream:
        console_sh = logging.StreamHandler()
        console_sh.setFormatter(LOG_FORMATTER)
        console_sh.setLevel(logging.INFO)
        project_root_logger.addHandler(console_sh)

    project_root_logger.setLevel(logging.DEBUG)
    project_root_logger.propagate = False

    project_root_logger.info("Logging to %s", relative_root)


def make_run_root() -> None:
    """Set the run root for the project.

    If there is a environment variable called "PULSE_RUN_COLLECTION", the run_root will be created
    in the specified path in the environment variable. Otherwise, a run root will be generated that
    is located in a folder called "run_data" in the directory where the process was started.

    The run root is a directory that contains all files related to a single run of the script. As
    such, if the run_root is already set in the module, this function does nothing.
    """
    global run_root
    try:
        if run_root:
            return
    except NameError:
        # We get here because "run_root" hasn't been defined as anything in the module yet
        # If run root hasn't been instantiated yet, then we'll continue on to the code below
        pass
    timestamp = datetime.now().strftime("%Y.%m.%d_%H%M%S")
    run_root_name = f"{getpass.getuser()}_{timestamp}"
    if "PULSE_RUN_COLLECTION" in os.environ:
        run_root = (
            Path(os.environ["PULSE_RUN_COLLECTION"]).expanduser().resolve()
            / run_root_name
        )
    else:
        run_root = Path(os.getcwd()).expanduser().resolve() / "run_data" / run_root_name

    run_root.mkdir(parents=True)
