"""Useful functions for running scripts.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
import argparse
import logging
from pathlib import Path
from typing import Any, cast
import yaml

from lib.log.log import ROOT_LOGGER_NAME
from lib.qiskit_dynamics.qd_simulator import QiskitDynamicsSimulator
from ql_simulator.abstract_dynamics_simulator import AbstractDynamicsSimulator

log = logging.getLogger(ROOT_LOGGER_NAME + "." + __name__)


def get_params() -> dict[str, Any]:
    """Get the testfile specified in the script arguments.

    Raises a ValueError if a "--testfile=" argument is not passed when running the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testfile",
        help="Testfile containing parameters to run the script.",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    testfile_path = args.testfile.expanduser().resolve()
    if not testfile_path.exists() or testfile_path.is_dir():
        raise ValueError(f"Couldn't find testfile: {args.testfile}")
    with testfile_path.open("r") as testfile:
        params: dict[str, Any] = cast(Any, yaml).load(testfile, yaml.FullLoader)
    return params


def log_marker(msg: str) -> None:
    """Log a message sandwiched by separators made of equal signs."""
    log.info(
        "===================================================================================="
    )
    log.info(msg)
    log.info(
        "===================================================================================="
    )


def parse_ql_backend(backend_string: str) -> type[AbstractDynamicsSimulator]:
    """Return a AbstractDynamicsSimulator based on string input from the user."""
    if backend_string == "qiskit_dynamics":
        return QiskitDynamicsSimulator
    else:
        raise ValueError(f"{backend_string} is not a valid backend option")
