# pyright: reportIncompatibleMethodOverride=false
"""Components to a pulse model that leverages Qiskit Dynamics.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
import logging
import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.parameter import Parameter

from lib.log.log import ROOT_LOGGER_NAME
from lib.pytorch.functions.dynamics_simulator_function import DynamicsSimulatorFunction
from ql_simulator.abstract_dynamics_simulator import AbstractDynamicsSimulator
from utils.pulse_utils import wide_sigmoid

log = logging.getLogger(ROOT_LOGGER_NAME + "." + __name__)


class DynamicsSimulatorLayer(torch.nn.Module):
    """Container for simulation and parameter storage."""

    def __init__(
        self,
        wrapper: AbstractDynamicsSimulator,
        pulse_array: Sequence[int],
        should_multiprocess: bool,
        initial_parameters: dict[str, list[list[float]]] | None = None,
    ) -> None:
        """Initialize a DynamicsSimulatorLayer with initial parameters.

        If the nested list structure of the values of the initial_parameters dictionary doesn't
        match the pulse_array parameter, a ValueError is thrown. If the length of the pulse_array
        tuple is not equal to the number of qubits that the wrapper supports, a ValueError is
        thrown.
        Note: the moduli should be the values that are AFTER the sigmoid (i.e. between 1 and -1).

        :param wrapper: The simulator to run the pulses.
        :param pulse_array: Tuple of ints that determine the pulses played on the simulator. The
            length of the list is the number of drive channels and the i-th element of the tuple is
            the number of pulses to be played on the i-th drive channel.
        :param should_multiprocess: Whether to multiprocess during the backward pass.
        :param initial_parameters: Dictionary of initial parameters of the layer/model. The keys
            should be one of: [durations, moduli, arguments, sigmas, betas, phases]. The values
            are nested lists where each inner list represents a drive channel and the i-th element
            of the inner lists are the pulse parameters for the i-th pulse.
        """
        super().__init__()
        if initial_parameters is None:
            initial_parameters = dict()
        if wrapper.num_qubits != len(pulse_array):
            raise ValueError(
                "Length of pulse_array is not equal to number of qubits in the "
                "simulator"
            )
        self.wrapper = wrapper
        self.should_multiprocess = should_multiprocess
        # ("parameter name", base value, half the range in which the value can be initialized)
        generic_parameters = [
            ("durations", 65, 15),
            ("moduli", 1.73, 0.7),
            ("arguments", 0, np.pi / 4),
            ("sigmas", 80, 10),
            ("betas", 0.7, 0.1),
            ("phases", 0, np.pi / 4),
        ]

        # Setting up RNG
        seed = os.environ.get("QUANTUMLEARNING_RANDOM_SEED", None)
        if seed is not None:
            log.debug("Found seed for initial parameters RNG: %s", seed)
        seed = int(seed) if seed is not None else None
        rng = np.random.default_rng(seed)

        for param_name, base_value, span in generic_parameters:
            if param_name in initial_parameters:
                param_array = initial_parameters[param_name]
                if len(param_array) != len(pulse_array):
                    raise ValueError(
                        f"Number of rows for the {param_name} initial parameter is "
                        "not equal to the length of pulse_array"
                    )
                for i, row in enumerate(param_array):
                    num_pulses = pulse_array[i]
                    if len(row) != num_pulses:
                        raise ValueError(
                            f"Not enough params in row {i} of the {param_name} "
                            "initial parameter"
                        )
                    direction = np.power(
                        -1, np.round(rng.random(num_pulses))  # pyright: ignore
                    )
                    noise = span * direction * rng.random(num_pulses)
                    # TODO Pass in device on which to create tensor
                    setattr(
                        self, f"{param_name}_{i}", Parameter(torch.tensor(row + noise))
                    )
            else:
                for param_name, base_value, span in generic_parameters:
                    for i, num_pulses in enumerate(pulse_array):
                        direction = np.power(
                            -1, np.round(rng.random(num_pulses))  # pyright: ignore
                        )
                        noise = span * direction * rng.random(num_pulses)
                        # Note: since noise is an ndarray, base_value is broadcasted
                        rand_param_array = torch.tensor(base_value + noise)
                        setattr(self, f"{param_name}_{i}", Parameter(rand_param_array))

    def forward(
        self, initial_comps: torch.Tensor, target_states: torch.Tensor
    ) -> torch.Tensor:
        """Pass the input and goal states to the differentiable pulse simulator.

        :param initial_comps: Tensor of components of coefficients (moduli and arguments)
            (i.e. [[a_1, b_1, phase_b_1, c_1, phase_c_1, ...], [[a_2, b_2, phase_b_2, ...]]]).
        :param target_states: Tensor of desired states as row statevectors.
        """
        fids: torch.Tensor = DynamicsSimulatorFunction.apply(  # pyright: ignore
            self.wrapper,
            initial_comps,
            target_states,
            self.should_multiprocess,
            self.wrapper.num_qubits,  # TODO: Find better way to determine number of drive pulses
            *self.get_args(),
        )
        return fids

    # TODO Look into caching the return value from this function
    # Since it returns a tuple of *objects*, the references should remain the same
    def get_args(self) -> tuple[Parameter, ...]:
        """Return the tuple of args for the DynamicsSimulatorFunction forward pass."""
        args_list: list[Parameter] = list()
        for i in range(self.wrapper.num_qubits):
            args_list.append(getattr(self, f"durations_{i}"))
            args_list.append(getattr(self, f"moduli_{i}"))
            args_list.append(getattr(self, f"arguments_{i}"))
            args_list.append(getattr(self, f"sigmas_{i}"))
            args_list.append(getattr(self, f"betas_{i}"))
            args_list.append(getattr(self, f"phases_{i}"))
        return tuple(args_list)


class DynamicsSimulatorNet(torch.nn.Module):
    """Wrapper neural net for the DynamicsSimulatorLayer."""

    def __init__(
        self,
        wrapper: AbstractDynamicsSimulator,
        pulse_array: Sequence[int],
        should_multiprocess: bool,
        initial_parameters: dict[str, list[list[float]]] | None = None,
    ) -> None:
        """Initialize a QiskitDynamicsNet."""
        super().__init__()
        self.qd_layer = DynamicsSimulatorLayer(
            wrapper, pulse_array, should_multiprocess, initial_parameters
        )

    def forward(
        self, initial_states: torch.Tensor, target_states: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate initial states and return the fidelity with the target states."""
        x: torch.Tensor = self.qd_layer(initial_states, target_states)
        return x

    def log_params_to_stream(self) -> None:
        """Log the current model parameters to the console."""
        log.info("Pulse model parameters:")
        for name, param in self.named_parameters():
            report = param.detach().numpy()
            log.info("    %s %s", name, report)
            if "moduli" in name:
                log.info("        Effective modulus: %s", wide_sigmoid(report))

    def log_params_to_tensorboard(self, writer: SummaryWriter, x_val: int) -> None:
        """Log the current model parameters to Tensorboard."""
        for name, param in self.named_parameters():
            short_name = name.split(".")[-1]
            report = param.detach().numpy()
            # report should be a 1D array
            for i, element in enumerate(report):
                writer.add_scalar(  # pyright: ignore
                    f"Parameter/{short_name}_{i}", element, x_val
                )
                if "moduli" in name:
                    writer.add_scalar(  # pyright: ignore
                        f"Proccessed_Parameter/{short_name}_{i}",
                        wide_sigmoid(element),
                        x_val,
                    )
