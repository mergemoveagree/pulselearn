# pyright: reportMissingTypeStubs=false
"""Implementation of a PyTorch function for the qiskit interface.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
import logging
import multiprocessing as mp
import time
from multiprocessing.pool import AsyncResult
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from lib.log.log import ROOT_LOGGER_NAME
from qiskit.quantum_info import (
    state_fidelity,
)  # pyright: reportUnknownVariableType=false
from ql_simulator.abstract_dynamics_simulator import AbstractDynamicsSimulator
from utils.pulse_utils import calculate_states, wide_sigmoid
from utils.type_aliases import ComplexArray, PulseArgsAlias, RealArray

log = logging.getLogger(ROOT_LOGGER_NAME + "." + __name__)

duration_grad_modifier = 0.05


class DynamicsSimulatorFunction(torch.autograd.Function):
    """PyTorch Function for an AbstractDynamicsSimulator with a custom gradient implementation."""

    # PyTorch doesn't have a mypy stub file to mask the violation of Liskov substitutability
    @staticmethod
    def forward(  # pyright: reportIncompatibleMethodOverride=false
        ctx: Any,  # should be torch.autograd.function.FunctionCtx,
        wrapper: AbstractDynamicsSimulator,
        initial_comps: torch.Tensor,
        target_states: torch.Tensor,
        num_drive_pulses: int,
        should_multiprocess: bool,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        """Simulate the DRAG pulse and return the fidelities between the resultant and goal states.

        The args param is a tuple of tensors whose length is a multiple of 6. For every set of 6,
        the elements follow this order: [durations, moduli, arguments, sigmas, betas, phases].
        Each of these elements is a row tensor where the index of the set of 6 of which it belongs
        is the index of the drive channel where it will be transmitted.
        Each element of the row tensor corresponds to a pulse on that drive channel, thus all
        elements within a set of 6 must be the same length.
        In this function, the moduli that are passed will be evaluated by a sigmoid function before
        being converted to an amp.

        :param ctx: Context to save information for the backward pass.
        :param wrapper: Simulator to run pulses.
        :param initial_comps: Tensor of components of coefficients (moduli and arguments)
            (i.e. [[a_1, b_1, phase_b_1, c_1, phase_c_1, ...], [a_2, b_2, phase_b_2, ...]]).
        :param num_drive_pulses: Number of drive pulse parameter tensors to consume before
            processing control pulse parameters.
        :param should_multiprocess: Whether to allow multiprocessing during the backward pass.
        :param target_states: Tensor of desired states as row statevectors.
        """
        if len(args) % 6 != 0:
            raise ValueError("Invalid number of pulse parameter tensors")

        # Separating the pulse parameters into lists
        # TODO: Add processing for control channels
        durations: list[RealArray] = [
            args[i].detach().numpy() for i in range(0, len(args), 6)
        ]
        moduli: list[RealArray] = [
            args[i].detach().numpy() for i in range(1, len(args), 6)
        ]
        arguments: list[RealArray] = [
            args[i].detach().numpy() for i in range(2, len(args), 6)
        ]
        sigmas: list[RealArray] = [
            args[i].detach().numpy() for i in range(3, len(args), 6)
        ]
        betas: list[RealArray] = [
            args[i].detach().numpy() for i in range(4, len(args), 6)
        ]
        phases: list[RealArray] = [
            args[i].detach().numpy() for i in range(5, len(args), 6)
        ]

        initial_states = calculate_states(initial_comps.detach().numpy())
        parsed_target_states: npt.NDArray[
            np.complex128
        ] = target_states.detach().numpy()

        # Saving parameters for backward pass
        ctx.wrapper = wrapper
        ctx.should_multiprocess = should_multiprocess

        ctx.initial_states = initial_states
        ctx.target_states = parsed_target_states

        ctx.durations = durations
        ctx.moduli = moduli
        ctx.arguments = arguments
        ctx.sigmas = sigmas
        ctx.betas = betas
        ctx.phases = phases

        # Ensuring that the moduli of the amps are < 1 so Qiskit won't throw an error
        amps = calculate_amps(wide_sigmoid(squarify(moduli)), squarify(arguments))

        resultant_states = wrapper.simulate_pulse(
            squarify(durations),
            amps,
            squarify(sigmas),
            squarify(betas),
            squarify(phases),
            initial_states,
        )

        fidelities = calculate_fidelities(resultant_states, parsed_target_states)

        # Saving fidelities for grad calculation in backward pass
        ctx.fidelities = fidelities

        return torch.tensor(fidelities)

    @staticmethod
    def backward(
        ctx: Any,  # should be torch.autograd.function.FunctionCtx
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        """Return the gradients w.r.t. the inputs of the foward function."""
        global duration_grad_modifier

        base_list: list[list[RealArray]] = [
            ctx.durations,
            ctx.moduli,
            ctx.arguments,
            ctx.sigmas,
            ctx.betas,
            ctx.phases,
        ]
        shifted_args = get_shifted_args(base_list, ctx.initial_states)

        evaluated_shifted_params: list[list[list[ComplexArray]]] = []

        log.debug("Starting to calculate grads")
        time_start = time.perf_counter()
        if ctx.should_multiprocess:
            pool = mp.Pool(mp.cpu_count() - 1)

            shifted_params_results: list[list[list[AsyncResult[ComplexArray]]]] = []
            for params_array in shifted_args:
                evaluated_array: list[list[AsyncResult[ComplexArray]]] = []
                for row in params_array:
                    evaluated_row_results: list[AsyncResult[ComplexArray]] = []
                    for args_list in row:
                        evaluated_row_results.append(
                            pool.apply_async(ctx.wrapper.simulate_pulse, args_list)
                        )
                    evaluated_array.append(evaluated_row_results)
                shifted_params_results.append(evaluated_array)

            # Each element of shifted_params_results is a list of `AsyncResult`s that return a
            # numpy array of complex numbers
            for results_array in shifted_params_results:
                retrieved_results_array: list[list[ComplexArray]] = []
                for results_row in results_array:
                    retrieved_row: list[ComplexArray] = []
                    for result in results_row:
                        retrieved_row.append(result.get())
                    retrieved_results_array.append(retrieved_row)
                evaluated_shifted_params.append(retrieved_results_array)

            # No need for pool.join() since we're getting all our AsyncResults anyways
            pool.close()
        else:
            for params_array in shifted_args:
                evaluated_single_array: list[list[ComplexArray]] = []
                for row in params_array:
                    # List of resultant states
                    evaluated_row: list[ComplexArray] = []
                    for args_list in row:
                        evaluated_row.append(ctx.wrapper.simulate_pulse(*args_list))
                    evaluated_single_array.append(evaluated_row)
                evaluated_shifted_params.append(evaluated_single_array)

        time_stop = time.perf_counter()
        log.debug("Finished calculating grads in %s seconds", time_stop - time_start)

        grads: list[list[torch.Tensor]] = []
        for index, evaluated_param_array in enumerate(evaluated_shifted_params):
            param_array_grads: list[torch.Tensor] = []
            for i, evaluated_row in enumerate(evaluated_param_array):
                row_grads: list[RealArray] = []
                for j, evaluated_column in enumerate(evaluated_row):
                    column_fidelity = calculate_fidelities(
                        evaluated_column, ctx.target_states
                    )
                    modifier = 0.05 if index > 0 else duration_grad_modifier
                    # Interval of the difference quotient
                    h: np.float64 = max(modifier * base_list[index][i][j], 0.001)
                    column_grads = (
                        column_fidelity - cast(RealArray, ctx.fidelities)
                    ) / h
                    row_grads.append(column_grads)
                # Generally, the resultant gradients of a function that implements custom
                # differentiation are multiplied on the right with the grad_ouput tensor that is
                # passed as an argument to the Function#backward function; however, the
                # structure of the pulse model should just have one layer, so there really
                # shouldn't be any gradients that need to be multiplied for the chain rule. If
                # the model seems to not train, try not multipling the grad_output with our
                # calculated grads.
                #
                # Accidentally calculated the transpose of the gradients instead of the actual
                # gradients, so we transpose it here to fix the design issue
                # TODO Calculate the gradients correctly in the first place (a LOT of work)
                #
                # row_grads_tensor = torch.tensor(np.array(row_grads).T)
                row_grads_tensor = grad_output * torch.tensor(np.array(row_grads).T)
                param_array_grads.append(row_grads_tensor)
            grads.append(param_array_grads)

        d_durations, d_moduli, d_arguments, d_sigmas, d_betas, d_phases = grads

        log.debug("Durations grads: %s", d_durations)
        log.debug("Moduli grads: %s", d_moduli)
        log.debug("Arguments grads: %s", d_arguments)
        log.debug("Sigmas grads: %s", d_sigmas)
        log.debug("Betas grads: %s", d_betas)
        log.debug("Phases grads: %s", d_phases)

        # Since the duration of the pulse in terms of time samplings is discrete (must be a multiple
        # of 16 due to Qiskit), it's possible to shift the duration of the pulse in nanoseconds but
        # keep the same duration in terms of *time samplings* (which, again, has to be a multiple of
        # 16). We increment the percent of the duration by which we shift (starts at 5%) whenever we
        # find that one of the duration gradients is zero. Else, we slowly decrease the modifier.

        # d_duration is a list of lists of tensors, so it has to be unpacked in this nested
        # list comprehension
        if any((0.0 in y for y in x) for x in d_durations):
            log.debug("Found zero in the durations grad tensor")
            duration_grad_modifier += 0.01
            log.debug(
                "Updated duration_grad_modifier: %f -> %f",
                duration_grad_modifier - 0.01,
                duration_grad_modifier,
            )
        else:
            duration_grad_modifier -= 0.001
            log.debug(
                "Duration grad modifier decreased: %f -> %f",
                duration_grad_modifier + 0.001,
                duration_grad_modifier,
            )

        # Unpacking the grads from the array of tensors into the order that they were passed to the
        # forward call (in the cycle: duration, modulus, argument, sigma, beta, phase)
        grads_list: list[torch.Tensor] = []
        # All the lists of tensors should be the same length
        for params_grads_tuple in zip(
            d_durations, d_moduli, d_arguments, d_sigmas, d_betas, d_phases
        ):
            grads_list.extend(params_grads_tuple)

        return None, None, None, None, None, *grads_list


def calculate_amps(moduli: RealArray, arguments: RealArray) -> ComplexArray:
    """Calculate complex amplitudes from an array of moduli and arguments.

    The moduli are NOT passed through a sigmoid function before calculation.
    """
    if moduli.shape != arguments.shape:
        raise ValueError("The moduli and arguments array must be the same shape.")
    phases = np.exp(1.0j * arguments)
    amps = moduli.astype(np.complex128) * phases
    return amps


def calculate_fidelities(states1: ComplexArray, states2: ComplexArray) -> RealArray:
    """Calculate the fidelities using the Qiskit state_fidelity function.

    Both state arrays should be arrays of statevectors as row vectors.

    :return: A 1D numpy array of fidelities.
    """
    if states1.shape != states2.shape:
        raise ValueError("Both state arrays must have the same shape")
    fidelities: RealArray = np.zeros(len(states1), dtype=np.float64)
    for i, (state1, state2) in enumerate(zip(states1, states2)):
        fidelities[i] = state_fidelity(state1, state2)
    return fidelities


def get_shifted_args(
    base_list: list[list[RealArray]],
    initial_states: ComplexArray,
) -> list[list[list[PulseArgsAlias]]]:
    """Return necessary args to calculate the gradients of pulse parameters.

    The highest level list of base_list is expected to resemble:
        [durations, moduli, arguments, sigmas, betas, phases]
    where all elements are lists of numpy arrays. This function will return a nested list with the
    same structure, but the elements of each array are replaced with a list of args needed to run
    the AbstractDynamicsSImulator#simulate_pulse function with the parameter in the corresponding
    position shifted by at least 0.001 (default 5%).
    """
    total_shifted_args: list[list[list[PulseArgsAlias]]] = []
    for index, candidate in enumerate(base_list):
        shifted_candidates: list[list[PulseArgsAlias]] = []
        for i, row in enumerate(candidate):
            shifted_rows: list[PulseArgsAlias] = []
            for j, column in enumerate(row):
                # Generating the row with a shifted column
                row_copy = row.copy()
                modifier = 0.05 if index > 0 else duration_grad_modifier
                row_copy[j] = column + max(column * modifier, 0.001)

                # Making the shifted candidate copy
                candidate_copy = candidate.copy()
                candidate_copy[i] = row_copy

                # Setting up components for AbstractDynamicsSimulator#simulate_pulse args list
                base_list_copy = base_list.copy()
                base_list_copy[index] = candidate_copy

                # Making the args list
                array_base_list = [squarify(x) for x in base_list_copy]
                shifted_amps = calculate_amps(
                    wide_sigmoid(array_base_list[1]), array_base_list[2]
                )
                args_list: PulseArgsAlias = (
                    array_base_list[0],
                    shifted_amps,
                    array_base_list[3],
                    array_base_list[4],
                    array_base_list[5],
                    initial_states,
                )

                # args_list serves the purpose of "shifted_column"
                shifted_rows.append(args_list)

            shifted_candidates.append(shifted_rows)

        total_shifted_args.append(shifted_candidates)

    return total_shifted_args


def squarify(arrays: list[RealArray]) -> RealArray:
    """Compress a list of arrays into an array where the missing columns from each row is zero."""
    columns = max([len(x) for x in arrays])
    # All of the arrays in the list should have the same dtype
    # Infer the dtype from the first array in the list
    compressed_array: RealArray = np.zeros((len(arrays), columns), dtype=np.float64)
    for i, array in enumerate(arrays):
        compressed_array[i][: len(array)] = array
    return compressed_array
