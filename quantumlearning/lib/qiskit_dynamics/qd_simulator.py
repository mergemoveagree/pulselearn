# type: ignore
"""Implementation of a dynamics simulator using Qiskit Dynamics.

Static type checking is disabled on this file since Qiskit does not play nice with pyright
(possibly because it doesn't implement a py.typed file).

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
import logging

import numpy as np
from lib.log.log import ROOT_LOGGER_NAME
from qiskit import pulse
from qiskit.pulse.library import Drag
from qiskit.quantum_info.operators import Operator
from ql_simulator.abstract_dynamics_simulator import AbstractDynamicsSimulator
from utils.type_aliases import ComplexArray, RealArray

from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_dynamics.solvers import Solver

module_log = logging.getLogger(ROOT_LOGGER_NAME + "." + __name__)


class QiskitDynamicsSimulator(AbstractDynamicsSimulator):
    """Interface to interact with Qiskit Dynamics with numpy arrays."""

    def __init__(
        self,
        qubit_frequencies: list[float],
        rabi_rate_strengths: list[float],
        time_sampling_rate: float,
        qubit_couplings: dict[str, int] | None = None,
    ) -> None:
        """Construct the drift hamiltonian and operators for a Solver object.

        See parent class for argument documentation.
        """
        self.qubit_frequencies = qubit_frequencies
        self.qubit_frequencies = qubit_frequencies
        self.time_sampling_rate = time_sampling_rate

        # Validating incoming arguments
        if self.num_qubits < 1:
            raise ValueError("Must provide at least one qubit frequency")
        if len(rabi_rate_strengths) != self.num_qubits:
            raise ValueError(
                "Must have same number of qubit frequencies and rabi rate strengths"
            )

        # Constructing the drift hamiltonian
        drift_hamiltonian = Operator(
            np.zeros((2**self.num_qubits, 2**self.num_qubits))
        )
        base_label = "I" * self.num_qubits
        for i, freq in enumerate(self.qubit_frequencies):
            drift_hamiltonian += (
                0.5 * freq * Operator.from_label(_replace_indices(base_label, "Z", [i]))
            )
        if qubit_couplings is not None:
            for pair, coupling in qubit_couplings.items():
                indices = [int(x) for x in pair.split(":")]
                if any([x + 1 > self.num_qubits for x in indices]):
                    raise ValueError(
                        "Qubit coupling dictionary had out of range indices"
                    )
                label = _replace_indices(base_label, "Z", indices)
                drift_hamiltonian += 0.5 * coupling * Operator.from_label(label)
        # Correcting units from being in terms of h to h-bar
        self.drift_hamiltonian = 2 * np.pi * drift_hamiltonian
        # Creating operators for the Solver
        # TODO: Are rabi rate strengths really what I think they are? Need more research
        # TODO Figure out control channel operators and when to add them
        operators = [
            0.5
            * rabi_rate_strengths[i]
            * Operator.from_label(_replace_indices(base_label, "X", [i]))
            for i in range(self.num_qubits)
        ]
        # Correcting units
        self.operators = [2 * np.pi * x for x in operators]

    # TODO: Implement control operators
    def simulate_pulse(
        self,
        drive_durations: RealArray,
        drive_amplitudes: ComplexArray,
        drive_sigmas: RealArray,
        drive_betas: RealArray,
        drive_phases: RealArray,
        initial_states: ComplexArray,
        control_durations: RealArray | None = None,
        control_amplitudes: ComplexArray | None = None,
        control_sigmas: RealArray | None = None,
        control_betas: RealArray | None = None,
        control_phases: RealArray | None = None,
        control_delays: RealArray | None = None,
    ) -> ComplexArray:
        """See parent class for usage.

        :raises ValueError: Pulse parameters (for either drive or channel) aren't the same shape.
        """
        sim_schedule = _generate_schedule(
            self.num_qubits,
            self.time_sampling_rate,
            pulse.DriveChannel,
            drive_durations,
            drive_amplitudes,
            drive_sigmas,
            drive_betas,
            drive_phases,
        )

        if control_amplitudes:
            control_schedule = _generate_schedule(
                self.num_qubits,
                self.time_sampling_rate,
                pulse.ControlChannel,
                control_durations,
                control_amplitudes,
                control_sigmas,
                control_betas,
                control_phases,
                control_delays,
            )
            sim_schedule.append(control_schedule, inplace=True)

        total_runtime = sim_schedule.duration * self.time_sampling_rate

        # TODO: Figure out what are control channel frequencies
        # TODO: Implement control channel frequencies
        converter = InstructionToSignals(
            self.time_sampling_rate, carriers=self.qubit_frequencies
        )
        signals = converter.get_signals(sim_schedule)

        solver = Solver(
            static_hamiltonian=self.drift_hamiltonian,
            hamiltonian_operators=self.operators,
            hamiltonian_signals=signals,
            rotating_frame=self.drift_hamiltonian,
            rwa_cutoff_freq=2 * max(self.qubit_frequencies),
        )

        final_states: ComplexArray = np.zeros(
            shape=(len(initial_states), int(np.exp2(self.num_qubits))),
            dtype=np.complex128,
        )
        for i, state in enumerate(initial_states):
            sol = solver.solve(
                t_span=[0.0, total_runtime], y0=state, atol=1e-8, rtol=1e-8
            )
            # sol.y[-1] is the latest statevector after the pulse ends
            # Taking end state out of rotating frame from the rotating wave approximation
            end_state = solver.model.rotating_frame.state_out_of_frame(
                total_runtime, sol.y[-1]
            )
            final_states[i] = end_state.tolist()

        return final_states

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits implemented in the simulator."""
        return len(self.qubit_frequencies)


def _get_16_multiple(number: float) -> int:
    """Return the nearest multiple of 16."""
    return int(np.floor((number / 16) + 0.5) * 16)


def _generate_schedule(
    num_qubits: int,
    time_sampling_rate: int,
    channel_type: type[pulse.channels.PulseChannel],
    durations: RealArray,
    amplitudes: ComplexArray,
    sigmas: RealArray,
    betas: RealArray,
    phases: RealArray,
    delays: RealArray | None = None,
) -> pulse.Schedule:
    """Build a Schedule of pulses on specified channel types.

    :raises ValueError: The parameters did not have the same shape/number of parameters.
    :return: Qiskit Schedule with pulses only on the drive channels.
    """
    if delays is None:
        delays = np.zeros(amplitudes.shape)
    # Ensuring that all parameters have same number of drive channels
    if not (
        num_qubits
        == len(durations)
        == len(amplitudes)
        == len(sigmas)
        == len(betas)
        == len(phases)
        == len(delays)
    ):
        raise ValueError(
            "Number of drive channels and number of rows in parameter arrays are"
            "not the same"
        )

    sim_schedule = pulse.Schedule()
    for chan_index, (
        chan_durations,
        chan_amps,
        chan_sigmas,
        chan_betas,
        chan_phases,
        chan_delays,
    ) in enumerate(zip(durations, amplitudes, sigmas, betas, phases, delays)):
        # Ensure that each array has enough parameters for each pulse of the drive channel
        if not (
            len(chan_durations)
            == len(chan_amps)
            == len(chan_sigmas)
            == len(chan_betas)
            == len(chan_phases)
            == len(chan_delays)
        ):
            raise ValueError(f"Not enough pulse parameters for channel {chan_index}")

        for duration, amp, sigma, beta, phase, delay in zip(
            chan_durations, chan_amps, chan_sigmas, chan_betas, chan_phases, chan_delays
        ):
            if amp == 0:
                # Assume that an amplitude of 0 means a placeholder value to prevent creating a
                # ragged array (this is ok since it is *very* unlikely that we will learn to zero).
                continue
            if delay:
                delay_samples = delay // time_sampling_rate
                sim_schedule += pulse.Delay(delay_samples, channel_type(chan_index))
            # Qiskit requires that the duration in terms of time samplings is a multiple of 16
            pulse_samples = _get_16_multiple(duration / time_sampling_rate)
            sim_schedule += pulse.ShiftPhase(phase, channel_type(chan_index))
            sim_schedule += pulse.Play(
                Drag(pulse_samples, amp, sigma, beta), channel_type(chan_index)
            )
            sim_schedule += pulse.ShiftPhase(-phase, pulse.DriveChannel(chan_index))

    return sim_schedule


def _replace_indices(base_string: str, new_char: str, indices: list[int]) -> str:
    """Replace the characters at specified indices with given new character."""
    if len(new_char) > 1:
        raise ValueError("Length of 'new_char' is greater than one")
    if any(x > len(base_string) for x in indices):
        raise ValueError("Found index out of bounds of the base string")
    char_array = [char for char in base_string]
    for i in indices:
        char_array[i] = new_char
    return "".join(char_array)
