"""Abstract dynamics simulator that all dynamics simulators implement.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
import abc
from typing import Dict, Sequence, Tuple

from utils.type_aliases import RealArray, ComplexArray


class AbstractDynamicsSimulator(abc.ABC):
    """Generic template for a dynamics simulator."""

    @abc.abstractmethod
    def __init__(
        self,
        qubit_frequencies: Sequence[float],
        rabi_rate_strenghts: Sequence[float],
        time_sampling_rate: float,
        qubit_coupling: Dict[Tuple[int, int], float],
    ) -> None:
        """Instantiate a dynamics simulator with necessary parameters.

        :param qubit_frequencies: Tunneling frequencies of the qubits in GHz. The length of the list
            of is the number of qubits of the system.
        :param rabi_rate_strenghts: Rabi rate frequency of the qubits in GHz.
        :param time_sampling_rate: Time sampling rate of the simulator in nanoseconds.
        :param qubit_coupling: Dictionary of two qubit indices and the coupling between them.
        """
        pass

    @abc.abstractmethod
    def simulate_pulse(
        self,
        drive_durations: RealArray,
        drive_amps: ComplexArray,
        drive_sigmas: RealArray,
        drive_betas: RealArray,
        drive_phases: RealArray,
        initial_states: ComplexArray,
        control_durations: RealArray | None = None,
        control_amps: ComplexArray | None = None,
        control_sigmas: RealArray | None = None,
        control_betas: RealArray | None = None,
        control_phases: RealArray | None = None,
        control_delays: RealArray | None = None,
    ) -> ComplexArray:
        """Simulate DRAG pulses being played on a quantum computer and return the end statevectors.

        Pulse parameters (durations, amps, sigmas, betas, phases, and delays) are numpy arrays where
        the Nth row corresponds to the Nth drive channel and the Kth column corresponds to the Kth
        pulse played on the Nth drive channel.If the amplitude of a pulse is zero, then the pulse
        is skipped (in order to support differing number of pulses per drive channel).

        :param drive_durations: Durations of the drive pulses in nanoseconds.
        :param drive_amps: Complex amplitudes of the drive pulses (with modulus <= 1).
        :param drive_sigmas: Variances (widths) of the drive pulses.
        :param drive_betas: Correction amplitudes of the drive pulses (for DRAG pulses).
        :param drive_phases: Phases to shift the drive channel before playing the pulse.
        :param initial_states: 2D numpy array where each row represents a quantum state. The number
            of columns should be equal to 2 to the power of the number of qubits to be simulated.
        :param control_durations: Durations of the control pulses in nanoseconds, defaults to None.
        :param control_amps: Complex amplitudes of the drive pulses (with modulus <= 1),
            defaults to None.
        :param control_sigmas: Variances (widths) of the control pulses, defaults to None.
        :param control_betas: Correction amplitudes of the control pulses (for DRAG pulses),
            defaults to None.
        :param control_phases: Phases to shift the control channel before playing the pulse,
            defaults to None.
        :param control_delays: Delays to wait before playing the corresponding control pulse,
            defaults to None.
        :return: 2D array where each row is the statevector of the end states (with the Nth row
            corresponding to the Nth row of the initial_states array).
        """
        pass

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits implemented in the simulator."""
        pass
