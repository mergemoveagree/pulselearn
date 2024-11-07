# pyright: reportMissingTypeStubs=false
"""PyTorch datasets for known gates.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
import functools
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import Operator

from lib.log.log import ROOT_LOGGER_NAME

from utils.pulse_utils import calculate_states, normalize_components
from utils.type_aliases import ComplexArray

log = logging.getLogger(ROOT_LOGGER_NAME + "." + __name__)


class MatrixDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """General datasets whose target states can be generated through matrix multiplication."""

    def __init__(
        self, num_qubits: int, num_points: int, operator: ComplexArray
    ) -> None:
        """Initialize a gate dataset of random initial states with a matrix operator."""
        self.num_points = num_points
        if operator.dtype != np.complex128:
            operator = operator.astype(np.complex128)

        # Because this is using a numpy function, the result is a np.float64, so we have to cast it
        # Can't have float types in shape tuple
        num_columns = int(np.exp2(num_qubits + 1) - 1)

        # Setting up RNG
        seed = os.environ.get("QUANTUMLEARNING_RANDOM_SEED", None)
        if seed is not None:
            log.debug("Found seed for dataset RNG: %s", seed)
        seed = int(seed) if seed is not None else None
        rng = np.random.default_rng(seed)

        # Generate random values between -1 and 1
        initial_comps_array = (
            2.0 * rng.random((num_points, num_columns), dtype=np.float64) - 1.0
        )
        # All odd columns besides the first one represent relative phases, so they are in  [0, 2pi]
        for i in range(2, num_columns, 2):
            initial_comps_array[:, i] *= 2 * np.pi

        # Normalizing the moduli
        normalize_components(initial_comps_array)

        self.initial_comps = torch.tensor(initial_comps_array)

        # Composing the operator matrix with the statevectors
        initial_states = calculate_states(initial_comps_array)

        target_states_array: ComplexArray = np.zeros(
            (num_points, 2**num_qubits), dtype=np.complex128
        )
        for i, state in enumerate(initial_states):
            # Statevectors are treated as column vectors during matrix multiplication
            target_state = operator @ state[np.newaxis].T
            # The result of matrix multiplication with a column statevector is another column vector
            # Putting the statevector back into a row vector
            target_states_array[i] = target_state.T[0]
        self.target_states = torch.tensor(target_states_array)

    def __len__(self) -> int:
        """Return the number of points in the dataset."""
        return self.num_points

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the point at the given index."""
        return self.initial_comps[index], self.target_states[index]


class SingleQubitDataset(MatrixDataset):
    """Dataset for single qubit transformations (i.e. input matrix is 2x2)."""

    def __init__(
        self,
        num_qubits: int,
        num_points: int,
        target_qubit: int,
        matrix: ComplexArray,
    ) -> None:
        """Initialize a SingleQubitDataset."""
        if matrix.shape != (2, 2):
            raise ValueError("SingleQUbitDataset matrix should be 2x2")
        if matrix.dtype != np.complex128:
            matrix = matrix.astype(np.complex128)
        base_list = [np.eye(2, dtype=np.complex128)] * num_qubits
        base_list[target_qubit] = matrix
        operator = functools.reduce(np.kron, base_list)
        super().__init__(num_qubits, num_points, operator)


class CXDataset(MatrixDataset):
    """Dataset for a CNOT transformation."""

    def __init__(
        self,
        num_qubits: int,
        num_points: int,
        target_qubit: int,
        control_qubits: list[int],
    ) -> None:
        """Initialize a CXDataset."""
        if len(control_qubits) > 1:
            raise ValueError("A CNOT gate cannot have more than one control qubit")
        if target_qubit >= num_qubits:
            raise ValueError("Target qubit index out of range")
        if any(x >= num_qubits for x in control_qubits):
            raise ValueError("Control qubit index out of range")
        if any(x == target_qubit for x in control_qubits):
            raise ValueError("Target and control qubits can't be the same qubit")
        qc = QuantumCircuit(num_qubits)
        qc.cx(control_qubits[0], target_qubit)
        cnot_matrix: ComplexArray = Operator.from_circuit(qc).data  # pyright: ignore
        super().__init__(num_qubits, num_points, cnot_matrix)
