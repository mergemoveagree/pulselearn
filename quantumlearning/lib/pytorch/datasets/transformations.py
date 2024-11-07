# pyright: reportMissingTypeStubs=false, reportIncompatibleMethodOverride=false
"""Mapping of qubit transformation to their appropriate PyTorch dataset.

.. codeauthor: Jaden Nola <jaden.nola@netapp.com>
"""
from abc import ABCMeta, abstractmethod
from enum import Enum, EnumMeta, auto
from functools import partial
from typing import Any

import numpy as np
from qiskit.circuit.library import RYGate, RZGate
from utils.type_aliases import ComplexArray, StatesDataset

from lib.pytorch.datasets.gate_datasets import CXDataset, SingleQubitDataset


class ABCEnumMeta(EnumMeta, ABCMeta):
    """Metaclass for an abstract Enum."""

    pass


class Transformation(Enum, metaclass=ABCEnumMeta):
    """Base class for qubit transformations."""

    # Have to type ignore to support polymorphism with different function args
    @abstractmethod
    def get_dataset(self, *args: Any, **kwargs: Any) -> StatesDataset:
        """Return the PyTorch dataset for the corresponding transformation."""
        pass


class SingleQubitStaticGates(Transformation):
    """Transformations on one qubit that don't depend on any input parameters."""

    def get_dataset(
        self, num_qubits: int, num_points: int, target_index: int
    ) -> StatesDataset:
        """Return the dataset for num_qubits qubits and num_points points."""
        return SingleQubitDataset(
            num_qubits, num_points, target_index, static_gate_values[self]
        )

    PAULIX = auto()
    HADAMARD = auto()
    SX = auto()


# Unfortunately, we can't put numpy arrays as enum values because Python requires that all enum
# entries with the same value will become aliases for the first entry with the same value. As a
# result, Python compares each value with an == operator, which numpy does not support between
# arrays, thus raising an exception when instantiating the enum class. Our workaround is to
# create a dictionary that associates the enum entries with their corresponding matrices and
# leverage this dictionary in the Transformation#get_dataset function. However, if the value of the
# enum is a *function* that generates the matrix (i.e. TRIPLE_ROTATION in  the
# SingleQubitParameterizedGates enum), then this workaround becomes unnecessary.
static_gate_values = {
    SingleQubitStaticGates.PAULIX: np.array([[0, 1], [1, 0]], dtype=np.complex128),
    SingleQubitStaticGates.HADAMARD: np.sqrt(
        np.array([[0.5, 0.5], [0.5, -0.5]]), dtype=np.complex128
    ),
    SingleQubitStaticGates.SX: np.array(
        [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=np.complex128
    ),
}


class SingleQubitParameterizedGates(Transformation):
    """Transformations on one qubit that depend on one or more input parameters."""

    def get_dataset(
        self, num_qubits: int, num_points: int, target_index: int, **kwargs: float
    ) -> StatesDataset:
        """Return the dataset for num_qubits qubits, num_points points, and input parameters."""
        # Generating the matrix to pass to the dataset
        matrix = self.value(**kwargs)  # Expect user to correctly pass the arguments
        return SingleQubitDataset(num_qubits, num_points, target_index, matrix)

    @staticmethod
    def _triple_rotation_matrix(w_5k_1: float, w_5k_3: float) -> ComplexArray:
        """Return the matrix of the single qubit gate sequence of the entanglement witness.

        The triple rotation sequence is derived from the following paper: arxiv.org/abs/1902.07754v2
        """
        opp_ry: ComplexArray = RYGate(-w_5k_1).to_matrix()  # pyright: ignore
        rz: ComplexArray = RZGate(w_5k_3).to_matrix()  # pyright: ignore
        ry: ComplexArray = RYGate(w_5k_1).to_matrix()  # pyright: ignore
        return ry @ rz @ opp_ry

    TRIPLE_ROTATION = partial(_triple_rotation_matrix)


class ControlledStaticGates(Transformation):
    """Transformations that depend on control qubits and take no input parameters."""

    def get_dataset(
        self,
        num_qubits: int,
        num_points: int,
        target_index: int,
        control_qubits: list[int],
    ) -> StatesDataset:
        """Return the dataset associated with the transformation."""
        return self.value(num_qubits, num_points, target_index, control_qubits)

    CX = CXDataset


# TODO: Maybe move to script_utils.py?
def get_gate(classifier: str, transformation: str) -> Transformation:
    """Get a transformation by a classifier and its string name."""
    try:
        if classifier.lower() == "singlestatic":
            return SingleQubitStaticGates[transformation.upper()]
        elif classifier.lower() == "singleparameterized":
            return SingleQubitParameterizedGates[transformation.upper()]
        elif classifier.lower() == "controlledstatic":
            return ControlledStaticGates[transformation.upper()]
        else:
            raise ValueError(f"'{classifier}' is not a supported classifier")
    except KeyError as e:
        raise ValueError(
            f"'{transformation}' is not a member of classifier '{classifier}'"
        ) from e
