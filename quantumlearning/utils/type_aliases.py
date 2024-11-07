"""
Type aliases for the quantumlearning project.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset


RealArray: TypeAlias = npt.NDArray[np.float64]
ComplexArray: TypeAlias = npt.NDArray[np.complex128]

# Args of the AbstractDynamicsSimulator#simulate_pulse function
PulseArgsAlias: TypeAlias = tuple[
    RealArray,
    ComplexArray,
    RealArray,
    RealArray,
    RealArray,
    ComplexArray,
]

StatesDataset: TypeAlias = Dataset[tuple[torch.Tensor, torch.Tensor]]
