"""Useful functions for the quantumlearning project.

.. codeauthor: Jaden Nola <a733p959@wichita.edu>
"""
from typing import TypeVar, cast

import numpy as np

from utils.type_aliases import ComplexArray, RealArray

ArrayOrNumber = TypeVar("ArrayOrNumber", RealArray, ComplexArray, float)


def calculate_states(components: RealArray) -> ComplexArray:
    """Calculate states of a quantum system from an array of coefficients.

    Each row of the input array should resemble:
        [a_1, b_1, phase_b_1, c_1, phase_c_1, ...]
    This function returns a 1D array with the calculated states in the place of corresponding rows.
    """
    # Checking if the number of coefficients looks like 2^n + (2^n - 1)
    # Inverse of f(n) = 2^n + (2^n - 1) = 2^(n + 1) - 1
    num_qubits: float = np.log2(components.shape[1] + 1) - 1
    if not num_qubits.is_integer():
        raise ValueError(f"Incorrect number of components -> {components.shape[1]}")
    if not np.isclose(components_norm(components), 1).all():  # pyright: ignore
        raise ValueError("Components array is not properly normalized.")

    states: ComplexArray = np.zeros(
        (len(components), int(np.exp2(num_qubits))), dtype=np.complex128
    )
    for i, row in enumerate(components):
        real_coeff: RealArray = row[0]
        remaining_coefficients: list[ComplexArray] = [
            row[k] * np.exp(1.0j * row[k + 1]) for k in range(1, len(row), 2)
        ]
        states[i, 0] = real_coeff
        states[i, 1:] = remaining_coefficients
    return states


def components_norm(array: RealArray) -> RealArray:
    """Return the norm of the components of each row.

    Each row of the components array should follow the schema:
        [a, b, theta_0, c, theta_1, d, theta_2, ...]
    where a, b, ... are the unnormalized coefficients of the statevector and theta_0, theta_1, ...
    are the relative phases of the preceding coefficient.

    :return: Column vector with each row being the norm of the corresponding row in the given array.
    """
    # Summing the squares of the even columns and the square of the first column
    # Then, taking the square root
    norms: RealArray = np.sqrt(
        sum(np.power(array[:, i], 2) for i in range(1, array.shape[1], 2))
        + np.power(array[:, 0], 2)
    )
    return norms


def inverse_wide_sigmoid(obj: ArrayOrNumber) -> ArrayOrNumber:
    r"""Take a float or an array of floats and return the inverse of the wide_sigmoid function.

    The inverse of the wide_sigmoid function $\frac{1-e^x}{1+e^x}$ is
    $$\ln\left(\frac{1-x}{1+x}\right)$$.
    """
    log_arg: RealArray = np.divide(1 + obj, 1 - obj)
    evaluated_log = cast(ArrayOrNumber, np.log(log_arg))
    return evaluated_log


def normalize_components(array: RealArray) -> None:
    """Normalize the coefficients of a components array in-place."""
    norms = components_norm(array)
    np.divide(array[:, 0], norms, out=array[:, 0])
    for i in range(1, array.shape[1], 2):
        np.divide(array[:, i], norms, out=array[:, i])


def wide_sigmoid(obj: ArrayOrNumber) -> ArrayOrNumber:
    r"""Take an array of floats and constrain them between -1 and 1.

    The usual sigmoid function is defined as $f(x) = \frac{1}{1+e^x}$. Here we take the
    wide_sigmoid function to be $\frac{2}{1 + e^x}-1=\frac{1-e^x}{1+e^x}$.
    """
    exponentiated: RealArray = np.exp(-obj)
    sigmoid_result = cast(
        ArrayOrNumber, np.divide(1 - exponentiated, 1 + exponentiated)
    )
    return sigmoid_result
