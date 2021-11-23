# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Gates for spin backends."""

from typing import Union, Optional, List
from fractions import Fraction
import numpy as np
from scipy.linalg import expm

from qiskit.circuit.gate import Instruction, Gate
from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_cold_atom import QiskitColdAtomError, add_gate


class SpinGate(Gate):
    """Unitary gates for spin circuits."""

    def __init__(
        self,
        name: str,
        num_modes: int,
        params: Optional[List] = None,
        label: Optional[str] = None,
        generator: Optional[SpinOp] = None,
    ) -> None:
        """Create a new spin gate.

        Args:
            name: The Qobj name of the gate.
            num_modes: The number of fermionic modes the gate acts on.
            params: A list of parameters.
            label: An optional label for the gate.
            generator: The generating Hamiltonian of the gate unitary given as a SpinOp
        """

        self._generator = generator

        self.num_modes = num_modes

        if params is None:
            params = []

        super().__init__(name=name, num_qubits=num_modes, params=params, label=label)

    def power(self, exponent: float):
        """Creates a spin gate as `gate^exponent`.

        Args:
            exponent (float): The exponent with which the gate is exponentiated

        Returns:
            SpinGate: To which `.generator` is self.generator*exponent.

        Raises:
            QiskitColdAtomError: Ff the gate generator is not defined.
        """
        if self.generator is None:
            raise QiskitColdAtomError(
                "Gate can not be exponentiated if the gate generator is not defined."
            )
        # the generator of the exponentiated gate is the old generator times the exponent
        exp_generator = exponent * self.generator

        exp_params = (
            None if not self.params else [exponent * param for param in self.params]
        )

        exp_label = None if not self.label else self.label + f"^{exponent}"

        return SpinGate(
            name=self.name + f"^{exponent}",
            num_modes=self.num_modes,
            params=exp_params,
            label=exp_label,
            generator=exp_generator,
        )

    # pylint: disable=arguments-differ
    def to_matrix(self, spin: Union[float, Fraction] = Fraction(1, 2)) -> np.ndarray:
        """Return a Numpy.array for the gate unitary matrix.

        Args:
            spin: The spin value of each wire that the gate acts on

        Returns:
            A dense np.array of the unitary of the gate
        """
        spin_op = SpinOp(
            self.generator.to_list(), spin=spin, register_length=self.num_qubits
        )
        return expm(-1j * spin_op.to_matrix())

    def control(
        self,
        num_ctrl_qubits: Optional[int] = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
    ):
        """Overwrite control method which is supposed to return a controlled version of the gate.
        This is not applicable in the spin setting."""
        raise QiskitColdAtomError("Spin gates have no controlled version")

    @property
    def generator(self) -> SpinOp:
        """The Hamiltonian that generates the unitary of the gate, given as a SpinOp."""
        return self._generator


class LXGate(SpinGate):
    r"""Rotation of the collective spin of a cold atomic Bose-Einstein
    condensate around the x-axis.

    The generating Hamiltonian of the LX gate is

    :math:`H = \omega L_x`

    where :math:`\omega` is the free gate parameter.

    **Circuit symbol:**

    .. parsed-literal::

             ┌────────────┐
        q_0: ┤ RLX(omega) ├
             └────────────┘
    """

    def __init__(self, omega, label=None):
        """Create new RLX gate."""
        super().__init__("rLx", 1, [omega], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LX gate."""
        return float(self.params[0]) * SpinOp("X")


@add_gate
def lx(self, omega, wire):
    # pylint: disable=invalid-name
    """add the RLX gate to a QuantumCircuit"""
    return self.append(LXGate(omega), [wire], [])


class LYGate(SpinGate):
    r"""Rotation of the collective spin of a cold atomic Bose-Einstein
    condensate around the y-axis.

    The generating Hamiltonian of the LY gate is

    :math:`H = \omega L_y`

    where :math:`\omega` is the free gate parameter

    **Circuit symbol:**

    .. parsed-literal::

             ┌────────────┐
        q_0: ┤ RLY(omega) ├
             └────────────┘
    """

    def __init__(self, omega, label=None):
        """Create new RLY gate."""
        super().__init__("rLy", 1, [omega], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LY gate."""
        return float(self.params[0]) * SpinOp("Y")


@add_gate
def ly(self, omega, wire):
    # pylint: disable=invalid-name
    """add the RLY gate to a QuantumCircuit"""
    return self.append(LYGate(omega), [wire], [])


class LZGate(SpinGate):
    r"""Rotation of the collective spin of a cold atomic Bose-Einstein condensate around the z-axis.

    The generating Hamiltonian of the LZ gate is

    :math:`H = \delta L_z`

    where :math:`\delta` is the free gate parameter

    **Circuit symbol:**

    .. parsed-literal::

             ┌────────────┐
        q_0: ┤ RLZ(delta) ├
             └────────────┘

    """

    def __init__(self, delta, label=None):
        """Create new RZ gate."""
        super().__init__("rLz", 1, [delta], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LZ gate."""
        return float(self.params[0]) * SpinOp("Z")


@add_gate
def lz(self, delta, wire):
    # pylint: disable=invalid-name
    """add the RLZ gate to a QuantumCircuit"""
    return self.append(LZGate(delta), [wire], [])


class LZ2Gate(SpinGate):
    r"""Evolution of a coherent spin under the twisting dynamic generated by Lz^2'.

    The generating Hamiltonian of the LZ2 gate is

    :math:`H = \chi L_z^2`

    where :math:`\chi` is the free gate parameter.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤ RLZ2(chi) ├
             └───────────┘
    """

    def __init__(self, chi, label=None):
        """Create new rz2 gate."""
        super().__init__("rLz2", 1, [chi], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LZ gate."""
        return float(self.params[0]) * SpinOp("Z_0^2", register_length=1)


@add_gate
def lz2(self, chi, wire):
    # pylint: disable=invalid-name
    """Add the RLZ2 gate to a QuantumCircuit."""
    return self.append(LZ2Gate(chi), [wire], [])


class OATGate(SpinGate):
    r"""Evolution of a coherent spin under the one-axis-twisting Hamiltonian.

    The generating Hamiltonian of the OATgate is

    :math:`H = \chi L_z^2 + \Delta L_z + \Omega L_x`

    where :math:`\chi`, :math:`\Delta` and :math:`\Omega` are the free gate parameters.
    """

    def __init__(self, chi: float, delta: float, omega: float, label=None):
        """Create new one-axis-twisting rotation gate."""
        super().__init__("OAT", 1, [chi, delta, omega], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the OAT gate."""
        return (
            self.params[0] * SpinOp("Z_0^2", register_length=1)
            + self.params[1] * SpinOp("Z")
            + self.params[2] * SpinOp("X")
        )


@add_gate
def oat(self, chi: float, delta: float, omega: float, wire: int, label=None):
    """Add the RLZ2 gate to a QuantumCircuit."""
    return self.append(
        OATGate(chi=chi, delta=delta, omega=omega, label=label), [wire], []
    )


class LZZGate(SpinGate):
    r"""Coupled ZZ-rotation of two collective spins.

    The generating Hamiltonian of the LZZGate is

    :math:`H = \gamma L_{z, i} + L_{z, j}`

    where :math:`\gamma` is the free gate parameter and :math:`i` and :math:`j` index the wires the gate
    acts on.
    """

    def __init__(self, gamma: float, label=None):
        """Create new LZZ gate."""
        super().__init__("rLzz", 2, [gamma], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LZZ gate."""
        return self.params[0] * SpinOp("Z_0 Z_1", register_length=2)


@add_gate
def lzz(self, gamma: float, wires: List[int], label=None):
    """Add the LZZ gate to a QuantumCircuit."""
    return self.append(LZZGate(gamma=gamma, label=label), qargs=wires)


class LxLyGate(SpinGate):
    r"""The spin exchange gate of two collective spins.

    The generating Hamiltonian of the LxLyGate is

    :math:`H = \gamma (L_{x, i}L_{x,j} + L_{y, i} L_{y,j})`

    where :math:`\gamma` is the free gate parameter and :math:`i` and :math:`j` index the wires the gate
    acts on. This gate is equivalently expressed through raising and lowering operators as:

    :math:`H = 2\gamma (L_{+, i}L_{-,j} + L_{-, i} L_{+,j})`
    """

    def __init__(self, gamma: float, label=None):
        """Create new LxLy gate."""
        super().__init__("rLxLy", 2, [gamma], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LxLy gate."""
        return self.params[0] * (
            SpinOp("X_0 X_1", register_length=2) + SpinOp("Y_0 Y_1", register_length=2)
        )


@add_gate
def lxly(self, gamma: float, wires: List[int], label=None):
    """Add the LxLy gate to a QuantumCircuit."""
    return self.append(LxLyGate(gamma=gamma, label=label), qargs=wires)


class LoadSpins(Instruction):
    """
    LoadSpins makes it possible to define the spin length of each qudit mode.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────┐
        q_0: ┤ Load ├
             └──────┘
    """
    def __init__(self, num_atoms:int) -> None:
        """Initialise new load instruction."""
        super().__init__(name="load", num_qubits=1, num_clbits=0, params=[num_atoms], label=None)
        
@add_gate
def load_spins(self, wire, num_atoms):
    # pylint: disable=invalid-name
    """Add the load spin gate to a QuantumCircuit."""
    return self.append(LoadSpins(num_atoms), [wire], [])
