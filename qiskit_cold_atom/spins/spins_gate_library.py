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

        exp_params = None if not self.params else [exponent * param for param in self.params]

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
        spin_op = SpinOp(self.generator.to_list(), spin=spin, register_length=self.num_qubits)
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


class RLXGate(SpinGate):
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
        super().__init__("rlx", 1, [omega], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LX gate."""
        return float(self.params[0]) * SpinOp("X")


@add_gate
def rlx(self, omega, wire):
    # pylint: disable=invalid-name
    """add the RLX gate to a QuantumCircuit"""
    return self.append(RLXGate(omega), [wire], [])


class RLYGate(SpinGate):
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
        super().__init__("rly", 1, [omega], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LY gate."""
        return float(self.params[0]) * SpinOp("Y")


@add_gate
def rly(self, omega, wire):
    # pylint: disable=invalid-name
    """add the RLY gate to a QuantumCircuit"""
    return self.append(RLYGate(omega), [wire], [])


class RLZGate(SpinGate):
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
        """Create new RLZ gate."""
        super().__init__("rlz", 1, [delta], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LZ gate."""
        return float(self.params[0]) * SpinOp("Z")


@add_gate
def rlz(self, delta, wire):
    # pylint: disable=invalid-name
    """add the RLZ gate to a QuantumCircuit"""
    return self.append(RLZGate(delta), [wire], [])


class RLZ2Gate(SpinGate):
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
        super().__init__("rlz2", 1, [chi], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LZ gate."""
        return float(self.params[0]) * SpinOp("Z_0^2", register_length=1)


@add_gate
def rlz2(self, chi, wire):
    # pylint: disable=invalid-name
    """Add the RLZ2 gate to a QuantumCircuit."""
    return self.append(RLZ2Gate(chi), [wire], [])


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
            float(self.params[0]) * SpinOp("Z_0^2", register_length=1)
            + float(self.params[1]) * SpinOp("Z")
            + float(self.params[2]) * SpinOp("X")
        )


@add_gate
def oat(self, chi: float, delta: float, omega: float, wire: int, label=None):
    """Add the RLZ2 gate to a QuantumCircuit."""
    return self.append(OATGate(chi=chi, delta=delta, omega=omega, label=label), [wire], [])


class RLZLZGate(SpinGate):
    r"""Coupled ZZ-rotation of two collective spins.

    The generating Hamiltonian of the RLZLZGate is

    :math:`H = \gamma L_{z, i} + L_{z, j}`

    where :math:`\gamma` is the free gate parameter and :math:`i` and :math:`j` index the wires
    the gate acts on.
    """

    def __init__(self, gamma: float, label=None):
        """Create new RLZLZ gate."""
        super().__init__("rlzlz", 2, [gamma], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LZZ gate."""
        return self.params[0] * SpinOp("Z_0 Z_1", register_length=2)


@add_gate
def rlzlz(self, gamma: float, wires: List[int], label=None):
    """Add the RLZLZ gate to a QuantumCircuit."""
    return self.append(RLZLZGate(gamma=gamma, label=label), qargs=wires)


class RLXLYGate(SpinGate):
    r"""The spin exchange gate of two collective spins.

    The generating Hamiltonian of the LxLyGate is

    :math:`H = \gamma (L_{x, i}L_{x,j} + L_{y, i} L_{y,j})`

    where :math:`\gamma` is the free gate parameter and :math:`i` and :math:`j` index the wires the gate
    acts on. This gate is equivalently expressed through raising and lowering operators as:

    :math:`H = 2\gamma (L_{+, i}L_{-,j} + L_{-, i} L_{+,j})`
    """

    def __init__(self, gamma: float, label=None):
        """Create new RLXLY gate."""
        super().__init__("rlxly", 2, [gamma], label=label)

    @property
    def generator(self) -> SpinOp:
        r"""The generating Hamiltonian of the LxLy gate."""
        return self.params[0] * (
            SpinOp("X_0 X_1", register_length=2) + SpinOp("Y_0 Y_1", register_length=2)
        )


@add_gate
def rlxly(self, gamma: float, wires: List[int], label=None):
    """Add the RLXLY gate to a QuantumCircuit."""
    return self.append(RLXLYGate(gamma=gamma, label=label), qargs=wires)


class LoadSpins(Instruction):
    r"""An instruction to define the spin length of each qudit mode.

    This gate loads `num_atoms` onto the wire of index `wire`. This results in a
    local spin length of :math:`\ell = N/2`.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────┐
        q_0: ┤ Load ├
             └──────┘
    """

    def __init__(self, num_atoms: int):
        """Initialise new load instruction.

        Args:
            num_atoms: The integer number of atoms loaded into this wire. n Qobj name of the gate.
        """
        super().__init__(name="load", num_qubits=1, num_clbits=0, params=[num_atoms], label=None)


@add_gate
def load_spins(self, num_atoms: int, wire: int):
    """Add the load spin gate to a QuantumCircuit."""
    return self.append(LoadSpins(num_atoms), [wire], [])


class RydbergFull(SpinGate):
    r"""
    Global 1D-Rydberg dynamic consisting of the detuning, Rabi coupling and Rydberg blockade.

    The generating Hamiltonian of the Fermi-Hubbard gate is

    :math:`H = \sum_{i=1,\sigma}^{L-1} - J_i (f^\dagger_{i,\sigma} f_{i+1,\sigma}
    + f^\dagger_{i+1,\sigma} f_{i,\sigma})
    + U \sum_{i=1}^{L} n_{i,\uparrow} n_{i,\downarrow}
    + \sum_{i=1,\sigma}^{L} \mu_i n_{i,\sigma}`

    where :math:`i` indexes the mode, :math:`\sigma` indexes the spin, :math:`L` gives the total number
    of sites, :math:`\Omega_i` are the Rabi couplings, :math:`U` is the interaction strength and
    :math:`\Delta_i` are the local detunings.

    **Circuit symbol:**

        .. parsed-literal::

                 ┌────────────────┐
            q_0: ┤   RydbergFull  ├
                 └────────────────┘

    """

    def __init__(self, num_modes: int, omega: float, delta: float, phi: float, label=None):
        """Initialize a global Rydberg gate

        Args:
            num_modes: number of tweezers on which the hopping acts, must be entire quantum register
            omega: global strength of the Rabi coupling on each site.
            delta: global detuning
            phi: global interaction strength
            label: optional
        """
       
        params = [omega, delta, phi]

        super().__init__(
            name="rydberg_full",
            num_modes=num_modes,
            params=params,
            label=label,
        )

    def inverse(self):
        """Get the inverse gate by reversing the sign of all gate parameters"""
        omega_val, delta_val, phi_val = self.params[0], self.params[1], self.params[2]

        return RydbergFull(num_modes=self.num_modes, omega=-omega_val, delta=-delta_val, phi=-phi_val)

    @property
    def generator(self) -> SpinOp:
        """The generating Hamiltonian of the Rydberg Hamiltonian."""
        params = [float(param) for param in self.params]
        omega, delta, phi = params[0], params[1], params[2]
        generators = []
        
        # add generators of Rabi coupling
        if omega != 0.0:
            for i in range(self.num_modes):
                generators.append((f"X_{i}", omega))
        # add generators of detuning
        if delta != 0.0:
            for i in range(self.num_modes):
                generators.append((f"Z_{i}", delta))

        # add generators of interaction term
        if phi != 0.0:
            for i in range(self.num_modes):
                for j in range(i+1, self.num_modes):
                    coeff = phi/np.abs(i-j)**6
                    generators.append((f"Z_{i} Z_{j}", coeff))
                    generators.append((f"Z_{i}", coeff/2))
                    generators.append((f"Z_{j}", coeff/2))
       
        if not generators:
            return SpinOp("I_0", register_length=self.num_modes)
        else:
            return sum(
                coeff * SpinOp(label, register_length=self.num_modes)
                for label, coeff in generators
            )


# pylint: disable=invalid-name
@add_gate
def rydberg_full(self, omega: float, delta: float, phi: float, modes: List[int], label=None):
    """Add the combined Rydberg Gate gate to a QuantumCircuit."""
    return self.append(
        RydbergFull(num_modes=len(modes), omega=omega, delta=delta, phi=phi, label=label), qargs=modes
    )