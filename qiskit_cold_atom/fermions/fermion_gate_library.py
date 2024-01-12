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

"""Gates for fermionic backends."""

from typing import List, Optional, Union
from copy import deepcopy
from scipy.sparse.linalg import expm
from scipy.sparse import csc_matrix
import numpy as np

from qiskit.circuit import Instruction, Gate
from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_cold_atom.exceptions import QiskitColdAtomError
from qiskit_cold_atom import add_gate
from qiskit_cold_atom.fermions.fermionic_basis import FermionicBasis


class FermionicGate(Gate):
    """Unitary gates for fermionic circuits."""

    def __init__(
        self,
        name: str,
        num_modes: int,
        params: Optional[List] = None,
        label: Optional[str] = None,
        generator: Optional[FermionicOp] = None,
    ) -> None:
        """Create a new fermionic gate.

        Args:
            name: The Qobj name of the gate.
            num_modes: The number of fermionic modes the gate acts on.
            params: A list of parameters.
            label: An optional label for the gate.
            generator: The generating Hamiltonian of the gate unitary given as a FermionicOp
        """

        self._generator = generator

        if params is None:
            params = []

        self.num_modes = num_modes

        super().__init__(name=name, num_qubits=num_modes, params=params, label=label)

    def power(self, exponent: float):
        """Creates a fermionic gate as `gate^exponent`

        Args:
            exponent (float): The exponent with which the gate is exponentiated

        Returns:
            FermionicGate: To which `.generator` is self.generator*exponent.

        Raises:
            QiskitColdAtomError: If the gate has no defined generator.
        """
        if self.generator is None:
            raise QiskitColdAtomError(
                "Gate can not be exponentiated if the gate generator is not defined."
            )
        # the generator of the exponentiated gate is the old generator times the exponent
        exp_generator = exponent * self.generator.simplify()

        exp_params = None if not self.params else [exponent * param for param in self.params]

        exp_label = None if not self.label else self.label + f"^{exponent}"

        return FermionicGate(
            name=self.name + f"^{exponent}",
            num_modes=self.num_modes,
            params=exp_params,
            label=exp_label,
            generator=exp_generator,
        )

    # pylint: disable=arguments-differ
    def to_matrix(self, num_species: int = 1, basis: Optional[FermionicBasis] = None) -> np.ndarray:
        """Return a Numpy.array for the gate unitary matrix. This function will compute :math:`exp(-i H)`
        where :math:`H` is the generator of the gate.

        Args:
            num_species: Number of fermion species which defaults to 1.
            basis: The basis in which to return the matrix. If None is given then the matrix will
                be returned in the full basis.

        Raises:
            QiskitColdAtomError: If the generator of the gate is not hermitian.

        Returns:
            np.array: The array of the gate unitary over the full fock basis, with states ordered like
            00...0, 00...1, ..., 11...0, 11...1
        """
        # This can be simplified when the FermionicOp gets a .to_matrix() method in a future release of
        # qiskit-nature
        if self.generator is None:
            raise QiskitColdAtomError(
                "Matrix of gate can not be computed if the gate generator is not defined."
            )

        generator_mat = self.operator_to_mat(self.generator, num_species, basis)

        if (generator_mat.H - generator_mat).count_nonzero() != 0:
            raise QiskitColdAtomError("generator of unitary gate is not hermitian!")

        unitary_mat = expm(-1j * generator_mat)

        return unitary_mat.toarray()

    @staticmethod
    def operator_to_mat(
        generator: FermionicOp, num_species: int, basis: Optional[FermionicBasis] = None
    ) -> csc_matrix:
        """Compute the matrix representation of the fermion operator.

        Args:
            generator: fermion operator of which to compute the matrix representation.
            num_species: Number of fermion species which defaults to 1.
            basis: The basis in which to return the matrix. If None is given then the matrix will
                be returned in the full basis.

        Returns:
            scipy.sparse matrix of the Hamiltonian.

        Raises:
            QiskitColdAtomError: If the type of the generator is not a FermionicOp.
            QiskitColdAtomError: If the fermion operator does not match the expected shape.
        """

        if not isinstance(generator, FermionicOp):
            raise QiskitColdAtomError(
                f"Expected FermionicOp; got {type(generator).__name__} instead."
            )

        if basis is None:
            basis = FermionicBasis.from_fermionic_op(generator)

        csc_data, csc_col, csc_row = [], [], []

        basis_occupations = basis.get_occupations()

        # loop over all individual terms in the generators
        for term in generator.to_list(display_format="dense"):
            opstring = term[0]
            prefactor = term[1]

            if len(opstring) != num_species * basis.sites:
                raise QiskitColdAtomError(
                    f"Length of operator {opstring} must match the number of "
                    f"modes in the basis {num_species*basis.sites}"
                )

            # loop over all basis states
            for i_basis, occupations in enumerate(basis_occupations):
                new_occupations = deepcopy(occupations)
                mapped_to_zero = (
                    False  # boolean flag to check whether the basis state is mapped to zero
                )
                sign = 1

                # in reverse, loop over all individual fermionic creators/annihilators in the opstring:
                for k, symbol in reversed(list(enumerate(opstring))):
                    if symbol == "I":
                        continue

                    if symbol == "-":
                        # If this mode is not occupied, the action of '-' on this state is zero
                        if occupations[k] == 0:
                            mapped_to_zero = True
                            break
                        sign *= (-1) ** sum(occupations[:k])
                        new_occupations[k] = 0

                    elif symbol == "+":
                        # If this mode is already occupied, the action of '+' on this state is zero
                        if occupations[k] == 1:
                            mapped_to_zero = True
                            break
                        sign *= (-1) ** sum(occupations[:k])
                        new_occupations[k] = 1

                    elif symbol == "N":
                        # If this mode is not occupied, the action of 'N' on this state is zero
                        if occupations[k] == 0:
                            mapped_to_zero = True
                            break

                    elif symbol == "E":
                        # If this mode is occupied, the action of 'E' on this state is zero
                        if occupations[k] == 1:
                            mapped_to_zero = True
                            break

                if not mapped_to_zero:
                    # find the index of the new basis state that the operator strings maps to
                    j_basis = basis_occupations.index(new_occupations)

                    csc_data.append(sign * prefactor)
                    csc_row.append(j_basis)
                    csc_col.append(i_basis)

        return csc_matrix(
            (csc_data, (csc_row, csc_col)),
            shape=(basis.dimension, basis.dimension),
            dtype=complex,
        )

    def control(
        self,
        num_ctrl_qubits: Optional[int] = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
    ):
        """Overwrite control method which is supposed to return a controlled version of the gate.
        This is not applicable in the fermionic setting."""
        raise QiskitColdAtomError("Fermionic gates have no controlled version")

    @property
    def generator(self) -> FermionicOp:
        """The Hamiltonian that generates the unitary of the gate, given as a FermionicOp"""
        return self._generator


class FermiHubbard(FermionicGate):
    r"""
    Global 1D-Fermi-Hubbard dynamic consisting of the hopping, interaction and local phase gates.

    The generating Hamiltonian of the Fermi-Hubbard gate is

    :math:`H = \sum_{i=1,\sigma}^{L-1} - J_i (f^\dagger_{i,\sigma} f_{i+1,\sigma}
    + f^\dagger_{i+1,\sigma} f_{i,\sigma})
    + U \sum_{i=1}^{L} n_{i,\uparrow} n_{i,\downarrow}
    + \sum_{i=1,\sigma}^{L} \mu_i n_{i,\sigma}`

    where :math:`i` indexes the mode, :math:`\sigma` indexes the spin, :math:`L` gives the total number
    of sites, :math:`J_i` are the hopping strengths, :math:`U` is the interaction strength and
    :math:`\mu_i` are the local potentials that lead to local phases.

    **Circuit symbol:**

        .. parsed-literal::

                 ┌──────────────┐
            q_0: ┤   FHubbard   ├
                 └──────────────┘

    """

    def __init__(self, num_modes: int, j: List[float], u: float, mu: List[float], label=None):
        """Initialize a global Fermi-Hubbard gate

        Args:
            num_modes: number of tweezers on which the hopping acts, must be entire quantum register
            j: list of hopping strengths between the tweezer. j[0] gives the strength of hopping
                between wires 0 and 1, j[1] gives the strength of hopping between wires 1 and 2, etc.,
                so len(j) has to be of length num_wires-1
            label: optional
            u: global interaction strength parameter
            mu: list of parameters that tune the local phases, must be on length num_wires
            label: optional

        Raises:
            QiskitColdAtomError:
                - If the given num_modes is not an even integer.
                - If length of j not num_modes/2 - 1.
        """
        if not (isinstance(num_modes, int) and num_modes % 2 == 0):
            raise QiskitColdAtomError("num_modes has to be even integer")
        if not len(j) == (num_modes / 2 - 1):
            raise QiskitColdAtomError("j has to be a list of length num_modes/2 -1")

        param_list = [j, [u], mu]
        params = [item for sublist in param_list for item in sublist]

        super().__init__(
            name="fhubbard",
            num_modes=num_modes,
            params=params,
            label=label,
        )

    def inverse(self):
        """Get the inverse gate by reversing the sign of all gate parameters"""
        j_val, u_val, mu_vals = self.params[0], self.params[1], self.params[2]

        return FermiHubbard(num_modes=self.num_modes, j=-j_val, u=-u_val, mu=-mu_vals)

    @property
    def generator(self) -> FermionicOp:
        """The generating Hamiltonian of the FH Gate."""
        params = [float(param) for param in self.params]
        generators = []
        sites = self.num_modes // 2
        # add generators of hopping term
        if not all(j == 0.0 for j in params[: sites - 1]):
            for i in range(sites - 1):
                generators.append((f"+_{i} -_{i+1}", -1 * params[i]))
                generators.append((f"-_{i} +_{i+1}", params[i]))
                generators.append((f"+_{i+sites} -_{i+sites+1}", -1 * params[i]))
                generators.append((f"-_{i+sites} +_{i+sites+1}", params[i]))
        # add generators of interaction term
        if params[sites - 1] != 0.0:
            for i in range(sites):
                generators.append((f"N_{i} N_{i + sites}", params[sites - 1]))
        # add generators of local phase term
        if not all(muval == 0.0 for muval in params[sites:]):
            for i in range(sites):
                generators.append((f"N_{i}", float(self.params[i + sites])))
                generators.append((f"N_{i+sites}", float(self.params[i + sites])))

        if not generators:
            return FermionicOp("I_0", register_length=self.num_modes)
        else:
            return sum(
                coeff * FermionicOp(label, register_length=self.num_modes)
                for label, coeff in generators
            )


# pylint: disable=invalid-name
@add_gate
def fhubbard(self, j: List[float], u: float, mu: List[float], modes: List[int], label=None):
    """Add the FermiHubbard gate to a QuantumCircuit."""
    return self.append(
        FermiHubbard(num_modes=len(modes), j=j, u=u, mu=mu, label=label), qargs=modes
    )


class Hop(FermionicGate):
    r"""
    Hopping of particles to neighbouring wells due to tunneling.

    The generating Hamiltonian of the hopping gate is

    :math:`H = \sum_{i=1,\sigma}^{L-1} - J_i (f^\dagger_{i,\sigma} f_{i+1,\sigma}
    + f^\dagger_{i+1,\sigma} f_{i,\sigma})`

    where :math:`i` indexes the mode, :math:`\sigma`
    indexes the spin, :math:`L` gives the total number of sites and :math:`J_i` are the hopping strengths
    """

    def __init__(self, num_modes, j: List[float], label=None):
        """
        Initialize hopping gate
        Args:
            num_modes: number of fermionic modes that are connected by the hopping
            (= 2* number of tweezers)
            j: list of hopping strengths between the tweezer. j[0] gives the strength of hopping
            between wires 0 and 1,
            j[1] gives the strength of hopping between wires 1 and 2, etc., so len(j) has to be
            of length num_wires-1
            label: optional
        Raises:
            QiskitColdAtomError: given num_modes not even integer
            QiskitColdAtomError: length of j not num_modes/2 - 1
        """
        if not (isinstance(num_modes, int) and num_modes % 2 == 0):
            raise QiskitColdAtomError("num_modes has to be even integer")
        if not len(j) == (num_modes / 2 - 1):
            raise QiskitColdAtomError("j has to be a list of length num_modes/2 -1")

        super().__init__(name="fhop", num_modes=num_modes, params=j, label=label)

    def inverse(self):
        """Get inverse gate by reversing the sign of all hopping strengths"""
        return Hop(num_modes=self.num_modes, j=self.params)

    @property
    def generator(self) -> FermionicOp:
        """The generating Hamiltonian of the hopping gate."""
        generator = FermiHubbard(
            num_modes=self.num_modes, j=self.params, u=0.0, mu=[0.0]
        ).generator.simplify()

        if generator == 0:
            return FermionicOp("I_0", register_length=self.num_modes)
        else:
            return generator


@add_gate
def fhop(self, j: List[float], modes: List[int], label=None):
    """Add the hopping gate to a QuantumCircuit."""
    return self.append(Hop(num_modes=len(modes), j=j, label=label), qargs=modes)


class Interaction(FermionicGate):
    r"""On-site interaction of particles of opposite spin species on the same site.

    The generating Hamiltonian of the interaction gate is

    :math:`H = U \sum_{i=1}^{L} n_{i,\uparrow} n_{i,\downarrow}`

    where :math:`i` indexes the mode,
    :math:`L` gives the total number of sites and :math:`U` is the interaction strength
    """

    def __init__(self, num_modes: int, u: float, label=None):
        """Initialize interaction gate.

        Args:
            num_modes: number of modes on which the gate acts
            u: global interaction strength parameter
            label: optional

        Raises:
            QiskitColdAtomError: If the number of wires the gate acts on is uneven
        """

        if not num_modes % 2 == 0:
            raise QiskitColdAtomError(f"number of modes must be even, {num_modes} given.")

        super().__init__(
            name="fint",
            num_modes=num_modes,
            params=[u],
            label=label,
        )

    def inverse(self):
        """Get inverse gate by reversing the sign of the interaction parameter"""
        return Interaction(num_modes=self.num_modes, u=-self.params[0])

    @property
    def generator(self) -> FermionicOp:
        """The generating Hamiltonian of the interaction gate."""
        generator = FermiHubbard(
            num_modes=self.num_modes,
            j=[0.0] * (int(self.num_modes / 2) - 1),
            u=self.params[0],
            mu=[0.0],
        ).generator.simplify()

        if generator == 0:
            return FermionicOp("I_0", register_length=self.num_modes)
        else:
            return generator


@add_gate
def fint(self, u: float, modes: List[int], label=None):
    """Add the interaction gate to a QuantumCircuit."""
    return self.append(Interaction(num_modes=len(modes), u=u, label=label), qargs=modes)


class Phase(FermionicGate):
    r"""
    Applying a local phase to individual tweezers through an external potential

    The generating Hamiltonian of the local phase gate is

    :math:`H = \sum_{i=1,\sigma}^{L} \mu_i n_{i,\sigma}`

    where :math:`i` indexes the mode,
    :math:`\sigma` indexes the spin, :math:`L` gives the total number of sites and
    :math:`\mu_i` are the local potentials that lead to local phases
    """

    def __init__(self, num_modes: int, mu: List[float], label=None):
        """
        Initialize a LocalPhase gate
        Args:
            num_modes: number of modes on which the local potential acts
            mu: list of parameters that tune the local phases, must be of length num_modes/2
            label: optional
        Raises:
            QiskitColdAtomError: If the length of mu does not match the given wire count num_modes
        """

        if not num_modes % 2 == 0:
            raise QiskitColdAtomError(f"number of modes must be even, {num_modes} given.")

        if not len(mu) == num_modes / 2:
            raise QiskitColdAtomError(
                f"list of pre-factors {mu} has to be same dimension as the wire count "
                f"of the gate {num_modes}"
            )

        super().__init__(
            name="fphase",
            num_modes=num_modes,
            params=mu,
            label=label,
        )

    def inverse(self):
        """Get inverse gate by reversing the sign of all potentials"""
        return Phase(num_modes=self.num_modes, mu=[-1 * param for param in self.params])

    @property
    def generator(self) -> FermionicOp:
        """The generating Hamiltonian of the local phase gate."""
        generator = FermiHubbard(
            num_modes=self.num_modes,
            j=[0.0] * (int(self.num_modes / 2) - 1),
            u=0.0,
            mu=self.params,
        ).generator.simplify()

        if generator == 0:
            return FermionicOp("I_0", register_length=self.num_modes)
        else:
            return generator


@add_gate
def fphase(self, mu: List[float], modes: List[int], label=None):
    """Add the local phase gate to a QuantumCircuit."""
    return self.append(Phase(num_modes=len(modes), mu=mu, label=label), qargs=modes)


class FRXGate(FermionicGate):
    r"""X-rotation between the spin-up and spin-down state at one tweezer site.

    The generating Hamiltonian of the FermionRx gate is

    :math:`H = \phi (f^\dagger_{x,\uparrow} f_{x,\downarrow} + f^\dagger_{x,\downarrow} f_{x,\uparrow})`

    where :math:`x` is the index of the mode and :math:`\phi` is the free gate parameter
    """

    def __init__(self, phi: float, label=None):
        """Initialize a FermionRX gate

        Args:
            phi: angle of the rotation
            label: optional
        """

        super().__init__(name="frx", num_modes=2, params=[phi], label=label)

    def inverse(self):
        """Get inverse gate by inverting the sign of the rotation angle"""
        return FRXGate(-self.params[0])

    @property
    def generator(self) -> FermionicOp:
        """The generating Hamiltonian of the FermionRX gate."""
        op = float(self.params[0]) * FermionicOp("+_0 -_1", register_length=2) - float(
            self.params[0]
        ) * FermionicOp("-_0 +_1", register_length=2)
        return op


@add_gate
def frx(self, phi: float, wires: List[int]):
    """Add the FermionRX gate to a QuantumCircuit."""
    return self.append(FRXGate(phi=phi), qargs=wires)


class FRYGate(FermionicGate):
    r"""Y-rotation between the spin-up and spin-down state at one tweezer site.

    The generating Hamiltonian of the FermionRy gate is

    :math:`H = i \phi (f^\dagger_{x,\downarrow} f_{x,\uparrow}-f^\dagger_{x,\uparrow} f_{x,\downarrow})`

    where :math:`x` is the index of the mode and :math:`\phi` is the free gate parameter
    """

    def __init__(self, phi, label=None):
        """
        Initialize a FermionRY gate
        Args:
            phi: angle of the rotation
            label: optional
        """
        super().__init__(name="fry", num_modes=2, params=[phi], label=label)

    def inverse(self):
        """Get inverse gate by inverting the sign of the rotation angle"""
        return FRYGate(-self.params[0])

    @property
    def generator(self) -> FermionicOp:
        """The generating Hamiltonian of the FermionRY gate."""
        op = -1j * float(self.params[0]) * FermionicOp("+_0 -_1", register_length=2) - 1j * float(
            self.params[0]
        ) * FermionicOp("-_0 +_1", register_length=2)
        return op


@add_gate
def fry(self, phi: float, wires: List[int]):
    """Add the FermionRY gate to a QuantumCircuit."""
    return self.append(FRYGate(phi=phi), qargs=wires)


class FRZGate(FermionicGate):
    r"""Z-rotation between the spin-up and spin-down state at one tweezer site.

    The generating Hamiltonian of the FermionRz gate is

    :math:`H = \phi (f^\dagger_{x,\uparrow} f_{x,\uparrow} - f^\dagger_{x,\downarrow} f_{x,\downarrow})`

    where :math:`x` is the index of the mode and :math:`\phi` is the free gate parameter.
    """

    def __init__(self, phi, label=None):
        """
        Initialize a FermionRZ gate
        Args:
            phi: angle of the rotation
            label: optional
        """
        super().__init__(name="frz", num_modes=2, params=[phi], label=label)

    def inverse(self):
        """Get inverse gate by inverting the sign of the rotation angle"""
        return FRZGate(-self.params[0])

    @property
    def generator(self) -> FermionicOp:
        """The generating Hamiltonian of the FermionRZ gate."""
        op = float(self.params[0]) * FermionicOp("N_0", register_length=2) - float(
            self.params[0]
        ) * FermionicOp("N_1", register_length=2)
        return op


@add_gate
def frz(self, phi: float, wires: List[int]):
    """Add the FermionRZ gate to a QuantumCircuit."""
    return self.append(FRZGate(phi=phi), qargs=wires)


class LoadFermions(Instruction):
    """
    LoadFermions places a particle in an empty fermionic mode.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────┐
        q_0: ┤ Load ├
             └──────┘
    """

    def __init__(self):
        """Initialise new load instruction."""
        super().__init__(name="load", num_qubits=1, num_clbits=0, params=[])


@add_gate
def fload(self, wire):
    """Add the load fermion gate to a QuantumCircuit."""
    return self.append(LoadFermions(), [wire], [])
