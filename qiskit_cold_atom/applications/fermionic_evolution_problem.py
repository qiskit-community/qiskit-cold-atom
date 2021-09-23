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

"""Class that holds a fermionic time-evolution problem."""

from typing import List, Union

from qiskit import QuantumCircuit
from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_cold_atom.fermions.fermionic_state import FermionicState
from qiskit_cold_atom.fermions.fermionic_basis import FermionicBasis
from qiskit_cold_atom.fermions.fermion_gate_library import FermionicGate
from qiskit_cold_atom.exceptions import QiskitColdAtomError
from qiskit_cold_atom.applications.fermi_hubbard import FermionicLattice


class FermionicEvolutionProblem:
    """
    Problem class corresponding to evaluating an observable of a fermionic system after a time
    evolution under a hamiltonian from an initial state in an occupation number representation.
    """

    def __init__(
        self,
        system: FermionicLattice,
        initial_state: FermionicState,
        evolution_times: Union[float, List[float]],
        observable: FermionicOp,
    ):
        """
        Initialize a fermionic time evolution problem.

        Args:
            system: The fermionic system under which the initial state will evolve.
            initial_state: The fermionic state at time t=0.
            evolution_times: List of times (or single time) after which the observable is measured.
            observable: The observable to measure after the time evolution, given as a FermionicOp.
                        The observable must be diagonal in the fermionic occupation number basis.

        Raises:
            QiskitColdAtomError: - If the sizes of the system, initial state and the observable
                                   do not match.
                                 - If the observables is not diagonal in the fermionic occupation number
                                   basis
        """

        if system.size != initial_state.sites:
            raise QiskitColdAtomError(
                f"The size of the system {system.size} does not match "
                f"the size of the initial state {initial_state.sites}."
            )

        if 2 * system.size != observable.register_length:
            raise QiskitColdAtomError(
                f"The fermionic modes of the system {2*system.size} do not match "
                f"the size of the observable {observable.register_length}."
            )

        # check if matrix is diagonal
        # can later be replaced when the FermionicOp from qiskit-nature has its own .to_matrix() method
        basis = FermionicBasis.from_fermionic_op(observable)
        observable_mat = FermionicGate.operator_to_mat(
            observable, num_species=1, basis=basis
        )

        if list(observable_mat.nonzero()[0]) != list(observable_mat.nonzero()[1]):
            raise QiskitColdAtomError(
                "The fermionic observable needs to be diagonal in the computational basis, "
                "as measuring general, non-diagonal observables is not yet implemented for "
                "fermionic backends. This requires non-trivial basis transformations that "
                "are in general difficult to find and depend on the backend's native gate set."
            )

        self._system = system
        self._initial_state = initial_state
        self._evolution_times = evolution_times
        self._observable = observable

    @property
    def system(self) -> FermionicLattice:
        """Return the system of the problem."""
        return self._system

    @property
    def initial_state(self) -> FermionicState:
        """Return the initial state of the system."""
        return self._initial_state

    @property
    def evolution_times(self) -> List[float]:
        """Return the evolution times to simulate."""
        return self._evolution_times

    @property
    def observable(self) -> FermionicOp:
        """Return the observable as a FermionicOp."""
        return self._observable

    def circuits(self, initial_state: QuantumCircuit) -> List[QuantumCircuit]:
        """
        The problem embedded in a quantum circuit.

        Args:
            initial_state: A quantum circuit which corresponds to the initial state for the
                time-evolution problem.

        Return:
            A list of quantum circuits. Circuit :math:`i` is a single instruction which
            corresponds to :math:`exp(-i*H*t_i)` where :math:`t_i` is the time of the
            the ith evolution time.
        """
        circuits = []

        for time in self.evolution_times:
            circ = QuantumCircuit(initial_state.num_qubits)
            circ.compose(initial_state, inplace=True)
            circ.compose(self.system.to_circuit(time), inplace=True)
            circuits.append(circ)

        return circuits
