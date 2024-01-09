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

"""A solver for time-evolution problems."""

from typing import List

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    BravyiKitaevMapper,
    ParityMapper,
)

from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit.algorithms import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers import TrotterQRTE
from qiskit.quantum_info import Statevector

from qiskit_cold_atom.applications.fermionic_evolution_problem import (
    FermionicEvolutionProblem,
)
from qiskit_cold_atom.fermions.base_fermion_backend import BaseFermionBackend


class TimeEvolutionSolver:
    """
    Solver class that solves time evolution problem by either analog simulation on fermionic
    hardware or trotterized time evolution on qubit hardware. The computation that this time
    evolution solver will do depends on the type of the backend.
    """

    MAPPER_DICT = {
        "bravyi_kitaev": BravyiKitaevMapper(),
        "jordan_wigner": JordanWignerMapper(),
        "parity": ParityMapper(),
    }

    def __init__(
        self,
        backend,
        map_type: str = None,
        trotter_steps: int = None,
        shots: int = 1000,
    ):
        """
        Initialize a time evolution solver

        Args:
            backend: The backend on which to execute the problem, may be qubit or fermionic.
            map_type: The fermion-to-qubit mapping required if a qubit backend is given
            trotter_steps: The amount of trotter steps to approximate time evolution on
                qubit backends
            shots: number of measurements taken of the constructed circuits
        """

        self.backend = backend
        self.map_type = map_type
        self.trotter_steps = trotter_steps
        self.shots = shots

    def solve(self, problem: FermionicEvolutionProblem) -> List[float]:
        """Solve the problem using the provided backend

        Args:
            problem: The FermionicEvolutionProblem to solve.

        Returns:
            A list of expectation values of the observable of the problem. This list has the
            same length as the list of times for which to compute the time evolution.
        """

        if isinstance(self.backend, BaseFermionBackend):
            qc_load = self.backend.initialize_circuit(problem.initial_state.occupations)
            circuits = problem.circuits(qc_load)

            observable_evs = self.backend.measure_observable_expectation(
                circuits, problem.observable, self.shots
            )

        else:
            # use qubit pipeline
            circuits = self.construct_qubit_circuits(problem)

            mapper = self.MAPPER_DICT[self.map_type]
            qubit_observable = mapper.map(problem.observable)
            observable_evs = [
                Statevector(qc).expectation_value(qubit_observable) for qc in circuits
            ]

        return observable_evs

    def construct_qubit_circuits(self, problem: FermionicEvolutionProblem) -> List[QuantumCircuit]:
        """Convert the problem to a trotterized qubit circuit using the specified map_type

        Args:
            problem: The fermionic evolution problem specifying the system, evolution-time
                and observable to be measured

        Returns:
            a list of quantum circuits that simulate the time evolution.
            There is one circuit for each evolution time specified in the problem'
        """

        psi_0 = problem.initial_state
        system = problem.system
        hamiltonian = system.to_fermionic_op()

        mapper = self.MAPPER_DICT[self.map_type]

        circuits = []

        # construct circuit of initial state:
        label = {f"+_{i}": 1.0 for i, bit in enumerate(psi_0.occupations_flat) if bit}
        bitstr_op = FermionicOp(label, num_spin_orbitals=len(psi_0.occupations_flat))
        qubit_op = mapper.map(bitstr_op)[0]
        init_circ = QuantumCircuit(QuantumRegister(qubit_op.num_qubits, "q"))

        for i, pauli_label in enumerate(qubit_op.paulis.to_labels()[0][::-1]):
            if pauli_label == "X":
                init_circ.x(i)
            elif pauli_label == "Y":
                init_circ.y(i)
            elif pauli_label == "Z":
                init_circ.z(i)

        for time in problem.evolution_times:
            # map fermionic hamiltonian to qubits
            qubit_hamiltonian = mapper.map(hamiltonian)
            # construct trotterization circuits
            evolution_problem = TimeEvolutionProblem(qubit_hamiltonian, time, init_circ)
            trotter_qrte = TrotterQRTE(num_timesteps=self.trotter_steps)
            evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state
            circuits.append(evolved_state)

        return circuits
